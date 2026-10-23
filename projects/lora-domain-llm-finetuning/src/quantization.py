"""Quantization module for INT8 optimization."""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert
from typing import Dict, Any, Optional, Tuple, Callable
import time
import numpy as np
from pathlib import Path
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handle model quantization for CPU inference optimization."""
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.quantized_model = None
        self.device = "cpu"
    
    def dynamic_quantize(self) -> nn.Module:
        """Apply dynamic INT8 quantization (recommended for transformers)."""
        logger.info("Applying dynamic INT8 quantization...")
        
        model_cpu = self.original_model.to("cpu")
        model_cpu.eval()
        
        self.quantized_model = quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization complete")
        return self.quantized_model
    
    def get_model_size_bytes(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        return size
    
    def compare_sizes(self) -> Dict[str, float]:
        """Compare FP32 and INT8 model sizes."""
        fp32_size = self.get_model_size_bytes(self.original_model)
        
        if self.quantized_model is None:
            self.dynamic_quantize()
        
        int8_size = self.get_model_size_bytes(self.quantized_model)
        
        return {
            "fp32_size_mb": fp32_size / (1024 * 1024),
            "int8_size_mb": int8_size / (1024 * 1024),
            "compression_ratio": fp32_size / int8_size if int8_size > 0 else 0,
            "size_reduction_pct": (1 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0
        }
    
    def benchmark_latency(
        self,
        input_fn: Callable[[], Dict[str, torch.Tensor]],
        num_runs: int = 50,
        warmup_runs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark inference latency for FP32 and INT8 models."""
        results = {}
        
        for name, model in [("fp32", self.original_model), 
                           ("int8", self.quantized_model or self.dynamic_quantize())]:
            model.eval()
            model.to("cpu")
            
            for _ in range(warmup_runs):
                inputs = input_fn()
                with torch.no_grad():
                    model(**{k: v.to("cpu") for k, v in inputs.items()})
            
            latencies = []
            for _ in range(num_runs):
                inputs = input_fn()
                start = time.perf_counter()
                with torch.no_grad():
                    model(**{k: v.to("cpu") for k, v in inputs.items()})
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            latencies = np.array(latencies)
            results[name] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "throughput_samples_per_sec": float(1000.0 / np.mean(latencies))
            }
        
        results["speedup"] = results["fp32"]["mean_ms"] / results["int8"]["mean_ms"]
        
        return results
    
    def measure_peak_memory(
        self,
        input_fn: Callable[[], Dict[str, torch.Tensor]],
        num_runs: int = 10
    ) -> Dict[str, float]:
        """Measure peak memory usage for both models."""
        import tracemalloc
        
        results = {}
        
        for name, model in [("fp32", self.original_model),
                           ("int8", self.quantized_model or self.dynamic_quantize())]:
            model.eval()
            model.to("cpu")
            
            tracemalloc.start()
            
            for _ in range(num_runs):
                inputs = input_fn()
                with torch.no_grad():
                    model(**{k: v.to("cpu") for k, v in inputs.items()})
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[f"{name}_peak_mb"] = peak / (1024 * 1024)
            results[f"{name}_current_mb"] = current / (1024 * 1024)
        
        return results
    
    def full_comparison(
        self,
        input_fn: Callable[[], Dict[str, torch.Tensor]],
        evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
        num_latency_runs: int = 50
    ) -> Dict[str, Any]:
        """Run full FP32 vs INT8 comparison."""
        if self.quantized_model is None:
            self.dynamic_quantize()
        
        size_comparison = self.compare_sizes()
        
        latency_comparison = self.benchmark_latency(
            input_fn, 
            num_runs=num_latency_runs
        )
        
        memory_comparison = self.measure_peak_memory(input_fn)
        
        accuracy_comparison = {}
        if evaluate_fn:
            logger.info("Evaluating FP32 model accuracy...")
            accuracy_comparison["fp32_metric"] = evaluate_fn(self.original_model)
            logger.info("Evaluating INT8 model accuracy...")
            accuracy_comparison["int8_metric"] = evaluate_fn(self.quantized_model)
            if accuracy_comparison["fp32_metric"] != 0:
                accuracy_comparison["metric_change_pct"] = (
                    (accuracy_comparison["int8_metric"] - accuracy_comparison["fp32_metric"]) 
                    / abs(accuracy_comparison["fp32_metric"]) * 100
                )
        
        return {
            "size": size_comparison,
            "latency": latency_comparison,
            "memory": memory_comparison,
            "accuracy": accuracy_comparison
        }


def generate_quantization_report(comparison: Dict[str, Any]) -> str:
    """Generate markdown report from comparison results."""
    lines = [
        "## FP32 vs INT8 Quantization Comparison\n",
        "### Model Size",
        "| Precision | Size (MB) | Compression |",
        "|-----------|-----------|-------------|",
        f"| FP32 | {comparison['size']['fp32_size_mb']:.2f} | 1.00x |",
        f"| INT8 | {comparison['size']['int8_size_mb']:.2f} | {comparison['size']['compression_ratio']:.2f}x |",
        "",
        "### Inference Latency (ms/sample)",
        "| Precision | Mean | P50 | P95 | Throughput |",
        "|-----------|------|-----|-----|------------|",
    ]
    
    for precision in ["fp32", "int8"]:
        lat = comparison["latency"][precision]
        lines.append(
            f"| {precision.upper()} | {lat['mean_ms']:.2f} | {lat['p50_ms']:.2f} | "
            f"{lat['p95_ms']:.2f} | {lat['throughput_samples_per_sec']:.1f}/s |"
        )
    
    lines.extend([
        f"\n**Speedup:** {comparison['latency']['speedup']:.2f}x",
        "",
        "### Peak Memory Usage",
        "| Precision | Peak RAM (MB) |",
        "|-----------|--------------|",
        f"| FP32 | {comparison['memory']['fp32_peak_mb']:.2f} |",
        f"| INT8 | {comparison['memory']['int8_peak_mb']:.2f} |",
    ])
    
    if comparison.get("accuracy"):
        acc = comparison["accuracy"]
        lines.extend([
            "",
            "### Accuracy/Metric Comparison",
            "| Precision | Metric Value | Change |",
            "|-----------|-------------|--------|",
            f"| FP32 | {acc.get('fp32_metric', 'N/A'):.4f} | - |",
            f"| INT8 | {acc.get('int8_metric', 'N/A'):.4f} | {acc.get('metric_change_pct', 0):.2f}% |"
        ])
    
    return "\n".join(lines)


def export_onnx_int8(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    output_path: str
) -> Optional[str]:
    """Export model to ONNX format (preparation for INT8 ONNX Runtime)."""
    try:
        import onnx
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        dummy_input = tuple(v for v in sample_input.values())
        input_names = list(sample_input.keys())
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={name: {0: "batch_size"} for name in input_names},
            opset_version=14
        )
        
        logger.info(f"ONNX model exported to {output_path}")
        return str(output_path)
        
    except ImportError:
        logger.warning("ONNX not installed, skipping ONNX export")
        return None
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        return None
