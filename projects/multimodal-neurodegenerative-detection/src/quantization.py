"""Quantization module for INT8 optimization."""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from typing import Dict, Any, Callable
import time
import numpy as np
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handle model quantization for CPU inference optimization."""
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.quantized_model = None
    
    def dynamic_quantize(self) -> nn.Module:
        """Apply dynamic INT8 quantization."""
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
        input_fn: Callable,
        num_runs: int = 50,
        warmup_runs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark inference latency."""
        results = {}
        
        for name, model in [("fp32", self.original_model),
                           ("int8", self.quantized_model or self.dynamic_quantize())]:
            model.eval()
            model.to("cpu")
            
            for _ in range(warmup_runs):
                inputs = input_fn()
                with torch.no_grad():
                    model(inputs["image"], inputs["clinical"])
            
            latencies = []
            for _ in range(num_runs):
                inputs = input_fn()
                start = time.perf_counter()
                with torch.no_grad():
                    model(inputs["image"], inputs["clinical"])
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            latencies = np.array(latencies)
            results[name] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "throughput_per_sec": float(1000.0 / np.mean(latencies))
            }
        
        results["speedup"] = results["fp32"]["mean_ms"] / results["int8"]["mean_ms"]
        
        return results
    
    def full_comparison(
        self,
        input_fn: Callable,
        evaluate_fn: Callable = None,
        num_latency_runs: int = 50
    ) -> Dict[str, Any]:
        """Run full comparison."""
        if self.quantized_model is None:
            self.dynamic_quantize()
        
        size_comparison = self.compare_sizes()
        latency_comparison = self.benchmark_latency(input_fn, num_runs=num_latency_runs)
        
        accuracy_comparison = {}
        if evaluate_fn:
            accuracy_comparison["fp32_metric"] = evaluate_fn(self.original_model)
            accuracy_comparison["int8_metric"] = evaluate_fn(self.quantized_model)
        
        return {
            "size": size_comparison,
            "latency": latency_comparison,
            "accuracy": accuracy_comparison
        }
