"""
INT8 Quantization utilities for CS Tutoring LLM.

Implements model quantization for efficient inference:
- Dynamic quantization (no calibration needed)
- Static quantization (with calibration)
- Quantization-aware training support

References:
- Jacob et al., 2018. "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference"
- Zafrir et al., 2019. "Q8BERT: Quantized 8Bit BERT"
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    Manager for model quantization operations.
    
    Supported methods:
    1. Dynamic Quantization:
       - Quantizes weights to INT8 at runtime
       - No calibration data needed
       - Good for models with variable input sizes
       
    2. Static Quantization:
       - Pre-computed quantization parameters
       - Requires calibration data
       - Better performance but more complex setup
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.original_model = None
        self.quantized_model = None
        self.calibration_data = None
    
    def dynamic_quantization(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Dynamic quantization:
        - Weights are quantized to INT8
        - Activations remain in FP32
        - Quantization happens at runtime
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype (qint8 or quint8)
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        self.original_model = model
        
        # Quantize Linear and LSTM layers
        quantized = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        
        self.quantized_model = quantized
        logger.info("Dynamic quantization complete")
        
        return quantized
    
    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        Calculate model size in memory.
        
        Returns:
            Dictionary with size information
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        # FP32: 4 bytes per parameter
        # INT8: 1 byte per parameter
        
        param_bytes = 0
        for p in model.parameters():
            if p.dtype == torch.float32:
                param_bytes += p.numel() * 4
            elif p.dtype == torch.float16:
                param_bytes += p.numel() * 2
            elif p.dtype in [torch.int8, torch.qint8, torch.quint8]:
                param_bytes += p.numel() * 1
            else:
                param_bytes += p.numel() * 4  # Default to float32
        
        # Buffer sizes
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "param_memory_mb": param_bytes / (1024 * 1024),
            "buffer_memory_mb": buffer_bytes / (1024 * 1024),
            "total_memory_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
        }
    
    def compare_sizes(self) -> Dict[str, Any]:
        """
        Compare sizes of original and quantized models.
        
        Returns:
            Comparison dictionary
        """
        if self.original_model is None or self.quantized_model is None:
            return {"error": "Models not available for comparison"}
        
        original_size = self.get_model_size(self.original_model)
        quantized_size = self.get_model_size(self.quantized_model)
        
        compression_ratio = (
            original_size["total_memory_mb"] / 
            max(quantized_size["total_memory_mb"], 1e-6)
        )
        
        return {
            "original": original_size,
            "quantized": quantized_size,
            "compression_ratio": compression_ratio,
            "memory_saved_mb": (
                original_size["total_memory_mb"] - 
                quantized_size["total_memory_mb"]
            ),
            "memory_saved_percent": (
                (1 - quantized_size["total_memory_mb"] / 
                 max(original_size["total_memory_mb"], 1e-6)) * 100
            ),
        }
    
    def benchmark_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Timing statistics
        """
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)
        
        # Create attention mask if needed
        attention_mask = torch.ones_like(sample_input)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                try:
                    _ = model(sample_input, attention_mask)
                except TypeError:
                    _ = model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                try:
                    _ = model(sample_input, attention_mask)
                except TypeError:
                    _ = model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        return {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "median_latency_ms": np.median(times),
            "p95_latency_ms": np.percentile(times, 95),
            "p99_latency_ms": np.percentile(times, 99),
            "throughput_per_sec": 1000 / np.mean(times),
        }
    
    def full_comparison(
        self,
        sample_input: torch.Tensor,
        num_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Run full comparison between original and quantized models.
        
        Returns:
            Comprehensive comparison results
        """
        if self.original_model is None or self.quantized_model is None:
            return {"error": "Models not available for comparison"}
        
        logger.info("Running full quantization comparison...")
        
        # Size comparison
        size_comparison = self.compare_sizes()
        
        # Speed comparison
        logger.info("Benchmarking original model...")
        original_speed = self.benchmark_inference(
            self.original_model.cpu(),
            sample_input.cpu(),
            num_iterations
        )
        
        logger.info("Benchmarking quantized model...")
        quantized_speed = self.benchmark_inference(
            self.quantized_model.cpu(),
            sample_input.cpu(),
            num_iterations
        )
        
        speedup = original_speed["mean_latency_ms"] / max(quantized_speed["mean_latency_ms"], 1e-6)
        
        return {
            "size_comparison": size_comparison,
            "original_speed": original_speed,
            "quantized_speed": quantized_speed,
            "speedup_factor": speedup,
            "summary": {
                "memory_reduction": f"{size_comparison['memory_saved_percent']:.1f}%",
                "speedup": f"{speedup:.2f}x",
                "original_latency_ms": f"{original_speed['mean_latency_ms']:.2f}",
                "quantized_latency_ms": f"{quantized_speed['mean_latency_ms']:.2f}",
            }
        }
    
    def save_quantized_model(self, path: str):
        """Save quantized model to disk."""
        if self.quantized_model is None:
            raise ValueError("No quantized model available")
        
        torch.save(self.quantized_model.state_dict(), path)
        logger.info(f"Quantized model saved to {path}")
    
    def load_quantized_model(
        self,
        model_class: type,
        path: str,
        **model_kwargs
    ) -> nn.Module:
        """Load quantized model from disk."""
        # Create model instance
        model = model_class(**model_kwargs)
        
        # Apply quantization structure
        quantized = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Load weights
        state_dict = torch.load(path, map_location='cpu')
        quantized.load_state_dict(state_dict)
        
        return quantized


def generate_quantization_report(comparison: Dict[str, Any]) -> str:
    """Generate human-readable quantization report."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("INT8 QUANTIZATION COMPARISON REPORT")
    report.append("=" * 70)
    
    # Summary
    summary = comparison.get("summary", {})
    report.append("\n## Summary")
    report.append("-" * 50)
    report.append(f"Memory Reduction: {summary.get('memory_reduction', 'N/A')}")
    report.append(f"Speedup Factor: {summary.get('speedup', 'N/A')}")
    
    # Size details
    size = comparison.get("size_comparison", {})
    report.append("\n## Model Size")
    report.append("-" * 50)
    
    orig = size.get("original", {})
    quant = size.get("quantized", {})
    
    report.append(f"{'Metric':<30} {'Original':<15} {'Quantized':<15}")
    report.append("-" * 60)
    report.append(f"{'Parameters':<30} {orig.get('total_parameters', 0):>13,} {quant.get('total_parameters', 0):>13,}")
    report.append(f"{'Memory (MB)':<30} {orig.get('total_memory_mb', 0):>13.2f} {quant.get('total_memory_mb', 0):>13.2f}")
    report.append(f"{'Compression Ratio':<30} {'1.00x':>13} {size.get('compression_ratio', 1):>12.2f}x")
    
    # Speed details
    report.append("\n## Inference Speed (CPU)")
    report.append("-" * 50)
    
    orig_speed = comparison.get("original_speed", {})
    quant_speed = comparison.get("quantized_speed", {})
    
    report.append(f"{'Metric':<30} {'Original':<15} {'Quantized':<15}")
    report.append("-" * 60)
    report.append(f"{'Mean Latency (ms)':<30} {orig_speed.get('mean_latency_ms', 0):>13.2f} {quant_speed.get('mean_latency_ms', 0):>13.2f}")
    report.append(f"{'P95 Latency (ms)':<30} {orig_speed.get('p95_latency_ms', 0):>13.2f} {quant_speed.get('p95_latency_ms', 0):>13.2f}")
    report.append(f"{'Throughput (samples/sec)':<30} {orig_speed.get('throughput_per_sec', 0):>13.2f} {quant_speed.get('throughput_per_sec', 0):>13.2f}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test quantization
    from model import TutoringModel
    from transformers import AutoTokenizer
    
    config = get_config()
    
    # Create model
    logger.info("Creating model...")
    model = TutoringModel(config)
    
    # Create sample input
    tokenizer = model.tokenizer
    sample_text = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=128)
    
    # Quantization manager
    manager = QuantizationManager(config)
    
    # Apply quantization
    quantized_model = manager.dynamic_quantization(model)
    
    # Full comparison
    comparison = manager.full_comparison(
        inputs["input_ids"],
        num_iterations=20
    )
    
    # Print report
    print(generate_quantization_report(comparison))
