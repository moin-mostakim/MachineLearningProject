"""Quantization utilities for INT8 optimization."""

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
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.quantized_model = None
    
    def dynamic_quantize(self) -> nn.Module:
        logger.info("Applying INT8 quantization...")
        self.quantized_model = quantize_dynamic(
            self.original_model.cpu().eval(), {nn.Linear}, dtype=torch.qint8
        )
        return self.quantized_model
    
    def get_model_size_bytes(self, model: nn.Module) -> int:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        return size
    
    def compare_sizes(self) -> Dict[str, float]:
        fp32_size = self.get_model_size_bytes(self.original_model)
        if not self.quantized_model:
            self.dynamic_quantize()
        int8_size = self.get_model_size_bytes(self.quantized_model)
        
        return {
            "fp32_size_mb": fp32_size / (1024 * 1024),
            "int8_size_mb": int8_size / (1024 * 1024),
            "compression_ratio": fp32_size / int8_size if int8_size > 0 else 0,
            "size_reduction_pct": (1 - int8_size / fp32_size) * 100
        }
    
    def benchmark_latency(self, input_fn: Callable, num_runs: int = 50) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, model in [("fp32", self.original_model), 
                           ("int8", self.quantized_model or self.dynamic_quantize())]:
            model.eval().cpu()
            
            for _ in range(5):  # Warmup
                with torch.no_grad():
                    model(input_fn())
            
            latencies = []
            for _ in range(num_runs):
                inp = input_fn()
                start = time.perf_counter()
                with torch.no_grad():
                    model(inp)
                latencies.append((time.perf_counter() - start) * 1000)
            
            latencies = np.array(latencies)
            results[name] = {
                "mean_ms": float(np.mean(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "throughput_per_sec": float(1000 / np.mean(latencies))
            }
        
        results["speedup"] = results["fp32"]["mean_ms"] / results["int8"]["mean_ms"]
        return results
    
    def full_comparison(self, input_fn: Callable, evaluate_fn: Callable = None) -> Dict[str, Any]:
        if not self.quantized_model:
            self.dynamic_quantize()
        
        result = {
            "size": self.compare_sizes(),
            "latency": self.benchmark_latency(input_fn)
        }
        
        if evaluate_fn:
            result["accuracy"] = {
                "fp32_metric": evaluate_fn(self.original_model),
                "int8_metric": evaluate_fn(self.quantized_model)
            }
        
        return result
