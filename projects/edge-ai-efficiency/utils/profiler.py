"""
Profiler utilities for latency measurement and resource monitoring.
Designed for edge AI benchmarking scenarios.
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import psutil
import os


@dataclass
class LatencyStats:
    """Latency statistics from benchmarking."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    num_runs: int
    
    def __str__(self) -> str:
        return (f"Latency: {self.mean_ms:.2f}±{self.std_ms:.2f}ms "
                f"(min={self.min_ms:.2f}, max={self.max_ms:.2f}, "
                f"p50={self.p50_ms:.2f}, p90={self.p90_ms:.2f}, p99={self.p99_ms:.2f})")


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False
) -> LatencyStats:
    """
    Measure inference latency with proper warmup and statistics.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on ('cpu' for edge simulation)
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        use_amp: Use automatic mixed precision
        
    Returns:
        LatencyStats with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp and device != 'cpu':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    
    # Synchronize if using CUDA
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if use_amp and device != 'cpu':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
            
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return LatencyStats(
        mean_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p90_ms=float(np.percentile(latencies, 90)),
        p99_ms=float(np.percentile(latencies, 99)),
        num_runs=num_runs
    )


def measure_throughput(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    duration_seconds: float = 5.0
) -> Dict[str, float]:
    """
    Measure inference throughput (samples per second).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        duration_seconds: Duration to run benchmark
        
    Returns:
        Dictionary with throughput metrics
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size).to(device)
    batch_size = input_size[0]
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure throughput
    num_batches = 0
    start_time = time.perf_counter()
    
    with torch.no_grad():
        while time.perf_counter() - start_time < duration_seconds:
            _ = model(x)
            num_batches += 1
    
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    total_samples = num_batches * batch_size
    
    return {
        "throughput_samples_per_sec": total_samples / elapsed,
        "throughput_batches_per_sec": num_batches / elapsed,
        "elapsed_seconds": elapsed,
        "total_samples": total_samples
    }


class ResourceMonitor:
    """Monitor system resource usage during model execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements: Dict[str, List[float]] = {
            "cpu_percent": [],
            "memory_mb": [],
            "memory_percent": []
        }
        self.gpu_available = torch.cuda.is_available()
    
    def sample(self):
        """Take a resource usage sample."""
        self.measurements["cpu_percent"].append(self.process.cpu_percent())
        mem_info = self.process.memory_info()
        self.measurements["memory_mb"].append(mem_info.rss / (1024 ** 2))
        self.measurements["memory_percent"].append(self.process.memory_percent())
        
        if self.gpu_available:
            if "gpu_memory_mb" not in self.measurements:
                self.measurements["gpu_memory_mb"] = []
                self.measurements["gpu_memory_percent"] = []
            
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            self.measurements["gpu_memory_mb"].append(gpu_mem)
            self.measurements["gpu_memory_percent"].append(100 * gpu_mem / gpu_total)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all measurements."""
        summary = {}
        for key, values in self.measurements.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "std": np.std(values)
                }
        return summary
    
    def reset(self):
        """Reset all measurements."""
        for key in self.measurements:
            self.measurements[key] = []


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Comprehensive model profiling: latency, throughput, and resource usage.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        
    Returns:
        Dictionary with all profiling results
    """
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize monitor
    monitor = ResourceMonitor()
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
            monitor.sample()
    
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    monitor.reset()
    
    # Measure with monitoring
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            monitor.sample()
    
    latencies = np.array(latencies)
    
    return {
        "latency": LatencyStats(
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p90_ms=float(np.percentile(latencies, 90)),
            p99_ms=float(np.percentile(latencies, 99)),
            num_runs=num_runs
        ),
        "resources": monitor.get_summary(),
        "device": device,
        "input_size": input_size
    }


def compare_models(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on latency and resource usage.
    
    Args:
        models: Dictionary mapping model names to models
        input_size: Input tensor size
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for name, model in models.items():
        print(f"Profiling {name}...")
        results[name] = profile_model(model, input_size, device, num_warmup, num_runs)
    
    return results


def estimate_edge_metrics(
    latency_ms: float,
    power_per_gflop: float = 0.1,  # Watts per GFLOP (estimate)
    gflops: float = 1.0
) -> Dict[str, float]:
    """
    Estimate edge-specific metrics based on latency and FLOPs.
    
    Args:
        latency_ms: Inference latency in milliseconds
        power_per_gflop: Estimated power consumption per GFLOP
        gflops: Number of GFLOPs in the model
        
    Returns:
        Dictionary with edge metrics
    """
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    power_watts = power_per_gflop * gflops
    energy_per_inference_mj = power_watts * latency_ms  # millijoules
    
    return {
        "fps": fps,
        "estimated_power_watts": power_watts,
        "energy_per_inference_mj": energy_per_inference_mj,
        "inferences_per_joule": 1000.0 / energy_per_inference_mj if energy_per_inference_mj > 0 else 0
    }
