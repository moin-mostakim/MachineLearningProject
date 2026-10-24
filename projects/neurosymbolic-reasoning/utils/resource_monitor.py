"""
Resource monitoring utilities for tracking compute usage.

Monitors:
- GPU memory usage
- CPU memory usage
- Training time
- Inference latency
"""

import os
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ResourceSnapshot:
    """Single resource measurement snapshot."""
    timestamp: float
    cpu_percent: float = 0.0
    cpu_memory_mb: float = 0.0
    cpu_memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "cpu_memory_mb": self.cpu_memory_mb,
            "cpu_memory_percent": self.cpu_memory_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_utilization": self.gpu_utilization
        }


@dataclass
class ResourceStats:
    """Aggregated resource statistics."""
    duration_seconds: float = 0.0
    
    # CPU stats
    cpu_percent_mean: float = 0.0
    cpu_percent_max: float = 0.0
    cpu_memory_mb_mean: float = 0.0
    cpu_memory_mb_max: float = 0.0
    
    # GPU stats
    gpu_memory_mb_mean: float = 0.0
    gpu_memory_mb_max: float = 0.0
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0
    
    # Timing
    num_samples: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "duration_seconds": self.duration_seconds,
            "cpu_percent_mean": self.cpu_percent_mean,
            "cpu_percent_max": self.cpu_percent_max,
            "cpu_memory_mb_mean": self.cpu_memory_mb_mean,
            "cpu_memory_mb_max": self.cpu_memory_mb_max,
            "gpu_memory_mb_mean": self.gpu_memory_mb_mean,
            "gpu_memory_mb_max": self.gpu_memory_mb_max,
            "gpu_utilization_mean": self.gpu_utilization_mean,
            "gpu_utilization_max": self.gpu_utilization_max,
            "num_samples": self.num_samples
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "RESOURCE USAGE SUMMARY",
            "=" * 50,
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "CPU:",
            f"  Usage: {self.cpu_percent_mean:.1f}% mean, {self.cpu_percent_max:.1f}% max",
            f"  Memory: {self.cpu_memory_mb_mean:.1f} MB mean, {self.cpu_memory_mb_max:.1f} MB max",
            "",
            "GPU:",
            f"  Memory: {self.gpu_memory_mb_mean:.1f} MB mean, {self.gpu_memory_mb_max:.1f} MB max",
            f"  Utilization: {self.gpu_utilization_mean:.1f}% mean, {self.gpu_utilization_max:.1f}% max",
            "=" * 50
        ]
        return "\n".join(lines)


class ResourceMonitor:
    """
    Monitor system resources during training/inference.
    
    Usage:
        monitor = ResourceMonitor()
        with monitor.track("training"):
            train_model()
        print(monitor.get_stats("training"))
    """
    
    def __init__(
        self,
        sample_interval: float = 1.0,
        memory_threshold_gb: float = 8.0,
        warn_callback: Optional[Callable[[str], None]] = None
    ):
        self.sample_interval = sample_interval
        self.memory_threshold_gb = memory_threshold_gb
        self.warn_callback = warn_callback or print
        
        self.snapshots: Dict[str, List[ResourceSnapshot]] = {}
        self.active_tracking: Dict[str, bool] = {}
        self.tracking_threads: Dict[str, threading.Thread] = {}
        
        # Check GPU availability
        self.has_gpu = False
        if HAS_TORCH:
            self.has_gpu = torch.cuda.is_available()
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource snapshot."""
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        # CPU metrics
        if HAS_PSUTIL:
            try:
                snapshot.cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                snapshot.cpu_memory_mb = memory.used / (1024 * 1024)
                snapshot.cpu_memory_percent = memory.percent
            except Exception:
                pass
        
        # GPU metrics
        if self.has_gpu and HAS_TORCH:
            try:
                snapshot.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                snapshot.gpu_memory_percent = (
                    torch.cuda.memory_allocated() / total_gpu_memory * 100
                )
                # Utilization requires nvidia-smi, approximate with memory %
                snapshot.gpu_utilization = snapshot.gpu_memory_percent
            except Exception:
                pass
        
        return snapshot
    
    def _monitor_loop(self, name: str):
        """Background monitoring loop."""
        while self.active_tracking.get(name, False):
            snapshot = self._take_snapshot()
            self.snapshots[name].append(snapshot)
            
            # Check memory threshold
            if snapshot.cpu_memory_mb > self.memory_threshold_gb * 1024:
                self.warn_callback(
                    f"Warning: CPU memory ({snapshot.cpu_memory_mb:.0f} MB) "
                    f"exceeds threshold ({self.memory_threshold_gb} GB)"
                )
            
            time.sleep(self.sample_interval)
    
    def start_tracking(self, name: str):
        """Start tracking resources for a named operation."""
        self.snapshots[name] = []
        self.active_tracking[name] = True
        
        thread = threading.Thread(target=self._monitor_loop, args=(name,), daemon=True)
        self.tracking_threads[name] = thread
        thread.start()
    
    def stop_tracking(self, name: str):
        """Stop tracking resources for a named operation."""
        self.active_tracking[name] = False
        
        if name in self.tracking_threads:
            self.tracking_threads[name].join(timeout=self.sample_interval * 2)
            del self.tracking_threads[name]
    
    @contextmanager
    def track(self, name: str):
        """Context manager for tracking a code block."""
        self.start_tracking(name)
        try:
            yield
        finally:
            self.stop_tracking(name)
    
    def get_stats(self, name: str) -> ResourceStats:
        """Compute aggregated statistics for a tracked operation."""
        stats = ResourceStats()
        
        snapshots = self.snapshots.get(name, [])
        if not snapshots:
            return stats
        
        stats.num_samples = len(snapshots)
        stats.duration_seconds = snapshots[-1].timestamp - snapshots[0].timestamp
        
        # CPU stats
        cpu_percents = [s.cpu_percent for s in snapshots]
        cpu_memory = [s.cpu_memory_mb for s in snapshots]
        
        stats.cpu_percent_mean = sum(cpu_percents) / len(cpu_percents)
        stats.cpu_percent_max = max(cpu_percents)
        stats.cpu_memory_mb_mean = sum(cpu_memory) / len(cpu_memory)
        stats.cpu_memory_mb_max = max(cpu_memory)
        
        # GPU stats
        gpu_memory = [s.gpu_memory_mb for s in snapshots]
        gpu_util = [s.gpu_utilization for s in snapshots]
        
        stats.gpu_memory_mb_mean = sum(gpu_memory) / len(gpu_memory)
        stats.gpu_memory_mb_max = max(gpu_memory)
        stats.gpu_utilization_mean = sum(gpu_util) / len(gpu_util)
        stats.gpu_utilization_max = max(gpu_util)
        
        return stats
    
    def get_current_usage(self) -> ResourceSnapshot:
        """Get current resource usage without tracking."""
        return self._take_snapshot()
    
    def save_snapshots(self, name: str, path: str):
        """Save snapshots to JSON file."""
        snapshots = self.snapshots.get(name, [])
        data = {
            "name": name,
            "snapshots": [s.to_dict() for s in snapshots],
            "stats": self.get_stats(name).to_dict()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_current_usage(self):
        """Print current resource usage."""
        snapshot = self.get_current_usage()
        
        print("\n" + "=" * 40)
        print("CURRENT RESOURCE USAGE")
        print("=" * 40)
        print(f"CPU: {snapshot.cpu_percent:.1f}%")
        print(f"CPU Memory: {snapshot.cpu_memory_mb:.0f} MB ({snapshot.cpu_memory_percent:.1f}%)")
        
        if self.has_gpu:
            print(f"GPU Memory: {snapshot.gpu_memory_mb:.0f} MB ({snapshot.gpu_memory_percent:.1f}%)")
        else:
            print("GPU: Not available")
        print("=" * 40 + "\n")


def get_gpu_memory_summary() -> str:
    """Get a summary of GPU memory usage."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return "GPU not available"
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    cached = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    return (
        f"GPU Memory: {allocated:.2f} GB allocated, "
        f"{cached:.2f} GB cached, "
        f"{total:.2f} GB total"
    )


def get_system_info() -> Dict:
    """Get system information for reproducibility."""
    info = {
        "python_version": "",
        "torch_version": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
        "cpu_count": 0,
        "total_memory_gb": 0.0
    }
    
    import sys
    info["python_version"] = sys.version
    
    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or ""
            info["gpu_name"] = torch.cuda.get_device_name(0)
    
    if HAS_PSUTIL:
        info["cpu_count"] = psutil.cpu_count()
        info["total_memory_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    
    return info


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {self.elapsed:.4f}s"
        return f"{self.elapsed:.4f}s"


if __name__ == "__main__":
    # Test resource monitoring
    print("System Info:")
    info = get_system_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\n" + get_gpu_memory_summary())
    
    monitor = ResourceMonitor(sample_interval=0.5)
    monitor.print_current_usage()
    
    print("\nTesting resource tracking...")
    with monitor.track("test"):
        # Simulate some work
        time.sleep(2)
    
    stats = monitor.get_stats("test")
    print(stats)
