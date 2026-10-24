"""Resource monitoring utilities."""

import time
import os
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class ResourceStats:
    """Statistics about resource usage."""
    cpu_mean: float
    cpu_max: float
    memory_mean_mb: float
    memory_max_mb: float
    gpu_memory_mean_mb: Optional[float] = None
    gpu_memory_max_mb: Optional[float] = None
    duration_seconds: float = 0


class ResourceMonitor:
    """Monitor CPU, memory, and GPU resources during training."""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.process = psutil.Process(os.getpid())
    
    def _get_gpu_stats(self) -> tuple:
        """Get GPU memory and utilization."""
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1e6  # MB
                # Note: torch doesn't directly provide utilization
                return gpu_memory, None
            except Exception:
                pass
        return None, None
    
    def _sample(self):
        """Take a single resource sample."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1e6
            memory_percent = self.process.memory_percent()
            
            gpu_memory, gpu_util = self._get_gpu_stats()
            
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                gpu_memory_mb=gpu_memory,
                gpu_utilization=gpu_util
            )
            self.snapshots.append(snapshot)
        except Exception as e:
            print(f"Warning: Resource sampling failed: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            self._sample()
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            return
        
        self.snapshots = []
        self.start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("Resource monitoring started")
    
    def stop(self) -> ResourceStats:
        """Stop monitoring and return statistics."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        return self.get_stats()
    
    def get_stats(self) -> ResourceStats:
        """Compute statistics from collected samples."""
        if not self.snapshots:
            return ResourceStats(0, 0, 0, 0, duration_seconds=0)
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        
        stats = ResourceStats(
            cpu_mean=sum(cpu_values) / len(cpu_values),
            cpu_max=max(cpu_values),
            memory_mean_mb=sum(memory_values) / len(memory_values),
            memory_max_mb=max(memory_values),
            duration_seconds=time.time() - self.start_time if self.start_time else 0
        )
        
        # GPU stats
        gpu_memory_values = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
        if gpu_memory_values:
            stats.gpu_memory_mean_mb = sum(gpu_memory_values) / len(gpu_memory_values)
            stats.gpu_memory_max_mb = max(gpu_memory_values)
        
        return stats
    
    def print_stats(self):
        """Print resource statistics."""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print(" Resource Usage Statistics")
        print("=" * 50)
        print(f"  Duration: {stats.duration_seconds:.1f} seconds")
        print(f"  CPU Usage: {stats.cpu_mean:.1f}% (mean), {stats.cpu_max:.1f}% (max)")
        print(f"  Memory: {stats.memory_mean_mb:.1f} MB (mean), {stats.memory_max_mb:.1f} MB (max)")
        
        if stats.gpu_memory_mean_mb is not None:
            print(f"  GPU Memory: {stats.gpu_memory_mean_mb:.1f} MB (mean), {stats.gpu_memory_max_mb:.1f} MB (max)")
        
        print("=" * 50 + "\n")
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def get_system_info() -> Dict:
    """Get system information."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / 1e9,
        "available_memory_gb": psutil.virtual_memory().available / 1e9,
    }
    
    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("\n" + "=" * 50)
    print(" System Information")
    print("=" * 50)
    
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_system_info()
    
    # Test monitoring
    with ResourceMonitor(interval=0.5) as monitor:
        # Simulate some work
        import numpy as np
        for _ in range(5):
            _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
            time.sleep(0.5)
    
    monitor.print_stats()
