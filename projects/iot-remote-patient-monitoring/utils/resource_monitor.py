"""
Resource Monitoring Utilities for Patient Monitoring System.

Tracks CPU, memory, and GPU usage during model training and inference.
"""

import os
import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


@dataclass
class ResourceSnapshot:
    """Single resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class ResourceSummary:
    """Summary statistics for resource usage."""
    duration_seconds: float
    cpu_mean: float
    cpu_max: float
    memory_mean_gb: float
    memory_max_gb: float
    memory_mean_percent: float
    gpu_memory_mean_gb: Optional[float] = None
    gpu_memory_max_gb: Optional[float] = None
    gpu_utilization_mean: Optional[float] = None


class ResourceMonitor:
    """
    Monitor system resources during model operations.
    
    Provides:
    - CPU utilization tracking
    - Memory usage monitoring
    - GPU memory and utilization (if available)
    - Background monitoring thread
    - Summary statistics
    """
    
    def __init__(self, config=None, interval: float = 1.0):
        self.config = config or get_config()
        self.interval = interval
        
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Check for GPU
        self.has_gpu = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_stats(self) -> Dict:
        """Get GPU memory and utilization stats."""
        if not self.has_gpu:
            return {}
        
        try:
            import torch
            
            if torch.cuda.is_available():
                # Memory stats
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                
                # Get total memory
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)  # GB
                
                return {
                    "gpu_memory_used_gb": memory_allocated,
                    "gpu_memory_percent": (memory_allocated / total_memory) * 100,
                    "gpu_utilization": None  # Would need pynvml for this
                }
        except Exception:
            pass
        
        return {}
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource usage snapshot."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        # GPU stats
        gpu_stats = self._get_gpu_stats()
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_percent=memory_percent,
            **gpu_stats
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def start(self):
        """Start background resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> ResourceSummary:
        """Stop monitoring and return summary."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        return self.get_summary()
    
    def get_summary(self) -> ResourceSummary:
        """Calculate summary statistics from snapshots."""
        if not self.snapshots:
            return ResourceSummary(
                duration_seconds=0,
                cpu_mean=0,
                cpu_max=0,
                memory_mean_gb=0,
                memory_max_gb=0,
                memory_mean_percent=0
            )
        
        # Calculate duration
        duration = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
        
        # CPU stats
        cpu_values = [s.cpu_percent for s in self.snapshots]
        cpu_mean = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)
        
        # Memory stats
        mem_values = [s.memory_used_gb for s in self.snapshots]
        mem_mean = sum(mem_values) / len(mem_values)
        mem_max = max(mem_values)
        
        mem_percent = [s.memory_percent for s in self.snapshots]
        mem_percent_mean = sum(mem_percent) / len(mem_percent)
        
        # GPU stats
        gpu_mem_values = [s.gpu_memory_used_gb for s in self.snapshots 
                         if s.gpu_memory_used_gb is not None]
        gpu_mem_mean = sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None
        gpu_mem_max = max(gpu_mem_values) if gpu_mem_values else None
        
        gpu_util_values = [s.gpu_utilization for s in self.snapshots 
                          if s.gpu_utilization is not None]
        gpu_util_mean = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None
        
        return ResourceSummary(
            duration_seconds=duration,
            cpu_mean=cpu_mean,
            cpu_max=cpu_max,
            memory_mean_gb=mem_mean,
            memory_max_gb=mem_max,
            memory_mean_percent=mem_percent_mean,
            gpu_memory_mean_gb=gpu_mem_mean,
            gpu_memory_max_gb=gpu_mem_max,
            gpu_utilization_mean=gpu_util_mean
        )
    
    def get_current(self) -> ResourceSnapshot:
        """Get current resource usage."""
        return self._take_snapshot()
    
    def format_summary(self, summary: Optional[ResourceSummary] = None) -> str:
        """Format summary as string."""
        if summary is None:
            summary = self.get_summary()
        
        lines = [
            "=" * 50,
            "RESOURCE USAGE SUMMARY",
            "=" * 50,
            f"Duration: {summary.duration_seconds:.1f} seconds",
            "",
            "CPU:",
            f"  Mean: {summary.cpu_mean:.1f}%",
            f"  Max: {summary.cpu_max:.1f}%",
            "",
            "Memory:",
            f"  Mean: {summary.memory_mean_gb:.2f} GB ({summary.memory_mean_percent:.1f}%)",
            f"  Max: {summary.memory_max_gb:.2f} GB",
        ]
        
        if summary.gpu_memory_mean_gb is not None:
            lines.extend([
                "",
                "GPU Memory:",
                f"  Mean: {summary.gpu_memory_mean_gb:.2f} GB",
                f"  Max: {summary.gpu_memory_max_gb:.2f} GB",
            ])
            
            if summary.gpu_utilization_mean is not None:
                lines.append(f"  Utilization: {summary.gpu_utilization_mean:.1f}%")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class ResourceMonitorContext:
    """Context manager for resource monitoring."""
    
    def __init__(self, name: str = "operation", interval: float = 1.0):
        self.name = name
        self.monitor = ResourceMonitor(interval=interval)
        self.summary: Optional[ResourceSummary] = None
        
    def __enter__(self):
        self.monitor.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.summary = self.monitor.stop()
        return False
    
    def print_summary(self):
        """Print resource usage summary."""
        print(f"\nResource Usage for '{self.name}':")
        print(self.monitor.format_summary(self.summary))


def measure_resources(func):
    """Decorator to measure resource usage of a function."""
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            summary = monitor.stop()
            print(f"\nResource usage for {func.__name__}:")
            print(monitor.format_summary(summary))
        
        return result
    
    return wrapper


if __name__ == "__main__":
    # Test resource monitoring
    print("Testing Resource Monitor...")
    
    with ResourceMonitorContext("test_operation") as ctx:
        # Simulate some work
        import numpy as np
        for i in range(5):
            _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
            time.sleep(0.5)
    
    ctx.print_summary()
