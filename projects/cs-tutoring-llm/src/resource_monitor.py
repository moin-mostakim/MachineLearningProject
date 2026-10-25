"""
Resource monitoring utilities for CS Tutoring LLM.

Monitors system resources during training and inference:
- CPU usage
- Memory (RAM) usage
- GPU usage and memory
- Disk I/O

References:
- Best practices for monitoring ML training workloads
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import psutil
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

try:
    import GPUtil
    HAS_GPUTIL = True
except Exception:
    HAS_GPUTIL = False


@dataclass
class ResourceSnapshot:
    """Single resource snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None


@dataclass
class ResourceSummary:
    """Summary statistics for resource usage."""
    duration_seconds: float = 0.0
    
    # CPU stats
    cpu_mean: float = 0.0
    cpu_max: float = 0.0
    cpu_min: float = 0.0
    
    # Memory stats
    memory_mean_percent: float = 0.0
    memory_max_percent: float = 0.0
    memory_peak_gb: float = 0.0
    
    # GPU stats
    gpu_mean: Optional[float] = None
    gpu_max: Optional[float] = None
    gpu_memory_peak_gb: Optional[float] = None
    gpu_temp_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "duration_seconds": self.duration_seconds,
            "cpu": {
                "mean_percent": self.cpu_mean,
                "max_percent": self.cpu_max,
                "min_percent": self.cpu_min,
            },
            "memory": {
                "mean_percent": self.memory_mean_percent,
                "max_percent": self.memory_max_percent,
                "peak_gb": self.memory_peak_gb,
            },
        }
        
        if self.gpu_mean is not None:
            result["gpu"] = {
                "mean_percent": self.gpu_mean,
                "max_percent": self.gpu_max,
                "memory_peak_gb": self.gpu_memory_peak_gb,
                "temp_max_celsius": self.gpu_temp_max,
            }
        
        return result


class ResourceMonitor:
    """
    Resource monitoring for ML training and inference.
    
    Features:
    - Background monitoring thread
    - Configurable sampling interval
    - Memory warning thresholds
    - GPU monitoring (if available)
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.snapshots: List[ResourceSnapshot] = []
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        
        # Check GPU availability
        self.has_gpu = self._check_gpu()
        if self.has_gpu:
            logger.info("GPU monitoring enabled")
        else:
            logger.info("GPU not available, CPU-only monitoring")
    
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except Exception:
            pass
        
        if HAS_NVML:
            try:
                pynvml.nvmlDeviceGetCount()
                return True
            except Exception:
                pass
        
        return False
    
    def _get_gpu_stats(self) -> Dict[str, Optional[float]]:
        """Get current GPU statistics."""
        result = {
            "gpu_percent": None,
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": None,
            "gpu_temperature": None,
        }
        
        if not self.has_gpu:
            return result
        
        try:
            if HAS_NVML:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result["gpu_percent"] = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result["gpu_memory_used_gb"] = mem_info.used / (1024**3)
                result["gpu_memory_total_gb"] = mem_info.total / (1024**3)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                result["gpu_temperature"] = temp
            
            elif HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    result["gpu_percent"] = gpu.load * 100
                    result["gpu_memory_used_gb"] = gpu.memoryUsed / 1024
                    result["gpu_memory_total_gb"] = gpu.memoryTotal / 1024
                    result["gpu_temperature"] = gpu.temperature
        
        except Exception as e:
            logger.debug(f"GPU stats collection failed: {e}")
        
        return result
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource snapshot."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_used_gb = mem.used / (1024**3)
        memory_available_gb = mem.available / (1024**3)
        
        # GPU
        gpu_stats = self._get_gpu_stats()
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            **gpu_stats
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            
            # Check memory warning threshold
            if snapshot.memory_percent >= self.config.resources.memory_warning_threshold * 100:
                logger.warning(
                    f"High memory usage: {snapshot.memory_percent:.1f}% "
                    f"({snapshot.memory_used_gb:.1f} GB used)"
                )
            
            self._stop_event.wait(self.config.resources.log_interval_seconds)
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.snapshots = []
        self._start_time = time.time()
        self._stop_event.clear()
        self.is_monitoring = True
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.is_monitoring:
            return
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.is_monitoring = False
        logger.info("Resource monitoring stopped")
    
    def get_current(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._take_snapshot()
    
    def log_current_usage(self):
        """Log current resource usage."""
        snapshot = self._take_snapshot()
        
        msg = (
            f"Resources - CPU: {snapshot.cpu_percent:.1f}%, "
            f"RAM: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.1f}GB)"
        )
        
        if snapshot.gpu_percent is not None:
            msg += f", GPU: {snapshot.gpu_percent:.1f}% ({snapshot.gpu_memory_used_gb:.1f}GB)"
        
        logger.info(msg)
    
    def get_summary(self) -> ResourceSummary:
        """Get summary statistics from monitoring session."""
        if not self.snapshots:
            return ResourceSummary()
        
        duration = (
            self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            if len(self.snapshots) > 1 else 0
        )
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        mem_values = [s.memory_percent for s in self.snapshots]
        mem_gb_values = [s.memory_used_gb for s in self.snapshots]
        
        summary = ResourceSummary(
            duration_seconds=duration,
            cpu_mean=np.mean(cpu_values),
            cpu_max=np.max(cpu_values),
            cpu_min=np.min(cpu_values),
            memory_mean_percent=np.mean(mem_values),
            memory_max_percent=np.max(mem_values),
            memory_peak_gb=np.max(mem_gb_values),
        )
        
        # GPU stats
        gpu_values = [s.gpu_percent for s in self.snapshots if s.gpu_percent is not None]
        if gpu_values:
            summary.gpu_mean = np.mean(gpu_values)
            summary.gpu_max = np.max(gpu_values)
            
            gpu_mem = [s.gpu_memory_used_gb for s in self.snapshots if s.gpu_memory_used_gb is not None]
            if gpu_mem:
                summary.gpu_memory_peak_gb = np.max(gpu_mem)
            
            gpu_temp = [s.gpu_temperature for s in self.snapshots if s.gpu_temperature is not None]
            if gpu_temp:
                summary.gpu_temp_max = np.max(gpu_temp)
        
        return summary
    
    def generate_report(self) -> str:
        """Generate human-readable resource report."""
        summary = self.get_summary()
        
        report = []
        report.append("\n" + "=" * 60)
        report.append("RESOURCE MONITORING REPORT")
        report.append("=" * 60)
        
        report.append(f"\nMonitoring Duration: {summary.duration_seconds:.1f} seconds")
        report.append(f"Samples Collected: {len(self.snapshots)}")
        
        report.append("\n## CPU Usage")
        report.append("-" * 40)
        report.append(f"Mean: {summary.cpu_mean:.1f}%")
        report.append(f"Max:  {summary.cpu_max:.1f}%")
        report.append(f"Min:  {summary.cpu_min:.1f}%")
        
        report.append("\n## Memory Usage")
        report.append("-" * 40)
        report.append(f"Mean: {summary.memory_mean_percent:.1f}%")
        report.append(f"Max:  {summary.memory_max_percent:.1f}%")
        report.append(f"Peak: {summary.memory_peak_gb:.2f} GB")
        
        if summary.gpu_mean is not None:
            report.append("\n## GPU Usage")
            report.append("-" * 40)
            report.append(f"Mean:        {summary.gpu_mean:.1f}%")
            report.append(f"Max:         {summary.gpu_max:.1f}%")
            if summary.gpu_memory_peak_gb:
                report.append(f"Memory Peak: {summary.gpu_memory_peak_gb:.2f} GB")
            if summary.gpu_temp_max:
                report.append(f"Max Temp:    {summary.gpu_temp_max:.0f}°C")
        else:
            report.append("\n## GPU: Not available")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class MemoryTracker:
    """Context manager for tracking memory usage of code blocks."""
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_memory = 0
        self.end_memory = 0
        self.peak_memory = 0
    
    def __enter__(self):
        import gc
        gc.collect()
        
        mem = psutil.virtual_memory()
        self.start_memory = mem.used / (1024**3)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import gc
        gc.collect()
        
        mem = psutil.virtual_memory()
        self.end_memory = mem.used / (1024**3)
        
        logger.info(
            f"[{self.name}] Memory: {self.start_memory:.2f}GB -> "
            f"{self.end_memory:.2f}GB (delta: {self.end_memory - self.start_memory:+.2f}GB)"
        )
        
        return False


if __name__ == "__main__":
    # Test resource monitor
    monitor = ResourceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some work
    logger.info("Starting test workload...")
    
    # CPU work
    result = 0
    for i in range(10):
        result += sum(range(1000000))
        time.sleep(0.5)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Print report
    print(monitor.generate_report())
    
    # Test memory tracker
    print("\nTesting memory tracker...")
    with MemoryTracker("test_allocation"):
        data = [list(range(100000)) for _ in range(100)]
    
    del data
