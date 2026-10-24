"""
Resource monitoring utilities.

Tracks:
- CPU/Memory usage
- GPU memory and utilization
- Training throughput
"""

import time
import os
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Resource monitoring will be limited.")


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage."""
    timestamp: float
    step: int
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_max_mb: float = 0.0
    gpu_utilization: float = 0.0
    throughput: float = 0.0  # samples/second


class ResourceTracker:
    """Track and log resource usage during training."""
    
    def __init__(
        self,
        log_interval: int = 10,
        log_gpu: bool = True,
        log_memory: bool = True,
        save_path: Optional[str] = None
    ):
        self.log_interval = log_interval
        self.log_gpu = log_gpu and torch.cuda.is_available()
        self.log_memory = log_memory and PSUTIL_AVAILABLE
        self.save_path = save_path
        
        self.history: List[ResourceSnapshot] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.samples_processed = 0
    
    def get_cpu_memory_stats(self) -> Dict[str, float]:
        """Get CPU and memory statistics."""
        if not PSUTIL_AVAILABLE:
            return {}
        
        process = psutil.Process()
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }
    
    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics."""
        if not self.log_gpu:
            return {}
        
        stats = {
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_memory_max_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
        
        # Try to get GPU utilization (requires nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats['gpu_utilization'] = util.gpu
        except Exception:
            stats['gpu_utilization'] = 0.0
        
        return stats
    
    def log(self, step: int, batch_size: int = 1) -> Optional[ResourceSnapshot]:
        """Log resource usage at current step."""
        self.samples_processed += batch_size
        
        if step % self.log_interval != 0:
            return None
        
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        throughput = batch_size * self.log_interval / elapsed if elapsed > 0 else 0
        
        # Get stats
        cpu_stats = self.get_cpu_memory_stats()
        gpu_stats = self.get_gpu_stats()
        
        snapshot = ResourceSnapshot(
            timestamp=current_time,
            step=step,
            throughput=throughput,
            **cpu_stats,
            **gpu_stats
        )
        
        self.history.append(snapshot)
        self.last_log_time = current_time
        
        return snapshot
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        total_time = time.time() - self.start_time
        
        summary = {
            'total_time_seconds': total_time,
            'total_samples': self.samples_processed,
            'avg_throughput': self.samples_processed / total_time if total_time > 0 else 0,
            'num_snapshots': len(self.history)
        }
        
        # Aggregate stats
        if self.log_memory:
            summary['peak_memory_mb'] = max(s.memory_mb for s in self.history)
            summary['avg_memory_mb'] = sum(s.memory_mb for s in self.history) / len(self.history)
            summary['avg_cpu_percent'] = sum(s.cpu_percent for s in self.history) / len(self.history)
        
        if self.log_gpu:
            summary['peak_gpu_memory_mb'] = max(s.gpu_memory_mb for s in self.history)
            summary['avg_gpu_memory_mb'] = sum(s.gpu_memory_mb for s in self.history) / len(self.history)
        
        return summary
    
    def save(self, path: Optional[str] = None):
        """Save resource history to file."""
        path = path or self.save_path
        if not path:
            return
        
        data = {
            'summary': self.get_summary(),
            'history': [vars(s) for s in self.history]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved resource history to {path}")
    
    def print_summary(self):
        """Print resource usage summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        print(f"Total Time: {summary.get('total_time_seconds', 0):.1f} seconds")
        print(f"Total Samples: {summary.get('total_samples', 0):,}")
        print(f"Avg Throughput: {summary.get('avg_throughput', 0):.1f} samples/sec")
        
        if self.log_memory:
            print(f"\nMemory Usage:")
            print(f"  Peak: {summary.get('peak_memory_mb', 0):.1f} MB")
            print(f"  Avg: {summary.get('avg_memory_mb', 0):.1f} MB")
            print(f"  Avg CPU: {summary.get('avg_cpu_percent', 0):.1f}%")
        
        if self.log_gpu:
            print(f"\nGPU Memory:")
            print(f"  Peak: {summary.get('peak_gpu_memory_mb', 0):.1f} MB")
            print(f"  Avg: {summary.get('avg_gpu_memory_mb', 0):.1f} MB")
        
        print("=" * 60)


def get_system_info() -> Dict[str, any]:
    """Get system information."""
    info = {
        'python_version': os.sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    
    if PSUTIL_AVAILABLE:
        info['cpu_count'] = psutil.cpu_count()
        info['total_memory_gb'] = psutil.virtual_memory().total / 1024 / 1024 / 1024
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
        print(f"GPU: {info.get('gpu_name', 'N/A')}")
        print(f"GPU Memory: {info.get('gpu_memory_total', 0):.0f} MB")
    
    if 'cpu_count' in info:
        print(f"\nCPU Cores: {info['cpu_count']}")
        print(f"System Memory: {info.get('total_memory_gb', 0):.1f} GB")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test resource monitoring
    print_system_info()
    
    tracker = ResourceTracker(log_interval=5)
    
    # Simulate training
    for step in range(25):
        time.sleep(0.1)
        snapshot = tracker.log(step, batch_size=32)
        if snapshot:
            print(f"Step {step}: Memory={snapshot.memory_mb:.1f}MB, "
                  f"Throughput={snapshot.throughput:.1f} samples/sec")
    
    tracker.print_summary()
