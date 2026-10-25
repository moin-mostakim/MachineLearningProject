"""
Resource monitoring utilities for tracking training efficiency.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
import threading


class ResourceMonitor:
    """
    Monitor system resources during training.
    
    Tracks:
    - CPU usage
    - Memory usage
    - GPU memory (if available)
    - Training time
    """
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Recorded stats
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
        self.timestamps = []
        
        # Summary stats
        self.start_time = None
        self.end_time = None
        self.peak_memory_mb = 0
        self.peak_gpu_memory_mb = 0
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            self._record_stats()
            time.sleep(self.interval)
    
    def _record_stats(self):
        """Record current resource usage."""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        self.memory_usage.append(memory_used_mb)
        self.peak_memory_mb = max(self.peak_memory_mb, memory_used_mb)
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.gpu_memory.append(gpu_memory_mb)
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_memory_mb)
            except Exception:
                self.gpu_memory.append(0)
        
        self.timestamps.append(timestamp)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.start_time = time.time()
        
        # Reset stats
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.peak_memory_mb = 0
        self.peak_gpu_memory_mb = 0
        
        # Start background thread
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring and return summary.
        
        Returns:
            Dictionary with resource statistics
        """
        self.running = False
        self.end_time = time.time()
        
        if self.thread:
            self.thread.join(timeout=2)
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get resource usage summary.
        
        Returns:
            Dictionary with summary statistics
        """
        import numpy as np
        
        total_time = (self.end_time or time.time()) - (self.start_time or 0)
        
        summary = {
            "total_time": total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }
        
        if self.cpu_usage:
            summary["avg_cpu_percent"] = np.mean(self.cpu_usage)
            summary["max_cpu_percent"] = np.max(self.cpu_usage)
        
        if self.memory_usage:
            summary["avg_memory_mb"] = np.mean(self.memory_usage)
        
        if self.gpu_memory:
            summary["avg_gpu_memory_mb"] = np.mean(self.gpu_memory)
        
        return summary
    
    def print_summary(self):
        """Print resource usage summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 50)
        print(f"Total Time: {summary['total_time']:.1f}s")
        print(f"Peak Memory: {summary['peak_memory_mb']:.1f} MB")
        
        if summary.get("avg_cpu_percent"):
            print(f"Avg CPU: {summary['avg_cpu_percent']:.1f}%")
            print(f"Max CPU: {summary['max_cpu_percent']:.1f}%")
        
        if summary.get("peak_gpu_memory_mb", 0) > 0:
            print(f"Peak GPU Memory: {summary['peak_gpu_memory_mb']:.1f} MB")
            print(f"Avg GPU Memory: {summary.get('avg_gpu_memory_mb', 0):.1f} MB")
        
        print("=" * 50)


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get model information including parameter counts.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size (FP32)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size_mb,
        "model_name": model.__class__.__name__,
    }


def print_model_info(model: torch.nn.Module):
    """Print model information."""
    info = get_model_info(model)
    
    print("\n" + "=" * 50)
    print(f"MODEL: {info['model_name']}")
    print("=" * 50)
    print(f"Total Parameters: {info['total_params']:,}")
    print(f"Trainable Parameters: {info['trainable_params']:,}")
    print(f"Estimated Size (FP32): {info['param_size_mb']:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    # Test resource monitor
    print("Testing Resource Monitor...")
    
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    
    # Simulate some work
    time.sleep(3)
    
    summary = monitor.stop()
    monitor.print_summary()
