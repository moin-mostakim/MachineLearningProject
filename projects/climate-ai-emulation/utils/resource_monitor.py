"""
Resource monitoring utilities for climate emulation experiments.

Tracks:
- GPU memory usage
- CPU/RAM usage
- Training time
- FLOPs estimation
- Model size and parameters
"""

import os
import time
import psutil
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    total_time_seconds: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    samples_per_second: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    peak_cpu_memory_gb: float = 0.0
    avg_gpu_utilization: float = 0.0


class ResourceMonitor:
    """
    Monitor system resources during training.
    
    Tracks CPU, memory, and GPU usage over time.
    """
    
    def __init__(
        self,
        log_interval_seconds: float = 1.0,
        enable_gpu: bool = True
    ):
        """
        Initialize resource monitor.
        
        Args:
            log_interval_seconds: How often to sample resources
            enable_gpu: Whether to monitor GPU
        """
        self.log_interval = log_interval_seconds
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        
        self.snapshots: List[ResourceSnapshot] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def _get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        gpu_used = 0.0
        gpu_total = 0.0
        gpu_util = 0.0
        
        if self.enable_gpu:
            try:
                gpu_used = torch.cuda.memory_allocated() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Note: nvidia-smi would give utilization, but we use memory as proxy
                gpu_util = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            except Exception:
                pass
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory.used / 1e9,
            memory_available_gb=memory.available / 1e9,
            gpu_memory_used_gb=gpu_used,
            gpu_memory_total_gb=gpu_total,
            gpu_utilization=gpu_util
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            snapshot = self._get_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.log_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        self._monitoring = True
        self.snapshots = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return snapshots."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        return self.snapshots
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage from snapshots."""
        if not self.snapshots:
            return {}
        
        return {
            'peak_cpu_percent': max(s.cpu_percent for s in self.snapshots),
            'peak_memory_gb': max(s.memory_used_gb for s in self.snapshots),
            'peak_gpu_memory_gb': max(s.gpu_memory_used_gb for s in self.snapshots),
            'avg_gpu_memory_gb': sum(s.gpu_memory_used_gb for s in self.snapshots) / len(self.snapshots)
        }
    
    @contextmanager
    def monitor(self):
        """Context manager for monitoring."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """Start timing a section."""
        self._start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time."""
        if name not in self._start_times:
            return 0.0
        
        elapsed = time.time() - self._start_times[name]
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        
        del self._start_times[name]
        return elapsed
    
    @contextmanager
    def time(self, name: str):
        """Context manager for timing."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics for a section."""
        if name not in self.times or not self.times[name]:
            return {}
        
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed sections."""
        return {name: self.get_stats(name) for name in self.times}


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def estimate_model_size(model: torch.nn.Module, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Estimate model size in memory.
    
    Args:
        model: PyTorch model
        dtype: Data type for size calculation
        
    Returns:
        Dictionary with size estimates in MB
    """
    param_count = count_parameters(model)
    
    # Bytes per element
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4
    
    params_mb = param_count['total'] * bytes_per_param / 1e6
    
    return {
        'params_mb': params_mb,
        'params_fp32_mb': param_count['total'] * 4 / 1e6,
        'params_fp16_mb': param_count['total'] * 2 / 1e6,
        'params_int8_mb': param_count['total'] * 1 / 1e6
    }


def estimate_flops(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Estimate FLOPs for a forward pass.
    
    This is a rough estimate based on layer types.
    For accurate profiling, use torch.profiler.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device for dummy forward pass
        
    Returns:
        Dictionary with FLOP estimates
    """
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, torch.nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * K * K * Hout * Wout
            batch_size = input[0].shape[0]
            cin = module.in_channels
            cout = module.out_channels
            k = module.kernel_size[0] * module.kernel_size[1]
            hout, wout = output.shape[2], output.shape[3]
            flops = 2 * cin * cout * k * hout * wout * batch_size
            total_flops += flops
            
        elif isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * in * out * batch
            batch_size = input[0].shape[0]
            flops = 2 * module.in_features * module.out_features * batch_size
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape, device=device)
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'flops': total_flops,
        'gflops': total_flops / 1e9,
        'tflops': total_flops / 1e12
    }


def get_parameter_table(model: torch.nn.Module) -> str:
    """
    Generate a table of model parameters by layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted string table
    """
    table = []
    table.append("-" * 80)
    table.append(f"{'Layer':<40} {'Shape':<20} {'Parameters':>15}")
    table.append("-" * 80)
    
    total = 0
    for name, param in model.named_parameters():
        shape = str(list(param.shape))
        num_params = param.numel()
        total += num_params
        table.append(f"{name[:40]:<40} {shape:<20} {num_params:>15,}")
    
    table.append("-" * 80)
    table.append(f"{'Total':<40} {'':<20} {total:>15,}")
    table.append("-" * 80)
    
    return "\n".join(table)


class ExperimentLogger:
    """
    Logger for experiment metrics and resources.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timer = Timer()
        self.resource_monitor = ResourceMonitor()
        
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: List[Dict[str, float]] = []
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def log_epoch(self, metrics: Dict[str, float], epoch: int):
        """Log metrics for an epoch."""
        metrics['epoch'] = epoch
        self.epoch_metrics.append(metrics)
    
    def get_summary(self) -> Dict:
        """Get summary of logged metrics and resources."""
        summary = {
            'timing': self.timer.get_all_stats(),
            'resources': self.resource_monitor.get_peak_usage(),
            'metrics': {
                name: {
                    'final': values[-1] if values else None,
                    'best': min(values) if values else None,
                    'count': len(values)
                }
                for name, values in self.metrics.items()
            }
        }
        return summary
    
    def save_summary(self, filename: str = "experiment_summary.txt"):
        """Save summary to file."""
        summary = self.get_summary()
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TIMING\n")
            f.write("-" * 40 + "\n")
            for name, stats in summary['timing'].items():
                f.write(f"{name}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nRESOURCE USAGE\n")
            f.write("-" * 40 + "\n")
            for name, value in summary['resources'].items():
                f.write(f"{name}: {value:.2f}\n")
            
            f.write("\nMETRICS\n")
            f.write("-" * 40 + "\n")
            for name, stats in summary['metrics'].items():
                f.write(f"{name}: final={stats['final']:.4f}, best={stats['best']:.4f}\n")
        
        return filepath


if __name__ == "__main__":
    # Test resource monitoring
    print("Testing resource monitoring...")
    
    monitor = ResourceMonitor()
    snapshot = monitor._get_snapshot()
    
    print(f"CPU: {snapshot.cpu_percent:.1f}%")
    print(f"Memory: {snapshot.memory_used_gb:.2f} GB used, {snapshot.memory_available_gb:.2f} GB available")
    print(f"GPU: {snapshot.gpu_memory_used_gb:.2f} GB used / {snapshot.gpu_memory_total_gb:.2f} GB total")
    
    # Test timer
    print("\nTesting timer...")
    timer = Timer()
    
    with timer.time("test_operation"):
        time.sleep(0.1)
    
    print(f"Timer stats: {timer.get_stats('test_operation')}")
    
    # Test parameter counting
    print("\nTesting parameter counting...")
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 3, 1)
    )
    
    params = count_parameters(model)
    print(f"Parameters: {params}")
    
    sizes = estimate_model_size(model)
    print(f"Model size: {sizes}")
    
    print("\nParameter table:")
    print(get_parameter_table(model))
    
    print("\nAll resource monitoring tests passed!")
