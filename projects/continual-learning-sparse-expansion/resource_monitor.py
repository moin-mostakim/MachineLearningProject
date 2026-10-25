"""
Resource Monitoring Utilities.

Tracks:
- GPU memory usage
- CPU memory usage
- Inference time
- Parameter counts over time
"""

import torch
import time
import os
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import config


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    task_id: int
    epoch: int
    batch_idx: int
    
    # Memory (in MB)
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    cpu_memory_used: float = 0.0
    cpu_memory_percent: float = 0.0
    
    # Timing (in ms)
    batch_time: float = 0.0
    
    # Model stats
    total_params: int = 0
    trainable_params: int = 0


@dataclass
class InferenceProfile:
    """Profiling results for inference."""
    model_name: str
    batch_size: int
    num_samples: int
    
    # Timing statistics (in ms)
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    throughput: float  # samples per second
    
    # Memory
    peak_memory_mb: float


class ResourceMonitor:
    """
    Monitor and log resource usage during training and inference.
    """
    
    def __init__(self, enabled: bool = None):
        self.enabled = enabled if enabled is not None else config.MONITOR_RESOURCES
        self.snapshots: List[ResourceSnapshot] = []
        self.inference_profiles: List[InferenceProfile] = []
        
        self._batch_start_time = None
        self._last_log_batch = 0
    
    def start_batch(self):
        """Mark the start of a batch for timing."""
        if self.enabled:
            self._batch_start_time = time.time()
    
    def end_batch(
        self,
        task_id: int,
        epoch: int,
        batch_idx: int,
        model: torch.nn.Module = None,
        force_log: bool = False
    ):
        """Record resource snapshot at end of batch."""
        if not self.enabled:
            return
        
        # Check if we should log this batch
        if not force_log and batch_idx - self._last_log_batch < config.RESOURCE_LOG_INTERVAL:
            return
        
        self._last_log_batch = batch_idx
        
        batch_time = 0.0
        if self._batch_start_time is not None:
            batch_time = (time.time() - self._batch_start_time) * 1000  # ms
        
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            task_id=task_id,
            epoch=epoch,
            batch_idx=batch_idx,
            batch_time=batch_time
        )
        
        # GPU memory
        if config.LOG_GPU_MEMORY and torch.cuda.is_available():
            snapshot.gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            snapshot.gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        # CPU memory
        if config.LOG_CPU_MEMORY:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            snapshot.cpu_memory_used = mem_info.rss / (1024 ** 2)
            snapshot.cpu_memory_percent = process.memory_percent()
        
        # Model parameters
        if model is not None:
            snapshot.total_params = sum(p.numel() for p in model.parameters())
            snapshot.trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        
        self.snapshots.append(snapshot)
    
    @torch.no_grad()
    def profile_inference(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        model_name: str = "model"
    ) -> InferenceProfile:
        """
        Profile model inference time.
        
        Args:
            model: Model to profile
            input_shape: Shape of input tensor (excluding batch dimension)
            batch_size: Batch size for profiling
            num_iterations: Number of timing iterations
            warmup_iterations: Number of warmup iterations (not timed)
            model_name: Name for the profile
        
        Returns:
            InferenceProfile with timing statistics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time iterations
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        import numpy as np
        latencies = np.array(latencies)
        
        # Peak memory
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        profile = InferenceProfile(
            model_name=model_name,
            batch_size=batch_size,
            num_samples=batch_size * num_iterations,
            mean_latency=float(np.mean(latencies)),
            std_latency=float(np.std(latencies)),
            min_latency=float(np.min(latencies)),
            max_latency=float(np.max(latencies)),
            throughput=batch_size * 1000 / float(np.mean(latencies)),
            peak_memory_mb=peak_memory
        )
        
        self.inference_profiles.append(profile)
        return profile
    
    def get_summary(self) -> Dict:
        """Get summary of resource usage."""
        if not self.snapshots:
            return {}
        
        import numpy as np
        
        gpu_mem = [s.gpu_memory_allocated for s in self.snapshots]
        cpu_mem = [s.cpu_memory_used for s in self.snapshots]
        batch_times = [s.batch_time for s in self.snapshots if s.batch_time > 0]
        
        summary = {
            "num_snapshots": len(self.snapshots),
            "gpu_memory_mb": {
                "mean": np.mean(gpu_mem) if gpu_mem else 0,
                "max": np.max(gpu_mem) if gpu_mem else 0,
                "min": np.min(gpu_mem) if gpu_mem else 0
            },
            "cpu_memory_mb": {
                "mean": np.mean(cpu_mem) if cpu_mem else 0,
                "max": np.max(cpu_mem) if cpu_mem else 0,
                "min": np.min(cpu_mem) if cpu_mem else 0
            },
            "batch_time_ms": {
                "mean": np.mean(batch_times) if batch_times else 0,
                "max": np.max(batch_times) if batch_times else 0,
                "min": np.min(batch_times) if batch_times else 0
            }
        }
        
        return summary
    
    def print_summary(self):
        """Print resource usage summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        
        if not summary:
            print("No resource data collected.")
            return
        
        print(f"\nSnapshots collected: {summary['num_snapshots']}")
        
        print("\nGPU Memory (MB):")
        gpu = summary['gpu_memory_mb']
        print(f"  Mean: {gpu['mean']:.2f}, Max: {gpu['max']:.2f}, Min: {gpu['min']:.2f}")
        
        print("\nCPU Memory (MB):")
        cpu = summary['cpu_memory_mb']
        print(f"  Mean: {cpu['mean']:.2f}, Max: {cpu['max']:.2f}, Min: {cpu['min']:.2f}")
        
        print("\nBatch Time (ms):")
        batch = summary['batch_time_ms']
        print(f"  Mean: {batch['mean']:.2f}, Max: {batch['max']:.2f}, Min: {batch['min']:.2f}")
        
        # Inference profiles
        if self.inference_profiles:
            print("\nInference Profiles:")
            for profile in self.inference_profiles:
                print(f"\n  {profile.model_name} (batch_size={profile.batch_size}):")
                print(f"    Latency: {profile.mean_latency:.2f} +/- {profile.std_latency:.2f} ms")
                print(f"    Throughput: {profile.throughput:.2f} samples/sec")
                print(f"    Peak Memory: {profile.peak_memory_mb:.2f} MB")
        
        print("=" * 60)


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> Dict[str, int]:
    """
    Count model parameters by layer type.
    
    Returns:
        Dictionary with parameter counts
    """
    counts = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "by_layer": {}
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        counts["total"] += num_params
        
        if param.requires_grad:
            counts["trainable"] += num_params
        else:
            counts["non_trainable"] += num_params
        
        # Group by layer name
        layer_name = name.split('.')[0]
        if layer_name not in counts["by_layer"]:
            counts["by_layer"][layer_name] = 0
        counts["by_layer"][layer_name] += num_params
    
    if verbose:
        print("\nParameter Count:")
        print(f"  Total: {counts['total']:,}")
        print(f"  Trainable: {counts['trainable']:,}")
        print(f"  Non-trainable: {counts['non_trainable']:,}")
        
        print("\nBy layer:")
        for layer, count in counts["by_layer"].items():
            print(f"  {layer}: {count:,}")
    
    return counts


def print_parameter_table(
    param_counts: Dict[str, Dict[str, int]],
    method_names: List[str] = None
):
    """
    Print a formatted table of parameter counts across methods/tasks.
    
    Args:
        param_counts: Dict mapping method/task to parameter counts
        method_names: Optional list of method names for ordering
    """
    if not param_counts:
        return
    
    print("\n" + "=" * 80)
    print("PARAMETER COUNT TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Method/Task':<30} {'Total':>15} {'Trainable':>15} {'Growth':>15}")
    print("-" * 80)
    
    # Get baseline for growth calculation
    methods = method_names or list(param_counts.keys())
    baseline = None
    
    for method in methods:
        if method not in param_counts:
            continue
        
        counts = param_counts[method]
        total = counts.get("total", 0)
        trainable = counts.get("trainable", total)
        
        if baseline is None:
            baseline = total
            growth = "baseline"
        else:
            growth = f"{(total / baseline - 1) * 100:+.1f}%"
        
        print(f"{method:<30} {total:>15,} {trainable:>15,} {growth:>15}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test resource monitor
    monitor = ResourceMonitor()
    
    # Simulate some batches
    for i in range(5):
        monitor.start_batch()
        time.sleep(0.01)  # Simulate work
        monitor.end_batch(task_id=0, epoch=0, batch_idx=i * 10, force_log=True)
    
    monitor.print_summary()
