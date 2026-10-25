"""Resource monitoring and profiling utilities."""

import os
import time
import json
import psutil
import tracemalloc
import platform
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from contextlib import contextmanager
import numpy as np

import torch


@dataclass
class HardwareInfo:
    """System hardware specifications."""
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    os_info: str = ""
    python_version: str = ""
    torch_version: str = ""
    
    @classmethod
    def collect(cls) -> "HardwareInfo":
        """Collect current hardware information."""
        cpu_freq = psutil.cpu_freq()
        return cls(
            cpu_model=platform.processor() or "Unknown CPU",
            cpu_cores_physical=psutil.cpu_count(logical=False) or 0,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 0,
            cpu_freq_mhz=cpu_freq.current if cpu_freq else 0.0,
            ram_total_gb=psutil.virtual_memory().total / (1024**3),
            os_info=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            torch_version=torch.__version__
        )
    
    def to_markdown_table(self) -> str:
        """Generate markdown table of hardware specs."""
        return f"""| Specification | Value |
|--------------|-------|
| CPU Model | {self.cpu_model} |
| Physical Cores | {self.cpu_cores_physical} |
| Logical Cores | {self.cpu_cores_logical} |
| CPU Frequency | {self.cpu_freq_mhz:.0f} MHz |
| Total RAM | {self.ram_total_gb:.1f} GB |
| OS | {self.os_info} |
| Python | {self.python_version} |
| PyTorch | {self.torch_version} |"""


@dataclass
class ResourceMetrics:
    """Collected resource metrics."""
    wall_time_seconds: float = 0.0
    cpu_percent_mean: float = 0.0
    cpu_percent_max: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Monitor CPU, memory, and time during execution."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self._monitoring = False
        self._thread = None
        self._cpu_samples = []
        self._start_time = None
        self._process = psutil.Process(os.getpid())
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                cpu_percent = self._process.cpu_percent()
                self._cpu_samples.append(cpu_percent)
            except Exception:
                pass
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start resource monitoring."""
        tracemalloc.start()
        self._cpu_samples = []
        self._start_time = time.perf_counter()
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return collected metrics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        wall_time = time.perf_counter() - self._start_time
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return ResourceMetrics(
            wall_time_seconds=wall_time,
            cpu_percent_mean=np.mean(self._cpu_samples) if self._cpu_samples else 0.0,
            cpu_percent_max=np.max(self._cpu_samples) if self._cpu_samples else 0.0,
            memory_peak_mb=peak / (1024 * 1024),
            memory_current_mb=current / (1024 * 1024),
            cpu_samples=self._cpu_samples
        )


@contextmanager
def monitor_resources(sample_interval: float = 0.1):
    """Context manager for resource monitoring."""
    monitor = ResourceMonitor(sample_interval)
    monitor.start()
    try:
        yield monitor
    finally:
        pass


class LatencyProfiler:
    """Profile inference latency with statistics."""
    
    def __init__(self, warmup_runs: int = 3, num_runs: int = 50):
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
    
    def profile(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """Profile a function's latency."""
        for _ in range(self.warmup_runs):
            fn(*args, **kwargs)
        
        latencies = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "num_runs": self.num_runs,
            "throughput_per_sec": float(1000.0 / np.mean(latencies))
        }


def setup_cpu_optimization(num_threads: Optional[int] = None):
    """Configure PyTorch for optimal CPU performance."""
    if num_threads is None:
        num_threads = min(psutil.cpu_count(logical=False) or 4, 8)
    
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    return {
        "num_threads": num_threads,
        "interop_threads": max(1, num_threads // 2)
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def save_resource_results(
    results: Dict[str, Any],
    output_path: str,
    hardware_info: Optional[HardwareInfo] = None
):
    """Save resource monitoring results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hardware_info is None:
        hardware_info = HardwareInfo.collect()
    
    full_results = {
        "hardware": asdict(hardware_info),
        "metrics": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    return output_path
