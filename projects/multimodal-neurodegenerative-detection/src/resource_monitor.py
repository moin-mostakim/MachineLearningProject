"""Resource monitoring utilities."""

import os
import time
import psutil
import tracemalloc
import platform
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import torch


@dataclass
class HardwareInfo:
    """System hardware specifications."""
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    python_version: str = ""
    torch_version: str = ""
    
    @classmethod
    def collect(cls) -> "HardwareInfo":
        return cls(
            cpu_model=platform.processor() or "Unknown",
            cpu_cores_physical=psutil.cpu_count(logical=False) or 0,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 0,
            ram_total_gb=psutil.virtual_memory().total / (1024**3),
            python_version=platform.python_version(),
            torch_version=torch.__version__
        )
    
    def to_markdown_table(self) -> str:
        return f"""| Specification | Value |
|--------------|-------|
| CPU Model | {self.cpu_model} |
| Physical Cores | {self.cpu_cores_physical} |
| Logical Cores | {self.cpu_cores_logical} |
| Total RAM | {self.ram_total_gb:.1f} GB |
| Python | {self.python_version} |
| PyTorch | {self.torch_version} |"""


@dataclass
class ResourceMetrics:
    """Resource metrics."""
    wall_time_seconds: float = 0.0
    cpu_percent_mean: float = 0.0
    memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Monitor resources during execution."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self._monitoring = False
        self._thread = None
        self._cpu_samples = []
        self._start_time = None
        self._process = psutil.Process(os.getpid())
    
    def _monitor_loop(self):
        while self._monitoring:
            try:
                self._cpu_samples.append(self._process.cpu_percent())
            except Exception:
                pass
            time.sleep(self.sample_interval)
    
    def start(self):
        tracemalloc.start()
        self._cpu_samples = []
        self._start_time = time.perf_counter()
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> ResourceMetrics:
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        wall_time = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return ResourceMetrics(
            wall_time_seconds=wall_time,
            cpu_percent_mean=np.mean(self._cpu_samples) if self._cpu_samples else 0.0,
            memory_peak_mb=peak / (1024 * 1024)
        )


def setup_cpu_optimization(num_threads: Optional[int] = None):
    """Configure PyTorch for CPU."""
    if num_threads is None:
        num_threads = min(psutil.cpu_count(logical=False) or 4, 8)
    
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    
    return {"num_threads": num_threads}
