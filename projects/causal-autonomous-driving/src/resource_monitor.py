"""Resource monitoring utilities."""

import os
import time
import psutil
import tracemalloc
import platform
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import torch


@dataclass
class HardwareInfo:
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    torch_version: str
    
    @classmethod
    def collect(cls):
        return cls(
            platform.processor() or "Unknown",
            psutil.cpu_count(logical=False) or 0,
            psutil.virtual_memory().total / (1024**3),
            torch.__version__
        )
    
    def to_markdown_table(self) -> str:
        return f"""| Spec | Value |
|------|-------|
| CPU | {self.cpu_model} |
| Cores | {self.cpu_cores} |
| RAM | {self.ram_gb:.1f} GB |
| PyTorch | {self.torch_version} |"""


@dataclass
class ResourceMetrics:
    wall_time_seconds: float
    cpu_percent_mean: float
    memory_peak_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    def __init__(self):
        self._monitoring = False
        self._cpu_samples = []
        self._start_time = None
        self._process = psutil.Process(os.getpid())
    
    def _monitor_loop(self):
        while self._monitoring:
            try:
                self._cpu_samples.append(self._process.cpu_percent())
            except:
                pass
            time.sleep(0.1)
    
    def start(self):
        tracemalloc.start()
        self._cpu_samples = []
        self._start_time = time.perf_counter()
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> ResourceMetrics:
        self._monitoring = False
        self._thread.join(timeout=1)
        wall_time = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return ResourceMetrics(
            wall_time, 
            np.mean(self._cpu_samples) if self._cpu_samples else 0,
            peak / (1024 * 1024)
        )


def setup_cpu_optimization(num_threads=None):
    if num_threads is None:
        num_threads = min(psutil.cpu_count(logical=False) or 4, 8)
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    return {"num_threads": num_threads}
