"""Resource monitoring for federated learning experiments."""

import time
import os
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class ResourceMonitor:
    """
    Monitor system resources during training.
    
    Tracks:
    - CPU usage
    - Memory usage
    - GPU memory and utilization (if available)
    - Timing
    """
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Phase tracking
        self.phases: Dict[str, Dict] = {}
        self.current_phase: Optional[str] = None
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.is_running = True
        self.snapshots = []
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> Dict:
        """Stop monitoring and return summary."""
        self.is_running = False
        self.end_time = time.time()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.interval * 2)
        
        return self.get_summary()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # GPU (if available)
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Note: GPU utilization requires nvidia-smi or pynvml
            except Exception:
                pass
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_utilization=gpu_utilization
        )
    
    def start_phase(self, phase_name: str):
        """Mark start of a training phase."""
        self.current_phase = phase_name
        self.phases[phase_name] = {
            "start_time": time.time(),
            "start_snapshot_idx": len(self.snapshots)
        }
    
    def end_phase(self, phase_name: str = None):
        """Mark end of a training phase."""
        if phase_name is None:
            phase_name = self.current_phase
        
        if phase_name and phase_name in self.phases:
            self.phases[phase_name]["end_time"] = time.time()
            self.phases[phase_name]["end_snapshot_idx"] = len(self.snapshots)
            self.phases[phase_name]["duration"] = (
                self.phases[phase_name]["end_time"] - 
                self.phases[phase_name]["start_time"]
            )
        
        self.current_phase = None
    
    def get_summary(self) -> Dict:
        """Get summary of resource usage."""
        if not self.snapshots:
            return {"error": "No snapshots recorded"}
        
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        memory_percents = [s.memory_percent for s in self.snapshots]
        memory_used = [s.memory_used_gb for s in self.snapshots]
        
        summary = {
            "duration_seconds": (self.end_time or time.time()) - (self.start_time or 0),
            "num_snapshots": len(self.snapshots),
            "cpu": {
                "mean_percent": sum(cpu_percents) / len(cpu_percents),
                "max_percent": max(cpu_percents),
                "min_percent": min(cpu_percents)
            },
            "memory": {
                "mean_percent": sum(memory_percents) / len(memory_percents),
                "max_percent": max(memory_percents),
                "peak_used_gb": max(memory_used),
                "mean_used_gb": sum(memory_used) / len(memory_used)
            }
        }
        
        # GPU stats
        gpu_memory = [s.gpu_memory_used_gb for s in self.snapshots if s.gpu_memory_used_gb is not None]
        if gpu_memory:
            summary["gpu"] = {
                "peak_memory_gb": max(gpu_memory),
                "mean_memory_gb": sum(gpu_memory) / len(gpu_memory),
                "total_memory_gb": self.snapshots[-1].gpu_memory_total_gb
            }
        
        # Phase stats
        if self.phases:
            summary["phases"] = {}
            for phase_name, phase_data in self.phases.items():
                if "duration" in phase_data:
                    summary["phases"][phase_name] = {
                        "duration_seconds": phase_data["duration"]
                    }
        
        return summary
    
    def get_timeline(self) -> List[Dict]:
        """Get timeline of all snapshots."""
        if not self.snapshots or not self.start_time:
            return []
        
        return [
            {
                "time_offset": s.timestamp - self.start_time,
                "cpu_percent": s.cpu_percent,
                "memory_percent": s.memory_percent,
                "memory_used_gb": s.memory_used_gb,
                "gpu_memory_gb": s.gpu_memory_used_gb
            }
            for s in self.snapshots
        ]


def get_gpu_info() -> Dict:
    """Get GPU information if available."""
    if not HAS_TORCH:
        return {"available": False, "reason": "torch not installed"}
    
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    
    try:
        device_count = torch.cuda.device_count()
        devices = []
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "device_count": device_count,
            "devices": devices,
            "current_device": torch.cuda.current_device()
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def get_system_info() -> Dict:
    """Get system information."""
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "platform": os.uname().sysname,
        "python_version": os.sys.version,
        "gpu": get_gpu_info()
    }
