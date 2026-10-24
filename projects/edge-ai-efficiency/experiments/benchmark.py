"""
Benchmarking utilities for edge AI models.

Measures:
- Accuracy
- Latency (CPU and GPU)
- Model size
- FLOPs
- Memory usage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import time
import json
import os

try:
    from utils.metrics import (
        count_parameters, 
        get_model_size_mb, 
        count_flops, 
        format_flops,
        create_metrics_table
    )
    from utils.profiler import (
        measure_latency,
        measure_throughput,
        profile_model,
        estimate_edge_metrics,
        LatencyStats
    )
except ImportError:
    from ..utils.metrics import (
        count_parameters, 
        get_model_size_mb, 
        count_flops, 
        format_flops,
        create_metrics_table
    )
    from ..utils.profiler import (
        measure_latency,
        measure_throughput,
        profile_model,
        estimate_edge_metrics,
        LatencyStats
    )


class Benchmarker:
    """
    Comprehensive model benchmarking.
    
    Measures accuracy, latency, model size, FLOPs, and
    estimates edge deployment metrics.
    """
    
    def __init__(
        self,
        val_loader: DataLoader,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
        device: str = 'cuda',
        num_warmup: int = 10,
        num_benchmark_runs: int = 100
    ):
        """
        Initialize benchmarker.
        
        Args:
            val_loader: Validation data loader
            input_size: Input tensor size for profiling
            device: Device for accuracy evaluation
            num_warmup: Number of warmup runs for latency
            num_benchmark_runs: Number of runs for latency measurement
        """
        self.val_loader = val_loader
        self.input_size = input_size
        self.device = device
        self.num_warmup = num_warmup
        self.num_benchmark_runs = num_benchmark_runs
        
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def measure_accuracy(self, model: nn.Module) -> float:
        """
        Measure model accuracy on validation set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Accuracy percentage
        """
        model.eval()
        model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100.0 * correct / total
    
    def benchmark_model(
        self,
        model: nn.Module,
        name: str,
        benchmark_cpu: bool = True,
        benchmark_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark of a single model.
        
        Args:
            model: Model to benchmark
            name: Name for the model
            benchmark_cpu: Measure CPU latency
            benchmark_gpu: Measure GPU latency (if available)
            
        Returns:
            Dictionary with all benchmark results
        """
        print(f"\nBenchmarking {name}...")
        
        results = {
            'name': name,
            'params': count_parameters(model),
            'size_mb': get_model_size_mb(model),
            'flops': count_flops(model, self.input_size, device='cpu')
        }
        
        # Accuracy
        print("  Measuring accuracy...")
        results['accuracy'] = self.measure_accuracy(model)
        
        # CPU latency (for edge simulation)
        if benchmark_cpu:
            print("  Measuring CPU latency...")
            cpu_latency = measure_latency(
                model, self.input_size, device='cpu',
                num_warmup=self.num_warmup, num_runs=self.num_benchmark_runs
            )
            results['cpu_latency'] = cpu_latency
            results['latency_ms'] = cpu_latency.mean_ms
        
        # GPU latency
        if benchmark_gpu and torch.cuda.is_available():
            print("  Measuring GPU latency...")
            gpu_latency = measure_latency(
                model, self.input_size, device='cuda',
                num_warmup=self.num_warmup, num_runs=self.num_benchmark_runs
            )
            results['gpu_latency'] = gpu_latency
        
        # Edge metrics estimation
        gflops = results['flops'] / 1e9
        results['edge_metrics'] = estimate_edge_metrics(
            latency_ms=results.get('latency_ms', 10.0),
            gflops=gflops
        )
        
        self.results[name] = results
        
        # Print summary
        print(f"  Results for {name}:")
        print(f"    Accuracy: {results['accuracy']:.2f}%")
        print(f"    Parameters: {results['params'] / 1e6:.2f}M")
        print(f"    Size: {results['size_mb']:.2f} MB")
        print(f"    FLOPs: {format_flops(results['flops'])}")
        if 'cpu_latency' in results:
            print(f"    CPU Latency: {results['cpu_latency'].mean_ms:.2f}±{results['cpu_latency'].std_ms:.2f} ms")
        if 'gpu_latency' in results:
            print(f"    GPU Latency: {results['gpu_latency'].mean_ms:.2f}±{results['gpu_latency'].std_ms:.2f} ms")
        
        return results
    
    def benchmark_multiple(
        self,
        models: Dict[str, nn.Module],
        benchmark_cpu: bool = True,
        benchmark_gpu: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple models.
        
        Args:
            models: Dictionary mapping names to models
            benchmark_cpu: Measure CPU latency
            benchmark_gpu: Measure GPU latency
            
        Returns:
            Dictionary with all results
        """
        for name, model in models.items():
            self.benchmark_model(model, name, benchmark_cpu, benchmark_gpu)
        
        return self.results
    
    def compare_quantization(
        self,
        float_model: nn.Module,
        int8_model: nn.Module,
        int4_model: Optional[nn.Module] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare float, INT8, and INT4 models.
        
        Args:
            float_model: Float32 model
            int8_model: INT8 quantized model
            int4_model: Optional INT4 quantized model
            
        Returns:
            Comparison results
        """
        models = {
            'Float32': float_model,
            'INT8': int8_model
        }
        
        if int4_model is not None:
            models['INT4'] = int4_model
        
        return self.benchmark_multiple(models)
    
    def get_summary_table(self) -> str:
        """
        Get a summary table of all benchmarked models.
        
        Returns:
            Formatted table string
        """
        return create_metrics_table(self.results)
    
    def get_pareto_models(self) -> List[str]:
        """
        Find Pareto-optimal models (accuracy vs latency).
        
        Returns:
            List of Pareto-optimal model names
        """
        if not self.results:
            return []
        
        pareto = []
        
        for name, metrics in self.results.items():
            acc = metrics.get('accuracy', 0)
            latency = metrics.get('latency_ms', float('inf'))
            
            is_dominated = False
            for other_name, other_metrics in self.results.items():
                if other_name == name:
                    continue
                
                other_acc = other_metrics.get('accuracy', 0)
                other_latency = other_metrics.get('latency_ms', float('inf'))
                
                # Check if other dominates this
                if (other_acc >= acc and other_latency <= latency and
                    (other_acc > acc or other_latency < latency)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(name)
        
        return pareto
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to JSON.
        
        Args:
            filepath: Path to save results
        """
        # Convert LatencyStats to dict for JSON serialization
        serializable_results = {}
        
        for name, metrics in self.results.items():
            serializable_results[name] = {}
            for key, value in metrics.items():
                if isinstance(value, LatencyStats):
                    serializable_results[name][key] = {
                        'mean_ms': value.mean_ms,
                        'std_ms': value.std_ms,
                        'min_ms': value.min_ms,
                        'max_ms': value.max_ms,
                        'p50_ms': value.p50_ms,
                        'p90_ms': value.p90_ms,
                        'p99_ms': value.p99_ms
                    }
                else:
                    serializable_results[name][key] = value
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load benchmark results from JSON.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)


def create_benchmarker(
    val_loader: DataLoader,
    input_size: Tuple[int, ...],
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Benchmarker:
    """Factory function for benchmarker."""
    return Benchmarker(
        val_loader=val_loader,
        input_size=input_size,
        device=device,
        num_warmup=config.get('num_warmup_runs', 10),
        num_benchmark_runs=config.get('num_benchmark_runs', 100)
    )
