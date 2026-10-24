"""
INT8 Quantization Utilities for Model Compression.

Implements:
- Dynamic quantization (PyTorch native)
- Static quantization with calibration
- Comparison of FP32 vs INT8 models

This allows comparison of model size and inference speed
with and without quantization.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Tuple, Optional
import time
import copy
import config


class QuantizationAnalyzer:
    """
    Analyze and compare models with INT8 quantization.
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        self.original_model = model
        self.device = device or config.DEVICE
        self.quantized_model = None
        
    def apply_dynamic_quantization(self) -> nn.Module:
        """
        Apply dynamic INT8 quantization.
        
        Dynamic quantization quantizes weights statically but
        activations dynamically during inference.
        
        Best for: Models with LSTM/Linear layers where compute
        is dominated by matrix multiplications.
        """
        # Move to CPU for quantization (PyTorch requirement)
        model_cpu = copy.deepcopy(self.original_model).cpu()
        
        # Quantize linear layers dynamically
        self.quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        return self.quantized_model
    
    def apply_static_quantization(
        self,
        calibration_data: torch.utils.data.DataLoader,
        num_calibration_batches: int = None
    ) -> nn.Module:
        """
        Apply static INT8 quantization with calibration.
        
        Static quantization quantizes both weights and activations
        using calibration data to determine optimal scale/zero-point.
        
        Best for: CNN models with batch normalization.
        """
        num_batches = num_calibration_batches or config.CALIBRATION_SAMPLES // config.BATCH_SIZE
        
        # Prepare model for static quantization
        model_cpu = copy.deepcopy(self.original_model).cpu()
        model_cpu.eval()
        
        # Set quantization configuration
        model_cpu.qconfig = quant.get_default_qconfig('fbgemm')
        
        # Fuse modules if possible (Conv + BN + ReLU)
        # Note: This requires model-specific fusion rules
        try:
            model_fused = quant.fuse_modules(
                model_cpu,
                [['conv', 'bn', 'relu']],  # Example fusion pattern
                inplace=False
            )
        except:
            model_fused = model_cpu
        
        # Prepare for calibration
        model_prepared = quant.prepare(model_fused, inplace=False)
        
        # Run calibration
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                if i >= num_batches:
                    break
                inputs = inputs.cpu()
                try:
                    model_prepared(inputs)
                except:
                    # Some models may not support forward without task_id
                    model_prepared(inputs, task_id=0)
        
        # Convert to quantized model
        self.quantized_model = quant.convert(model_prepared, inplace=False)
        
        return self.quantized_model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size in memory.
        
        Returns:
            Dictionary with size in MB and parameter count
        """
        # Count parameters
        param_size = 0
        param_count = 0
        
        for param in model.parameters():
            param_count += param.numel()
            param_size += param.numel() * param.element_size()
        
        # Count buffers
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "param_count": param_count,
            "param_size_mb": param_size / (1024 ** 2),
            "buffer_size_mb": buffer_size / (1024 ** 2),
            "total_size_mb": total_size / (1024 ** 2)
        }
    
    @torch.no_grad()
    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference time.
        
        Returns:
            Dictionary with timing statistics in milliseconds
        """
        model.eval()
        
        # Determine device (quantized models are CPU-only)
        is_quantized = hasattr(model, 'qconfig') or any(
            'quantized' in str(type(m)).lower() for m in model.modules()
        )
        device = 'cpu' if is_quantized else self.device
        
        model = model.to(device)
        dummy_input = torch.randn(1, *input_shape, device=device)
        
        # Warmup
        for _ in range(warmup):
            try:
                model(dummy_input)
            except:
                model(dummy_input, task_id=0)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            try:
                model(dummy_input)
            except:
                model(dummy_input, task_id=0)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        import numpy as np
        times = np.array(times)
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "device": device
        }
    
    def compare_fp32_int8(
        self,
        input_shape: Tuple,
        num_iterations: int = 50
    ) -> Dict[str, Dict]:
        """
        Compare FP32 original model with INT8 quantized model.
        
        Returns:
            Dictionary with comparison metrics
        """
        results = {}
        
        # FP32 model
        fp32_size = self.get_model_size(self.original_model)
        fp32_timing = self.benchmark_inference(
            self.original_model, input_shape, num_iterations
        )
        results["fp32"] = {
            "size": fp32_size,
            "timing": fp32_timing
        }
        
        # INT8 model (apply if not already done)
        if self.quantized_model is None:
            self.apply_dynamic_quantization()
        
        int8_size = self.get_model_size(self.quantized_model)
        int8_timing = self.benchmark_inference(
            self.quantized_model, input_shape, num_iterations
        )
        results["int8"] = {
            "size": int8_size,
            "timing": int8_timing
        }
        
        # Compute improvements
        size_reduction = 1 - (int8_size["total_size_mb"] / fp32_size["total_size_mb"])
        speedup = fp32_timing["mean_ms"] / int8_timing["mean_ms"]
        
        results["comparison"] = {
            "size_reduction_percent": size_reduction * 100,
            "speedup_factor": speedup
        }
        
        return results
    
    def print_comparison(self, comparison: Dict[str, Dict]):
        """Print formatted comparison table."""
        print("\n" + "=" * 70)
        print("FP32 vs INT8 QUANTIZATION COMPARISON")
        print("=" * 70)
        
        fp32 = comparison["fp32"]
        int8 = comparison["int8"]
        comp = comparison["comparison"]
        
        print(f"\n{'Metric':<30} {'FP32':>15} {'INT8':>15} {'Diff':>10}")
        print("-" * 70)
        
        # Size comparison
        print(f"{'Model Size (MB)':<30} "
              f"{fp32['size']['total_size_mb']:>15.2f} "
              f"{int8['size']['total_size_mb']:>15.2f} "
              f"{-comp['size_reduction_percent']:>+9.1f}%")
        
        # Parameter count
        print(f"{'Parameters':<30} "
              f"{fp32['size']['param_count']:>15,} "
              f"{int8['size']['param_count']:>15,}")
        
        # Timing comparison
        print(f"{'Inference Time (ms)':<30} "
              f"{fp32['timing']['mean_ms']:>15.2f} "
              f"{int8['timing']['mean_ms']:>15.2f} "
              f"{comp['speedup_factor']:>9.2f}x")
        
        print("-" * 70)
        print(f"Device: FP32 on {fp32['timing']['device']}, INT8 on {int8['timing']['device']}")
        print("=" * 70)


def create_parameter_count_table(
    models_dict: Dict[str, nn.Module],
    include_per_layer: bool = False
) -> str:
    """
    Create a formatted parameter count table for multiple models.
    
    Args:
        models_dict: Dictionary mapping model name to model
        include_per_layer: Whether to include per-layer breakdown
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PARAMETER COUNT TABLE (Showing Network Expansion)")
    lines.append("=" * 80)
    lines.append(f"{'Model/Config':<35} {'Total Params':>15} {'Size (MB)':>12} {'vs Base':>12}")
    lines.append("-" * 80)
    
    base_params = None
    
    for name, model in models_dict.items():
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate size (assuming FP32)
        size_mb = total_params * 4 / (1024 ** 2)
        
        if base_params is None:
            base_params = total_params
            vs_base = "baseline"
        else:
            vs_base = f"{(total_params / base_params - 1) * 100:+.1f}%"
        
        lines.append(f"{name:<35} {total_params:>15,} {size_mb:>11.2f} {vs_base:>12}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test quantization
    from models import ContinualCNN
    
    model = ContinualCNN(use_sparse_expansion=False)
    model.add_task(0)
    
    analyzer = QuantizationAnalyzer(model)
    
    print("Applying dynamic quantization...")
    quantized = analyzer.apply_dynamic_quantization()
    
    print("Comparing FP32 vs INT8...")
    comparison = analyzer.compare_fp32_int8(input_shape=(3, 32, 32), num_iterations=20)
    analyzer.print_comparison(comparison)
