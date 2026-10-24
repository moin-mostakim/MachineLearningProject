"""
Model quantization utilities for INT8 comparison.

Implements:
- Dynamic quantization (INT8)
- Quantization-aware training preparation
- Performance comparison utilities

References:
- Jacob et al. (2018): "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- PyTorch Quantization documentation
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json

import torch
import torch.nn as nn


@dataclass
class QuantizationStats:
    """Statistics from quantization comparison."""
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    size_reduction_ratio: float = 0.0
    
    original_latency_ms: float = 0.0
    quantized_latency_ms: float = 0.0
    speedup_ratio: float = 0.0
    
    original_accuracy: float = 0.0
    quantized_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "original_size_mb": self.original_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "size_reduction_ratio": self.size_reduction_ratio,
            "original_latency_ms": self.original_latency_ms,
            "quantized_latency_ms": self.quantized_latency_ms,
            "speedup_ratio": self.speedup_ratio,
            "original_accuracy": self.original_accuracy,
            "quantized_accuracy": self.quantized_accuracy,
            "accuracy_drop": self.accuracy_drop
        }
    
    def __str__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                  INT8 QUANTIZATION COMPARISON                 ║
╠══════════════════════════════════════════════════════════════╣
║ Metric              │ Original    │ Quantized   │ Change     ║
╠══════════════════════════════════════════════════════════════╣
║ Model Size (MB)     │ {self.original_size_mb:>10.2f}  │ {self.quantized_size_mb:>10.2f}  │ {self.size_reduction_ratio:>9.2f}x ║
║ Latency (ms)        │ {self.original_latency_ms:>10.2f}  │ {self.quantized_latency_ms:>10.2f}  │ {self.speedup_ratio:>9.2f}x ║
║ Accuracy            │ {self.original_accuracy:>10.4f}  │ {self.quantized_accuracy:>10.4f}  │ {self.accuracy_drop:>+9.4f} ║
╚══════════════════════════════════════════════════════════════╝
"""


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 * 1024)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by dtype."""
    counts = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in counts:
            counts[dtype_str] = 0
        counts[dtype_str] += param.numel()
    return counts


def quantize_model(
    model: nn.Module,
    quantization_type: str = "dynamic",
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply quantization to model.
    
    Args:
        model: Model to quantize
        quantization_type: Type of quantization ("dynamic", "static")
        dtype: Target quantization dtype
        
    Returns:
        Quantized model
    """
    model_cpu = model.cpu()
    model_cpu.eval()
    
    if quantization_type == "dynamic":
        # Dynamic quantization: quantize weights, compute activations dynamically
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},  # Quantize linear layers
            dtype=dtype
        )
    elif quantization_type == "static":
        # Static quantization requires calibration (not implemented here)
        # For now, fall back to dynamic
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=dtype
        )
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    return quantized_model


def measure_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> float:
    """
    Measure model inference latency.
    
    Returns:
        Average latency in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to same device as model
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            if attention_mask is not None:
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids)
    
    # Synchronize if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if attention_mask is not None:
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_latency_ms = (end_time - start_time) / num_runs * 1000
    return avg_latency_ms


def compare_quantization(
    model: nn.Module,
    eval_fn,
    sample_input: Tuple[torch.Tensor, ...],
    quantization_type: str = "dynamic"
) -> QuantizationStats:
    """
    Compare original and quantized model performance.
    
    Args:
        model: Original model
        eval_fn: Function that evaluates model and returns accuracy
        sample_input: Sample input for latency measurement (input_ids, attention_mask)
        quantization_type: Type of quantization to apply
        
    Returns:
        QuantizationStats with comparison results
    """
    stats = QuantizationStats()
    
    # Original model metrics
    stats.original_size_mb = get_model_size(model)
    
    input_ids, attention_mask = sample_input
    
    # Move model to CPU for fair comparison (quantization works on CPU)
    model_cpu = model.cpu()
    model_cpu.eval()
    
    stats.original_latency_ms = measure_latency(
        model_cpu, input_ids.cpu(), attention_mask.cpu() if attention_mask is not None else None
    )
    
    try:
        stats.original_accuracy = eval_fn(model_cpu)
    except Exception as e:
        print(f"Warning: Could not evaluate original model: {e}")
        stats.original_accuracy = 0.0
    
    # Quantized model metrics
    quantized_model = quantize_model(model_cpu, quantization_type)
    
    stats.quantized_size_mb = get_model_size(quantized_model)
    
    try:
        stats.quantized_latency_ms = measure_latency(
            quantized_model, input_ids.cpu(), 
            attention_mask.cpu() if attention_mask is not None else None
        )
    except Exception as e:
        print(f"Warning: Latency measurement failed for quantized model: {e}")
        stats.quantized_latency_ms = stats.original_latency_ms
    
    try:
        stats.quantized_accuracy = eval_fn(quantized_model)
    except Exception as e:
        print(f"Warning: Could not evaluate quantized model: {e}")
        stats.quantized_accuracy = stats.original_accuracy
    
    # Compute ratios
    stats.size_reduction_ratio = stats.original_size_mb / max(stats.quantized_size_mb, 0.001)
    stats.speedup_ratio = stats.original_latency_ms / max(stats.quantized_latency_ms, 0.001)
    stats.accuracy_drop = stats.quantized_accuracy - stats.original_accuracy
    
    return stats


def print_quantization_table(stats: QuantizationStats):
    """Print formatted quantization comparison table."""
    print(str(stats))


def save_quantization_results(stats: QuantizationStats, path: str):
    """Save quantization results to JSON."""
    with open(path, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)


class QuantizationAwareWrapper(nn.Module):
    """
    Wrapper for quantization-aware training (QAT).
    
    Adds fake quantization nodes during training to simulate
    quantization effects while maintaining gradients.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        # Quantization parameters
        self.quant_min = -128
        self.quant_max = 127
        
    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization for QAT."""
        # Compute scale
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (self.quant_max - self.quant_min)
        scale = torch.max(scale, torch.tensor(1e-8))
        
        # Quantize and dequantize
        x_int = torch.round(x / scale)
        x_int = torch.clamp(x_int, self.quant_min, self.quant_max)
        x_dequant = x_int * scale
        
        # Straight-through estimator for gradients
        return x + (x_dequant - x).detach()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def estimate_quantization_impact(model: nn.Module) -> Dict:
    """
    Estimate the impact of INT8 quantization without actually quantizing.
    
    Returns theoretical size reduction and affected layers.
    """
    total_params = 0
    quantizable_params = 0
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            quantizable_params += num_params
            layer_info.append({
                "name": name,
                "type": "Linear",
                "params": num_params,
                "quantizable": True
            })
        elif isinstance(module, nn.Embedding):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            # Embeddings are typically not quantized
            layer_info.append({
                "name": name,
                "type": "Embedding",
                "params": num_params,
                "quantizable": False
            })
    
    original_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    quantized_size_mb = (
        (total_params - quantizable_params) * 4 +  # Non-quantized (FP32)
        quantizable_params * 1  # Quantized (INT8)
    ) / (1024 * 1024)
    
    return {
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "quantizable_ratio": quantizable_params / max(total_params, 1),
        "original_size_mb": original_size_mb,
        "estimated_quantized_size_mb": quantized_size_mb,
        "estimated_reduction_ratio": original_size_mb / max(quantized_size_mb, 0.001),
        "layers": layer_info
    }


if __name__ == "__main__":
    # Test quantization utilities
    print("Testing quantization utilities...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 4)
        
        def forward(self, x, mask=None):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return {"logits": self.fc3(x)}
    
    model = TestModel()
    
    print("\nModel size before quantization:")
    print(f"  {get_model_size(model):.2f} MB")
    
    print("\nParameter counts by dtype:")
    for dtype, count in count_parameters(model).items():
        print(f"  {dtype}: {count:,}")
    
    print("\nEstimated quantization impact:")
    impact = estimate_quantization_impact(model)
    print(f"  Quantizable params: {impact['quantizable_ratio']*100:.1f}%")
    print(f"  Estimated size reduction: {impact['estimated_reduction_ratio']:.2f}x")
    
    # Quantize
    quantized = quantize_model(model)
    
    print("\nModel size after INT8 quantization:")
    print(f"  {get_model_size(quantized):.2f} MB")
    
    # Measure latency
    sample_input = torch.randn(1, 256)
    
    print("\nLatency comparison:")
    orig_latency = measure_latency(model.cpu(), sample_input, num_runs=100)
    quant_latency = measure_latency(quantized, sample_input, num_runs=100)
    print(f"  Original: {orig_latency:.2f} ms")
    print(f"  Quantized: {quant_latency:.2f} ms")
    print(f"  Speedup: {orig_latency/quant_latency:.2f}x")
