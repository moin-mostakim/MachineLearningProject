"""Model utilities for federated learning."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import copy


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Get model size in memory.
    
    Args:
        model: PyTorch model
        unit: Size unit ("B", "KB", "MB", "GB")
        
    Returns:
        Model size in specified unit
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return total_size / divisors.get(unit.upper(), 1024**2)


def get_parameter_table(model: nn.Module) -> Dict[str, Dict]:
    """
    Get detailed parameter information per layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with layer names and parameter counts
    """
    table = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip container modules
        
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        if num_params > 0:
            table[name if name else "root"] = {
                "type": module.__class__.__name__,
                "parameters": num_params,
                "trainable": trainable,
                "size_kb": sum(
                    p.nelement() * p.element_size() 
                    for p in module.parameters(recurse=False)
                ) / 1024
            }
    
    return table


def print_parameter_table(model: nn.Module, model_name: str = "Model"):
    """Print formatted parameter table."""
    table = get_parameter_table(model)
    
    print(f"\n{'='*70}")
    print(f"Parameter Table: {model_name}")
    print(f"{'='*70}")
    print(f"{'Layer':<30} {'Type':<15} {'Params':>12} {'Size (KB)':>10}")
    print(f"{'-'*70}")
    
    total_params = 0
    total_size = 0
    
    for name, info in table.items():
        print(f"{name:<30} {info['type']:<15} {info['parameters']:>12,} {info['size_kb']:>10.2f}")
        total_params += info['parameters']
        total_size += info['size_kb']
    
    print(f"{'-'*70}")
    print(f"{'Total':<30} {'':<15} {total_params:>12,} {total_size:>10.2f}")
    print(f"{'='*70}")
    
    return total_params, total_size


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Quantize model to INT8.
    
    Args:
        model: PyTorch model (CPU)
        method: Quantization method ("dynamic", "static")
        dtype: Quantization dtype
        
    Returns:
        Quantized model
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    
    if method == "dynamic":
        # Dynamic quantization - no calibration needed
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
    elif method == "static":
        # Static quantization would require calibration data
        # Here we use a simplified version
        model_cpu.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        torch.ao.quantization.prepare(model_cpu, inplace=True)
        # Would need calibration here
        quantized_model = torch.ao.quantization.convert(model_cpu, inplace=False)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return quantized_model


def compare_quantization(
    model: nn.Module,
    test_input: torch.Tensor = None,
    device: str = "cpu"
) -> Dict:
    """
    Compare original and quantized model.
    
    Args:
        model: Original model
        test_input: Optional test input for inference comparison
        device: Device for comparison
        
    Returns:
        Comparison results
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    
    # Original model stats
    original_params = count_parameters(model_cpu)
    original_size = get_model_size(model_cpu, "MB")
    
    # Quantize
    quantized_model = quantize_model(model_cpu, method="dynamic")
    quantized_size = get_model_size(quantized_model, "MB")
    
    results = {
        "original": {
            "parameters": original_params,
            "size_mb": original_size,
            "dtype": "float32"
        },
        "quantized": {
            "size_mb": quantized_size,
            "dtype": "int8"
        },
        "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
        "size_reduction_percent": (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    }
    
    # Inference comparison
    if test_input is not None:
        test_input = test_input.cpu()
        
        import time
        
        # Original model inference
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = model_cpu(test_input)
            original_time = (time.time() - start) / 100
        
        # Quantized model inference
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = quantized_model(test_input)
            quantized_time = (time.time() - start) / 100
        
        results["inference"] = {
            "original_ms": original_time * 1000,
            "quantized_ms": quantized_time * 1000,
            "speedup": original_time / quantized_time if quantized_time > 0 else 0
        }
    
    return results


def print_quantization_comparison(model: nn.Module, model_name: str = "Model"):
    """Print quantization comparison results."""
    # Create dummy input based on model type
    test_input = None
    try:
        # Try to infer input shape
        first_layer = list(model.modules())[1]
        if isinstance(first_layer, nn.Conv2d):
            in_channels = first_layer.in_channels
            # Use 28x28 for single channel (MNIST-like), 32x32 for 3 channels (CIFAR-like)
            img_size = 28 if in_channels == 1 else 32
            test_input = torch.randn(1, in_channels, img_size, img_size)
        elif isinstance(first_layer, nn.Linear):
            in_features = first_layer.in_features
            test_input = torch.randn(1, in_features)
    except Exception:
        pass
    
    results = compare_quantization(model, test_input)
    
    print(f"\n{'='*60}")
    print(f"Quantization Comparison: {model_name}")
    print(f"{'='*60}")
    print(f"Original Model (FP32):")
    print(f"  Parameters: {results['original']['parameters']:,}")
    print(f"  Size: {results['original']['size_mb']:.2f} MB")
    print(f"\nQuantized Model (INT8):")
    print(f"  Size: {results['quantized']['size_mb']:.2f} MB")
    print(f"\nCompression:")
    print(f"  Ratio: {results['compression_ratio']:.2f}x")
    print(f"  Size Reduction: {results['size_reduction_percent']:.1f}%")
    
    if "inference" in results:
        print(f"\nInference (CPU):")
        print(f"  Original: {results['inference']['original_ms']:.2f} ms")
        print(f"  Quantized: {results['inference']['quantized_ms']:.2f} ms")
        print(f"  Speedup: {results['inference']['speedup']:.2f}x")
    
    print(f"{'='*60}")
    
    return results
