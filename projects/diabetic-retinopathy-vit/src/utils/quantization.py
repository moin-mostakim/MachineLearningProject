"""
INT8 Quantization utilities for model compression.

Compares model size and inference speed before/after quantization.
"""

import torch
import torch.nn as nn
try:
    import torch.quantization as quant
    from torch.quantization import quantize_dynamic
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
import numpy as np
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import copy


def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in megabytes
    """
    torch.save(model.state_dict(), '/tmp/temp_model.pt')
    size = Path('/tmp/temp_model.pt').stat().st_size / (1024 * 1024)
    Path('/tmp/temp_model.pt').unlink()
    return size


def count_parameters(model: nn.Module) -> Dict[str, int]:
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
        'non_trainable': total - trainable,
    }


def dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Apply dynamic INT8 quantization.
    
    Dynamic quantization quantizes weights and activations at runtime.
    Best for models with small batch sizes.
    
    Args:
        model: Model to quantize
        dtype: Quantization dtype (qint8 or float16)
        
    Returns:
        Quantized model
    """
    if not HAS_QUANTIZATION:
        print("Warning: Quantization not available, returning copy of original model")
        return copy.deepcopy(model)
    
    model_fp32 = copy.deepcopy(model)
    model_fp32.eval()
    
    # Quantize linear layers (main compute in transformers)
    quantized_model = quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=dtype,
    )
    
    return quantized_model


def prepare_static_quantization(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    backend: str = 'fbgemm',
) -> nn.Module:
    """
    Prepare model for static quantization with calibration.
    
    Static quantization determines quantization parameters during calibration
    and uses fixed parameters during inference. Generally faster than dynamic.
    
    Args:
        model: Model to quantize
        calibration_loader: Data loader for calibration
        num_batches: Number of calibration batches
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    if not HAS_QUANTIZATION:
        print("Warning: Quantization not available, returning copy of original model")
        return copy.deepcopy(model)
    
    model_fp32 = copy.deepcopy(model)
    model_fp32.eval()
    
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Prepare for quantization
    model_fp32.qconfig = quant.get_default_qconfig(backend)
    
    # Fuse layers where possible (conv+bn+relu)
    # For ViT, mainly linear layers so limited fusion
    
    # Insert observers
    model_prepared = quant.prepare(model_fp32)
    
    # Calibrate with representative data
    device = next(model.parameters()).device
    model_prepared.to('cpu')  # Quantization works on CPU
    
    print("Calibrating for static quantization...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            images = images.cpu()
            model_prepared(images)
    
    # Convert to quantized model
    quantized_model = quant.convert(model_prepared)
    
    return quantized_model


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Benchmark inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to benchmark on
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    x = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    # Synchronize if GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
    }


def compare_quantization(
    model: nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    include_static: bool = False,
) -> Dict[str, Dict]:
    """
    Compare FP32 vs INT8 quantization.
    
    Args:
        model: Model to compare
        calibration_loader: Data loader for static quantization
        input_shape: Input shape for benchmarking
        include_static: Include static quantization comparison
        
    Returns:
        Comparison dictionary
    """
    results = {}
    
    # FP32 baseline
    print("Benchmarking FP32 model...")
    model_fp32 = copy.deepcopy(model)
    model_fp32.eval()
    
    results['fp32'] = {
        'size_mb': get_model_size(model_fp32),
        'parameters': count_parameters(model_fp32),
        'inference': benchmark_inference(model_fp32, input_shape, device='cpu'),
    }
    
    # Dynamic INT8 quantization
    print("Applying dynamic INT8 quantization...")
    model_int8_dynamic = dynamic_quantization(model)
    
    results['int8_dynamic'] = {
        'size_mb': get_model_size(model_int8_dynamic),
        'parameters': count_parameters(model_int8_dynamic),
        'inference': benchmark_inference(model_int8_dynamic, input_shape, device='cpu'),
    }
    
    # Static INT8 quantization (if calibration data provided)
    if include_static and calibration_loader is not None:
        print("Applying static INT8 quantization...")
        try:
            model_int8_static = prepare_static_quantization(
                model, calibration_loader
            )
            
            results['int8_static'] = {
                'size_mb': get_model_size(model_int8_static),
                'parameters': count_parameters(model_int8_static),
                'inference': benchmark_inference(model_int8_static, input_shape, device='cpu'),
            }
        except Exception as e:
            print(f"Static quantization failed: {e}")
            results['int8_static'] = {'error': str(e)}
    
    # Compute comparisons
    if 'int8_dynamic' in results and 'error' not in results['int8_dynamic']:
        results['comparison'] = {
            'size_reduction': (
                1 - results['int8_dynamic']['size_mb'] / results['fp32']['size_mb']
            ) * 100,
            'speedup': (
                results['fp32']['inference']['mean_ms'] /
                results['int8_dynamic']['inference']['mean_ms']
            ),
        }
    
    return results


def print_quantization_comparison(results: Dict[str, Dict]):
    """
    Print formatted quantization comparison.
    
    Args:
        results: Results from compare_quantization
    """
    print("\n" + "=" * 70)
    print(" INT8 Quantization Comparison")
    print("=" * 70)
    
    # Size comparison
    print("\nModel Size (MB):")
    print(f"  FP32:          {results['fp32']['size_mb']:.2f} MB")
    if 'int8_dynamic' in results:
        print(f"  INT8 Dynamic:  {results['int8_dynamic']['size_mb']:.2f} MB")
    if 'int8_static' in results and 'error' not in results.get('int8_static', {}):
        print(f"  INT8 Static:   {results['int8_static']['size_mb']:.2f} MB")
    
    # Parameter count
    print("\nParameter Count:")
    print(f"  Total:         {results['fp32']['parameters']['total']:,}")
    print(f"  Trainable:     {results['fp32']['parameters']['trainable']:,}")
    
    # Inference speed
    print("\nInference Speed (CPU, batch=1):")
    print(f"  FP32:          {results['fp32']['inference']['mean_ms']:.2f} ms "
          f"(±{results['fp32']['inference']['std_ms']:.2f})")
    if 'int8_dynamic' in results:
        print(f"  INT8 Dynamic:  {results['int8_dynamic']['inference']['mean_ms']:.2f} ms "
              f"(±{results['int8_dynamic']['inference']['std_ms']:.2f})")
    
    # Summary
    if 'comparison' in results:
        print("\nSummary:")
        print(f"  Size Reduction: {results['comparison']['size_reduction']:.1f}%")
        print(f"  Speedup:        {results['comparison']['speedup']:.2f}x")
    
    print("=" * 70 + "\n")
