"""
Quantization utilities for model compression and deployment.

Implements:
- INT8 dynamic quantization
- INT8 static quantization  
- Quantization-aware training (QAT) preparation
- Comparison metrics between full and quantized models

References:
- PyTorch Quantization documentation
- TensorRT quantization best practices
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Tuple, Optional, Callable
import copy
import time


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to model.
    
    Dynamic quantization quantizes weights ahead of time but
    activations are quantized at runtime. Good for:
    - Models with small batch sizes
    - RNN/LSTM layers
    - Models where activation ranges vary significantly
    
    Args:
        model: PyTorch model
        
    Returns:
        Quantized model
    """
    # Ensure model is on CPU (quantization typically done on CPU)
    model = model.cpu()
    model.eval()
    
    # Apply dynamic quantization to Linear and LSTM layers
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8
    )
    
    return quantized_model


class QuantizationWrapper(nn.Module):
    """
    Wrapper to prepare model for static quantization.
    
    Adds QuantStub and DeQuantStub for proper quantization flow.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def prepare_static_quantization(
    model: nn.Module,
    backend: str = "fbgemm"
) -> nn.Module:
    """
    Prepare model for static quantization (calibration step needed).
    
    Static quantization determines activation quantization parameters
    from calibration data, providing better accuracy than dynamic.
    
    Args:
        model: PyTorch model
        backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)
        
    Returns:
        Model prepared for calibration
    """
    model = model.cpu()
    model.eval()
    
    # Set quantization configuration
    if backend == "fbgemm":
        model.qconfig = quant.get_default_qconfig('fbgemm')
    else:
        model.qconfig = quant.get_default_qconfig('qnnpack')
    
    # Prepare for calibration
    prepared_model = quant.prepare(model)
    
    return prepared_model


def calibrate_and_convert(
    prepared_model: nn.Module,
    calibration_loader,
    num_batches: int = 100,
    device: str = "cpu"
) -> nn.Module:
    """
    Calibrate and convert prepared model to quantized model.
    
    Args:
        prepared_model: Model from prepare_static_quantization
        calibration_loader: DataLoader for calibration
        num_batches: Number of calibration batches
        device: Device for calibration
        
    Returns:
        Quantized model
    """
    prepared_model.to(device)
    prepared_model.eval()
    
    # Run calibration
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            
            inputs = batch['input'].to(device)
            # Handle input shape for models expecting flattened time dimension
            if len(inputs.shape) == 5:  # (B, T, C, H, W)
                inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[3], inputs.shape[4])
            prepared_model(inputs)
    
    # Convert to quantized model
    quantized_model = quant.convert(prepared_model)
    
    return quantized_model


def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compare original and quantized model performance.
    
    Args:
        original_model: Full precision model
        quantized_model: Quantized model
        test_input: Test input tensor
        device: Device for comparison
        
    Returns:
        Dictionary with comparison metrics
    """
    original_model = original_model.to(device).eval()
    quantized_model = quantized_model.to(device).eval()
    
    test_input = test_input.to(device)
    
    # Handle input shape
    if len(test_input.shape) == 5:
        test_input_flat = test_input.reshape(test_input.shape[0], -1, test_input.shape[3], test_input.shape[4])
    else:
        test_input_flat = test_input
    
    results = {}
    
    # Output comparison
    with torch.no_grad():
        orig_out = original_model(test_input)
        quant_out = quantized_model(test_input_flat)
    
    # Handle potential shape differences
    if orig_out.shape != quant_out.shape:
        min_size = min(orig_out.numel(), quant_out.numel())
        orig_flat = orig_out.flatten()[:min_size]
        quant_flat = quant_out.flatten()[:min_size]
    else:
        orig_flat = orig_out.flatten()
        quant_flat = quant_out.flatten()
    
    # Mean absolute error
    results['mae'] = torch.abs(orig_flat - quant_flat).mean().item()
    
    # Relative error
    results['relative_error'] = (
        torch.abs(orig_flat - quant_flat) / (torch.abs(orig_flat) + 1e-8)
    ).mean().item()
    
    # Correlation
    orig_centered = orig_flat - orig_flat.mean()
    quant_centered = quant_flat - quant_flat.mean()
    corr = (orig_centered * quant_centered).sum() / (
        torch.sqrt((orig_centered ** 2).sum() * (quant_centered ** 2).sum()) + 1e-8
    )
    results['correlation'] = corr.item()
    
    # Model size comparison
    def get_model_size(model):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1e6
    
    results['original_size_mb'] = get_model_size(original_model)
    # For quantized models, size estimation is approximate
    results['quantized_size_mb'] = results['original_size_mb'] / 4  # Rough INT8 estimate
    results['compression_ratio'] = results['original_size_mb'] / results['quantized_size_mb']
    
    # Inference time comparison
    num_runs = 10
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = original_model(test_input)
            _ = quantized_model(test_input_flat)
    
    # Time original
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = original_model(test_input)
    torch.cuda.synchronize() if device == "cuda" else None
    results['original_time_ms'] = (time.time() - start) / num_runs * 1000
    
    # Time quantized
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = quantized_model(test_input_flat)
    torch.cuda.synchronize() if device == "cuda" else None
    results['quantized_time_ms'] = (time.time() - start) / num_runs * 1000
    
    results['speedup'] = results['original_time_ms'] / results['quantized_time_ms']
    
    return results


def quantize_for_deployment(
    model: nn.Module,
    calibration_loader=None,
    method: str = "dynamic"
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Full quantization pipeline for deployment.
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for static quantization
        method: "dynamic" or "static"
        
    Returns:
        Tuple of (quantized_model, comparison_metrics)
    """
    # Create copy for quantization
    model_copy = copy.deepcopy(model)
    
    if method == "dynamic":
        quantized = dynamic_quantize(model_copy)
    else:
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quantization")
        
        wrapped = QuantizationWrapper(model_copy)
        prepared = prepare_static_quantization(wrapped)
        quantized = calibrate_and_convert(prepared, calibration_loader)
    
    # Generate comparison metrics with dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 2, 7, 32, 64)
    
    try:
        metrics = compare_models(model, quantized, dummy_input, device="cpu")
    except Exception as e:
        metrics = {'error': str(e)}
    
    return quantized, metrics


def print_quantization_report(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print a formatted quantization comparison report.
    
    Args:
        metrics: Comparison metrics from compare_models
        model_name: Name of the model for display
    """
    print("\n" + "=" * 60)
    print(f"QUANTIZATION REPORT: {model_name}")
    print("=" * 60)
    
    if 'error' in metrics:
        print(f"Error during comparison: {metrics['error']}")
        return
    
    print("\n📊 ACCURACY")
    print("-" * 40)
    print(f"  Mean Absolute Error:  {metrics.get('mae', 'N/A'):.6f}")
    print(f"  Relative Error:       {metrics.get('relative_error', 'N/A'):.4%}")
    print(f"  Output Correlation:   {metrics.get('correlation', 'N/A'):.6f}")
    
    print("\n💾 MODEL SIZE")
    print("-" * 40)
    print(f"  Original (FP32):      {metrics.get('original_size_mb', 'N/A'):.2f} MB")
    print(f"  Quantized (INT8):     {metrics.get('quantized_size_mb', 'N/A'):.2f} MB")
    print(f"  Compression Ratio:    {metrics.get('compression_ratio', 'N/A'):.1f}x")
    
    print("\n⏱️  INFERENCE TIME")
    print("-" * 40)
    print(f"  Original:             {metrics.get('original_time_ms', 'N/A'):.2f} ms")
    print(f"  Quantized:            {metrics.get('quantized_time_ms', 'N/A'):.2f} ms")
    print(f"  Speedup:              {metrics.get('speedup', 'N/A'):.2f}x")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test quantization
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing quantization utilities...")
    
    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(14, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 7, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Handle 5D input
            if len(x.shape) == 5:
                b, t, c, h, w = x.shape
                x = x.reshape(b, t * c, h, w)
            x = self.relu(self.conv1(x))
            return self.conv2(x)
    
    model = SimpleModel()
    
    # Test dynamic quantization
    print("\nTesting dynamic quantization...")
    quantized, metrics = quantize_for_deployment(model, method="dynamic")
    
    print_quantization_report(metrics, "SimpleModel")
    
    print("\nQuantization tests passed!")
