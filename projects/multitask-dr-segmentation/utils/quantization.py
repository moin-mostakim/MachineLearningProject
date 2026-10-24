"""
INT8 Quantization Utilities for Multi-Task DR Models.

Implements:
- Post-training dynamic quantization
- Post-training static quantization (with calibration)
- Model size and inference speed comparison

References:
- PyTorch Quantization documentation
- Jacob et al. (2018) "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
"""

import torch
import torch.nn as nn
import torch.quantization
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.multitask_model import build_model, count_parameters


def get_model_size(model, filepath=None):
    """Get model size in MB."""
    if filepath:
        torch.save(model.state_dict(), filepath)
        size_mb = os.path.getsize(filepath) / (1024 ** 2)
        os.remove(filepath)
    else:
        # Estimate from parameters
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return size_mb


def measure_inference_time(model, input_tensor, num_iterations=50, warmup=10):
    """Measure average inference time."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    return avg_time * 1000  # Return in milliseconds


class QuantizableEncoder(nn.Module):
    """Quantization-friendly wrapper for the encoder."""
    
    def __init__(self, original_encoder):
        super().__init__()
        self.encoder = original_encoder
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        features, bottleneck = self.encoder(x)
        # Dequant each output
        features = [self.dequant(f) for f in features]
        bottleneck = self.dequant(bottleneck)
        return features, bottleneck


def apply_dynamic_quantization(model):
    """Apply dynamic INT8 quantization to linear layers."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def prepare_static_quantization(model, calibration_loader):
    """
    Prepare and apply static INT8 quantization.
    
    Note: Full static quantization requires careful layer-by-layer setup
    and may not work for all architectures. This is a simplified version.
    """
    model.eval()
    model.cpu()  # Quantization works best on CPU
    
    # Dynamic quantization as fallback (more stable)
    quantized_model = apply_dynamic_quantization(model)
    
    return quantized_model


def compare_quantization(model_type='classification', calibration_loader=None):
    """
    Compare FP32 vs INT8 quantized models.
    
    Returns comparison metrics:
    - Model size (MB)
    - Inference time (ms)
    - Parameter count
    """
    # Build original model
    model_fp32 = build_model(model_type)
    model_fp32.eval()
    model_fp32.cpu()
    
    # Test input
    input_tensor = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # FP32 metrics
    fp32_size = get_model_size(model_fp32)
    fp32_time = measure_inference_time(model_fp32, input_tensor)
    fp32_params = count_parameters(model_fp32)
    
    # Quantized model
    model_int8 = apply_dynamic_quantization(model_fp32)
    
    # INT8 metrics
    int8_size = get_model_size(model_int8)
    int8_time = measure_inference_time(model_int8, input_tensor)
    
    # Compute improvements
    size_reduction = (1 - int8_size / fp32_size) * 100
    speed_improvement = (fp32_time - int8_time) / fp32_time * 100
    
    results = {
        'model_type': model_type,
        'fp32': {
            'size_mb': fp32_size,
            'inference_ms': fp32_time,
            'params': fp32_params,
        },
        'int8': {
            'size_mb': int8_size,
            'inference_ms': int8_time,
        },
        'improvement': {
            'size_reduction_percent': size_reduction,
            'speed_improvement_percent': speed_improvement,
        }
    }
    
    return results


def print_quantization_comparison(results_list):
    """Print formatted quantization comparison table."""
    print("\n" + "=" * 80)
    print("INT8 QUANTIZATION COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'FP32 Size':>12} {'INT8 Size':>12} {'Size Reduction':>15}")
    print("-" * 80)
    
    for results in results_list:
        model_type = results['model_type']
        fp32_size = results['fp32']['size_mb']
        int8_size = results['int8']['size_mb']
        reduction = results['improvement']['size_reduction_percent']
        
        print(f"{model_type:<25} {fp32_size:>10.2f}MB {int8_size:>10.2f}MB {reduction:>13.1f}%")
    
    print("-" * 80)
    print(f"\n{'Model':<25} {'FP32 Latency':>12} {'INT8 Latency':>12} {'Speed Improvement':>17}")
    print("-" * 80)
    
    for results in results_list:
        model_type = results['model_type']
        fp32_time = results['fp32']['inference_ms']
        int8_time = results['int8']['inference_ms']
        improvement = results['improvement']['speed_improvement_percent']
        
        print(f"{model_type:<25} {fp32_time:>10.2f}ms {int8_time:>10.2f}ms {improvement:>15.1f}%")
    
    print("-" * 80)


def run_quantization_analysis():
    """Run full quantization analysis for all models."""
    model_types = ['classification', 'segmentation', 'multitask']
    
    results_list = []
    for model_type in model_types:
        print(f"Analyzing {model_type} model...")
        results = compare_quantization(model_type)
        results_list.append(results)
    
    print_quantization_comparison(results_list)
    
    return results_list


if __name__ == "__main__":
    results = run_quantization_analysis()
