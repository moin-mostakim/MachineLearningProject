"""INT8 Quantization utilities for model efficiency comparison."""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert
from typing import Dict, Any, Optional, Tuple
import copy
import time


def quantize_model(
    model: nn.Module,
    quantization_type: str = "dynamic",
    calibration_loader=None,
    device: str = "cpu"
) -> nn.Module:
    """
    Quantize model to INT8.
    
    Args:
        model: Model to quantize
        quantization_type: 'dynamic' or 'static'
        calibration_loader: Data loader for static quantization calibration
        device: Device (quantization typically runs on CPU)
        
    Returns:
        Quantized model
    """
    model_cpu = copy.deepcopy(model).cpu().eval()
    
    if quantization_type == "dynamic":
        # Dynamic quantization - weights quantized, activations quantized dynamically
        quantized_model = quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
    elif quantization_type == "static":
        # Static quantization - requires calibration
        model_cpu.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for calibration
        prepared_model = prepare(model_cpu)
        
        # Calibration (run forward passes)
        if calibration_loader is not None:
            with torch.no_grad():
                for i, (images, _) in enumerate(calibration_loader):
                    if i >= 100:  # Calibrate on 100 batches
                        break
                    _ = prepared_model(images)
        
        # Convert to quantized
        quantized_model = convert(prepared_model)
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    return quantized_model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in memory."""
    # Count parameters
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / 1e6,
        "buffer_size_mb": buffer_size / 1e6,
        "total_size_mb": total_size / 1e6,
        "num_parameters": sum(p.numel() for p in model.parameters())
    }


def measure_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu"
) -> Dict[str, float]:
    """Measure model inference speed."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times_tensor = torch.tensor(times)
    
    return {
        "mean_ms": times_tensor.mean().item() * 1000,
        "std_ms": times_tensor.std().item() * 1000,
        "min_ms": times_tensor.min().item() * 1000,
        "throughput_per_sec": 1.0 / times_tensor.mean().item()
    }


def evaluate_accuracy(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    max_batches: Optional[int] = None
) -> float:
    """Evaluate model accuracy."""
    model = model.to(device).eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def compare_quantization(
    model: nn.Module,
    test_loader,
    calibration_loader=None,
    device: str = "cpu",
    max_batches: int = 50
) -> Dict[str, Any]:
    """
    Compare original and quantized model performance.
    
    Args:
        model: Original model
        test_loader: Test data loader
        calibration_loader: Calibration data loader (for static quantization)
        device: Device for evaluation
        max_batches: Maximum batches for evaluation
        
    Returns:
        Comparison results
    """
    results = {}
    
    # Original model (FP32)
    print("Evaluating original FP32 model...")
    model_fp32 = copy.deepcopy(model).cpu().eval()
    
    results["fp32"] = {
        "size": get_model_size(model_fp32),
        "speed": measure_inference_speed(model_fp32, device="cpu"),
        "accuracy": evaluate_accuracy(model_fp32, test_loader, device="cpu", max_batches=max_batches)
    }
    
    # Dynamic quantization (INT8)
    print("Evaluating dynamically quantized INT8 model...")
    try:
        model_int8_dynamic = quantize_model(model, quantization_type="dynamic")
        
        results["int8_dynamic"] = {
            "size": get_model_size(model_int8_dynamic),
            "speed": measure_inference_speed(model_int8_dynamic, device="cpu"),
            "accuracy": evaluate_accuracy(model_int8_dynamic, test_loader, device="cpu", max_batches=max_batches)
        }
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        results["int8_dynamic"] = {"error": str(e)}
    
    return results


def print_quantization_comparison(results: Dict[str, Any]):
    """Print quantization comparison results."""
    print("\n" + "=" * 70)
    print(" INT8 Quantization Comparison")
    print("=" * 70)
    
    # Header
    print(f"{'Metric':<25} {'FP32':>15} {'INT8 Dynamic':>15} {'Ratio':>12}")
    print("-" * 70)
    
    fp32 = results.get("fp32", {})
    int8 = results.get("int8_dynamic", {})
    
    if "error" not in int8:
        # Size comparison
        fp32_size = fp32.get("size", {}).get("total_size_mb", 0)
        int8_size = int8.get("size", {}).get("total_size_mb", 0)
        ratio = fp32_size / int8_size if int8_size > 0 else 0
        print(f"{'Model Size (MB)':<25} {fp32_size:>15.2f} {int8_size:>15.2f} {ratio:>11.2f}x")
        
        # Speed comparison
        fp32_speed = fp32.get("speed", {}).get("mean_ms", 0)
        int8_speed = int8.get("speed", {}).get("mean_ms", 0)
        speedup = fp32_speed / int8_speed if int8_speed > 0 else 0
        print(f"{'Inference Time (ms)':<25} {fp32_speed:>15.2f} {int8_speed:>15.2f} {speedup:>11.2f}x")
        
        # Accuracy comparison
        fp32_acc = fp32.get("accuracy", 0)
        int8_acc = int8.get("accuracy", 0)
        acc_diff = int8_acc - fp32_acc
        print(f"{'Accuracy':<25} {fp32_acc:>15.4f} {int8_acc:>15.4f} {acc_diff:>+11.4f}")
    else:
        print(f"INT8 quantization failed: {int8['error']}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test with a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    
    # Get model size
    size_info = get_model_size(model)
    print(f"Model size: {size_info}")
    
    # Measure speed
    speed_info = measure_inference_speed(model, device="cpu")
    print(f"Inference speed: {speed_info}")
    
    # Dynamic quantization
    quantized = quantize_model(model, "dynamic")
    quant_size = get_model_size(quantized)
    quant_speed = measure_inference_speed(quantized, device="cpu")
    
    print(f"\nQuantized model size: {quant_size}")
    print(f"Quantized inference speed: {quant_speed}")
    print(f"Size reduction: {size_info['total_size_mb'] / quant_size['total_size_mb']:.2f}x")
