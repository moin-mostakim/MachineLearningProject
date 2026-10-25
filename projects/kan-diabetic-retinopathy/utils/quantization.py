"""
INT8 Quantization utilities for model efficiency comparison.

Compares FP32 vs INT8 inference for:
- Model size
- Inference speed
- Accuracy
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, Any, Optional
import copy
import os

from .metrics import calculate_metrics


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    calibration_loader: Optional[DataLoader] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Quantize a model to INT8.
    
    Args:
        model: Model to quantize
        method: Quantization method (dynamic, static)
        calibration_loader: Data loader for static quantization
        backend: Quantization backend (fbgemm for x86, qnnpack for ARM)
    
    Returns:
        Quantized model
    """
    # Set backend
    torch.backends.quantized.engine = backend
    
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy = model_copy.cpu()
    
    if method == "dynamic":
        # Dynamic quantization (no calibration needed)
        # Quantizes weights, activations quantized at runtime
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8,
        )
    elif method == "static":
        # Static quantization (requires calibration)
        if calibration_loader is None:
            print("Warning: No calibration data, falling back to dynamic quantization")
            return quantize_model(model, method="dynamic", backend=backend)
        
        # Prepare model for static quantization
        model_copy.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Fuse modules if possible (Conv-BN-ReLU, etc.)
        # Note: This is simplified; real implementation would need model-specific fusion
        
        prepared_model = torch.quantization.prepare(model_copy)
        
        # Calibrate with data
        with torch.no_grad():
            for images, _ in calibration_loader:
                prepared_model(images.cpu())
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return quantized_model


def measure_model_size(model: nn.Module, save_path: str = "/tmp/model_size_test.pt") -> float:
    """
    Measure model size in MB.
    
    Args:
        model: Model to measure
        save_path: Temporary path to save model
    
    Returns:
        Size in MB
    """
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    os.remove(save_path)
    return size_mb


def measure_inference_speed(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    num_iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of inference iterations
        warmup: Warmup iterations
        device: Device (quantized models require CPU)
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "throughput_fps": 1000 / np.mean(times),
    }


def evaluate_quantized(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate quantized model accuracy.
    
    Args:
        model: Quantized model
        data_loader: Data loader
        device: Device (CPU for quantized)
    
    Returns:
        Evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_proba = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            proba = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_proba.extend(proba.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    
    return calculate_metrics(y_true, y_pred, y_proba)


def compare_quantized(
    model: nn.Module,
    test_loader: DataLoader,
    calibration_loader: Optional[DataLoader] = None,
    method: str = "dynamic",
) -> Dict[str, Any]:
    """
    Compare FP32 and INT8 models.
    
    Args:
        model: Original FP32 model
        test_loader: Test data loader
        calibration_loader: Calibration data for static quantization
        method: Quantization method
    
    Returns:
        Comparison results
    """
    print("\n" + "=" * 60)
    print("QUANTIZATION COMPARISON: FP32 vs INT8")
    print("=" * 60)
    
    # Ensure model is on CPU for fair comparison
    model_fp32 = copy.deepcopy(model).cpu()
    model_fp32.eval()
    
    # Quantize
    print("\nQuantizing model...")
    model_int8 = quantize_model(
        model_fp32, 
        method=method,
        calibration_loader=calibration_loader,
    )
    
    results = {}
    
    # Model size
    print("\nMeasuring model sizes...")
    fp32_size = measure_model_size(model_fp32)
    int8_size = measure_model_size(model_int8)
    
    results["fp32_size_mb"] = fp32_size
    results["int8_size_mb"] = int8_size
    results["size_reduction"] = fp32_size / int8_size
    
    print(f"  FP32 size: {fp32_size:.2f} MB")
    print(f"  INT8 size: {int8_size:.2f} MB")
    print(f"  Reduction: {results['size_reduction']:.2f}x")
    
    # Inference speed
    print("\nMeasuring inference speed...")
    fp32_speed = measure_inference_speed(model_fp32, device="cpu")
    int8_speed = measure_inference_speed(model_int8, device="cpu")
    
    results["fp32_latency_ms"] = fp32_speed["mean_ms"]
    results["int8_latency_ms"] = int8_speed["mean_ms"]
    results["speedup"] = fp32_speed["mean_ms"] / int8_speed["mean_ms"]
    
    print(f"  FP32 latency: {fp32_speed['mean_ms']:.2f} ms ({fp32_speed['throughput_fps']:.1f} FPS)")
    print(f"  INT8 latency: {int8_speed['mean_ms']:.2f} ms ({int8_speed['throughput_fps']:.1f} FPS)")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    # Accuracy
    print("\nEvaluating accuracy...")
    fp32_metrics = evaluate_quantized(model_fp32, test_loader, device="cpu")
    int8_metrics = evaluate_quantized(model_int8, test_loader, device="cpu")
    
    results["fp32_accuracy"] = fp32_metrics["accuracy"]
    results["fp32_qwk"] = fp32_metrics["qwk"]
    results["int8_accuracy"] = int8_metrics["accuracy"]
    results["int8_qwk"] = int8_metrics["qwk"]
    results["accuracy_drop"] = fp32_metrics["accuracy"] - int8_metrics["accuracy"]
    results["qwk_drop"] = fp32_metrics["qwk"] - int8_metrics["qwk"]
    
    print(f"  FP32 Accuracy: {fp32_metrics['accuracy']:.4f} | QWK: {fp32_metrics['qwk']:.4f}")
    print(f"  INT8 Accuracy: {int8_metrics['accuracy']:.4f} | QWK: {int8_metrics['qwk']:.4f}")
    print(f"  Accuracy drop: {results['accuracy_drop']:.4f}")
    print(f"  QWK drop: {results['qwk_drop']:.4f}")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    # Test quantization
    print("Testing quantization utilities...")
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 5)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # Quantize
    quantized = quantize_model(model, method="dynamic")
    
    # Measure
    fp32_size = measure_model_size(model)
    int8_size = measure_model_size(quantized)
    
    print(f"FP32 size: {fp32_size:.4f} MB")
    print(f"INT8 size: {int8_size:.4f} MB")
    print(f"Reduction: {fp32_size/int8_size:.2f}x")
    
    # Speed test
    fp32_speed = measure_inference_speed(model, input_shape=(1, 512))
    int8_speed = measure_inference_speed(quantized, input_shape=(1, 512))
    
    print(f"\nFP32 latency: {fp32_speed['mean_ms']:.3f} ms")
    print(f"INT8 latency: {int8_speed['mean_ms']:.3f} ms")
    print(f"Speedup: {fp32_speed['mean_ms']/int8_speed['mean_ms']:.2f}x")
