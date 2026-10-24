"""
INT8 Quantization Utilities for Edge Deployment.

Implements post-training quantization for LSTM and Transformer models
to enable deployment on resource-constrained IoT devices.

References:
- Jacob, B., et al. (2018). Quantization and Training of Neural Networks.
- Krishnamoorthi, R. (2018). Quantizing Deep Convolutional Networks for Efficient Inference.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, QuantizationConfig


class ModelQuantizer:
    """
    Post-training quantization for PyTorch models.
    
    Supports:
    - Dynamic quantization (weights quantized, activations computed in fp32)
    - Static quantization (both weights and activations quantized)
    
    Dynamic quantization is simpler and works well for RNN/LSTM models.
    Static quantization provides better latency but requires calibration.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.quant_config = self.config.quantization
        
    def dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic INT8 quantization to the model.
        
        Dynamic quantization quantizes weights statically but computes
        activations in floating point. Good for LSTM/GRU models.
        
        Returns:
            Quantized model copy
        """
        # Create a copy of the model
        model_copy = type(model)(
            *self._get_model_args(model)
        )
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear, nn.LSTM},  # Layers to quantize
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _get_model_args(self, model):
        """Extract model constructor arguments."""
        if hasattr(model, 'input_size'):
            # LSTM Autoencoder
            return (
                model.input_size,
                model.hidden_size,
                model.num_layers
            )
        elif hasattr(model, 'd_model'):
            # Transformer
            return (
                model.input_size,
                model.d_model,
                model.n_heads if hasattr(model, 'n_heads') else 4,
                model.n_layers if hasattr(model, 'n_layers') else 2,
                model.d_ff if hasattr(model, 'd_ff') else 128
            )
        else:
            raise ValueError("Unknown model type")
    
    def compare_model_sizes(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, float]:
        """
        Compare sizes of original and quantized models.
        
        Returns:
            Dictionary with size comparison metrics
        """
        # Calculate original model size
        original_size = sum(
            p.numel() * p.element_size() 
            for p in original_model.parameters()
        )
        
        # Calculate quantized model size (approximate)
        quantized_size = 0
        for name, module in quantized_model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight()
                if hasattr(weight, 'int_repr'):
                    # Quantized weight (int8)
                    quantized_size += weight.int_repr().numel() * 1  # 1 byte for int8
                else:
                    quantized_size += weight.numel() * weight.element_size()
            elif hasattr(module, '_packed_params'):
                # Quantized LSTM
                # Approximate size based on parameters
                for param_name in ['weight_ih_l0', 'weight_hh_l0']:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if param is not None:
                            quantized_size += param.numel() * 1
        
        # If we couldn't estimate quantized size, use heuristic
        if quantized_size == 0:
            # INT8 is 4x smaller than FP32
            quantized_size = original_size / 4
        
        compression_ratio = original_size / quantized_size
        size_reduction = (1 - quantized_size / original_size) * 100
        
        return {
            "original_size_bytes": original_size,
            "original_size_mb": original_size / (1024**2),
            "quantized_size_bytes": quantized_size,
            "quantized_size_mb": quantized_size / (1024**2),
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction
        }
    
    def benchmark_inference(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed of original vs quantized models.
        
        Returns:
            Dictionary with timing comparison
        """
        original_model.eval()
        quantized_model.eval()
        
        # Move to CPU for fair comparison (quantized models run on CPU)
        original_model = original_model.cpu()
        sample_input = sample_input.cpu()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(sample_input)
                _ = quantized_model(sample_input)
        
        # Benchmark original
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = original_model(sample_input)
                original_times.append(time.perf_counter() - start)
        
        # Benchmark quantized
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = quantized_model(sample_input)
                quantized_times.append(time.perf_counter() - start)
        
        original_mean = np.mean(original_times) * 1000  # ms
        original_std = np.std(original_times) * 1000
        quantized_mean = np.mean(quantized_times) * 1000
        quantized_std = np.std(quantized_times) * 1000
        
        speedup = original_mean / quantized_mean
        
        return {
            "original_latency_ms": original_mean,
            "original_latency_std_ms": original_std,
            "quantized_latency_ms": quantized_mean,
            "quantized_latency_std_ms": quantized_std,
            "speedup": speedup,
            "num_runs": num_runs
        }
    
    def evaluate_accuracy(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compare reconstruction accuracy of original vs quantized models.
        
        Returns:
            Dictionary with accuracy comparison
        """
        original_model.eval()
        quantized_model.eval()
        
        original_model = original_model.cpu()
        test_data = test_data.cpu()
        
        with torch.no_grad():
            original_output = original_model(test_data)
            quantized_output = quantized_model(test_data)
        
        # Calculate MSE
        original_mse = torch.mean((test_data - original_output) ** 2).item()
        quantized_mse = torch.mean((test_data - quantized_output) ** 2).item()
        
        # Calculate output difference
        output_diff = torch.mean(torch.abs(original_output - quantized_output)).item()
        
        # Relative accuracy degradation
        accuracy_degradation = (quantized_mse - original_mse) / original_mse * 100
        
        return {
            "original_mse": original_mse,
            "quantized_mse": quantized_mse,
            "output_difference": output_diff,
            "accuracy_degradation_percent": accuracy_degradation
        }


def quantize_and_compare(
    model: nn.Module,
    test_data: torch.Tensor,
    debug: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Quantize a model and compare with original.
    
    Returns:
        - Quantized model
        - Comparison metrics dictionary
    """
    quantizer = ModelQuantizer()
    
    if debug:
        print("\nApplying INT8 Quantization...")
    
    # Quantize model
    quantized_model = quantizer.dynamic_quantize(model)
    
    # Compare sizes
    size_metrics = quantizer.compare_model_sizes(model, quantized_model)
    
    # Benchmark inference
    sample_input = test_data[:32]  # Use batch of 32 for benchmarking
    timing_metrics = quantizer.benchmark_inference(
        model, quantized_model, sample_input, num_runs=50
    )
    
    # Evaluate accuracy
    accuracy_metrics = quantizer.evaluate_accuracy(
        model, quantized_model, test_data[:100]
    )
    
    all_metrics = {
        **size_metrics,
        **timing_metrics,
        **accuracy_metrics
    }
    
    if debug:
        print("\n" + "=" * 50)
        print("INT8 QUANTIZATION COMPARISON")
        print("=" * 50)
        print(f"\nModel Size:")
        print(f"  Original: {size_metrics['original_size_mb']:.2f} MB")
        print(f"  Quantized: {size_metrics['quantized_size_mb']:.2f} MB")
        print(f"  Compression: {size_metrics['compression_ratio']:.2f}x")
        print(f"  Size Reduction: {size_metrics['size_reduction_percent']:.1f}%")
        print(f"\nInference Latency:")
        print(f"  Original: {timing_metrics['original_latency_ms']:.3f} ± {timing_metrics['original_latency_std_ms']:.3f} ms")
        print(f"  Quantized: {timing_metrics['quantized_latency_ms']:.3f} ± {timing_metrics['quantized_latency_std_ms']:.3f} ms")
        print(f"  Speedup: {timing_metrics['speedup']:.2f}x")
        print(f"\nReconstruction Accuracy:")
        print(f"  Original MSE: {accuracy_metrics['original_mse']:.6f}")
        print(f"  Quantized MSE: {accuracy_metrics['quantized_mse']:.6f}")
        print(f"  Accuracy Degradation: {accuracy_metrics['accuracy_degradation_percent']:.2f}%")
        print("=" * 50)
    
    return quantized_model, all_metrics


if __name__ == "__main__":
    # Test quantization
    from models.lstm_detector import LSTMAutoencoder
    
    print("Testing INT8 Quantization...")
    
    # Create sample model
    model = LSTMAutoencoder(
        input_size=6,
        hidden_size=64,
        num_layers=2
    )
    
    # Create sample data
    test_data = torch.randn(100, 60, 6)  # (batch, seq_len, features)
    
    quantized_model, metrics = quantize_and_compare(model, test_data, debug=True)
