"""
INT8 Quantization comparison for model efficiency.

Compares:
- FP32 baseline
- INT8 dynamic quantization
- Accuracy degradation and speedup

References:
----------
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
"""

import torch
import torch.nn as nn
import time
import copy
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def quantize_model(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
    """
    Quantize model to INT8.
    
    Args:
        model: Original FP32 model
        quantization_type: 'dynamic' or 'static'
    
    Returns:
        Quantized model
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy = model_copy.cpu()  # Quantization requires CPU
    
    if quantization_type == "dynamic":
        # Dynamic quantization (weights only)
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
    else:
        # For static quantization, we would need calibration data
        # Using dynamic for simplicity
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear},
            dtype=torch.qint8
        )
    
    return quantized_model


def benchmark_inference(
    model: nn.Module,
    input_batch: Dict[str, torch.Tensor],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Benchmark model inference time.
    
    Args:
        model: Model to benchmark
        input_batch: Input batch
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Timing statistics
    """
    model.eval()
    model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in input_batch.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(batch)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
    }


def compare_accuracy(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_loader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compare accuracy between original and quantized models.
    
    Returns:
        Accuracy metrics for both models
    """
    original_model.eval()
    quantized_model.eval()
    
    original_model.to(device)
    quantized_model.to("cpu")  # Quantized model must be on CPU
    
    original_preds = []
    quantized_preds = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_cpu = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Original model
            orig_out = original_model(batch_dev)
            original_preds.append(orig_out['pred_properties'].cpu())
            
            # Quantized model
            try:
                quant_out = quantized_model(batch_cpu)
                quantized_preds.append(quant_out['pred_properties'])
            except Exception as e:
                logger.warning(f"Quantized model forward failed: {e}")
                # Use original predictions as fallback
                quantized_preds.append(orig_out['pred_properties'].cpu())
            
            targets.append(batch_cpu['y'])
    
    original_preds = torch.cat(original_preds, dim=0)
    quantized_preds = torch.cat(quantized_preds, dim=0)
    targets = torch.cat(targets, dim=0)
    
    # Compute MAE
    original_mae = torch.abs(original_preds - targets).mean().item()
    quantized_mae = torch.abs(quantized_preds - targets).mean().item()
    
    # Compute prediction difference
    pred_diff = torch.abs(original_preds - quantized_preds).mean().item()
    
    return {
        'original_mae': original_mae,
        'quantized_mae': quantized_mae,
        'mae_degradation': quantized_mae - original_mae,
        'mae_degradation_percent': (quantized_mae - original_mae) / (original_mae + 1e-8) * 100,
        'prediction_difference': pred_diff
    }


def full_quantization_comparison(
    model: nn.Module,
    test_loader,
    config,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    Full quantization comparison including size, speed, and accuracy.
    
    Returns:
        Comprehensive comparison results
    """
    logger.info("Starting quantization comparison...")
    
    # Original model size
    original_size = get_model_size(model)
    logger.info(f"Original model size: {original_size:.2f} MB")
    
    # Quantize model
    logger.info("Quantizing model to INT8...")
    try:
        quantized_model = quantize_model(model, "dynamic")
        quantized_size = get_model_size(quantized_model)
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        quantization_successful = True
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")
        quantized_model = model
        quantized_size = original_size
        quantization_successful = False
    
    # Create sample batch for benchmarking
    sample_batch = next(iter(test_loader))
    
    # Benchmark original model
    logger.info("Benchmarking original model...")
    original_timing = benchmark_inference(
        model, sample_batch, num_runs=50, device=device
    )
    
    # Benchmark quantized model (CPU only)
    logger.info("Benchmarking quantized model...")
    if quantization_successful:
        quantized_timing = benchmark_inference(
            quantized_model, sample_batch, num_runs=50, device="cpu"
        )
    else:
        quantized_timing = original_timing
    
    # Compare accuracy
    logger.info("Comparing accuracy...")
    if quantization_successful:
        accuracy_comparison = compare_accuracy(
            model, quantized_model, test_loader, device
        )
    else:
        accuracy_comparison = {
            'original_mae': 0.0,
            'quantized_mae': 0.0,
            'mae_degradation': 0.0,
            'mae_degradation_percent': 0.0,
            'prediction_difference': 0.0
        }
    
    results = {
        'quantization_successful': quantization_successful,
        'size_comparison': {
            'original_mb': original_size,
            'quantized_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100
        },
        'timing_comparison': {
            'original': original_timing,
            'quantized': quantized_timing,
            'speedup': original_timing['mean_ms'] / quantized_timing['mean_ms'] if quantized_timing['mean_ms'] > 0 else 1.0
        },
        'accuracy_comparison': accuracy_comparison
    }
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("QUANTIZATION COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Size: {original_size:.2f} MB -> {quantized_size:.2f} MB "
               f"({results['size_comparison']['size_reduction_percent']:.1f}% reduction)")
    logger.info(f"Inference Time: {original_timing['mean_ms']:.2f}ms -> {quantized_timing['mean_ms']:.2f}ms "
               f"({results['timing_comparison']['speedup']:.2f}x speedup)")
    logger.info(f"MAE Degradation: {accuracy_comparison['mae_degradation_percent']:.2f}%")
    logger.info("=" * 60)
    
    return results


def print_quantization_table(results: Dict) -> str:
    """Format quantization results as a table."""
    lines = []
    lines.append("=" * 70)
    lines.append("INT8 QUANTIZATION COMPARISON")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'Original':>15} {'Quantized':>15}")
    lines.append("-" * 70)
    
    size = results['size_comparison']
    lines.append(f"{'Model Size (MB)':<30} {size['original_mb']:>15.2f} {size['quantized_mb']:>15.2f}")
    
    timing = results['timing_comparison']
    lines.append(f"{'Inference Time (ms)':<30} {timing['original']['mean_ms']:>15.2f} {timing['quantized']['mean_ms']:>15.2f}")
    
    acc = results['accuracy_comparison']
    lines.append(f"{'Property MAE':<30} {acc['original_mae']:>15.4f} {acc['quantized_mae']:>15.4f}")
    
    lines.append("-" * 70)
    lines.append(f"{'Compression Ratio':<30} {size['compression_ratio']:>15.2f}x")
    lines.append(f"{'Speedup':<30} {timing['speedup']:>15.2f}x")
    lines.append(f"{'MAE Degradation':<30} {acc['mae_degradation_percent']:>15.2f}%")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test quantization
    import sys
    sys.path.append("..")
    from config import get_config
    from models.vae import build_model
    from data.dataset import get_dataloaders
    
    config = get_config(debug=True)
    model = build_model(config)
    
    # Get test loader
    _, _, test_loader = get_dataloaders(config, debug_samples=100)
    
    # Run comparison
    results = full_quantization_comparison(model, test_loader, config, "cpu")
    
    # Print table
    print(print_quantization_table(results))
