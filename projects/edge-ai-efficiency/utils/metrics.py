"""
Metrics utilities for model evaluation and benchmarking.
Includes FLOPs counting, parameter counting, and accuracy metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional
import numpy as np
from collections import defaultdict


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters per layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    layer_params = {}
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            layer_params[name] = params
    return layer_params


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def count_flops(model: nn.Module, input_size: Tuple[int, ...], 
                device: str = 'cpu') -> int:
    """
    Count FLOPs (Floating Point Operations) for a forward pass.
    
    This is a simplified FLOPs counter focusing on common operations:
    - Conv2d: 2 * H * W * Cin * Cout * Kh * Kw / stride^2
    - Linear: 2 * in_features * out_features
    - BatchNorm: 2 * num_features * H * W
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        
    Returns:
        Total FLOPs
    """
    model = model.to(device)
    model.eval()
    
    total_flops = 0
    hooks = []
    
    def conv_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        output_channels, output_height, output_width = output.size()[1:]
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0
        
        flops = batch_size * output_channels * output_height * output_width * (2 * kernel_ops + bias_ops)
        total_flops += flops
    
    def linear_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        flops = batch_size * (2 * module.in_features * module.out_features)
        if module.bias is not None:
            flops += batch_size * module.out_features
        total_flops += flops
    
    def bn_hook(module, input, output):
        nonlocal total_flops
        # BN: 2 ops per element (normalize and scale)
        total_flops += 2 * input[0].numel()
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(bn_hook))
    
    # Forward pass
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def format_flops(flops: int) -> str:
    """Format FLOPs with appropriate unit."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    return f"{flops} FLOPs"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:{self.fmt}}"


def accuracy(output: torch.Tensor, target: torch.Tensor, 
             topk: Tuple[int, ...] = (1,)) -> list:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model predictions (B, num_classes)
        target: Ground truth labels (B,)
        topk: Tuple of k values
        
    Returns:
        List of accuracy values for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def compute_calibration_error(predictions: np.ndarray, labels: np.ndarray,
                              num_bins: int = 15) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        predictions: Predicted probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        num_bins: Number of confidence bins
        
    Returns:
        Dictionary with ECE and MCE
    """
    confidences = np.max(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracies = predicted_labels == labels
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            bin_error = np.abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
    
    return {"ece": ece, "mce": mce}


def create_metrics_table(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a formatted metrics table for multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        Formatted table string
    """
    headers = ["Model", "Params (M)", "Size (MB)", "FLOPs (M)", 
               "Accuracy (%)", "Latency (ms)"]
    
    rows = []
    for model_name, metrics in results.items():
        row = [
            model_name,
            f"{metrics.get('params', 0) / 1e6:.2f}",
            f"{metrics.get('size_mb', 0):.2f}",
            f"{metrics.get('flops', 0) / 1e6:.2f}",
            f"{metrics.get('accuracy', 0):.2f}",
            f"{metrics.get('latency_ms', 0):.2f}"
        ]
        rows.append(row)
    
    # Calculate column widths
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    
    # Build table
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    
    table = [separator, header_row, separator]
    for row in rows:
        row_str = "|" + "|".join(f" {r:^{col_widths[i]}} " for i, r in enumerate(row)) + "|"
        table.append(row_str)
    table.append(separator)
    
    return "\n".join(table)
