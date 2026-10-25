"""Metrics computation and reporting."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import time


def compute_metrics(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Compute basic classification metrics."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    n_batches = batch_idx + 1 if max_batches else len(dataloader)
    
    return {
        "accuracy": correct / total,
        "loss": total_loss / n_batches,
        "total_samples": total
    }


def compute_adversarial_metrics(
    model: nn.Module,
    dataloader,
    attacker,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Compute metrics under adversarial attack."""
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()
        
        # Adversarial accuracy
        adv_images = attacker(images, labels)
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_correct += (adv_preds == labels).sum().item()
        
        total += labels.size(0)
    
    return {
        "clean_accuracy": clean_correct / total,
        "adversarial_accuracy": adv_correct / total,
        "robustness_gap": (clean_correct - adv_correct) / total,
        "total_samples": total
    }


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """Measure model inference time."""
    model.eval()
    model = model.to(device)
    
    x = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = torch.tensor(times)
    
    return {
        "mean_ms": times.mean().item() * 1000,
        "std_ms": times.std().item() * 1000,
        "min_ms": times.min().item() * 1000,
        "max_ms": times.max().item() * 1000
    }


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics"):
    """Print metrics in a formatted table."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:>20.4f}")
        elif isinstance(value, int):
            print(f"  {key:<30} {value:>20,}")
        else:
            print(f"  {key:<30} {str(value):>20}")
    
    print("=" * 60 + "\n")


def print_comparison_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Method Comparison"
):
    """Print comparison table across multiple methods."""
    if not results:
        return
    
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())
    
    # Header
    print("\n" + "=" * (30 + 15 * len(methods)))
    print(f" {title}")
    print("=" * (30 + 15 * len(methods)))
    
    header = f"{'Metric':<30}"
    for method in methods:
        header += f"{method:>15}"
    print(header)
    print("-" * (30 + 15 * len(methods)))
    
    # Rows
    for metric in metrics:
        row = f"{metric:<30}"
        for method in methods:
            value = results[method].get(metric, "N/A")
            if isinstance(value, float):
                row += f"{value:>15.4f}"
            else:
                row += f"{str(value):>15}"
        print(row)
    
    print("=" * (30 + 15 * len(methods)) + "\n")


def generate_results_summary(
    results: Dict[str, Any],
    config
) -> str:
    """Generate a markdown summary of results."""
    summary = []
    summary.append("# Adversarial Robustness Results\n")
    summary.append(f"**Dataset:** {config.data.dataset}")
    summary.append(f"**Model:** {config.model.vision_encoder}")
    summary.append(f"**Epochs:** {config.training.epochs}")
    summary.append(f"**Epsilon:** {config.attack.epsilon:.4f} ({config.attack.epsilon*255:.1f}/255)\n")
    
    summary.append("## Training Methods\n")
    
    for method, method_results in results.items():
        summary.append(f"### {method}\n")
        if "clean_accuracy" in method_results:
            summary.append(f"- Clean Accuracy: {method_results['clean_accuracy']:.4f}")
        if "adversarial_accuracy" in method_results:
            summary.append(f"- Adversarial Accuracy: {method_results['adversarial_accuracy']:.4f}")
        if "certified_accuracy" in method_results:
            summary.append(f"- Certified Accuracy: {method_results['certified_accuracy']:.4f}")
        summary.append("")
    
    summary.append("## Literature Comparison\n")
    summary.append("""
| Method | Clean Acc | PGD Acc (ε=8/255) | Source |
|--------|-----------|-------------------|--------|
| Madry et al. (2018) | 87.3% | 45.8% | ICLR 2018 |
| TRADES (Zhang et al.) | 84.9% | 56.6% | ICML 2019 |
| Cohen et al. (certified) | 75% | 60% (certified r=0.5) | ICML 2019 |
| Our Standard | {clean_std:.1%} | {adv_std:.1%} | This work |
| Our Adversarial | {clean_adv:.1%} | {adv_adv:.1%} | This work |
""".format(
        clean_std=results.get("standard", {}).get("clean_accuracy", 0),
        adv_std=results.get("standard", {}).get("adversarial_accuracy", 0),
        clean_adv=results.get("adversarial", {}).get("clean_accuracy", 0),
        adv_adv=results.get("adversarial", {}).get("adversarial_accuracy", 0)
    ))
    
    return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    metrics = {
        "accuracy": 0.8543,
        "loss": 0.3421,
        "total_samples": 10000
    }
    print_metrics_table(metrics, "Test Metrics")
    
    comparison = {
        "Standard": {"clean_acc": 0.93, "adv_acc": 0.10},
        "Adversarial": {"clean_acc": 0.87, "adv_acc": 0.45},
        "Certified": {"clean_acc": 0.75, "adv_acc": 0.50}
    }
    print_comparison_table(comparison, "Method Comparison")
