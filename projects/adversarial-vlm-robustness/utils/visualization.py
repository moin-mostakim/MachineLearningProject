"""Visualization utilities for adversarial robustness experiments."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional


def plot_accuracy_comparison(evaluation_results: Dict[str, Dict], 
                            output_dir: str = "figures"):
    """Plot clean vs adversarial accuracy comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(evaluation_results.keys())
    clean_accs = [evaluation_results[m]["clean_accuracy"] for m in methods]
    adv_accs = [evaluation_results[m]["adversarial_accuracy"] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', color='coral')
    
    ax.set_xlabel('Training Method')
    ax.set_ylabel('Accuracy')
    ax.set_title('Clean vs Adversarial Accuracy by Training Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "accuracy_comparison.png")


def plot_robustness_gap(evaluation_results: Dict[str, Dict],
                        output_dir: str = "figures"):
    """Plot the robustness gap (clean - adversarial accuracy)."""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(evaluation_results.keys())
    gaps = [evaluation_results[m]["clean_accuracy"] - evaluation_results[m]["adversarial_accuracy"] 
            for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gaps]
    bars = ax.bar(methods, gaps, color=colors, edgecolor='black')
    
    ax.set_xlabel('Training Method')
    ax.set_ylabel('Robustness Gap (Clean - Adversarial)')
    ax.set_title('Robustness Gap Analysis')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_gap.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "robustness_gap.png")


def plot_resource_usage(resource_stats: Dict[str, Any],
                       output_dir: str = "figures"):
    """Plot resource usage comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(resource_stats.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Training time
    times = [resource_stats[m].duration_seconds for m in methods]
    axes[0].bar(methods, times, color='steelblue', edgecolor='black')
    axes[0].set_title('Training Time (seconds)')
    axes[0].set_ylabel('Seconds')
    for i, t in enumerate(times):
        axes[0].annotate(f'{t:.1f}s', xy=(i, t), ha='center', va='bottom')
    
    # CPU usage
    cpu_mean = [resource_stats[m].cpu_mean for m in methods]
    cpu_max = [resource_stats[m].cpu_max for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    axes[1].bar(x - width/2, cpu_mean, width, label='Mean', color='green')
    axes[1].bar(x + width/2, cpu_max, width, label='Max', color='lightgreen')
    axes[1].set_title('CPU Usage (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].legend()
    
    # Memory usage
    mem_mean = [resource_stats[m].memory_mean_mb for m in methods]
    mem_max = [resource_stats[m].memory_max_mb for m in methods]
    axes[2].bar(x - width/2, mem_mean, width, label='Mean', color='purple')
    axes[2].bar(x + width/2, mem_max, width, label='Max', color='plum')
    axes[2].set_title('Memory Usage (MB)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_usage.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "resource_usage.png")


def plot_literature_comparison(our_results: Dict[str, Dict],
                              output_dir: str = "figures"):
    """Plot comparison with literature results."""
    os.makedirs(output_dir, exist_ok=True)
    
    literature = {
        "Madry et al.": {"clean": 0.873, "robust": 0.458},
        "TRADES": {"clean": 0.849, "robust": 0.566},
        "Wong et al.": {"clean": 0.833, "robust": 0.432},
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Literature results
    lit_names = list(literature.keys())
    lit_clean = [literature[n]["clean"] for n in lit_names]
    lit_robust = [literature[n]["robust"] for n in lit_names]
    
    # Our results  
    our_names = [f"Ours ({n})" for n in our_results.keys()]
    our_clean = [our_results[n]["clean_accuracy"] for n in our_results.keys()]
    our_robust = [our_results[n]["adversarial_accuracy"] for n in our_results.keys()]
    
    all_names = lit_names + our_names
    all_clean = lit_clean + our_clean
    all_robust = lit_robust + our_robust
    
    x = np.arange(len(all_names))
    width = 0.35
    
    colors_clean = ['lightblue'] * len(lit_names) + ['steelblue'] * len(our_names)
    colors_robust = ['lightsalmon'] * len(lit_names) + ['coral'] * len(our_names)
    
    ax.bar(x - width/2, all_clean, width, label='Clean Accuracy', 
           color=colors_clean, edgecolor='black')
    ax.bar(x + width/2, all_robust, width, label='Adversarial Accuracy',
           color=colors_robust, edgecolor='black')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison with Published Results (CIFAR-10, ε=8/255)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "literature_comparison.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "literature_comparison.png")


def plot_certified_accuracy(certified_results: Dict[str, Dict],
                           output_dir: str = "figures"):
    """Plot certified accuracy at different radii."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not certified_results:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method, results in certified_results.items():
        radii = []
        accs = []
        for key, value in sorted(results.items()):
            if key.startswith("certified_acc_r"):
                r = float(key.split("r")[-1])
                radii.append(r)
                accs.append(value)
        
        ax.plot(radii, accs, 'o-', linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Certified Radius')
    ax.set_ylabel('Certified Accuracy')
    ax.set_title('Certified Accuracy vs Perturbation Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "certified_accuracy.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "certified_accuracy.png")


def plot_quantization_comparison(quant_results: Dict,
                                output_dir: str = "figures"):
    """Plot INT8 quantization comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['Model Size\n(MB)', 'Inference Time\n(ms)', 'Accuracy']
    fp32_values = [
        quant_results['fp32']['size_mb'],
        quant_results['fp32']['inference_time_ms'],
        quant_results['fp32']['accuracy']
    ]
    int8_values = [
        quant_results['int8_dynamic']['size_mb'],
        quant_results['int8_dynamic']['inference_time_ms'],
        quant_results['int8_dynamic']['accuracy']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (metric, fp32, int8) in enumerate(zip(metrics, fp32_values, int8_values)):
        axes[i].bar(['FP32', 'INT8'], [fp32, int8], color=['steelblue', 'coral'])
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        
        for j, v in enumerate([fp32, int8]):
            axes[i].annotate(f'{v:.2f}', xy=(j, v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantization_comparison.png"), dpi=150)
    plt.close()
    
    return os.path.join(output_dir, "quantization_comparison.png")


def generate_all_figures(results_file: str, output_dir: str = "figures"):
    """Generate all figures from saved results."""
    with open(results_file) as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    figures = []
    
    if "evaluation" in results:
        figures.append(plot_accuracy_comparison(results["evaluation"], output_dir))
        figures.append(plot_robustness_gap(results["evaluation"], output_dir))
        figures.append(plot_literature_comparison(results["evaluation"], output_dir))
    
    if "certified" in results and results["certified"]:
        figures.append(plot_certified_accuracy(results["certified"], output_dir))
    
    print(f"Generated {len(figures)} figures in {output_dir}/")
    return figures
