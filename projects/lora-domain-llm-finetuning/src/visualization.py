"""Visualization module for training results and comparisons."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

plt.style.use('seaborn-v0_8-whitegrid')


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    learning_rates: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation loss curves."""
    num_plots = 1 + (val_losses is not None and len(val_losses) > 0) + (learning_rates is not None and len(learning_rates) > 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    axes[ax_idx].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[ax_idx].set_xlabel('Steps')
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].set_title('Training Loss')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1
    
    if val_losses is not None and len(val_losses) > 0:
        axes[ax_idx].plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
        axes[ax_idx].set_xlabel('Epochs')
        axes[ax_idx].set_ylabel('Loss')
        axes[ax_idx].set_title('Validation Loss')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
    
    if learning_rates is not None and len(learning_rates) > 0:
        axes[ax_idx].plot(learning_rates, label='Learning Rate', color='green', linewidth=2)
        axes[ax_idx].set_xlabel('Steps')
        axes[ax_idx].set_ylabel('Learning Rate')
        axes[ax_idx].set_title('Learning Rate Schedule')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_lora_analysis(
    lora_stats: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize LoRA configuration and parameter efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    trainable = lora_stats.get("trainable_params", 0)
    frozen = lora_stats.get("total_params", 0) - trainable
    
    axes[0].pie(
        [trainable, frozen],
        labels=['Trainable (LoRA)', 'Frozen'],
        autopct='%1.2f%%',
        colors=['#2ecc71', '#e74c3c'],
        explode=(0.05, 0),
        shadow=True
    )
    axes[0].set_title('Parameter Distribution')
    
    configs = ['Rank', 'Alpha', 'Dropout %']
    values = [
        lora_stats.get("rank", 0),
        lora_stats.get("alpha", 0),
        lora_stats.get("dropout", 0) * 100
    ]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(configs)))
    bars = axes[1].bar(configs, values, color=colors, edgecolor='black')
    axes[1].set_ylabel('Value')
    axes[1].set_title('LoRA Configuration')
    
    for bar, val in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LoRA analysis to {save_path}")
    
    return fig


def plot_method_comparison(
    results: Dict[str, Dict[str, Any]],
    metric_name: str = "perplexity",
    lower_is_better: bool = True,
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    published_results: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """Create comparison bar chart of different methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = []
    values = []
    colors = []
    
    our_methods = []
    our_values = []
    
    for method, data in results.items():
        if isinstance(data, dict) and "value" in data:
            our_methods.append(method)
            our_values.append(data["value"])
    
    if published_results:
        for method, value in published_results.items():
            if value is not None:
                methods.append(f"{method}\n(Published)")
                values.append(value)
                colors.append('#3498db')
    
    for method, value in zip(our_methods, our_values):
        methods.append(f"{method}\n(Ours)")
        values.append(value)
        colors.append('#e74c3c')
    
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.02),
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(f'{metric_name} ({"lower" if lower_is_better else "higher"} is better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Published Results'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Our Results')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison chart to {save_path}")
    
    return fig


def plot_perplexity_comparison(
    base_perplexity: float,
    finetuned_perplexity: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare perplexity between base and fine-tuned models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Base Model', 'Fine-tuned (LoRA)']
    perplexities = [base_perplexity, finetuned_perplexity]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(models, perplexities, color=colors, edgecolor='black', linewidth=2)
    
    for bar, ppl in zip(bars, perplexities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{ppl:.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    improvement = ((base_perplexity - finetuned_perplexity) / base_perplexity) * 100
    ax.set_title(f'Perplexity Comparison\n(Improvement: {improvement:.1f}%)', fontsize=14)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved perplexity comparison to {save_path}")
    
    return fig


def plot_parameter_efficiency(
    methods_params: Dict[str, Dict[str, int]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare parameter efficiency across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(methods_params.keys())
    trainable = [methods_params[m].get("trainable", 0) for m in methods]
    total = [methods_params[m].get("total", 1) for m in methods]
    efficiency = [100 * t / tot for t, tot in zip(trainable, total)]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    
    bars1 = axes[0].bar(methods, [t / 1e6 for t in trainable], color=colors, edgecolor='black')
    axes[0].set_ylabel('Trainable Parameters (Millions)')
    axes[0].set_title('Trainable Parameters by Method')
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    
    for bar, val in zip(bars1, trainable):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)
    
    bars2 = axes[1].bar(methods, efficiency, color=colors, edgecolor='black')
    axes[1].set_ylabel('% of Total Parameters')
    axes[1].set_title('Parameter Efficiency (% Trainable)')
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    
    for bar, val in zip(bars2, efficiency):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved parameter efficiency chart to {save_path}")
    
    return fig


def plot_generation_examples(
    generations: List[Dict[str, str]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize generation examples as text."""
    n_examples = min(len(generations), 4)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    
    if n_examples == 1:
        axes = [axes]
    
    for idx, (ax, gen) in enumerate(zip(axes, generations[:n_examples])):
        ax.axis('off')
        
        text = f"Prompt: {gen['prompt'][:100]}...\n\n"
        text += f"Generated: {gen['generated_only'][:200]}..."
        
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax.set_title(f'Generation Example {idx + 1}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved generation examples to {save_path}")
    
    return fig


def create_summary_report(
    results: Dict[str, Any],
    output_dir: str
) -> str:
    """Create a summary report of the experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 60)
    report.append("LoRA Fine-Tuning Experiment Summary")
    report.append("=" * 60)
    report.append("")
    
    if "train_losses" in results:
        report.append(f"Final Training Loss: {results['train_losses'][-1]:.4f}")
    
    if "val_losses" in results and results["val_losses"]:
        report.append(f"Final Validation Loss: {results['val_losses'][-1]:.4f}")
        report.append(f"Best Validation Loss: {min(results['val_losses']):.4f}")
    
    if "benchmark_results" in results:
        report.append("\nBenchmark Results:")
        for bench, data in results["benchmark_results"].items():
            if isinstance(data, dict) and "value" in data:
                report.append(f"  {bench}: {data['metric']} = {data['value']:.2f}")
    
    if "comparison" in results:
        report.append("\nMethod Comparison:")
        for method, data in results["comparison"].items():
            if isinstance(data, dict) and "value" in data:
                report.append(f"  {method}: {data['value']:.2f}")
    
    if "lora_stats" in results:
        stats = results["lora_stats"]
        report.append(f"\nLoRA Configuration:")
        report.append(f"  Rank: {stats.get('rank', 'N/A')}")
        report.append(f"  Alpha: {stats.get('alpha', 'N/A')}")
        report.append(f"  Trainable Parameters: {stats.get('trainable_params', 0):,}")
        report.append(f"  Parameter Efficiency: {stats.get('trainable_percentage', 0):.2f}%")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    report_path = output_dir / "experiment_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, list, dict, bool, type(None))):
                serializable_results[k] = v
        json.dump(serializable_results, f, indent=2, default=str)
    
    return report_text
