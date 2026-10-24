"""Visualization utilities for multimodal analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_accuracies, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix."""
    if class_names is None:
        class_names = ["Normal", "MCI", "AD"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_method_comparison(
    results: Dict[str, float],
    metric_name: str = "Accuracy",
    published_results: Dict[str, float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot comparison of methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = []
    values = []
    colors = []
    
    if published_results:
        for method, value in published_results.items():
            methods.append(f"{method}\n(Published)")
            values.append(value)
            colors.append('#3498db')
    
    for method, value in results.items():
        methods.append(f"{method}\n(Ours)")
        values.append(value)
        colors.append('#e74c3c')
    
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison: Published vs Our Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Published'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Our Methods')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison chart to {save_path}")
    
    return fig


def plot_modality_importance(
    modality_weights: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot modality importance from attention weights."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modalities = list(modality_weights.keys())
    weights = list(modality_weights.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(modalities)))
    
    bars = ax.bar(modalities, weights, color=colors, edgecolor='black')
    
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{w:.3f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Attention Weight')
    ax.set_title('Modality Importance')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved modality importance to {save_path}")
    
    return fig


def plot_quantization_comparison(
    comparison: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot FP32 vs INT8 comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(['FP32', 'INT8'],
               [comparison['size']['fp32_size_mb'],
                comparison['size']['int8_size_mb']],
               color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[0].set_ylabel('Size (MB)')
    axes[0].set_title('Model Size')
    
    axes[1].bar(['FP32', 'INT8'],
               [comparison['latency']['fp32']['mean_ms'],
                comparison['latency']['int8']['mean_ms']],
               color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title(f"Inference Latency (Speedup: {comparison['latency']['speedup']:.2f}x)")
    
    if comparison.get('accuracy'):
        axes[2].bar(['FP32', 'INT8'],
                   [comparison['accuracy'].get('fp32_metric', 0),
                    comparison['accuracy'].get('int8_metric', 0)],
                   color=['#3498db', '#e74c3c'], edgecolor='black')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Accuracy Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved quantization comparison to {save_path}")
    
    return fig
