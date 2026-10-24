"""Visualization utilities for federated learning experiments."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

plt.style.use('seaborn-v0_8-whitegrid')


def plot_convergence(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot convergence curves for different FL algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'FedAvg': '#3498db', 'FedProx': '#e74c3c', 'PersonalizedFL+DP': '#2ecc71'}
    
    for name, data in results.items():
        if 'accuracies' in data:
            ax.plot(data['accuracies'], label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Federated Learning Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_method_comparison(results: Dict[str, float], published: Dict[str, float] = None,
                          save_path: Optional[str] = None):
    """Plot comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods, values, colors = [], [], []
    
    if published:
        for m, v in published.items():
            methods.append(f"{m}\n(Published)")
            values.append(v)
            colors.append('#3498db')
    
    for m, v in results.items():
        methods.append(f"{m}\n(Ours)")
        values.append(v)
        colors.append('#e74c3c')
    
    bars = ax.bar(methods, values, color=colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Federated Learning Methods Comparison')
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_privacy_accuracy_tradeoff(epsilons: List[float], accuracies: List[float],
                                   save_path: Optional[str] = None):
    """Plot privacy-accuracy tradeoff curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(epsilons, accuracies, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Privacy-Accuracy Tradeoff')
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.annotate('More Private', xy=(min(epsilons), 0.1), fontsize=10, color='green')
    ax.annotate('Less Private', xy=(max(epsilons)*0.7, 0.1), fontsize=10, color='red')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_quantization_comparison(comparison: Dict[str, Any], save_path: Optional[str] = None):
    """Plot FP32 vs INT8 comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].bar(['FP32', 'INT8'], 
               [comparison['size']['fp32_size_mb'], comparison['size']['int8_size_mb']],
               color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[0].set_ylabel('Size (MB)')
    axes[0].set_title('Model Size')
    
    axes[1].bar(['FP32', 'INT8'],
               [comparison['latency']['fp32']['mean_ms'], comparison['latency']['int8']['mean_ms']],
               color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title(f"Latency (Speedup: {comparison['latency']['speedup']:.2f}x)")
    
    if comparison.get('accuracy'):
        axes[2].bar(['FP32', 'INT8'],
                   [comparison['accuracy']['fp32_metric'], comparison['accuracy']['int8_metric']],
                   color=['#3498db', '#e74c3c'], edgecolor='black')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Accuracy')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
