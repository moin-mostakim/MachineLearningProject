"""
Visualization utilities for KAN Diabetic Retinopathy experiments.

Generates publication-quality figures comparing KAN vs MLP architectures.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all results from subdirectories."""
    results = {'kan': None, 'mlp': None}
    
    for run_dir in glob.glob(os.path.join(results_dir, '*')):
        if os.path.isdir(run_dir):
            kan_file = os.path.join(run_dir, 'kan_results.json')
            mlp_file = os.path.join(run_dir, 'mlp_results.json')
            
            if os.path.exists(kan_file) and results['kan'] is None:
                with open(kan_file, 'r') as f:
                    results['kan'] = json.load(f)
            
            if os.path.exists(mlp_file) and results['mlp'] is None:
                with open(mlp_file, 'r') as f:
                    results['mlp'] = json.load(f)
    
    return results


def generate_accuracy_comparison(results: Dict[str, Any], output_dir: str):
    """Generate accuracy comparison between KAN and MLP."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = []
    accuracies = []
    qwks = []
    
    for method, data in results.items():
        if data:
            methods.append(method.upper())
            accuracies.append(data.get('accuracy', 0) * 100)
            qwks.append(data.get('qwk', 0))
    
    colors = ['#3498db', '#e74c3c'][:len(methods)]
    
    bars1 = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}%', ha='center', fontsize=10)
    
    bars2 = ax2.bar(methods, qwks, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Quadratic Weighted Kappa')
    ax2.set_title('QWK Comparison')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_parameter_comparison(results: Dict[str, Any], output_dir: str):
    """Generate parameter count comparison."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    params = []
    
    for method, data in results.items():
        if data:
            methods.append(method.upper())
            params.append(data.get('params', 0) / 1e6)
    
    colors = ['#3498db', '#e74c3c'][:len(methods)]
    
    bars = ax.bar(methods, params, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Size Comparison: KAN vs MLP')
    ax.set_yscale('log')
    
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
               f'{p:.2f}M', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'parameter_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_training_curves(results: Dict[str, Any], output_dir: str):
    """Generate training loss curves."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'kan': '#3498db', 'mlp': '#e74c3c'}
    
    for method, data in results.items():
        if data and 'history' in data:
            history = data['history']
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            
            if history.get('train_loss'):
                ax1.plot(epochs, history['train_loss'], '-o', 
                        label=f'{method.upper()} (train)', color=colors.get(method, 'gray'),
                        markersize=6)
            if history.get('val_loss'):
                ax1.plot(epochs, history['val_loss'], '--s', 
                        label=f'{method.upper()} (val)', color=colors.get(method, 'gray'),
                        markersize=6, alpha=0.7)
            
            if history.get('train_acc'):
                train_acc = [a * 100 for a in history['train_acc']]
                ax2.plot(epochs, train_acc, '-o', 
                        label=f'{method.upper()} (train)', color=colors.get(method, 'gray'),
                        markersize=6)
            if history.get('val_acc'):
                val_acc = [a * 100 for a in history['val_acc']]
                ax2.plot(epochs, val_acc, '--s', 
                        label=f'{method.upper()} (val)', color=colors.get(method, 'gray'),
                        markersize=6, alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_resource_usage(results: Dict[str, Any], output_dir: str):
    """Generate resource usage comparison."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = []
    times = []
    memory = []
    
    for method, data in results.items():
        if data and 'resource_stats' in data:
            methods.append(method.upper())
            stats = data['resource_stats']
            times.append(stats.get('total_time', 0))
            memory.append(stats.get('peak_memory_mb', 0) / 1024)
    
    colors = ['#3498db', '#e74c3c'][:len(methods)]
    
    bars1 = ax1.bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bar.get_height():.1f}s', ha='center', fontsize=10)
    
    bars2 = ax2.bar(methods, memory, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('Memory Usage Comparison')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{bar.get_height():.2f} GB', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'resource_usage.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_literature_comparison(results: Dict[str, Any], output_dir: str):
    """Generate literature comparison figure."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    literature_data = {
        'ResNet-50 (EyePACS)': 0.82,
        'EfficientNet (APTOS)': 0.91,
        'ViT-B/16 (DR)': 0.85,
        'DenseNet-121 (EyePACS)': 0.80,
        'InceptionResNetV2': 0.79,
        'SE-ResNeXt-50': 0.83,
    }
    
    kan_qwk = results.get('kan', {}).get('qwk', 0.039)
    mlp_qwk = results.get('mlp', {}).get('qwk', 0.0)
    
    literature_data['Ours (KAN)'] = kan_qwk
    literature_data['Ours (MLP)'] = mlp_qwk
    
    names = list(literature_data.keys())
    values = list(literature_data.values())
    
    colors = ['#3498db'] * (len(names) - 2) + ['#e74c3c', '#2ecc71']
    
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{bar.get_width():.3f}', va='center', fontsize=9)
    
    ax.set_xlabel('Quadratic Weighted Kappa')
    ax.set_title('Literature Comparison: DR Classification (QWK)\n(Note: Different datasets - not directly comparable)')
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'literature_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_figures(results: Dict[str, Any], output_dir: str):
    """Generate all visualization figures."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating figures to {output_dir}/")
    
    try:
        generate_accuracy_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate accuracy_comparison: {e}")
    
    try:
        generate_parameter_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate parameter_comparison: {e}")
    
    try:
        generate_training_curves(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate training_curves: {e}")
    
    try:
        generate_resource_usage(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate resource_usage: {e}")
    
    try:
        generate_literature_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate literature_comparison: {e}")
    
    print("Figure generation complete.")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    results = load_results(results_dir)
    
    output_dir = os.path.join(results_dir, 'figures')
    generate_all_figures(results, output_dir)
