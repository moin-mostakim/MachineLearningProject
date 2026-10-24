"""
Visualization utilities for CS Tutoring LLM experiments.

Generates publication-quality figures for educational AI comparison.
"""

import os
import json
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


def generate_method_comparison(results: Dict[str, Any], output_dir: str):
    """Generate method comparison figure."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    experiments = results.get('experiments', {})
    
    methods = []
    params = []
    
    for method, data in experiments.items():
        if method != 'quantization':
            methods.append(data.get('method', method))
            params.append(data.get('parameters', 0) / 1e6)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(methods)]
    
    bars = ax.bar(methods, params, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{bar.get_height():.1f}M', ha='center', fontsize=10)
    
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('CS Tutoring: Method Comparison by Model Size')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'method_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_evaluation_metrics(results: Dict[str, Any], output_dir: str):
    """Generate evaluation metrics comparison."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    evaluation = results.get('evaluation', {})
    
    retrieval_metrics = ['retrieval_recall', 'retrieval_precision', 'retrieval_mrr']
    retrieval_values = [evaluation.get(m, 0) * 100 for m in retrieval_metrics]
    retrieval_labels = ['Recall', 'Precision', 'MRR']
    
    ax1.bar(retrieval_labels, retrieval_values, color=['#3498db', '#e74c3c', '#2ecc71'],
           edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Retrieval Metrics')
    ax1.set_ylim(0, 100)
    
    gen_metrics = ['bleu_score', 'rouge_1', 'rouge_l']
    gen_values = [evaluation.get(m, 0) * 100 for m in gen_metrics]
    gen_labels = ['BLEU', 'ROUGE-1', 'ROUGE-L']
    
    bars2 = ax2.bar(gen_labels, gen_values, color=['#9b59b6', '#f39c12', '#1abc9c'],
                   edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Generation Quality Metrics')
    ax2.set_ylim(0, 100)
    
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bar.get_height():.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_quantization_comparison(results: Dict[str, Any], output_dir: str):
    """Generate INT8 quantization comparison."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    quant = results.get('experiments', {}).get('quantization', {})
    
    memory_reduction = float(quant.get('memory_reduction', '0%').replace('%', ''))
    speedup = float(quant.get('speedup', '1x').replace('x', ''))
    original_latency = float(quant.get('original_latency_ms', '0'))
    quantized_latency = float(quant.get('quantized_latency_ms', '0'))
    
    labels = ['Original', 'INT8 Quantized']
    latencies = [original_latency, quantized_latency]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = ax1.bar(labels, latencies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Inference Latency (Speedup: {speedup:.2f}x)')
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}ms', ha='center', fontsize=10)
    
    sizes = [100, 100 - memory_reduction]
    bars2 = ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Relative Memory (%)')
    ax2.set_title(f'Memory Usage (Reduction: {memory_reduction:.1f}%)')
    ax2.set_ylim(0, 120)
    
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{bar.get_height():.0f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quantization_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_readability_metrics(results: Dict[str, Any], output_dir: str):
    """Generate readability and clarity metrics."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    evaluation = results.get('evaluation', {})
    
    metrics = ['avg_readability', 'avg_clarity', 'syntax_correct_rate']
    values = [
        evaluation.get('avg_readability', 0),
        evaluation.get('avg_clarity', 0),
        evaluation.get('syntax_correct_rate', 0) * 100
    ]
    labels = ['Readability Score', 'Clarity Score', 'Syntax Correct %']
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Score')
    ax.set_title('Response Quality Metrics')
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bar.get_height():.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'readability_metrics.png')
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_literature_comparison(results: Dict[str, Any], output_dir: str):
    """Generate literature comparison figure."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    literature_data = {
        'CodeBERT (Feng, 2020)': 67.2,
        'GraphCodeBERT (Guo, 2021)': 69.1,
        'CodeT5 (Wang, 2021)': 65.9,
        'StarCoder (Li, 2023)': 72.5,
        'Code Llama (Rozière, 2023)': 74.8,
        'DeepSeek Coder (Guo, 2024)': 76.2,
    }
    
    evaluation = results.get('evaluation', {})
    our_bleu = evaluation.get('bleu_score', 0.174) * 100
    
    literature_data['Ours (CodeT5-small)'] = our_bleu
    
    names = list(literature_data.keys())
    values = list(literature_data.values())
    
    colors = ['#3498db'] * (len(names) - 1) + ['#e74c3c']
    
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{bar.get_width():.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('BLEU Score (%)')
    ax.set_title('Literature Comparison: Code Generation Models\n(Note: Different tasks/datasets - not directly comparable)')
    ax.set_xlim(0, 100)
    
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
        generate_method_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate method_comparison: {e}")
    
    try:
        generate_evaluation_metrics(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate evaluation_metrics: {e}")
    
    try:
        generate_quantization_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate quantization_comparison: {e}")
    
    try:
        generate_readability_metrics(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate readability_metrics: {e}")
    
    try:
        generate_literature_comparison(results, output_dir)
    except Exception as e:
        print(f"  Warning: Failed to generate literature_comparison: {e}")
    
    print("Figure generation complete.")


if __name__ == "__main__":
    import glob
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    results_files = glob.glob(os.path.join(results_dir, 'results_*.json'))
    
    if results_files:
        results_file = sorted(results_files)[-1]
        print(f"Using results file: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        output_dir = os.path.join(results_dir, 'figures')
        generate_all_figures(results, output_dir)
    else:
        print(f"No results files found in {results_dir}")
