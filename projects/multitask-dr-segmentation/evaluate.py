"""
Evaluation Script for Multi-Task DR Learning.

Performs comprehensive evaluation:
- Classification metrics (Accuracy, Kappa, F1, AUC)
- Segmentation metrics (Dice, IoU per lesion type)
- INT8 quantization comparison
- Resource profiling
- Visualization generation
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import get_dataloaders
from models.multitask_model import build_model, print_model_comparison
from models.encoder import count_parameters
from utils.metrics import MultiTaskMetrics, ClassificationMetrics, SegmentationMetrics
from utils.metrics import print_metrics_summary
from utils.quantization import run_quantization_analysis, print_quantization_comparison
from utils.visualization import (
    save_segmentation_examples, plot_confusion_matrix,
    create_literature_comparison_table, plot_model_comparison
)


def load_model(model_type, checkpoint_path=None, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(model_type)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    else:
        print(f"No checkpoint found, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, model_type='multitask', device=None):
    """Evaluate model on test set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    if model_type == 'multitask':
        metrics = MultiTaskMetrics()
    elif model_type == 'classification':
        metrics = ClassificationMetrics()
    else:
        metrics = SegmentationMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            grades = batch['grade'].to(device)
            
            outputs = model(images)
            
            if model_type == 'multitask':
                targets = {'mask': masks, 'grade': grades}
                metrics.update(outputs, targets)
            elif model_type == 'classification':
                metrics.update(
                    outputs['classification'].argmax(dim=1),
                    grades,
                    torch.softmax(outputs['classification'], dim=1)
                )
            else:
                metrics.update(outputs['segmentation'], masks)
    
    return metrics.compute()


def generate_parameter_table():
    """Generate and print parameter count table for all models."""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    model_types = ['classification', 'segmentation', 'multitask']
    
    print(f"\n{'Model':<35} {'Total Params':>15} {'Size (MB)':>12}")
    print("-" * 70)
    
    param_data = {}
    
    for model_type in model_types:
        model = build_model(model_type)
        total_params = count_parameters(model)
        size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        name = {
            'classification': 'Classification-only',
            'segmentation': 'Segmentation-only (U-Net)',
            'multitask': 'Multi-task (Ours)'
        }[model_type]
        
        print(f"{name:<35} {total_params:>15,} {size_mb:>12.2f}")
        param_data[model_type] = {'params': total_params, 'size_mb': size_mb}
        
        # Component breakdown
        if model_type == 'multitask':
            print("\n  Component Breakdown:")
            for comp_name, module in model.named_children():
                comp_params = count_parameters(module)
                comp_size = comp_params * 4 / (1024 ** 2)
                print(f"    {comp_name:<30} {comp_params:>12,} ({comp_size:.2f} MB)")
    
    print("-" * 70)
    
    # Efficiency comparison
    print("\n  Multi-task Efficiency:")
    cls_params = param_data['classification']['params']
    seg_params = param_data['segmentation']['params']
    mtl_params = param_data['multitask']['params']
    
    combined_baseline = cls_params + seg_params
    param_savings = (1 - mtl_params / combined_baseline) * 100
    
    print(f"    Combined baseline parameters:   {combined_baseline:,}")
    print(f"    Multi-task parameters:          {mtl_params:,}")
    print(f"    Parameter reduction:            {param_savings:.1f}%")
    
    return param_data


def run_full_evaluation():
    """Run comprehensive evaluation."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get test data
    _, _, test_loader = get_dataloaders()
    
    results = {}
    
    # 1. Parameter table
    print("\n[1/4] Model Parameter Analysis")
    param_data = generate_parameter_table()
    
    # 2. Evaluate all models
    print("\n[2/4] Model Evaluation on Test Set")
    print("-" * 70)
    
    model_configs = [
        ('classification', 'Classification-only'),
        ('segmentation', 'Segmentation-only (U-Net)'),
        ('multitask', 'Multi-task (Ours)')
    ]
    
    for model_type, display_name in model_configs:
        print(f"\nEvaluating {display_name}...")
        
        # Try to load from checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{model_type}_best.pth')
        model = load_model(model_type, checkpoint_path, device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, model_type, device)
        results[display_name] = metrics
        
        print_metrics_summary(metrics, prefix=f"{display_name} ")
    
    # 3. Quantization analysis
    print("\n[3/4] INT8 Quantization Analysis")
    quant_results = run_quantization_analysis()
    
    # 4. Generate visualizations
    print("\n[4/4] Generating Evaluation Figures")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Confusion matrix for multi-task model
    mtl_metrics = results.get('Multi-task (Ours)', {})
    if 'confusion_matrix' in mtl_metrics and mtl_metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(
            mtl_metrics['confusion_matrix'],
            save_path=os.path.join(config.FIGURES_DIR, 'test_confusion_matrix.png')
        )
    
    # Segmentation examples
    mtl_model = load_model('multitask', 
                           os.path.join(config.CHECKPOINT_DIR, 'multitask_best.pth'),
                           device)
    save_segmentation_examples(test_loader, mtl_model, num_examples=5,
                               save_dir=config.FIGURES_DIR)
    
    # Literature comparison
    create_literature_comparison_table(
        mtl_metrics,
        save_path=os.path.join(config.FIGURES_DIR, 'literature_comparison.png')
    )
    
    # Model comparison charts
    if mtl_metrics:
        for metric in ['cls_kappa', 'cls_accuracy', 'seg_dice_mean', 'seg_iou_mean']:
            if metric in mtl_metrics:
                comparison_data = {
                    name: m.get(metric, 0) 
                    for name, m in results.items() 
                    if m and metric in m
                }
                if comparison_data:
                    plot_model_comparison(
                        comparison_data, metric,
                        save_path=os.path.join(config.FIGURES_DIR, f'comparison_{metric}.png')
                    )
    
    # Print final summary
    print_final_summary(results, param_data, quant_results)
    
    # Save all results
    save_evaluation_results(results, param_data, quant_results)
    
    return results


def print_final_summary(results, param_data, quant_results):
    """Print comprehensive final summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Best results per metric
    print("\n📊 CLASSIFICATION PERFORMANCE")
    print("-" * 60)
    
    best_kappa = 0
    best_kappa_model = ""
    
    for model_name, metrics in results.items():
        if metrics and 'cls_kappa' in metrics:
            kappa = metrics['cls_kappa']
            if kappa > best_kappa:
                best_kappa = kappa
                best_kappa_model = model_name
            
            print(f"  {model_name}:")
            print(f"    Kappa:    {kappa:.4f}")
            print(f"    Accuracy: {metrics.get('cls_accuracy', 0):.4f}")
            print(f"    F1:       {metrics.get('cls_f1_macro', 0):.4f}")
    
    print(f"\n  🏆 Best Classification: {best_kappa_model} (Kappa: {best_kappa:.4f})")
    
    print("\n📊 SEGMENTATION PERFORMANCE")
    print("-" * 60)
    
    best_dice = 0
    best_dice_model = ""
    
    for model_name, metrics in results.items():
        if metrics and 'seg_dice_mean' in metrics:
            dice = metrics['seg_dice_mean']
            if dice > best_dice:
                best_dice = dice
                best_dice_model = model_name
            
            print(f"  {model_name}:")
            print(f"    Mean Dice: {dice:.4f}")
            print(f"    Mean IoU:  {metrics.get('seg_iou_mean', 0):.4f}")
    
    print(f"\n  🏆 Best Segmentation: {best_dice_model} (Dice: {best_dice:.4f})")
    
    print("\n📊 EFFICIENCY COMPARISON")
    print("-" * 60)
    
    if quant_results:
        for qr in quant_results:
            print(f"  {qr['model_type']}:")
            print(f"    FP32 Size:    {qr['fp32']['size_mb']:.2f} MB")
            print(f"    INT8 Size:    {qr['int8']['size_mb']:.2f} MB")
            print(f"    Reduction:    {qr['improvement']['size_reduction_percent']:.1f}%")
            print(f"    FP32 Latency: {qr['fp32']['inference_ms']:.2f} ms")
            print(f"    INT8 Latency: {qr['int8']['inference_ms']:.2f} ms")


def save_evaluation_results(results, param_data, quant_results):
    """Save all evaluation results to JSON."""
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    
    # Convert to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    output = {
        'model_results': convert_to_serializable(results),
        'parameter_data': convert_to_serializable(param_data),
        'quantization_results': convert_to_serializable(quant_results)
    }
    
    output_path = os.path.join(config.EXPERIMENTS_DIR, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    results = run_full_evaluation()
