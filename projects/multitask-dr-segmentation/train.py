"""
Training Script for Multi-Task DR Learning.

Supports:
- Classification-only baseline
- Segmentation-only baseline (U-Net)
- Multi-task model with shared encoder (our method)
- Multiple task weighting strategies

References:
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.synthetic_data import create_synthetic_dataset
from data.dataset import get_dataloaders
from models.multitask_model import build_model, print_model_comparison
from utils.losses import MultiTaskLoss, DiceBCELoss, FocalLoss
from utils.metrics import ClassificationMetrics, SegmentationMetrics, MultiTaskMetrics
from utils.monitoring import ResourceMonitor, TrainingLogger
from utils.visualization import (
    plot_training_curves, plot_confusion_matrix, 
    save_segmentation_examples, create_literature_comparison_table
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, model_type='multitask'):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    if model_type == 'multitask':
        metrics = MultiTaskMetrics()
    elif model_type == 'classification':
        metrics = ClassificationMetrics()
    else:
        metrics = SegmentationMetrics()
    
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        grades = batch['grade'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Compute loss based on model type
        if model_type == 'multitask':
            targets = {'mask': masks, 'grade': grades}
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total']
            metrics.update(outputs, targets)
        elif model_type == 'classification':
            loss = criterion(outputs['classification'], grades)
            metrics.update(
                outputs['classification'].argmax(dim=1),
                grades,
                torch.softmax(outputs['classification'], dim=1)
            )
        else:  # segmentation
            loss = criterion(outputs['segmentation'], masks)
            metrics.update(outputs['segmentation'], masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if model_type == 'multitask':
        computed_metrics = metrics.compute()
    elif model_type == 'classification':
        raw_metrics = metrics.compute()
        computed_metrics = {f'cls_{k}': v for k, v in raw_metrics.items() 
                          if k != 'confusion_matrix'}
    else:
        raw_metrics = metrics.compute()
        computed_metrics = {f'seg_{k}': v for k, v in raw_metrics.items()}
    
    return avg_loss, computed_metrics


def validate_epoch(model, dataloader, criterion, device, model_type='multitask'):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    if model_type == 'multitask':
        metrics = MultiTaskMetrics()
    elif model_type == 'classification':
        metrics = ClassificationMetrics()
    else:
        metrics = SegmentationMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            grades = batch['grade'].to(device)
            
            outputs = model(images)
            
            # Compute loss based on model type
            if model_type == 'multitask':
                targets = {'mask': masks, 'grade': grades}
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total']
                metrics.update(outputs, targets)
            elif model_type == 'classification':
                loss = criterion(outputs['classification'], grades)
                metrics.update(
                    outputs['classification'].argmax(dim=1),
                    grades,
                    torch.softmax(outputs['classification'], dim=1)
                )
            else:  # segmentation
                loss = criterion(outputs['segmentation'], masks)
                metrics.update(outputs['segmentation'], masks)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if model_type == 'multitask':
        computed_metrics = metrics.compute()
    elif model_type == 'classification':
        raw_metrics = metrics.compute()
        computed_metrics = {f'cls_{k}': v for k, v in raw_metrics.items() 
                          if k != 'confusion_matrix'}
        computed_metrics['confusion_matrix'] = raw_metrics.get('confusion_matrix')
    else:
        raw_metrics = metrics.compute()
        computed_metrics = {f'seg_{k}': v for k, v in raw_metrics.items()}
    
    return avg_loss, computed_metrics


def train_model(model_type='multitask', weighting_strategy='uncertainty', device=None):
    """Train a model with specified configuration."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_type.upper()} MODEL")
    if model_type == 'multitask':
        print(f"Task Weighting Strategy: {weighting_strategy}")
    print(f"{'='*70}")
    
    # Configuration
    eff_config = config.get_effective_config()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Debug Mode: {config.DEBUG_MODE}")
    print(f"Epochs: {eff_config['num_epochs']}")
    print(f"Batch Size: {eff_config['batch_size']}")
    
    # Data (load first to get class weights)
    train_loader, val_loader, _, class_weights = get_dataloaders(use_huggingface=config.USE_HUGGINGFACE)
    class_weights = class_weights.to(device)
    
    # Build model with pretrained encoder
    pretrained = getattr(config, 'USE_PRETRAINED_ENCODER', True)
    model = build_model(model_type, weighting_strategy, pretrained=pretrained)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
    
    # Loss function with class weights for imbalanced data
    use_weights = getattr(config, 'USE_CLASS_WEIGHTS', True)
    if model_type == 'multitask':
        criterion = MultiTaskLoss(strategy=weighting_strategy, class_weights=class_weights if use_weights else None)
    elif model_type == 'classification':
        criterion = FocalLoss(alpha=class_weights if use_weights else None)
    else:
        criterion = DiceBCELoss()
    criterion = criterion.to(device) if hasattr(criterion, 'to') else criterion
    
    # Optimizer
    params_to_optimize = list(model.parameters())
    if hasattr(criterion, 'parameters'):
        params_to_optimize += list(criterion.parameters())
    
    optimizer = optim.Adam(params_to_optimize, lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
                                   patience=config.SCHEDULER_PATIENCE)
    
    # Monitoring
    monitor = ResourceMonitor()
    logger = TrainingLogger(experiment_name=f"{model_type}_{weighting_strategy}")
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(eff_config['num_epochs']):
        monitor.start_epoch()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, model_type
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, model_type
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Get task weights if applicable
        task_weights = None
        if model_type == 'multitask' and hasattr(criterion, 'get_weights'):
            task_weights = criterion.get_weights()
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics,
                        current_lr, task_weights)
        logger.print_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
        
        if task_weights:
            print(f"  Task Weights: Seg={task_weights[0]:.4f}, Cls={task_weights[1]:.4f}")
        
        # Timing
        epoch_time = monitor.end_epoch(len(train_loader.dataset))
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{model_type}_best.pth')
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Final monitoring summary
    monitor.print_summary()
    
    # Save training history
    logger.save_history()
    
    return model, logger.history, best_metrics


def run_all_experiments():
    """Run all model experiments and generate comparisons."""
    
    print("\n" + "=" * 70)
    print("MULTI-TASK LEARNING FOR DR GRADING AND LESION SEGMENTATION")
    print("=" * 70)
    
    # Set seed
    set_seed(config.SEED)
    
    # Generate synthetic data (only if not using HuggingFace)
    if not config.USE_HUGGINGFACE:
        print("\n[1/5] Generating synthetic dataset...")
        create_synthetic_dataset()
    else:
        print("\n[1/5] Using real IDRiD dataset from HuggingFace...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Results storage
    all_results = {}
    
    # Train classification baseline
    print("\n[2/5] Training classification-only baseline...")
    cls_model, cls_history, cls_metrics = train_model('classification', device=device)
    all_results['Classification-only'] = cls_metrics
    
    # Train segmentation baseline
    print("\n[3/5] Training segmentation-only baseline (U-Net)...")
    seg_model, seg_history, seg_metrics = train_model('segmentation', device=device)
    all_results['Segmentation-only (U-Net)'] = seg_metrics
    
    # Train multi-task model
    print("\n[4/5] Training multi-task model (Our Method)...")
    mtl_model, mtl_history, mtl_metrics = train_model(
        'multitask', weighting_strategy='uncertainty', device=device
    )
    all_results['Multi-task (Ours)'] = mtl_metrics
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Training curves
    plot_training_curves(mtl_history, 
                        os.path.join(config.FIGURES_DIR, 'training_curves.png'))
    
    # Confusion matrix
    if mtl_metrics and 'confusion_matrix' in mtl_metrics and mtl_metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(mtl_metrics['confusion_matrix'],
                             save_path=os.path.join(config.FIGURES_DIR, 'confusion_matrix.png'))
    
    # Literature comparison
    create_literature_comparison_table(
        mtl_metrics,
        save_path=os.path.join(config.FIGURES_DIR, 'literature_comparison.png')
    )
    
    # Save segmentation examples
    _, _, test_loader, _ = get_dataloaders(use_huggingface=config.USE_HUGGINGFACE)
    mtl_model.eval()
    save_segmentation_examples(test_loader, mtl_model, num_examples=3, 
                               save_dir=config.FIGURES_DIR)
    
    # Print final comparison
    print_results_comparison(all_results)
    
    # Save results
    results_path = os.path.join(config.EXPERIMENTS_DIR, 'final_results.json')
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    
    # Convert to serializable format
    serializable_results = {}
    for model_name, metrics in all_results.items():
        if metrics:
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items() if k != 'confusion_matrix'
            }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return all_results


def print_results_comparison(results):
    """Print final results comparison table."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    
    # Classification metrics
    print("\nClassification Metrics:")
    print("-" * 80)
    print(f"{'Model':<30} {'Accuracy':>12} {'Kappa':>12} {'F1 (macro)':>12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        if metrics and 'cls_accuracy' in metrics:
            acc = metrics.get('cls_accuracy', 0)
            kappa = metrics.get('cls_kappa', 0)
            f1 = metrics.get('cls_f1_macro', 0)
            print(f"{model_name:<30} {acc:>12.4f} {kappa:>12.4f} {f1:>12.4f}")
    
    # Segmentation metrics
    print("\nSegmentation Metrics:")
    print("-" * 80)
    print(f"{'Model':<30} {'Dice (mean)':>12} {'IoU (mean)':>12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        if metrics and 'seg_dice_mean' in metrics:
            dice = metrics.get('seg_dice_mean', 0)
            iou = metrics.get('seg_iou_mean', 0)
            print(f"{model_name:<30} {dice:>12.4f} {iou:>12.4f}")
    
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Task DR Learning')
    parser.add_argument('--model', type=str, default='all',
                       choices=['classification', 'segmentation', 'multitask', 'all'],
                       help='Model type to train')
    parser.add_argument('--weighting', type=str, default='uncertainty',
                       choices=['fixed', 'uncertainty', 'gradnorm'],
                       help='Task weighting strategy')
    args = parser.parse_args()
    
    if args.model == 'all':
        run_all_experiments()
    else:
        set_seed(config.SEED)
        create_synthetic_dataset()
        train_model(args.model, args.weighting)
