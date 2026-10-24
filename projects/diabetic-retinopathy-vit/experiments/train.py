"""
Main training script for Diabetic Retinopathy grading models.

Trains and compares:
1. ResNet-50 baseline
2. Standard ViT baseline
3. Multi-Scale Lesion-Aware ViT (our method)

Outputs:
- Training logs
- Model checkpoints
- Metrics comparison
- Attention visualizations
- Quantization comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

from configs.config import Config, get_config, get_debug_config
from src.data.dataset import get_dataloaders, DRDataset
from src.models import (
    create_resnet_baseline,
    create_vit_baseline,
    create_multiscale_lesion_vit,
)
from src.utils import (
    OrdinalCrossEntropyLoss,
    ResourceMonitor,
    EarlyStopping,
    train_epoch,
    validate_epoch,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    compute_all_metrics,
    print_metrics_table,
    compare_quantization,
    print_quantization_comparison,
    count_parameters,
)
from src.visualization import (
    create_explainability_report,
    visualize_attention_maps,
    visualize_lesion_attention,
)


def setup_directories(config: Config):
    """Create output directories."""
    for dir_path in [config.output_dir, config.checkpoint_dir, 
                     config.log_dir, config.figure_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_class_weights(train_loader) -> torch.Tensor:
    """Get class weights from training data."""
    return train_loader.dataset.get_class_weights()


def print_parameter_table(models: dict):
    """Print parameter count comparison table."""
    print("\n" + "=" * 70)
    print(" Model Parameter Comparison")
    print("=" * 70)
    
    headers = ["Model", "Total Params", "Trainable", "Size (MB)"]
    print(f"{headers[0]:<30} {headers[1]:>15} {headers[2]:>15} {headers[3]:>12}")
    print("-" * 70)
    
    for name, model in models.items():
        params = count_parameters(model)
        # Estimate size (4 bytes per float32 param)
        size_mb = params['total'] * 4 / (1024 * 1024)
        
        print(f"{name:<30} {params['total']:>15,} {params['trainable']:>15,} {size_mb:>12.2f}")
    
    print("=" * 70 + "\n")


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    config: Config,
    device: torch.device,
) -> dict:
    """
    Train a single model.
    
    Args:
        model: Model to train
        model_name: Name for logging
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration
        device: Device to train on
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*70}")
    print(f" Training: {model_name}")
    print(f"{'='*70}\n")
    
    model = model.to(device)
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(train_loader).to(device)
    
    # Loss function
    criterion = OrdinalCrossEntropyLoss(
        num_classes=config.data.num_classes,
        ordinal_weight=config.training.ordinal_weight if config.training.use_ordinal_loss else 0,
        label_smoothing=config.training.label_smoothing,
        class_weights=class_weights if config.data.use_class_weights else None,
    )
    
    # Optimizer and scheduler
    optimizer = get_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
    )
    
    scheduler = get_scheduler(
        optimizer,
        num_epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
        scheduler_type=config.training.scheduler,
    )
    
    # Mixed precision
    scaler = GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    
    # Callbacks
    resource_monitor = ResourceMonitor()
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode='max',
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_qwk': [],
        'val_loss': [],
        'val_qwk': [],
        'val_accuracy': [],
        'learning_rates': [],
    }
    
    best_qwk = 0
    best_metrics = {}
    
    # Determine number of batches for debug mode
    debug_batches = config.training.debug_batches if config.training.debug_mode else None
    
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        print("-" * 40)
        
        resource_monitor.start_epoch()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            max_grad_norm=config.training.max_grad_norm,
            resource_monitor=resource_monitor,
            debug_batches=debug_batches,
        )
        
        # Validate
        val_metrics, val_preds, val_targets, val_probs = validate_epoch(
            model, val_loader, criterion, device,
            debug_batches=debug_batches,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Resource stats
        resource_stats = resource_monitor.end_epoch()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_qwk'].append(train_metrics['qwk'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_qwk'].append(val_metrics['quadratic_weighted_kappa'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train QWK: {train_metrics['qwk']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val QWK: {val_metrics['quadratic_weighted_kappa']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Epoch time: {resource_stats['epoch_time_sec']:.1f}s")
        
        if device.type == 'cuda':
            print(f"GPU Memory: {resource_stats.get('gpu_memory_peak_gb', 0):.2f} GB")
        
        # Save best model
        if val_metrics['quadratic_weighted_kappa'] > best_qwk:
            best_qwk = val_metrics['quadratic_weighted_kappa']
            best_metrics = val_metrics.copy()
            
            checkpoint_path = Path(config.checkpoint_dir) / f'{model_name}_best.pt'
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(checkpoint_path), scheduler
            )
            print(f"Saved best model with QWK: {best_qwk:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['quadratic_weighted_kappa'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Get resource summary
    resource_summary = resource_monitor.get_summary()
    
    return {
        'history': history,
        'best_metrics': best_metrics,
        'best_qwk': best_qwk,
        'resource_summary': resource_summary,
    }


def evaluate_model(
    model: nn.Module,
    model_name: str,
    test_loader,
    config: Config,
    device: torch.device,
) -> dict:
    """Evaluate model on test set."""
    print(f"\n{'='*70}")
    print(f" Evaluating: {model_name}")
    print(f"{'='*70}\n")
    
    model = model.to(device)
    model.eval()
    
    # Load best checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / f'{model_name}_best.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch'] + 1}")
    
    # Simple CE loss for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    debug_batches = config.training.debug_batches if config.training.debug_mode else None
    metrics, preds, targets, probs = validate_epoch(
        model, test_loader, criterion, device,
        debug_batches=debug_batches,
    )
    
    print_metrics_table(metrics, f"{model_name} Test Results")
    
    return {
        'metrics': metrics,
        'predictions': preds,
        'targets': targets,
        'probabilities': probs,
    }


def generate_visualizations(
    model: nn.Module,
    model_name: str,
    test_loader,
    config: Config,
    device: torch.device,
):
    """Generate attention visualizations."""
    print(f"\nGenerating visualizations for {model_name}...")
    
    model = model.to(device)
    model.eval()
    
    # Load best checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / f'{model_name}_best.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get sample images
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Get one sample from test set
    images, labels = next(iter(test_loader))
    
    for i in range(min(config.visualization.num_samples_to_visualize, len(images))):
        image = images[i:i+1]
        label = labels[i].item()
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = (images[i] * std + mean).permute(1, 2, 0).numpy()
        original_image = (original_image * 255).astype(np.uint8)
        
        # Create explainability report
        save_path = Path(config.figure_dir) / f'{model_name}_sample_{i}_explainability.png'
        try:
            create_explainability_report(
                model, image, original_image, label, class_names,
                str(save_path)
            )
        except Exception as e:
            print(f"Error creating explainability report: {e}")
        
        # Lesion attention (for our method)
        if hasattr(model, 'get_lesion_heatmaps'):
            save_path = Path(config.figure_dir) / f'{model_name}_sample_{i}_lesion_attention.png'
            try:
                visualize_lesion_attention(
                    model, image, original_image, str(save_path)
                )
            except Exception as e:
                print(f"Error creating lesion attention visualization: {e}")


def run_quantization_comparison(
    models: dict,
    test_loader,
    config: Config,
):
    """Run INT8 quantization comparison for all models."""
    print("\n" + "=" * 70)
    print(" INT8 Quantization Comparison")
    print("=" * 70)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nQuantizing {name}...")
        
        try:
            quant_results = compare_quantization(
                model,
                calibration_loader=test_loader,
                input_shape=(1, 3, config.data.image_size, config.data.image_size),
                include_static=False,  # Skip static for speed
            )
            
            results[name] = quant_results
            print_quantization_comparison(quant_results)
            
        except Exception as e:
            print(f"Quantization failed for {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


def save_results(results: dict, config: Config):
    """Save all results to JSON."""
    output_path = Path(config.output_dir) / 'results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def plot_training_curves(results: dict, config: Config):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for name, result in results['training'].items():
        history = result['history']
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label=f'{name} (train)')
        axes[0, 0].plot(history['val_loss'], '--', label=f'{name} (val)')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # QWK
    for name, result in results['training'].items():
        history = result['history']
        axes[0, 1].plot(history['train_qwk'], label=f'{name} (train)')
        axes[0, 1].plot(history['val_qwk'], '--', label=f'{name} (val)')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Quadratic Weighted Kappa')
    axes[0, 1].set_title('Training and Validation QWK')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final metrics comparison
    model_names = list(results['evaluation'].keys())
    metrics_to_plot = ['quadratic_weighted_kappa', 'accuracy', 'auc_macro']
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results['evaluation'][name]['metrics'].get(metric, 0) 
                  for name in model_names]
        axes[1, 0].bar(x + i * width, values, width, label=metric)
    
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Test Set Metrics Comparison')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Learning rate schedule
    for name, result in results['training'].items():
        history = result['history']
        axes[1, 1].plot(history['learning_rates'], label=name)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    save_path = Path(config.figure_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print(" Adaptive Multi-Scale Explainable Vision Transformer")
    print(" with Lesion-Aware Attention for Early DR Grading")
    print("=" * 70 + "\n")
    
    # Get configuration
    config = get_debug_config()  # Use debug config for quick testing
    print(f"Running in {'DEBUG' if config.training.debug_mode else 'FULL'} mode")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.epochs}")
    
    # Setup
    setup_directories(config)
    device = torch.device(config.device)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Create data loaders
    print("\nLoading data...")
    data_dir = Path(config.output_dir) / 'data'
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=config.data.batch_size,
        image_size=config.data.image_size,
        num_workers=0,  # Use 0 workers for faster startup in debug
        num_synthetic_samples=100 if config.training.debug_mode else 2000,
        seed=config.seed,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class distribution: {train_loader.dataset.get_class_distribution()}")
    
    # Create models
    print("\nCreating models...")
    
    models = {
        'ResNet50': create_resnet_baseline(
            num_classes=config.data.num_classes,
            variant=config.model.resnet_variant,
            pretrained=config.model.resnet_pretrained,
        ),
        'ViT_Baseline': create_vit_baseline(
            img_size=config.data.image_size,
            num_classes=config.data.num_classes,
            embed_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            num_heads=config.model.num_heads,
            drop_rate=config.model.dropout,
            attn_drop_rate=config.model.attention_dropout,
            drop_path_rate=config.model.drop_path,
        ),
        'MultiScale_LesionViT': create_multiscale_lesion_vit(
            img_size=config.data.image_size,
            num_classes=config.data.num_classes,
            embed_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            num_heads=config.model.num_heads,
            num_lesion_types=config.model.num_lesion_types,
            lesion_attention_heads=config.model.lesion_attention_heads,
            drop_rate=config.model.dropout,
            attn_drop_rate=config.model.attention_dropout,
            drop_path_rate=config.model.drop_path,
        ),
    }
    
    # Print parameter comparison
    print_parameter_table(models)
    
    # Results storage
    results = {
        'config': {
            'debug_mode': config.training.debug_mode,
            'epochs': config.training.epochs,
            'batch_size': config.data.batch_size,
            'image_size': config.data.image_size,
            'learning_rate': config.training.learning_rate,
        },
        'training': {},
        'evaluation': {},
        'quantization': {},
    }
    
    # Train each model
    for name, model in models.items():
        train_result = train_model(
            model, name, train_loader, val_loader, config, device
        )
        results['training'][name] = train_result
    
    # Evaluate each model
    for name, model in models.items():
        eval_result = evaluate_model(
            model, name, test_loader, config, device
        )
        results['evaluation'][name] = eval_result
    
    # Generate visualizations
    if config.visualization.save_attention_maps:
        for name, model in models.items():
            generate_visualizations(
                model, name, test_loader, config, device
            )
    
    # Quantization comparison
    results['quantization'] = run_quantization_comparison(
        models, test_loader, config
    )
    
    # Plot training curves
    plot_training_curves(results, config)
    
    # Save results
    save_results(results, config)
    
    # Print final summary
    print("\n" + "=" * 70)
    print(" Final Results Summary")
    print("=" * 70)
    
    print("\nTest Set Performance (Quadratic Weighted Kappa):")
    for name in models.keys():
        qwk = results['evaluation'][name]['metrics']['quadratic_weighted_kappa']
        acc = results['evaluation'][name]['metrics']['accuracy']
        print(f"  {name}: QWK={qwk:.4f}, Accuracy={acc:.4f}")
    
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70 + "\n")
    
    return results


if __name__ == '__main__':
    main()
