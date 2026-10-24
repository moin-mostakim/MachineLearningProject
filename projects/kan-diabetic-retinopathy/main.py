"""
Main experiment runner for KAN Diabetic Retinopathy Detection.

Compares:
1. MLP baseline
2. CNN baseline (ResNet)
3. KAN classifier
4. Hybrid CNN + KAN

Metrics: Quadratic Weighted Kappa, Accuracy, AUC, F1
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config, LITERATURE_REFERENCES
from data.dataset import get_data_loaders, create_synthetic_dataset
from models import MLPClassifier, CNNClassifier, KANClassifier, HybridCNNKAN
from models.hybrid_model import HybridCNNMLP
from utils.training import Trainer, create_optimizer, create_scheduler, evaluate
from utils.metrics import print_metrics, calculate_metrics
from utils.quantization import compare_quantized
from utils.resource_monitor import ResourceMonitor, print_model_info, get_model_info
from utils.visualization import plot_results, plot_training_curves, plot_confusion_matrix


def create_model(model_name: str, config: Config, feature_dim: int = 512) -> nn.Module:
    """
    Create model by name.
    
    Args:
        model_name: mlp, cnn, kan, hybrid_kan, hybrid_mlp
        config: Configuration
        feature_dim: Input feature dimension for classifiers
    
    Returns:
        Model instance
    """
    num_classes = config.data.num_classes
    
    if model_name == "mlp":
        return MLPClassifier(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dims=config.mlp.hidden_dims,
            dropout=config.mlp.dropout,
            activation=config.mlp.activation,
        )
    
    elif model_name == "cnn":
        return CNNClassifier(
            num_classes=num_classes,
            backbone=config.cnn.backbone,
            pretrained=config.cnn.pretrained,
            freeze_backbone=config.cnn.freeze_backbone,
        )
    
    elif model_name == "kan":
        return KANClassifier(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dims=config.kan.hidden_dims,
            grid_size=config.kan.grid_size,
            spline_order=config.kan.spline_order,
            scale_noise=config.kan.scale_noise,
            scale_base=config.kan.scale_base,
            scale_spline=config.kan.scale_spline,
            base_activation=config.kan.base_activation,
        )
    
    elif model_name == "hybrid_kan":
        return HybridCNNKAN(
            num_classes=num_classes,
            backbone=config.hybrid.cnn_backbone,
            pretrained=config.cnn.pretrained,
            freeze_backbone=config.hybrid.freeze_cnn,
            kan_hidden_dims=config.hybrid.kan_hidden_dims,
            grid_size=config.kan.grid_size,
            spline_order=config.kan.spline_order,
        )
    
    elif model_name == "hybrid_mlp":
        return HybridCNNMLP(
            num_classes=num_classes,
            backbone=config.hybrid.cnn_backbone,
            pretrained=config.cnn.pretrained,
            freeze_backbone=config.hybrid.freeze_cnn,
            mlp_hidden_dims=[256, 128],
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(
    model_name: str,
    config: Config,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    device: torch.device,
    save_dir: str,
) -> dict:
    """
    Train and evaluate a model.
    
    Args:
        model_name: Model name
        config: Configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        class_weights: Class weights for loss
        device: Device
        save_dir: Directory to save results
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create model
    if model_name in ["mlp", "kan"]:
        # These models work on features extracted by a small CNN encoder
        # Use a simple conv encoder for fair comparison
        feature_dim = 512
        
        class SimpleEncoder(nn.Module):
            """Simple CNN encoder for feature extraction."""
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),  # 1x1
                    nn.Flatten(),
                    nn.Linear(128, feature_dim),
                    nn.LayerNorm(feature_dim),
                )
            def forward(self, x):
                return self.encoder(x)
        
        if model_name == "mlp":
            model = nn.Sequential(
                SimpleEncoder(),
                MLPClassifier(
                    in_features=feature_dim,
                    num_classes=config.data.num_classes,
                    hidden_dims=config.mlp.hidden_dims,
                    dropout=config.mlp.dropout,
                ),
            )
        else:  # kan
            model = nn.Sequential(
                SimpleEncoder(),
                KANClassifier(
                    in_features=feature_dim,
                    num_classes=config.data.num_classes,
                    hidden_dims=config.kan.hidden_dims,
                    grid_size=config.kan.grid_size,
                    spline_order=config.kan.spline_order,
                ),
            )
    else:
        model = create_model(model_name, config)
    
    model = model.to(device)
    print_model_info(model)
    
    # Loss function with class weights
    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_name=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config.training.scheduler,
        num_epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )
    
    # Training
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")
    max_batches = config.experiment.debug_batches if config.experiment.debug_mode else None
    
    train_results = trainer.train(
        num_epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        log_interval=config.experiment.log_interval,
        save_path=checkpoint_path,
        max_batches=max_batches,
    )
    
    # Load best model for evaluation
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, max_batches=max_batches)
    print_metrics(test_metrics)
    
    # Get model info
    model_info = get_model_info(model)
    
    # Compile results
    results = {
        "model_name": model_name,
        "params": model_info["total_params"],
        "trainable_params": model_info["trainable_params"],
        "qwk": test_metrics["qwk"],
        "accuracy": test_metrics["accuracy"],
        "auc_macro": test_metrics.get("auc_macro"),
        "f1_macro": test_metrics["f1_macro"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": train_results["history"],
        "best_epoch": train_results["best_epoch"],
        "best_val_qwk": train_results["best_val_qwk"],
        "resource_stats": train_results["resource_stats"],
    }
    
    return results


def compare_parameters(config: Config) -> dict:
    """
    Compare parameter counts across models.
    
    Returns:
        Dictionary with parameter comparison
    """
    print("\n" + "=" * 70)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 70)
    
    models = {
        "MLP Classifier": MLPClassifier(
            in_features=512,
            num_classes=5,
            hidden_dims=config.mlp.hidden_dims,
        ),
        "KAN Classifier": KANClassifier(
            in_features=512,
            num_classes=5,
            hidden_dims=config.kan.hidden_dims,
            grid_size=config.kan.grid_size,
            spline_order=config.kan.spline_order,
        ),
        "CNN (ResNet-18)": CNNClassifier(
            num_classes=5,
            backbone="resnet18",
            pretrained=False,
        ),
        "Hybrid CNN+KAN": HybridCNNKAN(
            num_classes=5,
            backbone="resnet18",
            pretrained=False,
            kan_hidden_dims=config.hybrid.kan_hidden_dims,
        ),
        "Hybrid CNN+MLP": HybridCNNMLP(
            num_classes=5,
            backbone="resnet18",
            pretrained=False,
            mlp_hidden_dims=[256, 128],
        ),
    }
    
    comparison = {}
    
    print(f"\n{'Model':<25} {'Total Params':>15} {'Classifier Params':>20} {'Efficiency':>12}")
    print("-" * 75)
    
    mlp_classifier_params = None
    
    for name, model in models.items():
        info = get_model_info(model)
        total = info["total_params"]
        
        # Get classifier params
        if hasattr(model, "count_parameters"):
            counts = model.count_parameters()
            if "classifier" in counts:
                classifier_params = counts["classifier"]
            elif "kan_classifier" in counts:
                classifier_params = counts["kan_classifier"]
            elif "mlp_classifier" in counts:
                classifier_params = counts["mlp_classifier"]
            else:
                classifier_params = total
        else:
            classifier_params = total
        
        if "MLP" in name and mlp_classifier_params is None:
            mlp_classifier_params = classifier_params
        
        efficiency = "-"
        if mlp_classifier_params and "KAN" in name and classifier_params > 0:
            efficiency = f"{mlp_classifier_params / classifier_params:.2f}x"
        
        comparison[name] = {
            "total_params": total,
            "classifier_params": classifier_params,
        }
        
        print(f"{name:<25} {total:>15,} {classifier_params:>20,} {efficiency:>12}")
    
    print("-" * 75)
    print("\nNote: KAN uses ~2-4x fewer parameters in classifier head compared to MLP")
    print("      with equivalent or better expressiveness (Liu et al., 2024)")
    
    return comparison


def run_quantization_comparison(
    model: nn.Module,
    test_loader,
    model_name: str,
    save_dir: str,
) -> dict:
    """Run INT8 quantization comparison."""
    print(f"\n{'='*60}")
    print(f"QUANTIZATION COMPARISON: {model_name}")
    print(f"{'='*60}")
    
    results = compare_quantized(
        model,
        test_loader,
        method="dynamic",
    )
    
    return results


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="KAN DR Detection Experiment")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights")
    parser.add_argument("--models", nargs="+", 
                        default=["mlp", "kan", "cnn", "hybrid_kan"],
                        help="Models to train")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    args = parser.parse_args()
    
    # Configuration
    config = get_config(debug=args.debug)
    if args.no_pretrained:
        config.cnn.pretrained = False
    
    # Setup
    device = torch.device(config.experiment.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("HYBRID FOUNDATION MODEL WITH KAN FOR DIABETIC RETINOPATHY DIAGNOSIS")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Device: {device}")
    print(f"Debug mode: {config.experiment.debug_mode}")
    print(f"Output directory: {save_dir}")
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Print literature references
    print(LITERATURE_REFERENCES)
    
    # Parameter comparison
    param_comparison = compare_parameters(config)
    
    # Create data loaders
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    num_samples = config.experiment.debug_samples if config.experiment.debug_mode else 1000
    
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        num_samples=num_samples,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        num_workers=0,  # Use 0 for debugging
        seed=config.training.seed,
    )
    
    print(f"Dataset: Synthetic DR-like data")
    print(f"Total samples: {num_samples}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {class_weights.tolist()}")
    
    # Train models
    all_results = {}
    
    for model_name in args.models:
        try:
            results = train_model(
                model_name=model_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                device=device,
                save_dir=save_dir,
            )
            all_results[model_name] = results
            
            # Save individual results
            result_path = os.path.join(save_dir, f"{model_name}_results.json")
            save_results = {k: v for k, v in results.items() if k != "history"}
            save_results["history"] = {
                k: [float(x) if x is not None else None for x in v] 
                for k, v in results["history"].items()
            }
            with open(result_path, "w") as f:
                json.dump(save_results, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
    
    # Quantization comparison for best model
    if "hybrid_kan" in all_results and "error" not in all_results["hybrid_kan"]:
        print("\n" + "=" * 60)
        print("INT8 QUANTIZATION ANALYSIS")
        print("=" * 60)
        
        # Load best hybrid model
        model = create_model("hybrid_kan", config)
        checkpoint_path = os.path.join(save_dir, "hybrid_kan_best.pt")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
        
        quant_results = run_quantization_comparison(
            model, test_loader, "Hybrid CNN+KAN", save_dir
        )
        all_results["quantization"] = quant_results
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        plot_results(all_results, output_dir=save_dir, show=False)
        print(f"Visualizations saved to {save_dir}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'Params':>12} {'QWK':>10} {'Accuracy':>10} {'AUC':>10}")
    print("-" * 65)
    
    for model_name, results in all_results.items():
        if model_name == "quantization" or "error" in results:
            continue
        
        params = results.get("params", 0)
        qwk = results.get("qwk", 0)
        acc = results.get("accuracy", 0)
        auc = results.get("auc_macro", 0) or 0
        
        print(f"{model_name:<20} {params:>12,} {qwk:>10.4f} {acc:>10.4f} {auc:>10.4f}")
    
    print("-" * 65)
    
    # Save final summary
    summary_path = os.path.join(save_dir, "summary.json")
    summary = {
        "timestamp": timestamp,
        "config": config.to_dict(),
        "param_comparison": param_comparison,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "history"} 
                   for k, v in all_results.items() if isinstance(v, dict)},
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"All results saved to {save_dir}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
