"""
Main experiment runner for Adversarial Robustness of Vision-Language Models.

This script runs experiments comparing:
1. Standard training (baseline)
2. Adversarial training (PGD-based, Madry et al. 2018)
3. Certified training (Randomized smoothing, Cohen et al. 2019)
4. Distributionally robust optimization (Sinha et al. 2018)

Usage:
    python main.py --debug  # Run in debug mode with reduced epochs/batches
    python main.py          # Run full experiment
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, REFERENCES
from models.vlm import create_vlm, get_parameter_table, count_parameters
from data.datasets import get_dataloaders
from trainers import StandardTrainer, AdversarialTrainer, CertifiedTrainer, DROTrainer
from attacks.pgd import PGDAttack, evaluate_robustness
from attacks.certified import RandomizedSmoothing, evaluate_certified_accuracy
from utils.metrics import (
    compute_metrics, compute_adversarial_metrics, 
    print_metrics_table, print_comparison_table,
    generate_results_summary
)
from utils.monitoring import ResourceMonitor, print_system_info
from utils.quantization import compare_quantization, print_quantization_comparison


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_standard(config, train_loader, test_loader, device):
    """Train standard baseline model."""
    print("\n" + "=" * 60)
    print(" Training: Standard Baseline")
    print("=" * 60)
    
    model = create_vlm(config)
    trainer = StandardTrainer(model, train_loader, test_loader, config, device)
    
    with ResourceMonitor() as monitor:
        results = trainer.train()
    
    resource_stats = monitor.get_stats()
    
    return model, results, resource_stats


def train_adversarial(config, train_loader, test_loader, device):
    """Train with adversarial training (PGD)."""
    print("\n" + "=" * 60)
    print(" Training: Adversarial (PGD-based)")
    print("=" * 60)
    
    model = create_vlm(config)
    trainer = AdversarialTrainer(model, train_loader, test_loader, config, device)
    
    with ResourceMonitor() as monitor:
        results = trainer.train()
    
    resource_stats = monitor.get_stats()
    
    return model, results, resource_stats


def train_certified(config, train_loader, test_loader, device):
    """Train with certified defense (randomized smoothing)."""
    print("\n" + "=" * 60)
    print(" Training: Certified (Randomized Smoothing)")
    print("=" * 60)
    
    # Use smooth version of model
    config.training.mode = "certified"
    model = create_vlm(config)
    trainer = CertifiedTrainer(model, train_loader, test_loader, config, device)
    
    with ResourceMonitor() as monitor:
        results = trainer.train()
    
    resource_stats = monitor.get_stats()
    
    return model, results, resource_stats


def train_dro(config, train_loader, test_loader, device):
    """Train with distributionally robust optimization."""
    print("\n" + "=" * 60)
    print(" Training: Distributionally Robust Optimization")
    print("=" * 60)
    
    model = create_vlm(config)
    trainer = DROTrainer(model, train_loader, test_loader, config, device)
    
    with ResourceMonitor() as monitor:
        results = trainer.train()
    
    resource_stats = monitor.get_stats()
    
    return model, results, resource_stats


def evaluate_all_models(models, config, test_loader, device):
    """Evaluate all models on clean and adversarial accuracy."""
    print("\n" + "=" * 60)
    print(" Evaluating Models")
    print("=" * 60)
    
    results = {}
    
    # Create attacker for evaluation
    dummy_model = list(models.values())[0]
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.eval()
        
        # Create attacker for this model
        attacker = PGDAttack(
            model=model,
            epsilon=config.attack.epsilon,
            alpha=config.attack.alpha,
            num_steps=config.attack.num_steps * 2,  # Stronger attack for evaluation
            random_start=True
        )
        
        # Compute metrics
        max_batches = config.training.debug_batches if config.training.debug else None
        metrics = compute_adversarial_metrics(
            model, test_loader, attacker, device, max_batches=max_batches
        )
        
        results[name] = metrics
        print(f"  Clean Accuracy: {metrics['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {metrics['adversarial_accuracy']:.4f}")
    
    return results


def evaluate_certified_models(models, config, test_loader, device):
    """Evaluate certified accuracy for models trained with smoothing."""
    print("\n" + "=" * 60)
    print(" Evaluating Certified Accuracy")
    print("=" * 60)
    
    results = {}
    
    for name, model in models.items():
        if "certified" not in name.lower():
            continue
        
        print(f"\nCertifying {name}...")
        
        # Evaluate certified accuracy at different radii
        radii = [0.0, 0.25, 0.5]
        max_samples = 100 if config.training.debug else 500
        
        certified_results = evaluate_certified_accuracy(
            model=model,
            dataloader=test_loader,
            sigma=config.certified.sigma,
            radii=radii,
            n_samples=config.certified.n_samples_certify,
            num_classes=10,
            device=device,
            max_samples=max_samples
        )
        
        results[name] = certified_results
        
        for radius in radii:
            acc = certified_results.get(f"certified_acc_r{radius}", 0)
            print(f"  Certified Accuracy (r={radius}): {acc:.4f}")
    
    return results


def run_quantization_comparison(model, test_loader, config):
    """Run INT8 quantization comparison."""
    print("\n" + "=" * 60)
    print(" INT8 Quantization Comparison")
    print("=" * 60)
    
    max_batches = config.training.debug_batches if config.training.debug else 50
    
    results = compare_quantization(
        model=model,
        test_loader=test_loader,
        max_batches=max_batches
    )
    
    print_quantization_comparison(results)
    
    return results


def print_literature_comparison():
    """Print comparison with literature results."""
    print("\n" + "=" * 60)
    print(" Literature Comparison (CIFAR-10, ε=8/255)")
    print("=" * 60)
    
    literature = {
        "Madry et al. (2018)": {"clean": 0.873, "robust": 0.458, "certified": "-"},
        "TRADES (Zhang 2019)": {"clean": 0.849, "robust": 0.566, "certified": "-"},
        "Cohen et al. (2019)": {"clean": 0.750, "robust": "-", "certified": "0.60@r=0.5"},
        "Wong et al. (2020)": {"clean": 0.833, "robust": 0.432, "certified": "-"},
    }
    
    print(f"\n{'Method':<25} {'Clean Acc':>12} {'PGD Acc':>12} {'Certified':>15}")
    print("-" * 65)
    
    for method, results in literature.items():
        clean = f"{results['clean']:.1%}" if isinstance(results['clean'], float) else results['clean']
        robust = f"{results['robust']:.1%}" if isinstance(results['robust'], float) else results['robust']
        print(f"{method:<25} {clean:>12} {robust:>12} {results['certified']:>15}")
    
    print("-" * 65)
    print("\nReferences:")
    print("- Madry et al. (2018): 'Towards Deep Learning Models Resistant to Adversarial Attacks'")
    print("- Zhang et al. (2019): 'Theoretically Principled Trade-off (TRADES)'")
    print("- Cohen et al. (2019): 'Certified Adversarial Robustness via Randomized Smoothing'")
    print("- Wong et al. (2020): 'Fast is Better than Free'")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Adversarial VLM Robustness Experiments")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--methods", nargs="+", 
                        default=["standard", "adversarial", "certified", "dro"],
                        help="Training methods to run")
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(debug=args.debug)
    config.seed = args.seed
    
    # Set seeds
    set_seed(config.seed)
    
    # Print setup info
    print("\n" + "=" * 60)
    print(" Adversarial Robustness of Vision-Language Models")
    print("=" * 60)
    print(f"Dataset: {config.data.dataset}")
    print(f"Debug mode: {config.training.debug}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Epsilon: {config.attack.epsilon:.4f} ({config.attack.epsilon * 255:.1f}/255)")
    print(f"Methods: {args.methods}")
    
    # Print system info
    print_system_info()
    
    # Create model and print parameter table
    sample_model = create_vlm(config)
    print(get_parameter_table(sample_model))
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Store trained models
    models = {}
    training_results = {}
    resource_stats = {}
    
    # Train each method
    start_time = time.time()
    
    if "standard" in args.methods:
        model, results, stats = train_standard(config, train_loader, test_loader, config.device)
        models["Standard"] = model
        training_results["Standard"] = results
        resource_stats["Standard"] = stats
    
    if "adversarial" in args.methods:
        model, results, stats = train_adversarial(config, train_loader, test_loader, config.device)
        models["Adversarial"] = model
        training_results["Adversarial"] = results
        resource_stats["Adversarial"] = stats
    
    if "certified" in args.methods:
        model, results, stats = train_certified(config, train_loader, test_loader, config.device)
        models["Certified"] = model
        training_results["Certified"] = results
        resource_stats["Certified"] = stats
    
    if "dro" in args.methods:
        model, results, stats = train_dro(config, train_loader, test_loader, config.device)
        models["DRO"] = model
        training_results["DRO"] = results
        resource_stats["DRO"] = stats
    
    total_training_time = time.time() - start_time
    
    # Evaluate all models
    evaluation_results = evaluate_all_models(models, config, test_loader, config.device)
    
    # Evaluate certified accuracy for certified models
    certified_results = evaluate_certified_models(models, config, test_loader, config.device)
    
    # Print comparison table
    print_comparison_table(evaluation_results, "Method Comparison on CIFAR-10")
    
    # Run quantization comparison on best model
    if models:
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]["adversarial_accuracy"])
        print(f"\nRunning quantization comparison on {best_model_name} model...")
        quant_results = run_quantization_comparison(models[best_model_name], test_loader, config)
    
    # Print literature comparison
    print_literature_comparison()
    
    # Print resource usage summary
    print("\n" + "=" * 60)
    print(" Resource Usage Summary")
    print("=" * 60)
    for method, stats in resource_stats.items():
        print(f"\n{method}:")
        print(f"  Duration: {stats.duration_seconds:.1f}s")
        print(f"  CPU: {stats.cpu_mean:.1f}% mean, {stats.cpu_max:.1f}% max")
        print(f"  Memory: {stats.memory_mean_mb:.1f} MB mean, {stats.memory_max_mb:.1f} MB max")
        if stats.gpu_memory_mean_mb:
            print(f"  GPU Memory: {stats.gpu_memory_mean_mb:.1f} MB mean")
    
    # Final summary
    print("\n" + "=" * 60)
    print(" Final Summary")
    print("=" * 60)
    print(f"Total training time: {total_training_time:.1f}s")
    print(f"Models trained: {list(models.keys())}")
    
    for name, results in evaluation_results.items():
        print(f"\n{name}:")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"  Adversarial Accuracy (PGD): {results['adversarial_accuracy']:.4f}")
    
    if certified_results:
        for name, results in certified_results.items():
            print(f"\n{name} (Certified):")
            for key, value in results.items():
                print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print(" Experiment Complete!")
    print("=" * 60)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    
    # Prepare results for JSON serialization
    json_results = {
        "config": {
            "dataset": config.data.dataset,
            "debug": config.training.debug,
            "epochs": config.training.epochs,
            "epsilon": config.attack.epsilon,
            "sigma": config.certified.sigma
        },
        "evaluation": evaluation_results,
        "certified": certified_results,
        "training_time": total_training_time
    }
    
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return evaluation_results


if __name__ == "__main__":
    main()
