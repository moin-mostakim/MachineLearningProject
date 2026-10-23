#!/usr/bin/env python3
"""
Generative AI for Inverse Design of Functional Materials
with Multi-Objective Optimization

Main entry point for training and evaluation.

Literature References:
---------------------
1. Junction Tree VAE (Jin et al., ICML 2018)
   - Hierarchical molecular generation with 100% validity
   - Uses tree decomposition and message passing

2. GraphVAE (Simonovsky & Komodakis, 2018)
   - Direct graph generation approach
   - Probabilistic graph generation with node/edge prediction

3. MolGAN (De Cao & Kipf, 2018)
   - Adversarial training for molecular generation
   - Reinforcement learning for property optimization

4. SELFIES (Krenn et al., 2020)
   - Self-referencing embedded strings
   - Guarantees 100% valid molecules

5. Multi-objective Optimization (Deb et al., 2002)
   - NSGA-II for Pareto optimization
   - Non-dominated sorting and crowding distance

Dataset:
--------
QM9 (Ramakrishnan et al., 2014)
- ~134,000 small organic molecules
- Up to 9 heavy atoms (C, N, O, F)
- 12 computed quantum mechanical properties:
  * mu: Dipole moment
  * alpha: Polarizability
  * HOMO/LUMO: Frontier orbital energies
  * Gap: HOMO-LUMO gap
  * And more...

Usage:
------
    # Debug mode (small dataset, few epochs)
    python main.py --debug
    
    # Full training
    python main.py
    
    # Evaluation only
    python main.py --eval-only --checkpoint best_model.pt
"""

import argparse
import os
import sys
import json
import torch
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, estimate_parameters
from data import get_dataloaders, get_property_stats
from models import build_model, build_single_objective_model
from training import Trainer, build_loss
from optimization import compare_baselines, ParetoOptimizer
from evaluation import evaluate_model, full_quantization_comparison, print_quantization_table
from utils import (
    print_system_info, 
    ResourceTracker, 
    create_results_report,
    plot_training_curves
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header():
    """Print project header."""
    header = """
╔══════════════════════════════════════════════════════════════════════╗
║     Generative AI for Inverse Design of Functional Materials        ║
║              with Multi-Objective Optimization                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Dataset: QM9 (~134k molecules)                                      ║
║  Method: VAE + Multi-Objective Optimization                          ║
║  Objectives: HOMO, LUMO, Gap optimization                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(header)


def print_parameter_table(config):
    """Print parameter count table."""
    params = estimate_parameters(config)
    
    print("\n" + "=" * 70)
    print("PARAMETER COUNT TABLE")
    print("=" * 70)
    print(f"{'Component':<30} {'Parameters':>15} {'Size (MB)':>12} {'%':>8}")
    print("-" * 70)
    
    total = params['total']
    for name, count in params.items():
        if name not in ['total', 'total_millions']:
            mb = count * 4 / 1e6  # 4 bytes per FP32 param
            pct = count / total * 100
            print(f"{name:<30} {count:>15,} {mb:>12.2f} {pct:>8.1f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<30} {total:>15,} {total * 4 / 1e6:>12.2f} {'100.0':>8}")
    print(f"Total: {params['total_millions']:.2f}M parameters")
    print("=" * 70)


def print_literature_comparison():
    """Print comparison with literature methods."""
    print("\n" + "=" * 70)
    print("LITERATURE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Validity':>12} {'Uniqueness':>12} {'Novelty':>12}")
    print("-" * 70)
    
    # Literature results (from published papers)
    methods = [
        ("Junction Tree VAE*", "100.0%", "N/A", "100.0%"),
        ("GraphVAE*", "55.7%", "76.0%", "61.6%"),
        ("MolGAN*", "98.1%", "10.4%", "94.2%"),
        ("CGVAE*", "100.0%", "99.6%", "100.0%"),
        ("Our VAE (target)", ">90%", ">80%", ">90%"),
    ]
    
    for method, validity, unique, novel in methods:
        print(f"{method:<25} {validity:>12} {unique:>12} {novel:>12}")
    
    print("-" * 70)
    print("* Results from published papers on QM9 dataset")
    print("  - Junction Tree VAE: Jin et al., ICML 2018")
    print("  - GraphVAE: Simonovsky & Komodakis, 2018")
    print("  - MolGAN: De Cao & Kipf, 2018")
    print("  - CGVAE: Liu et al., NeurIPS 2018")
    print("=" * 70)


def train(config, args):
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config.training.checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({
            'model': vars(config.model),
            'training': vars(config.training),
            'optimization': vars(config.optimization),
            'data': vars(config.data),
        }, f, indent=2, default=str)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    logger.info("Loading QM9 dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    logger.info(f"Test: {len(test_loader.dataset)} samples")
    
    # Build model
    logger.info("Building VAE model...")
    model = build_model(config)
    
    # Build loss
    criterion = build_loss(config, "multi_objective")
    
    # Resource tracker
    resource_tracker = ResourceTracker(
        log_interval=config.resources.log_interval,
        log_gpu=config.resources.log_gpu,
        log_memory=config.resources.log_memory,
        save_path=os.path.join(output_dir, "resources.json")
    )
    
    # Train
    logger.info("Starting training...")
    trainer = Trainer(model, criterion, config, config.device)
    
    num_epochs = 5 if args.debug else config.training.num_epochs
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Save training history plot
    plot_training_curves(
        [vars(m) for m in history],
        os.path.join(output_dir, "training_curves.png")
    )
    
    # Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    eval_results = evaluate_model(model, test_loader, config, config.device)
    
    print("\nGeneration Metrics:")
    for key, value in eval_results['generation'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nProperty Prediction (MAE):")
    if 'overall' in eval_results['property_prediction']:
        overall = eval_results['property_prediction']['overall']
        print(f"  Overall MAE: {overall['mae']:.4f}")
        print(f"  Overall RMSE: {overall['rmse']:.4f}")
        print(f"  Overall R²: {overall['r2']:.4f}")
    
    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        # Convert tensors to lists for JSON serialization
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(eval_results), f, indent=2)
    
    return model, history, eval_results, output_dir


def run_optimization(model, config, output_dir):
    """Run multi-objective optimization experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-OBJECTIVE OPTIMIZATION")
    logger.info("=" * 60)
    
    # Reduce generations for debug mode
    if config.training.debug_mode:
        config.optimization.num_generations = 10
        config.optimization.population_size = 20
        config.optimization.random_search_samples = 100
    
    # Compare baselines
    results = compare_baselines(model, config, config.device)
    
    # Print results
    print("\nRandom Search Baseline:")
    random_props = results['random_search']['target_properties']
    print(f"  Mean HOMO: {random_props[:, 0].mean():.4f}")
    print(f"  Mean LUMO: {random_props[:, 1].mean():.4f}")
    print(f"  Mean Gap: {random_props[:, 2].mean():.4f}")
    
    print("\nMulti-Objective Optimization (Pareto):")
    if 'pareto_objectives' in results['multi_objective']:
        pareto_obj = results['multi_objective']['pareto_objectives']
        print(f"  Pareto front size: {len(pareto_obj)}")
        print(f"  Mean HOMO: {pareto_obj[:, 0].mean():.4f}")
        print(f"  Mean LUMO: {pareto_obj[:, 1].mean():.4f}")
        print(f"  Mean Gap: {pareto_obj[:, 2].mean():.4f}")
    
    print("\nSingle-Objective Optimization (HOMO):")
    single_props = results['single_objective_homo']['target_properties']
    print(f"  Mean HOMO: {single_props[:, 0].mean():.4f}")
    print(f"  Mean LUMO: {single_props[:, 1].mean():.4f}")
    print(f"  Mean Gap: {single_props[:, 2].mean():.4f}")
    
    # Save results
    opt_results = {
        'random_search': {
            'mean_objectives': random_props.mean(dim=0).tolist()
        },
        'multi_objective': {
            'pareto_front_size': len(results['multi_objective'].get('pareto_objectives', [])),
            'mean_objectives': results['multi_objective'].get('pareto_objectives', torch.zeros(1, 3)).mean(dim=0).tolist()
        },
        'single_objective': {
            'mean_objectives': single_props.mean(dim=0).tolist()
        }
    }
    
    with open(os.path.join(output_dir, "optimization_results.json"), 'w') as f:
        json.dump(opt_results, f, indent=2)
    
    return results


def run_quantization(model, test_loader, config, output_dir):
    """Run INT8 quantization comparison."""
    logger.info("\n" + "=" * 60)
    logger.info("INT8 QUANTIZATION COMPARISON")
    logger.info("=" * 60)
    
    quant_results = full_quantization_comparison(model, test_loader, config, "cpu")
    
    # Print table
    print(print_quantization_table(quant_results))
    
    # Save results
    with open(os.path.join(output_dir, "quantization_results.json"), 'w') as f:
        json.dump(quant_results, f, indent=2, default=str)
    
    return quant_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Molecular VAE Training")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--no-optimization', action='store_true', help='Skip optimization')
    parser.add_argument('--no-quantization', action='store_true', help='Skip quantization')
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Get config
    config = get_config(debug=args.debug)
    
    # Print system info
    print_system_info()
    
    # Print parameter table
    print_parameter_table(config)
    
    # Print literature comparison
    print_literature_comparison()
    
    if args.eval_only:
        # Load model and evaluate
        if not args.checkpoint:
            logger.error("--checkpoint required for --eval-only")
            return
        
        model = build_model(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, _, test_loader = get_dataloaders(config)
        eval_results = evaluate_model(model, test_loader, config, config.device)
        print(json.dumps(eval_results, indent=2, default=str))
    else:
        # Full training
        model, history, eval_results, output_dir = train(config, args)
        
        # Load test data for additional experiments
        _, _, test_loader = get_dataloaders(config)
        
        # Multi-objective optimization
        if not args.no_optimization:
            opt_results = run_optimization(model, config, output_dir)
        
        # Quantization comparison
        if not args.no_quantization:
            quant_results = run_quantization(model, test_loader, config, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if not args.eval_only:
        print(f"Results saved to: {output_dir}")
    
    # Final summary
    print("\nFinal Metrics Summary:")
    print(f"  Validity: {eval_results['generation']['validity']:.4f}")
    print(f"  Uniqueness: {eval_results['generation']['uniqueness']:.4f}")
    print(f"  Novelty: {eval_results['generation']['novelty']:.4f}")
    if 'overall' in eval_results['property_prediction']:
        print(f"  Property MAE: {eval_results['property_prediction']['overall']['mae']:.4f}")
    
    print("\nDataset: QM9 (synthetic data used due to torch_geometric availability)")
    print("Method: Molecular VAE with Multi-Objective Optimization")
    print("=" * 60)


if __name__ == "__main__":
    main()
