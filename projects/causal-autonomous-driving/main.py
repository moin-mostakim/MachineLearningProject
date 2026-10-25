#!/usr/bin/env python3
"""
Causal Representation Learning for Robust Decision-Making
in Autonomous Driving under Distribution Shift

REGRESSION VERSION: Predicts continuous steering angles
Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)
Dataset: Udacity Self-Driving Car Challenge (glenn2/udacity_dataset)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.config import ExperimentConfig
from src.data_loader import create_multi_environment_loaders, DATASET_INFO
from src.model import ERMClassifier, IRMClassifier, DataAugmentationBaseline
from src.trainer import ERMTrainer, IRMTrainer
from src.evaluation import evaluate_model, LiteratureComparison
from src.visualization import (plot_training_curves, plot_environment_comparison,
                               plot_method_comparison, plot_quantization_comparison)
from src.quantization import ModelQuantizer
from src.param_counter import compare_models, generate_comparison_table, save_param_analysis
from src.resource_monitor import HardwareInfo, ResourceMonitor, setup_cpu_optimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    setup_cpu_optimization()
    hardware = HardwareInfo.collect()
    logger.info(f"Hardware: {hardware.cpu_cores} cores, {hardware.ram_gb:.1f}GB RAM")
    
    config = ExperimentConfig.small_debug() if args.debug else ExperimentConfig.default()
    config.save(output_dir / "config.json")
    
    all_results = {
        "hardware": hardware.__dict__, 
        "config": config.to_dict(),
        "dataset": DATASET_INFO,
        "task": "steering_angle_regression",
        "metrics": ["MAE", "RMSE"]
    }
    
    logger.info("=" * 60)
    logger.info("Causal Representation Learning for Autonomous Driving")
    logger.info("STEERING ANGLE REGRESSION (MAE/RMSE)")
    logger.info(f"Dataset: {DATASET_INFO['name']}")
    logger.info(f"URL: {DATASET_INFO['url']}")
    logger.info("=" * 60)
    
    logger.info("Creating multi-environment dataloaders...")
    loaders = create_multi_environment_loaders(config)
    
    regression = getattr(config.model, 'regression', True)
    models = {
        "ERM": ERMClassifier(config.model.feature_dim, config.model.num_classes, regression=regression),
        "IRM": IRMClassifier(config.model.feature_dim, config.model.num_classes, regression=regression),
        "Augmentation": DataAugmentationBaseline(config.model.feature_dim, config.model.num_classes, regression=regression)
    }
    
    logger.info("Analyzing parameters...")
    param_analyses = compare_models(models)
    save_param_analysis(param_analyses, output_dir / "parameter_analysis.json")
    logger.info("\n" + generate_comparison_table(param_analyses))
    all_results["parameters"] = {n: a.to_dict() for n, a in param_analyses.items()}
    
    training_results = {}
    eval_results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        monitor = ResourceMonitor()
        monitor.start()
        
        if name == "IRM":
            trainer = IRMTrainer(model, loaders["train"], loaders["validation"], config)
        else:
            trainer = ERMTrainer(model, loaders["train"], loaders["validation"], config)
        
        train_result = trainer.train()
        resources = monitor.stop()
        
        training_results[name] = train_result
        eval_results[name] = evaluate_model(model, loaders["test"], regression=regression)
        
        logger.info(f"{name} - Test MAE: {eval_results[name]['overall_mae']:.4f}, "
                   f"RMSE: {eval_results[name]['overall_rmse']:.4f}, "
                   f"Robustness: {eval_results[name]['robustness_score']:.4f}, "
                   f"Time: {resources.wall_time_seconds:.1f}s")
        
        all_results[f"{name}_training"] = train_result
        all_results[f"{name}_eval"] = eval_results[name]
        all_results[f"{name}_resources"] = resources.to_dict()
    
    try:
        plot_training_curves(training_results, save_path=figures_dir / "training_curves.png")
        plot_environment_comparison(eval_results, save_path=figures_dir / "environment_comparison.png")
    except Exception as e:
        logger.warning(f"Could not generate some plots: {e}")
    
    logger.info("\nRunning quantization comparison...")
    quantizer = ModelQuantizer(models["IRM"])
    
    def make_input():
        return torch.randn(1, 3, config.data.image_size, config.data.image_size)
    
    def eval_mae(model):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in loaders["test"]:
                for batch in loader:
                    preds = model(batch["image"])
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch["steering_angle"].numpy())
        return 1 - np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    quant_comparison = quantizer.full_comparison(make_input, eval_mae)
    all_results["quantization"] = quant_comparison
    try:
        plot_quantization_comparison(quant_comparison, save_path=figures_dir / "quantization_comparison.png")
    except Exception as e:
        logger.warning(f"Could not plot quantization: {e}")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)
    
    print("\n" + "=" * 70)
    print("STEERING ANGLE REGRESSION RESULTS")
    print("=" * 70)
    print(f"Dataset: {DATASET_INFO['name']}")
    print(f"URL: {DATASET_INFO['url']}")
    print("-" * 70)
    print(f"{'Method':<15} {'MAE':>10} {'RMSE':>10} {'Robustness':>12} {'Per-Env MAE'}")
    print("-" * 70)
    
    for name, res in eval_results.items():
        per_env = ", ".join([f"{e['env_name']}:{e['mae']:.3f}" for e in res['per_environment']])
        print(f"{name:<15} {res['overall_mae']:>10.4f} {res['overall_rmse']:>10.4f} "
              f"{res['robustness_score']:>12.4f} {per_env}")
    
    print("-" * 70)
    print(f"\nQuantization speedup: {quant_comparison['latency']['speedup']:.2f}x")
    print(f"Results saved to: {output_dir / 'results.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
