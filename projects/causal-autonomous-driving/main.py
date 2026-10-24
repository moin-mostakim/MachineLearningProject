#!/usr/bin/env python3
"""
Causal Representation Learning for Robust Decision-Making
in Autonomous Driving under Distribution Shift
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')

from src.config import ExperimentConfig
from src.data_loader import create_multi_environment_loaders
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
    
    all_results = {"hardware": hardware.__dict__, "config": config.to_dict()}
    
    logger.info("=" * 60)
    logger.info("Causal Representation Learning for Autonomous Driving")
    logger.info("=" * 60)
    
    logger.info("Creating multi-environment dataloaders...")
    loaders = create_multi_environment_loaders(config)
    
    models = {
        "ERM": ERMClassifier(config.model.feature_dim, config.model.num_classes),
        "IRM": IRMClassifier(config.model.feature_dim, config.model.num_classes),
        "Augmentation": DataAugmentationBaseline(config.model.feature_dim, config.model.num_classes)
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
        eval_results[name] = evaluate_model(model, loaders["test"])
        
        logger.info(f"{name} - Test Acc: {eval_results[name]['overall_accuracy']:.4f}, "
                   f"Robustness: {eval_results[name]['robustness_score']:.4f}, "
                   f"Time: {resources.wall_time_seconds:.1f}s")
        
        all_results[f"{name}_training"] = train_result
        all_results[f"{name}_eval"] = eval_results[name]
        all_results[f"{name}_resources"] = resources.to_dict()
    
    plot_training_curves(training_results, save_path=figures_dir / "training_curves.png")
    plot_environment_comparison(eval_results, save_path=figures_dir / "environment_comparison.png")
    
    our_results = {n: r["overall_accuracy"] for n, r in eval_results.items()}
    published = {"DANN": 0.72, "IRM (Arjovsky)": 0.75}
    plot_method_comparison(our_results, published, save_path=figures_dir / "method_comparison.png")
    
    logger.info("\nRunning quantization comparison...")
    quantizer = ModelQuantizer(models["IRM"])
    
    def make_input():
        return torch.randn(1, 3, config.data.image_size, config.data.image_size)
    
    def eval_acc(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for loader in loaders["test"]:
                for batch in loader:
                    logits = model(batch["image"])
                    correct += (logits.argmax(1) == batch["label"]).sum().item()
                    total += batch["label"].size(0)
        return correct / total if total > 0 else 0
    
    quant_comparison = quantizer.full_comparison(make_input, eval_acc)
    all_results["quantization"] = quant_comparison
    plot_quantization_comparison(quant_comparison, save_path=figures_dir / "quantization_comparison.png")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)
    
    print("\nSUMMARY")
    print("=" * 40)
    for name, res in eval_results.items():
        print(f"{name}: Acc={res['overall_accuracy']:.4f}, Robustness={res['robustness_score']:.4f}")
    print(f"\nQuantization speedup: {quant_comparison['latency']['speedup']:.2f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
