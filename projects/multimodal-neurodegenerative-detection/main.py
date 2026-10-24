#!/usr/bin/env python3
"""
Multimodal Foundation Models for Early Detection of Neurodegenerative Diseases
from Imaging, Genomic, and Clinical Data

Main entry point for training, evaluation, and benchmarking.
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
from src.data_loader import create_dataloaders
from src.model import (
    MultimodalClassifier, ImageOnlyClassifier,
    ClinicalOnlyClassifier, ConcatFusionClassifier
)
from src.trainer import Trainer
from src.evaluation import compute_metrics, BenchmarkEvaluator
from src.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_method_comparison, plot_modality_importance, plot_quantization_comparison
)
from src.quantization import ModelQuantizer
from src.param_counter import ParameterCounter, compare_models, generate_comparison_table, save_param_analysis
from src.resource_monitor import HardwareInfo, ResourceMonitor, setup_cpu_optimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_and_evaluate(model, dataloaders, config, device="cpu"):
    """Train and evaluate a model."""
    trainer = Trainer(model, dataloaders["train"], dataloaders["validation"], config, device)
    training_results = trainer.train()
    
    test_metrics = trainer.evaluate(dataloaders["test"])
    full_metrics = compute_metrics(test_metrics["labels"], test_metrics["predictions"])
    
    return {
        "training": training_results,
        "test_metrics": full_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal Neurodegenerative Detection")
    parser.add_argument("--debug", action="store_true", help="Use debug config")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    setup_cpu_optimization(args.num_threads)
    hardware_info = HardwareInfo.collect()
    logger.info(f"Hardware: {hardware_info.cpu_cores_physical} cores, {hardware_info.ram_total_gb:.1f}GB RAM")
    
    config = ExperimentConfig.small_debug() if args.debug else ExperimentConfig.default()
    config.save(output_dir / "config.json")
    
    all_results = {"hardware": hardware_info.__dict__, "config": config.to_dict()}
    
    logger.info("=" * 60)
    logger.info("Multimodal Neurodegenerative Disease Detection")
    logger.info("=" * 60)
    
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(config)
    
    logger.info("Initializing models...")
    models = {
        "Image Only": ImageOnlyClassifier(config),
        "Clinical Only": ClinicalOnlyClassifier(config),
        "Concat Fusion": ConcatFusionClassifier(config),
        "Attention Fusion (Ours)": MultimodalClassifier(config)
    }
    
    logger.info("Analyzing model parameters...")
    counter = ParameterCounter()
    param_analyses = {name: counter.analyze_multimodal_model(model, name) for name, model in models.items()}
    save_param_analysis(param_analyses, output_dir / "parameter_analysis.json")
    logger.info("\n" + generate_comparison_table(param_analyses))
    all_results["parameter_analysis"] = {n: a.to_dict() for n, a in param_analyses.items()}
    
    logger.info("Training and evaluating models...")
    model_results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        monitor = ResourceMonitor()
        monitor.start()
        
        results = train_and_evaluate(model, dataloaders, config)
        
        resources = monitor.stop()
        results["resources"] = resources.to_dict()
        model_results[name] = results
        
        logger.info(f"{name} - Test Accuracy: {results['test_metrics']['accuracy']:.4f}, "
                   f"Time: {resources.wall_time_seconds:.1f}s")
    
    all_results["model_results"] = model_results
    
    best_model_name = "Attention Fusion (Ours)"
    best_results = model_results[best_model_name]
    
    plot_training_curves(
        best_results["training"]["train_losses"],
        best_results["training"]["val_losses"],
        best_results["training"]["val_accuracies"],
        save_path=figures_dir / "training_curves.png"
    )
    
    cm = np.array(best_results["test_metrics"]["confusion_matrix"])
    plot_confusion_matrix(cm, save_path=figures_dir / "confusion_matrix.png")
    
    our_results = {name: res["test_metrics"]["accuracy"] for name, res in model_results.items()}
    published = {"Attention (Sharma 2021)": 0.87, "ResNet+Clinical (Qiu 2020)": 0.82}
    plot_method_comparison(our_results, "Accuracy", published, save_path=figures_dir / "method_comparison.png")
    
    logger.info("Running quantization comparison...")
    best_model = models[best_model_name]
    quantizer = ModelQuantizer(best_model)
    
    def make_input():
        return {
            "image": torch.randn(1, 1, config.data.image_size, config.data.image_size),
            "clinical": torch.randn(1, config.clinical_encoder.input_dim)
        }
    
    def evaluate_accuracy(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloaders["test"]:
                logits, _ = model(batch["image"], batch["clinical"])
                preds = logits.argmax(dim=1)
                correct += (preds == batch["label"]).sum().item()
                total += batch["label"].size(0)
        return correct / total if total > 0 else 0
    
    quant_comparison = quantizer.full_comparison(make_input, evaluate_accuracy, num_latency_runs=30)
    all_results["quantization"] = quant_comparison
    plot_quantization_comparison(quant_comparison, save_path=figures_dir / "quantization_comparison.png")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nTest Accuracy by Method:")
    for name, res in model_results.items():
        print(f"  {name}: {res['test_metrics']['accuracy']:.4f}")
    
    print(f"\nQuantization (Attention Fusion):")
    print(f"  Size reduction: {quant_comparison['size']['size_reduction_pct']:.1f}%")
    print(f"  Speedup: {quant_comparison['latency']['speedup']:.2f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
