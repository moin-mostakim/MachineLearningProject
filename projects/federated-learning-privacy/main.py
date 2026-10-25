#!/usr/bin/env python3
"""
Privacy-Preserving Federated Learning with Heterogeneous Clients
via Personalized Aggregation and Differential Privacy
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
from src.data_loader import create_client_dataloaders
from src.model import create_model
from src.trainer import FederatedTrainer
from src.evaluation import LiteratureComparison
from src.visualization import (plot_convergence, plot_method_comparison, 
                               plot_quantization_comparison)
from src.quantization import ModelQuantizer
from src.param_counter import ParameterCounter, generate_comparison_table, save_param_analysis
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
    logger.info("Privacy-Preserving Federated Learning")
    logger.info("=" * 60)
    
    logger.info("Creating client dataloaders...")
    client_loaders, test_loader = create_client_dataloaders(config)
    logger.info(f"Created {len(client_loaders)} clients")
    
    # Analyze model
    model = create_model(config)
    counter = ParameterCounter()
    analysis = counter.analyze(model, "SimpleCNN")
    save_param_analysis({"SimpleCNN": analysis}, output_dir / "parameter_analysis.json")
    logger.info(f"Model: {analysis.total_params:,} params, {analysis.size_mb:.2f} MB")
    all_results["model_params"] = analysis.to_dict()
    
    # Train with different algorithms
    trainer = FederatedTrainer(config, client_loaders, test_loader)
    
    monitor = ResourceMonitor()
    monitor.start()
    
    results = {}
    
    logger.info("\n--- Training FedAvg ---")
    results["FedAvg"] = trainer.train_fedavg()
    
    logger.info("\n--- Training FedProx ---")
    results["FedProx"] = trainer.train_fedprox(mu=0.01)
    
    logger.info("\n--- Training PersonalizedFL+DP ---")
    results["PersonalizedFL+DP"] = trainer.train_personalized_dp()
    
    resources = monitor.stop()
    all_results["resources"] = resources.to_dict()
    
    # Save results
    for name, res in results.items():
        all_results[f"{name}_accuracies"] = res["accuracies"]
        all_results[f"{name}_final"] = res["final_accuracy"]
    
    # Plots
    plot_convergence(results, save_path=figures_dir / "convergence.png")
    
    our_results = {name: res["final_accuracy"] for name, res in results.items()}
    published = {"FedAvg (McMahan)": 0.77, "FedProx (Li)": 0.78}
    plot_method_comparison(our_results, published, save_path=figures_dir / "method_comparison.png")
    
    # Quantization
    logger.info("\nRunning quantization comparison...")
    best_model = results["FedAvg"]["model"]
    quantizer = ModelQuantizer(best_model)
    
    def make_input():
        return torch.randn(1, 3, 32, 32)
    
    def eval_acc(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += len(target)
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
    for name, res in results.items():
        print(f"{name}: {res['final_accuracy']:.4f}")
    print(f"\nDP Privacy Budget: ε={config.dp.epsilon}")
    print(f"Training time: {resources.wall_time_seconds:.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
