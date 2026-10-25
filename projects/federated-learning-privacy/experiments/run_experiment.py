"""Experiment runner for federated learning with differential privacy."""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import copy

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, get_debug_config, get_full_config
from models import get_model, CIFAR10CNN
from data import get_dataset, partition_data, FederatedDataset
from trainers.federated_trainer import FederatedTrainer
from aggregation import FedAvgAggregator, FedProxAggregator, PersonalizedAggregator
from utils import ResourceMonitor, count_parameters, print_parameter_table, print_quantization_comparison
from utils.resource_monitor import get_system_info
from experiments.benchmark import BenchmarkComparison


def run_experiment(
    config: Config,
    method: str = "fedavg",
    experiment_name: str = None
) -> Dict:
    """
    Run a single federated learning experiment.
    
    Args:
        config: Experiment configuration
        method: FL method ("fedavg", "fedprox", "personalized_dp")
        experiment_name: Name for logging
        
    Returns:
        Experiment results dictionary
    """
    if experiment_name is None:
        experiment_name = f"{method}_{config.data.dataset}"
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Method: {method}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Clients: {config.data.num_clients}")
    print(f"Rounds: {config.training.num_rounds}")
    print(f"DP Enabled: {config.dp.enabled}")
    if config.dp.enabled:
        print(f"Privacy Budget: ε={config.dp.epsilon}, δ={config.dp.delta}")
    print(f"{'='*60}\n")
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    device = config.experiment.device
    print(f"Using device: {device}")
    
    # Start resource monitoring
    monitor = ResourceMonitor(interval=2.0)
    monitor.start()
    
    try:
        # Load dataset
        print("Loading dataset...")
        monitor.start_phase("data_loading")
        
        train_dataset, test_dataset = get_dataset(
            dataset_name=config.data.dataset,
            data_dir=config.data.data_dir
        )
        
        # Partition data among clients
        print(f"Partitioning data (non-IID alpha={config.data.non_iid_alpha})...")
        client_indices = partition_data(
            train_dataset=train_dataset,
            num_clients=config.data.num_clients,
            method="dirichlet",
            alpha=config.data.non_iid_alpha,
            seed=config.training.seed
        )
        
        # Create federated dataset
        fed_dataset = FederatedDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            client_indices=client_indices,
            num_clients=config.data.num_clients
        )
        
        monitor.end_phase("data_loading")
        
        # Print data distribution
        data_sizes = fed_dataset.get_client_data_sizes()
        print(f"Data distribution: min={min(data_sizes.values())}, max={max(data_sizes.values())}, avg={np.mean(list(data_sizes.values())):.0f}")
        
        # Create model
        print("\nCreating model...")
        
        if config.data.dataset == "cifar10":
            model = CIFAR10CNN(num_classes=10, dropout=config.model.dropout)
        else:
            model = get_model(
                config.model.model_name,
                num_classes=config.model.num_classes
            )
        
        model = model.to(device)
        
        # Print parameter table
        total_params, total_size = print_parameter_table(model, config.model.model_name)
        
        # Configure method
        if method == "fedavg":
            config.dp.enabled = False
            config.personalization.enabled = False
        elif method == "fedprox":
            config.dp.enabled = False
            config.personalization.enabled = False
        elif method == "personalized_dp":
            config.dp.enabled = True
            config.personalization.enabled = True
        elif method == "dp_fedavg":
            config.dp.enabled = True
            config.personalization.enabled = False
        
        # Create trainer
        print("\nInitializing federated trainer...")
        monitor.start_phase("training")
        
        trainer = FederatedTrainer(
            model=model,
            federated_dataset=fed_dataset,
            config=config,
            device=device
        )
        
        # Run training
        history = trainer.train(
            num_rounds=config.training.num_rounds,
            eval_every=config.experiment.eval_interval
        )
        
        monitor.end_phase("training")
        
        # Final evaluation
        final_metrics = trainer.evaluate()
        
        # Get privacy spent
        epsilon_spent, delta = trainer.get_privacy_spent()
        
        # Stop monitoring
        resource_summary = monitor.stop()
        
        # Compile results
        results = {
            "experiment_name": experiment_name,
            "method": method,
            "config": config.to_dict(),
            "model": {
                "name": config.model.model_name,
                "parameters": total_params,
                "size_kb": total_size
            },
            "data": {
                "dataset": config.data.dataset,
                "num_clients": config.data.num_clients,
                "non_iid_alpha": config.data.non_iid_alpha,
                "samples_per_client": dict(data_sizes)
            },
            "training": {
                "rounds": config.training.num_rounds,
                "history": history
            },
            "final_metrics": final_metrics,
            "privacy": {
                "epsilon_spent": epsilon_spent,
                "delta": delta,
                "budget": config.dp.epsilon
            },
            "resources": resource_summary,
            "system_info": get_system_info()
        }
        
        # INT8 quantization comparison
        print("\nRunning INT8 quantization comparison...")
        quant_results = print_quantization_comparison(model, config.model.model_name)
        results["quantization"] = quant_results
        
        return results
        
    except Exception as e:
        monitor.stop()
        raise e


def run_all_experiments(
    config: Config = None,
    methods: List[str] = None,
    save_results: bool = True
) -> Dict[str, Dict]:
    """
    Run all federated learning experiments.
    
    Args:
        config: Base configuration (uses debug config if None)
        methods: List of methods to run
        save_results: Whether to save results to file
        
    Returns:
        Dictionary of results per method
    """
    if config is None:
        config = get_debug_config()
    
    if methods is None:
        methods = ["fedavg", "fedprox", "personalized_dp"]
    
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    benchmark = BenchmarkComparison()
    
    for method in methods:
        print(f"\n{'#'*60}")
        print(f"# Running {method.upper()} experiment")
        print(f"{'#'*60}")
        
        try:
            method_config = copy.deepcopy(config)
            results = run_experiment(method_config, method=method)
            all_results[method] = results
            
            # Add to benchmark
            benchmark.add_result(
                method=method,
                dataset=config.data.dataset,
                accuracy=results["final_metrics"]["test_accuracy"],
                rounds=config.training.num_rounds,
                num_clients=config.data.num_clients,
                participation_rate=config.data.participation_rate,
                epsilon=results["privacy"]["epsilon_spent"] if config.dp.enabled else None,
                notes=f"debug_mode={config.experiment.debug_mode}"
            )
            
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {"error": str(e)}
    
    # Print benchmark comparison
    benchmark.print_comparison_table()
    
    # Save results
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_path = results_dir / f"experiment_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        # Save benchmark comparison
        benchmark_path = results_dir / f"benchmark_comparison_{timestamp}.json"
        benchmark.save_results(str(benchmark_path))
    
    return all_results


if __name__ == "__main__":
    # Run in debug mode
    config = get_debug_config()
    results = run_all_experiments(config)
