"""
Main Entry Point for Continual Learning Experiments.

Runs experiments comparing:
1. Fine-tuning baseline (catastrophic forgetting)
2. Elastic Weight Consolidation (EWC)
3. Dynamic Sparse Expansion (our method)

Usage:
    python main.py                    # Run all experiments
    python main.py --method ewc       # Run specific method
    python main.py --debug            # Run in debug mode (fast)
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List

import config
from datasets import get_dataset
from models import get_model
from methods import get_method
from metrics import MetricsTracker, compare_methods
from resource_monitor import ResourceMonitor, count_parameters, print_parameter_table
from quantization import QuantizationAnalyzer, create_parameter_count_table


def set_seed(seed: int = None):
    """Set random seeds for reproducibility."""
    seed = seed or config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_method(
    method_name: str,
    dataset,
    input_size: tuple,
    num_classes: int,
    resource_monitor: ResourceMonitor
) -> Dict:
    """
    Run continual learning experiment for a single method.
    
    Returns:
        Dictionary with results including metrics and parameter counts
    """
    print(f"\n{'='*60}")
    print(f"RUNNING METHOD: {method_name.upper()}")
    print(f"{'='*60}")
    
    # Create model
    use_sparse = (method_name == "sparse_expansion")
    model = get_model(
        dataset_name=config.DATASET,
        method=method_name,
        input_size=input_size,
        num_classes=num_classes
    )
    
    # Create learner
    learner = get_method(method_name, model)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(num_tasks=config.NUM_TASKS)
    
    # Track parameter counts over tasks
    param_history = {}
    
    # Train on each task sequentially
    for task_id in range(config.NUM_TASKS):
        print(f"\n--- Training Task {task_id} ---")
        
        # Get task dataloader
        train_loader = dataset.get_task_dataloader(task_id, train=True)
        task_info = dataset.get_task_info(task_id)
        print(f"Task info: {task_info}")
        
        # Train on task
        history = learner.train_task(
            task_id=task_id,
            train_loader=train_loader,
            epochs=config.EPOCHS_PER_TASK,
            verbose=config.VERBOSE
        )
        
        # Record parameter count after this task
        param_counts = model.count_parameters() if hasattr(model, 'count_parameters') else {
            "total": sum(p.numel() for p in model.parameters())
        }
        param_history[f"after_task_{task_id}"] = param_counts
        
        # Evaluate on all tasks seen so far
        print(f"\nEvaluating after Task {task_id}...")
        for eval_task_id in range(task_id + 1):
            test_loader = dataset.get_task_dataloader(eval_task_id, train=False)
            acc = learner.evaluate(eval_task_id, test_loader)
            metrics_tracker.record_accuracy(task_id, eval_task_id, acc)
            print(f"  Task {eval_task_id}: {acc:.2f}%")
    
    # Print metrics
    metrics_tracker.print_summary()
    metrics = metrics_tracker.get_metrics()
    
    # Print parameter growth (for sparse expansion)
    if use_sparse and hasattr(learner, 'get_expansion_summary'):
        expansion = learner.get_expansion_summary()
        print("\nParameter Expansion History:")
        for task_id, info in expansion.items():
            print(f"  Task {task_id}: {info['active_params']:,} active params "
                  f"(sparsity: {info['sparsity']*100:.1f}%)")
    
    return {
        "method": method_name,
        "metrics": {
            "average_accuracy": metrics.average_accuracy,
            "forgetting": metrics.forgetting_measure,
            "backward_transfer": metrics.backward_transfer,
            "per_task_accuracy": metrics.per_task_accuracy
        },
        "accuracy_matrix": metrics.accuracy_matrix.tolist(),
        "param_history": param_history,
        "model": model  # Keep for quantization analysis
    }


def run_quantization_comparison(results: Dict[str, Dict]):
    """Run INT8 quantization comparison for each method."""
    print("\n" + "=" * 60)
    print("INT8 QUANTIZATION COMPARISON")
    print("=" * 60)
    
    input_shape = (3, 32, 32) if config.DATASET == "split_cifar100" else (1, 28, 28)
    
    for method_name, result in results.items():
        model = result.get("model")
        if model is None:
            continue
        
        print(f"\n--- {method_name.upper()} ---")
        
        try:
            analyzer = QuantizationAnalyzer(model)
            comparison = analyzer.compare_fp32_int8(
                input_shape=input_shape,
                num_iterations=30 if config.DEBUG_MODE else 100
            )
            analyzer.print_comparison(comparison)
            
            # Store quantization results
            result["quantization"] = {
                "fp32_size_mb": comparison["fp32"]["size"]["total_size_mb"],
                "int8_size_mb": comparison["int8"]["size"]["total_size_mb"],
                "size_reduction_percent": comparison["comparison"]["size_reduction_percent"],
                "speedup_factor": comparison["comparison"]["speedup_factor"]
            }
        except Exception as e:
            print(f"Quantization failed for {method_name}: {e}")
            result["quantization"] = {"error": str(e)}


def print_final_comparison(results: Dict[str, Dict]):
    """Print final comparison across all methods."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: ALL METHODS")
    print("=" * 80)
    
    # Metrics comparison
    print(f"\n{'Method':<25} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10} {'Params':>15}")
    print("-" * 80)
    
    for method_name, result in results.items():
        m = result["metrics"]
        
        # Get final parameter count
        param_hist = result.get("param_history", {})
        final_task_key = f"after_task_{config.NUM_TASKS - 1}"
        final_params = param_hist.get(final_task_key, {}).get("total", "N/A")
        if isinstance(final_params, int):
            final_params = f"{final_params:,}"
        
        print(
            f"{method_name:<25} "
            f"{m['average_accuracy']:>9.2f}% "
            f"{m['forgetting']:>11.2f}% "
            f"{m['backward_transfer']:>+9.2f}% "
            f"{final_params:>15}"
        )
    
    print("-" * 80)
    
    # Parameter growth table
    print("\n" + "=" * 80)
    print("PARAMETER COUNT TABLE (Showing Expansion)")
    print("=" * 80)
    print(f"{'Method':<25} ", end="")
    for t in range(config.NUM_TASKS):
        print(f"{'Task '+str(t):>12}", end=" ")
    print()
    print("-" * 80)
    
    for method_name, result in results.items():
        param_hist = result.get("param_history", {})
        print(f"{method_name:<25} ", end="")
        for t in range(config.NUM_TASKS):
            key = f"after_task_{t}"
            if key in param_hist:
                count = param_hist[key].get("total", 0)
                print(f"{count:>12,}", end=" ")
            else:
                print(f"{'--':>12}", end=" ")
        print()
    
    print("=" * 80)
    
    # Quantization comparison (if available)
    quant_available = any("quantization" in r for r in results.values())
    if quant_available:
        print("\n" + "=" * 80)
        print("QUANTIZATION SUMMARY")
        print("=" * 80)
        print(f"{'Method':<25} {'FP32 (MB)':>12} {'INT8 (MB)':>12} {'Reduction':>12} {'Speedup':>10}")
        print("-" * 80)
        
        for method_name, result in results.items():
            quant = result.get("quantization", {})
            if "error" in quant:
                print(f"{method_name:<25} {'Error':>12}")
            elif quant:
                print(
                    f"{method_name:<25} "
                    f"{quant['fp32_size_mb']:>11.2f} "
                    f"{quant['int8_size_mb']:>11.2f} "
                    f"{quant['size_reduction_percent']:>11.1f}% "
                    f"{quant['speedup_factor']:>9.2f}x"
                )
        
        print("=" * 80)


def print_literature_comparison():
    """Print comparison with literature methods."""
    print("\n" + "=" * 80)
    print("LITERATURE COMPARISON")
    print("=" * 80)
    print("""
Method                          Key Idea                                    Limitations
------------------------------------------------------------------------------------------------
EWC (Kirkpatrick 2017)         Fisher Information regularization           Fixed capacity, 
                               to protect important weights                 scalability issues
                               
PackNet (Mallya 2018)          Iterative pruning after each task,          Fixed total capacity,
                               use freed params for new tasks               pruning can be brittle
                               
Progressive Nets (Rusu 2016)   Add new columns per task with               Linear param growth,
                               lateral connections to old                   computationally heavy
                               
Lottery Ticket (Frankle 2019)  Sparse subnetworks can match                Finding tickets requires
                               full network performance                     expensive training
                               
Our Method: Dynamic Sparse     Sparse expansion per task,                  Requires task identity
Expansion                      preserves old while adding new              at inference time
                               efficiently
------------------------------------------------------------------------------------------------

Key Advantages of Dynamic Sparse Expansion:
1. Sublinear parameter growth (sparsity_ratio < 1 per task)
2. Zero forgetting (old weights frozen)
3. Efficient inference (sparse activations)
4. Compatible with INT8 quantization
""")
    print("=" * 80)


def save_results(results: Dict[str, Dict], output_dir: str = None):
    """Save results to JSON file."""
    output_dir = output_dir or config.LOG_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Remove non-serializable items (model)
    serializable_results = {}
    for method_name, result in results.items():
        serializable_results[method_name] = {
            k: v for k, v in result.items() if k != "model"
        }
    
    # Add experiment config
    output = {
        "config": {
            "dataset": config.DATASET,
            "num_tasks": config.NUM_TASKS,
            "epochs_per_task": config.EPOCHS_PER_TASK,
            "debug_mode": config.DEBUG_MODE,
            "device": config.DEVICE,
            "seed": config.SEED
        },
        "results": serializable_results
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continual Learning with Dynamic Sparse Expansion"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["finetune", "ewc", "sparse_expansion", "all"],
        default="all",
        help="Which method to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (fast, fewer epochs/tasks)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["split_cifar100", "permuted_mnist"],
        default=None,
        help="Dataset to use"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization comparison"
    )
    args = parser.parse_args()
    
    # Override config with command line args
    if args.debug:
        config.DEBUG_MODE = True
        config.EPOCHS_PER_TASK = 2
        config.NUM_TASKS = 3
        config.FISHER_SAMPLE_SIZE = 20
        config.BATCH_SIZE = 32
    
    if args.dataset:
        config.DATASET = args.dataset
    
    # Print experiment info
    print("=" * 60)
    print("CONTINUAL LEARNING WITH DYNAMIC SPARSE EXPANSION")
    print("=" * 60)
    print(f"Dataset: {config.DATASET}")
    print(f"Number of tasks: {config.NUM_TASKS}")
    print(f"Epochs per task: {config.EPOCHS_PER_TASK}")
    print(f"Debug mode: {config.DEBUG_MODE}")
    print(f"Device: {config.DEVICE}")
    print(f"Seed: {config.SEED}")
    print("=" * 60)
    
    # Set seed
    set_seed()
    
    # Load dataset
    print("\nLoading dataset...")
    dataset, input_size, num_classes = get_dataset()
    print(f"Input size: {input_size}")
    print(f"Classes per task: {num_classes}")
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor()
    
    # Determine which methods to run
    if args.method == "all":
        methods = config.METHODS
    else:
        methods = [args.method]
    
    # Run experiments
    results = {}
    for method_name in methods:
        result = run_single_method(
            method_name=method_name,
            dataset=dataset,
            input_size=input_size,
            num_classes=num_classes,
            resource_monitor=resource_monitor
        )
        results[method_name] = result
    
    # Run quantization comparison
    if not args.no_quantize and config.QUANTIZE_MODEL:
        run_quantization_comparison(results)
    
    # Print final comparison
    print_final_comparison(results)
    
    # Print literature comparison
    print_literature_comparison()
    
    # Print resource summary
    resource_monitor.print_summary()
    
    # Save results
    save_results(results)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
