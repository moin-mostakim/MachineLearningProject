"""
Edge AI Efficiency: Neural Architecture Search + Quantization-Aware Training

Main experiment runner that implements:
1. Baseline models (MobileNetV2, EfficientNet-Lite)
2. Neural Architecture Search (Random, Evolutionary)
3. Quantization-Aware Training (QAT)
4. Combined NAS + QAT pipeline

References:
- MobileNetV2: Sandler et al., CVPR 2018
- EfficientNet: Tan & Le, ICML 2019
- Once-for-All NAS: Cai et al., ICLR 2020
- Quantization: Jacob et al., CVPR 2018
"""

import os
import sys
import argparse
import json
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Add project to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also ensure parent is in path
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

from config import get_config, Config, REFERENCES
from data.datasets import get_dataloaders, get_calibration_loader
from models.mobilenet import MobileNetV2, MobileNetV2Small, mobilenetv2
from models.efficientnet import EfficientNetLite, efficientnet_lite0, efficientnet_lite_small
from models.search_space import SuperNet, create_supernet
from nas.searcher import NASSearcher
from nas.random_search import RandomSearch
from nas.evolutionary import EvolutionarySearch
from quantization.qat import QuantizationAwareTraining, prepare_qat, convert_to_quantized
from quantization.ptq import PostTrainingQuantization
from experiments.trainer import Trainer, create_trainer
from experiments.benchmark import Benchmarker, create_benchmarker
from utils.metrics import count_parameters, get_model_size_mb, count_flops, format_flops


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_model_info(model: nn.Module, name: str, input_size: tuple):
    """Print model information."""
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    flops = count_flops(model, input_size, device='cpu')
    
    print(f"\n{name}:")
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  FLOPs: {format_flops(flops)}")
    
    return {'params': params, 'size_mb': size_mb, 'flops': flops}


def run_baseline_experiment(
    config: Config,
    train_loader,
    val_loader,
    device: str,
    results_dir: str
) -> Dict[str, Any]:
    """
    Run baseline experiments with MobileNetV2 and EfficientNet-Lite.
    """
    print("\n" + "="*60)
    print("PHASE 1: Baseline Models")
    print("="*60)
    
    results = {}
    input_size = (1, 3, config.get_image_size(), config.get_image_size())
    
    # Create models
    models = {
        'MobileNetV2': mobilenetv2(
            num_classes=config.data.num_classes,
            width_mult=config.model.width_multiplier,
            input_size=config.get_image_size()
        ),
        'MobileNetV2-Small': MobileNetV2Small(
            num_classes=config.data.num_classes,
            width_mult=0.5,
            input_size=config.get_image_size()
        ),
        'EfficientNet-Lite0': efficientnet_lite0(
            num_classes=config.data.num_classes,
            input_size=config.get_image_size()
        ),
        'EfficientNet-Lite-Small': efficientnet_lite_small(
            num_classes=config.data.num_classes,
            input_size=config.get_image_size()
        )
    }
    
    # Print model info
    print("\n--- Model Architecture Summary ---")
    for name, model in models.items():
        info = print_model_info(model, name, input_size)
        results[name] = {'model_info': info}
    
    # Train and benchmark MobileNetV2-Small (representative baseline)
    print("\n--- Training MobileNetV2-Small (Baseline) ---")
    baseline_model = models['MobileNetV2-Small']
    
    trainer = create_trainer(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'momentum': config.training.momentum,
            'weight_decay': config.training.weight_decay,
            'use_amp': config.training.use_amp and device != 'cpu',
            'label_smoothing': config.training.label_smoothing,
            'gradient_clip_norm': config.training.gradient_clip_norm
        },
        device=device
    )
    
    trained_model = trainer.train(verbose=True)
    results['MobileNetV2-Small']['training_history'] = trainer.get_training_history()
    
    # Benchmark
    benchmarker = create_benchmarker(
        val_loader=val_loader,
        input_size=input_size,
        config={
            'num_warmup_runs': config.experiment.num_warmup_runs,
            'num_benchmark_runs': config.experiment.num_benchmark_runs
        },
        device=device
    )
    
    benchmark_result = benchmarker.benchmark_model(
        trained_model, 'MobileNetV2-Small',
        benchmark_cpu=True, benchmark_gpu=torch.cuda.is_available()
    )
    results['MobileNetV2-Small']['benchmark'] = {
        'accuracy': benchmark_result['accuracy'],
        'latency_ms': benchmark_result.get('latency_ms', 0),
        'params': benchmark_result['params'],
        'flops': benchmark_result['flops']
    }
    
    # Save baseline model
    torch.save(trained_model.state_dict(), os.path.join(results_dir, 'baseline_model.pth'))
    
    return results, trained_model


def run_nas_experiment(
    config: Config,
    train_loader,
    val_loader,
    device: str,
    results_dir: str
) -> Dict[str, Any]:
    """
    Run Neural Architecture Search experiments.
    """
    print("\n" + "="*60)
    print("PHASE 2: Neural Architecture Search")
    print("="*60)
    
    results = {}
    input_size = (1, 3, config.get_image_size(), config.get_image_size())
    
    # Create supernet
    print("\n--- Creating Supernet ---")
    supernet = create_supernet(
        num_classes=config.data.num_classes,
        input_size=config.get_image_size(),
        config={
            'kernel_sizes': config.nas.kernel_sizes,
            'expansion_ratios': config.nas.expansion_ratios,
            'depth_options': config.nas.depth_options,
            'width_mult': config.model.width_multiplier
        }
    )
    
    print_model_info(supernet, "SuperNet (max config)", input_size)
    
    # Random Search
    print("\n--- Running Random Search ---")
    random_searcher = NASSearcher(
        supernet=supernet,
        train_loader=train_loader,
        val_loader=val_loader,
        algorithm='random',
        num_samples=config.nas.num_random_samples,
        latency_constraint_ms=config.nas.latency_constraint_ms,
        device=device
    )
    
    random_results = random_searcher.search(verbose=True)
    best_random = random_searcher.get_best_architecture()
    
    results['random_search'] = {
        'num_evaluated': len(random_results),
        'best_accuracy': best_random.accuracy if best_random else 0,
        'best_latency': best_random.latency_ms if best_random else 0,
        'best_config': best_random.config if best_random else None
    }
    
    # Evolutionary Search
    print("\n--- Running Evolutionary Search ---")
    # Need fresh supernet for evolutionary search
    supernet_evo = create_supernet(
        num_classes=config.data.num_classes,
        input_size=config.get_image_size(),
        config={
            'kernel_sizes': config.nas.kernel_sizes,
            'expansion_ratios': config.nas.expansion_ratios,
            'depth_options': config.nas.depth_options,
            'width_mult': config.model.width_multiplier
        }
    )
    
    evo_searcher = NASSearcher(
        supernet=supernet_evo,
        train_loader=train_loader,
        val_loader=val_loader,
        algorithm='evolutionary',
        population_size=config.nas.population_size,
        num_generations=config.nas.num_generations,
        mutation_prob=config.nas.mutation_prob,
        latency_constraint_ms=config.nas.latency_constraint_ms,
        device=device
    )
    
    evo_results = evo_searcher.search(verbose=True)
    best_evo = evo_searcher.get_best_architecture()
    
    results['evolutionary_search'] = {
        'num_evaluated': len(evo_results),
        'best_accuracy': best_evo.accuracy if best_evo else 0,
        'best_latency': best_evo.latency_ms if best_evo else 0,
        'best_config': best_evo.config if best_evo else None
    }
    
    # Get best model from NAS
    best_nas_model = None
    best_nas_config = None
    
    if best_evo and best_random:
        if best_evo.accuracy >= best_random.accuracy:
            best_nas_config = best_evo.config
            supernet_evo.set_active_config(best_nas_config)
            best_nas_model = supernet_evo
            results['best_nas_method'] = 'evolutionary'
        else:
            best_nas_config = best_random.config
            supernet.set_active_config(best_nas_config)
            best_nas_model = supernet
            results['best_nas_method'] = 'random'
    elif best_evo:
        best_nas_config = best_evo.config
        supernet_evo.set_active_config(best_nas_config)
        best_nas_model = supernet_evo
        results['best_nas_method'] = 'evolutionary'
    elif best_random:
        best_nas_config = best_random.config
        supernet.set_active_config(best_nas_config)
        best_nas_model = supernet
        results['best_nas_method'] = 'random'
    
    return results, best_nas_model


def run_quantization_experiment(
    config: Config,
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    results_dir: str
) -> Dict[str, Any]:
    """
    Run Quantization-Aware Training and PTQ experiments.
    """
    print("\n" + "="*60)
    print("PHASE 3: Quantization")
    print("="*60)
    
    results = {}
    input_size = (1, 3, config.get_image_size(), config.get_image_size())
    
    # Get calibration data
    cal_loader = get_calibration_loader(
        dataset=config.data.dataset,
        batch_size=32,
        num_samples=config.quantization.observer_batches * 32,
        use_synthetic=config.experiment.debug_mode
    )
    
    # Post-Training Quantization comparison
    print("\n--- Post-Training Quantization Comparison ---")
    ptq = PostTrainingQuantization(
        model=model,
        calibration_loader=cal_loader,
        num_calibration_batches=config.quantization.observer_batches,
        device='cpu'
    )
    
    quant_comparison = ptq.compare_quantization_methods(val_loader)
    
    print("\nQuantization Comparison Results:")
    print("-" * 50)
    for method, metrics in quant_comparison.items():
        print(f"{method}: Accuracy={metrics['accuracy']:.2f}%, Bits={metrics['bits']}")
    
    results['ptq_comparison'] = quant_comparison
    
    # Size reduction
    size_info = ptq.get_model_size_reduction()
    print(f"\nModel Size Reduction:")
    print(f"  Float32: {size_info['float32_size_mb']:.2f} MB")
    print(f"  INT8: {size_info['int8_size_mb']:.2f} MB ({size_info['int8_reduction']*100:.1f}% reduction)")
    print(f"  INT4: {size_info['int4_size_mb']:.2f} MB ({size_info['int4_reduction']*100:.1f}% reduction)")
    
    results['size_reduction'] = size_info
    
    # Quantization-Aware Training
    print("\n--- Quantization-Aware Training ---")
    
    # Create fresh model for QAT
    qat_model = MobileNetV2Small(
        num_classes=config.data.num_classes,
        width_mult=0.5,
        input_size=config.get_image_size()
    )
    qat_model.load_state_dict(model.state_dict())
    
    qat_trainer = QuantizationAwareTraining(
        model=qat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        qat_epochs=config.quantization.qat_epochs,
        learning_rate=config.quantization.qat_learning_rate,
        warmup_epochs=config.quantization.qat_warmup_epochs,
        device=device
    )
    
    # Train with QAT
    qat_model_trained = qat_trainer.train(verbose=True)
    results['qat_history'] = qat_trainer.training_history
    
    # Compare accuracy
    accuracy_comparison = qat_trainer.compare_accuracy()
    print("\nQAT Accuracy Comparison:")
    for key, value in accuracy_comparison.items():
        print(f"  {key}: {value:.2f}%")
    
    results['qat_accuracy'] = accuracy_comparison
    
    return results


def run_combined_experiment(
    config: Config,
    nas_model: Optional[nn.Module],
    train_loader,
    val_loader,
    device: str,
    results_dir: str
) -> Dict[str, Any]:
    """
    Run combined NAS + QAT experiment.
    """
    print("\n" + "="*60)
    print("PHASE 4: Combined NAS + QAT")
    print("="*60)
    
    results = {}
    
    if nas_model is None:
        print("No NAS model available, skipping combined experiment")
        return results
    
    # Train NAS model from scratch first
    print("\n--- Training NAS-discovered architecture ---")
    
    trainer = create_trainer(
        model=nas_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'momentum': config.training.momentum,
            'weight_decay': config.training.weight_decay,
            'use_amp': config.training.use_amp and device != 'cpu',
        },
        device=device
    )
    
    nas_trained = trainer.train(verbose=True)
    results['nas_training'] = trainer.get_training_history()
    
    # Apply QAT to NAS model
    print("\n--- Applying QAT to NAS model ---")
    
    qat_trainer = QuantizationAwareTraining(
        model=nas_trained,
        train_loader=train_loader,
        val_loader=val_loader,
        qat_epochs=config.quantization.qat_epochs,
        learning_rate=config.quantization.qat_learning_rate,
        device=device
    )
    
    nas_qat_model = qat_trainer.train(verbose=True)
    results['nas_qat_history'] = qat_trainer.training_history
    
    # Final accuracy
    accuracy_comparison = qat_trainer.compare_accuracy()
    print("\nNAS + QAT Final Results:")
    for key, value in accuracy_comparison.items():
        print(f"  {key}: {value:.2f}%")
    
    results['nas_qat_accuracy'] = accuracy_comparison
    
    return results


def create_results_summary(all_results: Dict[str, Any], config: Config) -> str:
    """Create a comprehensive results summary."""
    summary = []
    summary.append("\n" + "="*70)
    summary.append("EDGE AI EFFICIENCY: EXPERIMENT SUMMARY")
    summary.append("="*70)
    
    summary.append(f"\nDataset: {config.data.dataset}")
    summary.append(f"Debug Mode: {config.experiment.debug_mode}")
    summary.append(f"Training Epochs: {config.training.num_epochs}")
    
    # Baseline results
    if 'baseline' in all_results:
        baseline = all_results['baseline']
        summary.append("\n--- Baseline Results ---")
        if 'MobileNetV2-Small' in baseline:
            bench = baseline['MobileNetV2-Small'].get('benchmark', {})
            summary.append(f"MobileNetV2-Small:")
            summary.append(f"  Accuracy: {bench.get('accuracy', 'N/A'):.2f}%")
            summary.append(f"  Latency: {bench.get('latency_ms', 'N/A'):.2f} ms")
            summary.append(f"  Parameters: {bench.get('params', 0)/1e6:.2f}M")
    
    # NAS results
    if 'nas' in all_results:
        nas = all_results['nas']
        summary.append("\n--- NAS Results ---")
        
        if 'random_search' in nas:
            rs = nas['random_search']
            summary.append(f"Random Search:")
            summary.append(f"  Evaluated: {rs['num_evaluated']} architectures")
            summary.append(f"  Best Accuracy: {rs['best_accuracy']:.2f}%")
            summary.append(f"  Best Latency: {rs['best_latency']:.2f} ms")
        
        if 'evolutionary_search' in nas:
            es = nas['evolutionary_search']
            summary.append(f"Evolutionary Search:")
            summary.append(f"  Evaluated: {es['num_evaluated']} architectures")
            summary.append(f"  Best Accuracy: {es['best_accuracy']:.2f}%")
            summary.append(f"  Best Latency: {es['best_latency']:.2f} ms")
    
    # Quantization results
    if 'quantization' in all_results:
        quant = all_results['quantization']
        summary.append("\n--- Quantization Results ---")
        
        if 'ptq_comparison' in quant:
            summary.append("PTQ Comparison:")
            for method, metrics in quant['ptq_comparison'].items():
                summary.append(f"  {method}: {metrics['accuracy']:.2f}% ({metrics['bits']}-bit)")
        
        if 'qat_accuracy' in quant:
            summary.append("QAT Results:")
            for key, value in quant['qat_accuracy'].items():
                summary.append(f"  {key}: {value:.2f}%")
    
    # Combined results
    if 'combined' in all_results and 'nas_qat_accuracy' in all_results['combined']:
        combined = all_results['combined']
        summary.append("\n--- Combined NAS + QAT Results ---")
        for key, value in combined['nas_qat_accuracy'].items():
            summary.append(f"  {key}: {value:.2f}%")
    
    # Literature comparison
    summary.append("\n--- Literature Comparison ---")
    summary.append("Reference results from papers (ImageNet):")
    summary.append("  MobileNetV2 (1.0): 72.0% Top-1, 3.4M params")
    summary.append("  MobileNetV2 (0.5): 65.4% Top-1, 1.95M params")
    summary.append("  EfficientNet-B0: 77.1% Top-1, 5.3M params")
    summary.append("  Once-for-All (OFA): Up to 80.0% Top-1 with 595M FLOPs")
    summary.append("  Note: Our results are on CIFAR-10, not directly comparable")
    
    summary.append("\n" + "="*70)
    
    return "\n".join(summary)


def main():
    """Main entry point."""
    import sys
    
    parser = argparse.ArgumentParser(description="Edge AI Efficiency Experiments")
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Run in debug mode (reduced epochs)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup - ensure stdout is unbuffered
    print("="*70, flush=True)
    print("EDGE AI EFFICIENCY: NAS + QAT FOR REAL-TIME INFERENCE", flush=True)
    print("="*70, flush=True)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Get configuration
    config = get_config(debug_mode=args.debug)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")
    
    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print(f"\n--- Loading Data ({config.data.dataset}) ---")
    subset_size = 2000 if config.experiment.debug_mode else None
    # Use synthetic data in debug mode for fast iteration
    use_synthetic = config.experiment.debug_mode
    train_loader, val_loader = get_dataloaders(
        dataset=config.data.dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        subset_size=subset_size,
        use_synthetic=use_synthetic
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # All results
    all_results = {}
    
    try:
        # Phase 1: Baseline
        baseline_results, baseline_model = run_baseline_experiment(
            config, train_loader, val_loader, device, results_dir
        )
        all_results['baseline'] = baseline_results
        
        # Phase 2: NAS
        nas_results, nas_model = run_nas_experiment(
            config, train_loader, val_loader, device, results_dir
        )
        all_results['nas'] = nas_results
        
        # Phase 3: Quantization
        quant_results = run_quantization_experiment(
            config, baseline_model, train_loader, val_loader, device, results_dir
        )
        all_results['quantization'] = quant_results
        
        # Phase 4: Combined NAS + QAT
        combined_results = run_combined_experiment(
            config, nas_model, train_loader, val_loader, device, results_dir
        )
        all_results['combined'] = combined_results
        
        # Generate summary
        summary = create_results_summary(all_results, config)
        print(summary)
        
        # Save summary
        with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
            f.write(summary)
        
        # Save all results
        results_file = os.path.join(results_dir, 'all_results.json')
        
        # Convert non-serializable items
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        serializable_results = make_serializable(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_dir}/")
        
        # Print literature references
        print("\n" + REFERENCES)
        
        success = True
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
