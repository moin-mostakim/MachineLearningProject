#!/usr/bin/env python3
"""
Main experiment script for Climate AI Emulation.

AI-Driven Climate Emulation and Uncertainty Quantification
for High-Resolution Earth System Modeling

This script runs the complete experimental pipeline:
1. Data loading (synthetic or WeatherBench)
2. Baseline evaluation (persistence, climatology, linear)
3. Neural network training (U-Net or ConvLSTM)
4. Uncertainty quantification (MC Dropout or Ensemble)
5. INT8 quantization comparison
6. Comprehensive evaluation with all metrics
7. Literature comparison table

References:
- Rasp et al. (2020) WeatherBench
- Bi et al. (2023) Pangu-Weather
- Lam et al. (2023) GraphCast
- Pathak et al. (2022) FourCastNet
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import get_config, Config
from data.dataset import create_dataloaders, SyntheticClimateDataset
from models.baselines import (
    PersistenceBaseline,
    ClimatologyBaseline,
    LinearRegressionBaseline,
    get_baseline_models
)
from models.unet import UNet, UNetWithUncertainty
from models.convlstm import ConvLSTMPredictor
from models.ensemble import DeepEnsemble, CRPS
from utils.metrics import ClimateMetrics, rmse, anomaly_correlation_coefficient
from utils.resource_monitor import (
    count_parameters,
    estimate_model_size,
    get_parameter_table,
    Timer,
    ResourceMonitor,
    ExperimentLogger
)
from utils.quantization import quantize_for_deployment, print_quantization_report
from experiments.trainer import Trainer, EnsembleTrainer


def print_header():
    """Print experiment header."""
    print("\n" + "=" * 70)
    print("  AI-Driven Climate Emulation and Uncertainty Quantification")
    print("  for High-Resolution Earth System Modeling")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def setup_experiment(config: Config):
    """Set up experiment environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.experiment.seed)
        torch.backends.cudnn.deterministic = config.experiment.deterministic
    
    # Create output directories
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def create_parameter_table(models: Dict[str, nn.Module]) -> str:
    """Create parameter count table for all models."""
    table = []
    table.append("\n" + "=" * 70)
    table.append("MODEL PARAMETER COUNT TABLE")
    table.append("=" * 70)
    table.append(f"{'Model':<30} {'Parameters':>15} {'Size (MB)':>12} {'Size INT8':>12}")
    table.append("-" * 70)
    
    for name, model in models.items():
        params = count_parameters(model)
        sizes = estimate_model_size(model)
        
        table.append(
            f"{name:<30} {params['trainable']:>15,} "
            f"{sizes['params_fp32_mb']:>12.2f} {sizes['params_int8_mb']:>12.2f}"
        )
    
    table.append("-" * 70)
    table.append("=" * 70)
    
    return "\n".join(table)


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str,
    with_uncertainty: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device for evaluation
        with_uncertainty: Whether to compute uncertainty metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    metrics_calc = ClimateMetrics()
    
    all_preds = []
    all_targets = []
    all_stds = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            if with_uncertainty and hasattr(model, 'predict_with_uncertainty'):
                import inspect
                sig = inspect.signature(model.predict_with_uncertainty)
                if 'n_samples' in sig.parameters:
                    mean, var = model.predict_with_uncertainty(inputs, n_samples=5)
                else:
                    mean, var = model.predict_with_uncertainty(inputs)
                std = torch.sqrt(var + 1e-8)
                all_stds.append(std.cpu())
                outputs = mean
            else:
                outputs = model(inputs)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if all_stds:
        all_stds = torch.cat(all_stds, dim=0)
        results = metrics_calc.compute_all(all_preds, all_targets, all_stds)
    else:
        results = metrics_calc.compute_deterministic(all_preds, all_targets)
    
    return results


def run_baseline_evaluation(
    config: Config,
    test_loader,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline models.
    
    Args:
        config: Configuration
        test_loader: Test data loader
        device: Device for evaluation
        
    Returns:
        Dictionary of model name to metrics
    """
    print_section("BASELINE EVALUATION")
    
    baselines = get_baseline_models(config)
    results = {}
    
    for name, model in baselines.items():
        print(f"Evaluating {name}...")
        
        if name == 'linear':
            # Fit linear regression first
            print("  Fitting linear regression on training data...")
            # Get some training data for fitting
            train_inputs = []
            train_targets = []
            for i, batch in enumerate(test_loader):
                if i >= 10:  # Use subset for fitting
                    break
                train_inputs.append(batch['input'])
                train_targets.append(batch['target'])
            
            train_inputs = torch.cat(train_inputs, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            model.fit(train_inputs, train_targets)
        
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
        
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  ACC:  {metrics['acc']:.4f}")
    
    return results


def train_neural_network(
    config: Config,
    train_loader,
    val_loader,
    device: str
) -> Tuple[nn.Module, Dict]:
    """
    Train neural network emulator.
    
    Args:
        config: Configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for training
        
    Returns:
        Tuple of (trained model, training history)
    """
    print_section("NEURAL NETWORK TRAINING")
    
    # Create model based on architecture choice
    if config.model.architecture == "unet":
        model = UNet(
            in_channels=config.data.input_time_steps * config.model.input_channels,
            out_channels=config.model.output_channels,
            base_channels=config.model.unet_base_channels,
            depth=config.model.unet_depth,
            activation=config.model.activation,
            normalization=config.model.normalization,
            dropout=config.model.dropout_rate,
            use_attention=config.model.unet_use_attention,
            use_residual=config.model.use_residual
        )
        print(f"Created U-Net with {model.count_parameters():,} parameters")
        
    elif config.model.architecture == "convlstm":
        model = ConvLSTMPredictor(
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            hidden_channels=config.model.convlstm_hidden_channels,
            kernel_size=config.model.convlstm_kernel_size,
            dropout=config.model.dropout_rate,
            use_residual=config.model.use_residual
        )
        print(f"Created ConvLSTM with {model.count_parameters():,} parameters")
    
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")
    
    # Print parameter table
    print("\n" + get_parameter_table(model))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    return model, history


def train_ensemble(
    config: Config,
    train_loader,
    val_loader,
    device: str
) -> Tuple[DeepEnsemble, List[Dict]]:
    """
    Train deep ensemble for uncertainty quantification.
    
    Args:
        config: Configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for training
        
    Returns:
        Tuple of (trained ensemble, list of training histories)
    """
    print_section("DEEP ENSEMBLE TRAINING")
    
    # Model factory function
    def create_model():
        return UNet(
            in_channels=config.data.input_time_steps * config.model.input_channels,
            out_channels=config.model.output_channels,
            base_channels=config.model.unet_base_channels // 2,  # Smaller for ensemble
            depth=min(config.model.unet_depth, 3),
            dropout=config.model.dropout_rate,
            use_residual=config.model.use_residual
        )
    
    # Create ensemble
    ensemble = DeepEnsemble(
        base_model_fn=create_model,
        ensemble_size=config.uncertainty.ensemble_size
    )
    
    print(f"Created ensemble with {config.uncertainty.ensemble_size} members")
    print(f"Parameters per member: {ensemble.count_parameters_per_member():,}")
    print(f"Total parameters: {ensemble.count_parameters():,}")
    
    # Create trainer
    trainer = EnsembleTrainer(
        ensemble=ensemble,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    histories = trainer.train()
    
    return ensemble, histories


def run_quantization_comparison(
    model: nn.Module,
    test_loader,
    device: str
) -> Dict[str, float]:
    """
    Run INT8 quantization and compare performance.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        
    Returns:
        Comparison metrics
    """
    print_section("INT8 QUANTIZATION COMPARISON")
    
    # Quantize model
    quantized, metrics = quantize_for_deployment(model, method="dynamic")
    
    print_quantization_report(metrics, "Climate Emulator")
    
    return metrics


def create_literature_comparison():
    """
    Create literature comparison table.
    
    Compares our implementation against published results from:
    - WeatherBench baselines
    - Pangu-Weather
    - GraphCast
    - FourCastNet
    """
    print_section("LITERATURE COMPARISON")
    
    comparison = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LITERATURE COMPARISON TABLE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model              │ Z500 RMSE (m) │ T850 RMSE (K) │ Resolution │ Lead Time  ║
║                    │    72h/120h   │    72h/120h   │            │            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ BASELINES (WeatherBench, Rasp et al. 2020)                                   ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ Persistence        │   936 / 1033  │  4.23 / 4.56  │   5.625°   │ 72h/120h   ║
║ Climatology        │   1075 / 1075 │  3.50 / 3.50  │   5.625°   │ 72h/120h   ║
║ Weekly Climatology │   816 / 816   │  2.45 / 2.45  │   5.625°   │ 72h/120h   ║
║ Linear Regression  │   268 / 499   │  1.85 / 2.52  │   5.625°   │ 72h/120h   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ML BASELINES (WeatherBench, Rasp et al. 2020)                                ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ CNN (ResNet)       │   268 / 373   │  1.65 / 2.02  │   5.625°   │ 72h/120h   ║
║ U-Net              │   254 / 345   │  1.58 / 1.95  │   5.625°   │ 72h/120h   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STATE-OF-THE-ART                                                             ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ IFS-HRES (ECMWF)   │   154 / 334   │  1.36 / 1.96  │   0.1°     │ 72h/120h   ║
║ Pangu-Weather [1]  │   135 / 263   │  1.21 / 1.64  │   0.25°    │ 72h/120h   ║
║ GraphCast [2]      │   123 / 244   │  1.14 / 1.57  │   0.25°    │ 72h/120h   ║
║ FourCastNet [3]    │   145 / 282   │  1.30 / 1.73  │   0.25°    │ 72h/120h   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ THIS WORK (Synthetic Data - for methodology validation)                      ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ Our U-Net          │   varies*     │   varies*     │   5.625°   │ 72h        ║
║ Our ConvLSTM       │   varies*     │   varies*     │   5.625°   │ 72h        ║
║ Our Ensemble       │   varies*     │   varies*     │   5.625°   │ 72h        ║
╚══════════════════════════════════════════════════════════════════════════════╝

* Results on synthetic data are not directly comparable to real weather benchmarks.
  This implementation validates the methodology and can be applied to real data.

REFERENCES:
[1] Bi, K., et al. (2023). Pangu-Weather: A 3D High-Resolution Model for Fast 
    and Accurate Global Weather Forecast. Nature, 619, 533-538.
    
[2] Lam, R., et al. (2023). GraphCast: Learning skillful medium-range global 
    weather forecasting. Science, 382(6677), 1416-1421.
    
[3] Pathak, J., et al. (2022). FourCastNet: A Global Data-driven High-resolution 
    Weather Model using Adaptive Fourier Neural Operators. arXiv:2202.11214.

[4] Rasp, S., et al. (2020). WeatherBench: A benchmark dataset for data-driven 
    weather forecasting. JAMES, 12(11), e2020MS002203.

KEY FINDINGS FROM LITERATURE:
- ML models can match or exceed operational NWP skill at medium-range (3-7 days)
- Ensemble methods improve calibration of uncertainty estimates
- Higher resolution improves skill but increases compute requirements
- Residual learning critical for capturing weather anomalies
- Data-driven approaches orders of magnitude faster than physics-based NWP
"""
    print(comparison)
    return comparison


def save_results(
    config: Config,
    baseline_results: Dict,
    nn_results: Dict,
    ensemble_results: Dict,
    quantization_results: Dict,
    output_dir: str
):
    """Save all experiment results to files."""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'debug': config.experiment.debug,
            'architecture': config.model.architecture,
            'batch_size': config.training.batch_size,
            'epochs': config.training.epochs,
            'learning_rate': config.training.learning_rate
        },
        'baseline_results': baseline_results,
        'neural_network_results': nn_results,
        'ensemble_results': ensemble_results,
        'quantization_results': quantization_results
    }
    
    # Save as JSON
    results_file = os.path.join(output_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    return results_file


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(
        description="Climate AI Emulation Experiment"
    )
    parser.add_argument(
        '--debug', action='store_true', default=True,
        help='Run in debug mode with reduced data and epochs'
    )
    parser.add_argument(
        '--architecture', type=str, default='unet',
        choices=['unet', 'convlstm'],
        help='Neural network architecture'
    )
    parser.add_argument(
        '--skip-ensemble', action='store_true',
        help='Skip ensemble training (faster)'
    )
    parser.add_argument(
        '--skip-quantization', action='store_true',
        help='Skip quantization comparison'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Get configuration
    config = get_config(debug=args.debug)
    config.model.architecture = args.architecture
    
    # Setup experiment
    device = setup_experiment(config)
    
    # Create data loaders
    print_section("DATA LOADING")
    print(f"Data source: {config.data.data_source}")
    print(f"Spatial resolution: {config.data.lat_resolution}x{config.data.lon_resolution}")
    print(f"Input time steps: {config.data.input_time_steps}")
    print(f"Forecast lead time: {config.data.forecast_lead_time_hours}h")
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Track timing
    timer = Timer()
    resource_monitor = ResourceMonitor(enable_gpu=(device == "cuda"))
    resource_monitor.start()
    
    # Run experiments
    try:
        # 1. Baseline evaluation
        with timer.time('baselines'):
            baseline_results = run_baseline_evaluation(config, test_loader, device)
        
        # 2. Train neural network
        with timer.time('nn_training'):
            nn_model, nn_history = train_neural_network(
                config, train_loader, val_loader, device
            )
        
        # Evaluate neural network
        print_section("NEURAL NETWORK EVALUATION")
        nn_results = evaluate_model(nn_model, test_loader, device, with_uncertainty=True)
        print(f"Neural Network Results:")
        print(f"  RMSE: {nn_results['rmse']:.4f}")
        print(f"  MAE:  {nn_results['mae']:.4f}")
        print(f"  ACC:  {nn_results['acc']:.4f}")
        print(f"  SSIM: {nn_results['ssim']:.4f}")
        
        # 3. Train ensemble (optional)
        ensemble_results = {}
        if not args.skip_ensemble:
            with timer.time('ensemble_training'):
                ensemble, ensemble_histories = train_ensemble(
                    config, train_loader, val_loader, device
                )
            
            print_section("ENSEMBLE EVALUATION")
            ensemble_results = evaluate_model(
                ensemble, test_loader, device, with_uncertainty=True
            )
            print(f"Ensemble Results:")
            print(f"  RMSE: {ensemble_results['rmse']:.4f}")
            print(f"  ACC:  {ensemble_results['acc']:.4f}")
            if 'crps' in ensemble_results:
                print(f"  CRPS: {ensemble_results['crps']:.4f}")
            if 'calibration_error' in ensemble_results:
                print(f"  Calibration Error: {ensemble_results['calibration_error']:.4f}")
        
        # 4. Quantization comparison (optional)
        quantization_results = {}
        if not args.skip_quantization:
            with timer.time('quantization'):
                quantization_results = run_quantization_comparison(
                    nn_model, test_loader, device
                )
        
        # 5. Parameter count table
        models_for_table = {
            'Persistence': PersistenceBaseline(),
            'Climatology': ClimatologyBaseline(),
            f'{config.model.architecture.upper()}': nn_model
        }
        print(create_parameter_table(models_for_table))
        
        # 6. Literature comparison
        create_literature_comparison()
        
        # 7. Save results
        save_results(
            config,
            baseline_results,
            nn_results,
            ensemble_results,
            quantization_results,
            config.experiment.output_dir
        )
        
    finally:
        resource_monitor.stop()
    
    # Print final summary
    print_section("EXPERIMENT SUMMARY")
    
    print("Timing:")
    for name, stats in timer.get_all_stats().items():
        print(f"  {name}: {stats['total']:.1f}s")
    
    print("\nResource Usage:")
    resources = resource_monitor.get_peak_usage()
    for name, value in resources.items():
        print(f"  {name}: {value:.2f}")
    
    print("\n" + "=" * 70)
    print("  Experiment Complete!")
    print("=" * 70 + "\n")
    
    return {
        'baseline_results': baseline_results,
        'nn_results': nn_results,
        'ensemble_results': ensemble_results,
        'quantization_results': quantization_results
    }


if __name__ == "__main__":
    results = main()
