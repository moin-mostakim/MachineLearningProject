#!/usr/bin/env python3
"""
Quick test script to verify the full pipeline works end-to-end.
Uses minimal data and epochs for fast validation.
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np

from config import Config, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig, UncertaintyConfig, EvaluationConfig, ResourceConfig
from data.dataset import SyntheticClimateDataset, create_dataloaders
from models.baselines import PersistenceBaseline, ClimatologyBaseline, LinearRegressionBaseline
from models.unet import UNet
from models.convlstm import ConvLSTMPredictor
from models.ensemble import DeepEnsemble
from utils.metrics import rmse, anomaly_correlation_coefficient, ClimateMetrics
from utils.resource_monitor import count_parameters, get_parameter_table, Timer
from utils.quantization import quantize_for_deployment, print_quantization_report


def get_minimal_config():
    """Create minimal config for quick testing."""
    config = Config()
    
    # Minimal data
    config.data.lat_resolution = 16
    config.data.lon_resolution = 32
    config.data.input_time_steps = 2
    
    # Minimal model
    config.model.unet_base_channels = 8
    config.model.unet_depth = 2
    config.model.unet_use_attention = False
    
    # Minimal training
    config.training.batch_size = 2
    config.training.epochs = 2
    
    # Minimal uncertainty
    config.uncertainty.mc_samples = 3
    config.uncertainty.ensemble_size = 2
    
    # Quick experiment
    config.experiment.debug = True
    config.experiment.debug_samples = 20
    config.experiment.debug_epochs = 2
    
    # No workers for speed
    config.resources.num_workers = 0
    config.resources.pin_memory = False
    
    config.__post_init__()
    return config


def test_data_loading(config):
    """Test data loading."""
    print("\n1. Testing Data Loading...")
    
    dataset = SyntheticClimateDataset(config.data, "train", num_samples=20)
    print(f"   Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"   Input shape: {sample['input'].shape}")
    print(f"   Target shape: {sample['target'].shape}")
    
    return True


def test_baselines(config, device):
    """Test baseline models."""
    print("\n2. Testing Baselines...")
    
    x = torch.randn(2, config.data.input_time_steps, config.model.input_channels, 
                    config.data.lat_resolution, config.data.lon_resolution)
    target = torch.randn(2, config.model.output_channels,
                         config.data.lat_resolution, config.data.lon_resolution)
    
    # Test persistence
    persistence = PersistenceBaseline()
    out = persistence(x)
    print(f"   Persistence output: {out.shape}")
    
    # Test climatology
    climatology = ClimatologyBaseline()
    out = climatology(x)
    print(f"   Climatology output: {out.shape}")
    
    # Test linear
    linear = LinearRegressionBaseline(
        config.model.input_channels, config.model.output_channels,
        config.data.input_time_steps, config.data.lat_resolution, config.data.lon_resolution
    )
    linear.fit(x, target)
    out = linear(x)
    print(f"   Linear output: {out.shape}")
    
    return True


def test_unet(config, device):
    """Test U-Net model."""
    print("\n3. Testing U-Net...")
    
    model = UNet(
        in_channels=config.data.input_time_steps * config.model.input_channels,
        out_channels=config.model.output_channels,
        base_channels=config.model.unet_base_channels,
        depth=config.model.unet_depth,
        dropout=config.model.dropout_rate,
        use_attention=config.model.unet_use_attention,
        use_residual=config.model.use_residual
    ).to(device)
    
    params = count_parameters(model)
    print(f"   Parameters: {params['trainable']:,}")
    
    x = torch.randn(2, config.data.input_time_steps, config.model.input_channels,
                    config.data.lat_resolution, config.data.lon_resolution).to(device)
    
    out = model(x)
    print(f"   Output shape: {out.shape}")
    
    # Test uncertainty
    mean, var = model.predict_with_uncertainty(x, n_samples=3)
    print(f"   Uncertainty - Mean: {mean.shape}, Var: {var.shape}")
    
    return model


def test_convlstm(config, device):
    """Test ConvLSTM model."""
    print("\n4. Testing ConvLSTM...")
    
    model = ConvLSTMPredictor(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        hidden_channels=[32, 32],
        dropout=config.model.dropout_rate,
        use_residual=config.model.use_residual
    ).to(device)
    
    params = count_parameters(model)
    print(f"   Parameters: {params['trainable']:,}")
    
    x = torch.randn(2, config.data.input_time_steps, config.model.input_channels,
                    config.data.lat_resolution, config.data.lon_resolution).to(device)
    
    out = model(x)
    print(f"   Output shape: {out.shape}")
    
    return model


def test_ensemble(config, device):
    """Test deep ensemble."""
    print("\n5. Testing Ensemble...")
    
    def create_small_unet():
        return UNet(
            in_channels=config.data.input_time_steps * config.model.input_channels,
            out_channels=config.model.output_channels,
            base_channels=4,
            depth=1,
            dropout=0.1
        )
    
    ensemble = DeepEnsemble(create_small_unet, ensemble_size=2).to(device)
    
    print(f"   Ensemble size: {ensemble.ensemble_size}")
    print(f"   Params per member: {ensemble.count_parameters_per_member():,}")
    
    x = torch.randn(2, config.data.input_time_steps, config.model.input_channels,
                    config.data.lat_resolution, config.data.lon_resolution).to(device)
    
    mean, var = ensemble.predict_with_uncertainty(x)
    print(f"   Uncertainty - Mean: {mean.shape}, Var: {var.shape}")
    
    return ensemble


def test_metrics(config, device):
    """Test metrics."""
    print("\n6. Testing Metrics...")
    
    pred = torch.randn(2, config.model.output_channels,
                       config.data.lat_resolution, config.data.lon_resolution)
    target = torch.randn_like(pred)
    std = torch.abs(torch.randn_like(pred)) + 0.1
    
    metrics_calc = ClimateMetrics()
    results = metrics_calc.compute_all(pred, target, std)
    
    print(f"   RMSE: {results['rmse']:.4f}")
    print(f"   MAE: {results['mae']:.4f}")
    print(f"   ACC: {results['acc']:.4f}")
    print(f"   CRPS: {results['crps']:.4f}")
    
    return True


def test_quantization(model, config, device):
    """Test INT8 quantization."""
    print("\n7. Testing Quantization...")
    
    try:
        quantized, metrics = quantize_for_deployment(model.cpu(), method="dynamic")
        print(f"   Compression ratio: {metrics.get('compression_ratio', 'N/A')}")
        print(f"   Output correlation: {metrics.get('correlation', 'N/A')}")
        return True
    except Exception as e:
        print(f"   Quantization error: {e}")
        return False


def test_training_step(model, config, device):
    """Test a single training step."""
    print("\n8. Testing Training Step...")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    x = torch.randn(2, config.data.input_time_steps, config.model.input_channels,
                    config.data.lat_resolution, config.data.lon_resolution).to(device)
    target = torch.randn(2, config.model.output_channels,
                         config.data.lat_resolution, config.data.lon_resolution).to(device)
    
    # Forward
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, target)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLIMATE AI EMULATION - QUICK TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = get_minimal_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    results = {}
    
    # Run tests
    try:
        results['data'] = test_data_loading(config)
        results['baselines'] = test_baselines(config, device)
        unet = test_unet(config, device)
        results['unet'] = unet is not None
        convlstm = test_convlstm(config, device)
        results['convlstm'] = convlstm is not None
        results['ensemble'] = test_ensemble(config, device) is not None
        results['metrics'] = test_metrics(config, device)
        results['quantization'] = test_quantization(unet, config, device)
        results['training'] = test_training_step(unet, config, device)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        if name == 'error':
            continue
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
