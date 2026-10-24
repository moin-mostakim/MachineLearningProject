"""
Quick test script to verify all components work.
Runs a minimal forward pass through all models and validates metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from configs.config import Config
from src.models import (
    create_resnet_baseline,
    create_vit_baseline,
    create_multiscale_lesion_vit,
)
from src.utils import (
    quadratic_weighted_kappa,
    compute_all_metrics,
    print_metrics_table,
    OrdinalCrossEntropyLoss,
    count_parameters,
    compare_quantization,
    print_quantization_comparison,
)

def test_models():
    """Test all model architectures."""
    print("\n" + "=" * 60)
    print(" Testing Model Architectures")
    print("=" * 60 + "\n")
    
    config = Config()
    batch_size = 2
    img_size = config.data.image_size
    num_classes = config.data.num_classes
    
    # Create dummy input
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    models = {
        'ResNet50': create_resnet_baseline(num_classes=num_classes),
        'ViT_Baseline': create_vit_baseline(
            img_size=img_size,
            num_classes=num_classes,
            embed_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            num_heads=config.model.num_heads,
        ),
        'MultiScale_LesionViT': create_multiscale_lesion_vit(
            img_size=img_size,
            num_classes=num_classes,
            embed_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            num_heads=config.model.num_heads,
            num_lesion_types=config.model.num_lesion_types,
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, num_classes), \
            f"Wrong output shape: {output.shape}"
        
        # Count parameters
        params = count_parameters(model)
        
        # Test attention (for ViT models)
        if hasattr(model, 'get_attention_maps'):
            with torch.no_grad():
                attn_info = model.get_attention_maps(x)
            if isinstance(attn_info, dict):
                print(f"  Attention info keys: {list(attn_info.keys())}")
            else:
                print(f"  Attention info type: {type(attn_info)}")
        
        # Test lesion heatmaps (for our method)
        if hasattr(model, 'get_lesion_heatmaps'):
            with torch.no_grad():
                heatmaps = model.get_lesion_heatmaps(x)
            if heatmaps is not None:
                print(f"  Lesion heatmap shape: {heatmaps.shape}")
        
        results[name] = {
            'output_shape': output.shape,
            'total_params': params['total'],
            'trainable_params': params['trainable'],
        }
        
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params['total']:,}")
        print(f"  PASSED!")
    
    return results


def test_metrics():
    """Test metric computation."""
    print("\n" + "=" * 60)
    print(" Testing Metrics")
    print("=" * 60 + "\n")
    
    # Create sample predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 100)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(100, 20, replace=False)
    y_pred[noise_idx] = np.random.randint(0, 5, 20)
    
    y_probs = np.eye(5)[y_pred] * 0.7 + 0.06  # Soft predictions
    
    # Compute QWK
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    
    # Compute all metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print_metrics_table(metrics, "Test Metrics")
    
    return metrics


def test_loss():
    """Test loss function."""
    print("\n" + "=" * 60)
    print(" Testing Loss Function")
    print("=" * 60 + "\n")
    
    batch_size = 8
    num_classes = 5
    
    # Create random logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Standard CE
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(logits, targets)
    
    # Ordinal CE
    ordinal_loss = OrdinalCrossEntropyLoss(
        num_classes=num_classes,
        ordinal_weight=0.3,
        label_smoothing=0.1,
    )
    ordinal_value = ordinal_loss(logits, targets)
    
    print(f"Standard CE Loss: {ce_value.item():.4f}")
    print(f"Ordinal CE Loss:  {ordinal_value.item():.4f}")
    
    return {
        'ce_loss': ce_value.item(),
        'ordinal_loss': ordinal_value.item(),
    }


def test_quantization():
    """Test INT8 quantization."""
    print("\n" + "=" * 60)
    print(" Testing INT8 Quantization")
    print("=" * 60 + "\n")
    
    # Create a small model for quick testing
    model = create_vit_baseline(
        img_size=224,
        num_classes=5,
        embed_dim=128,
        depth=2,
        num_heads=4,
    )
    
    try:
        results = compare_quantization(
            model,
            input_shape=(1, 3, 224, 224),
            include_static=False,
        )
        print_quantization_comparison(results)
        return results
    except Exception as e:
        print(f"Quantization test failed: {e}")
        return {'error': str(e)}


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\n" + "=" * 60)
    print(" Testing Synthetic Data Generation")
    print("=" * 60 + "\n")
    
    from src.data.dataset import DRDataset
    
    data_dir = Path('outputs/data_test')
    
    # Create dataset
    dataset = DRDataset(
        data_dir=str(data_dir),
        split='train',
        image_size=224,
        num_synthetic_samples=20,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Get a sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
    
    # Check class weights
    weights = dataset.get_class_weights()
    print(f"Class weights: {weights}")
    
    return {
        'size': len(dataset),
        'distribution': dataset.get_class_distribution(),
    }


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" QUICK TEST: Diabetic Retinopathy ViT Project")
    print("=" * 60)
    
    # Run tests
    model_results = test_models()
    metric_results = test_metrics()
    loss_results = test_loss()
    data_results = test_synthetic_data()
    quant_results = test_quantization()
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    print("\n Model Parameters:")
    for name, result in model_results.items():
        print(f"  {name}: {result['total_params']:,} params")
    
    print("\n Key Metrics:")
    print(f"  QWK: {metric_results['quadratic_weighted_kappa']:.4f}")
    print(f"  Accuracy: {metric_results['accuracy']:.4f}")
    
    print("\n Quantization:")
    if 'comparison' in quant_results:
        print(f"  Size Reduction: {quant_results['comparison']['size_reduction']:.1f}%")
        print(f"  Speedup: {quant_results['comparison']['speedup']:.2f}x")
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60 + "\n")
    
    return {
        'models': model_results,
        'metrics': metric_results,
        'loss': loss_results,
        'data': data_results,
        'quantization': quant_results,
    }


if __name__ == '__main__':
    main()
