"""
Multi-Task Model for Joint DR Grading and Lesion Segmentation.

Implements three model variants:
1. Classification-only baseline
2. Segmentation-only baseline (U-Net)
3. Multi-task model with shared encoder

References:
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
- Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
- Liu et al. (2019) "End-to-End Multi-Task Learning with Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.encoder import SharedEncoder, PretrainedResNetEncoder, get_encoder, count_parameters
from models.decoder import SegmentationDecoder, ClassificationHead


class ClassificationOnlyModel(nn.Module):
    """Baseline: Classification-only model for DR grading."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = get_encoder(pretrained=pretrained)
        self.classifier = ClassificationHead()
    
    def forward(self, x):
        _, bottleneck = self.encoder(x)
        logits = self.classifier(bottleneck)
        return {'classification': logits}


class SegmentationOnlyModel(nn.Module):
    """Baseline: U-Net style segmentation-only model."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = get_encoder(pretrained=pretrained)
        self.decoder = SegmentationDecoder(encoder_channels=self.encoder.channels)
    
    def forward(self, x):
        features, bottleneck = self.encoder(x)
        masks = self.decoder(features, bottleneck)
        return {'segmentation': masks}


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder for joint DR grading and segmentation.
    
    Architecture:
    - Shared ResNet-style encoder extracts hierarchical features
    - U-Net decoder with attention for lesion segmentation
    - Classification head for DR grading
    - Optional learned task weights for balancing losses
    """
    
    def __init__(self, use_uncertainty_weights=True, pretrained=True):
        super().__init__()
        
        self.use_uncertainty_weights = use_uncertainty_weights
        
        # Shared encoder (pretrained for transfer learning)
        self.encoder = get_encoder(pretrained=pretrained)
        
        # Task-specific heads
        self.seg_decoder = SegmentationDecoder(encoder_channels=self.encoder.channels)
        self.classifier = ClassificationHead()
        
        # Learned uncertainty weights (log variance)
        if use_uncertainty_weights:
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_cls = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Shared encoding
        features, bottleneck = self.encoder(x)
        
        # Task-specific decoding
        seg_masks = self.seg_decoder(features, bottleneck)
        cls_logits = self.classifier(bottleneck)
        
        outputs = {
            'segmentation': seg_masks,
            'classification': cls_logits
        }
        
        if self.use_uncertainty_weights:
            outputs['log_var_seg'] = self.log_var_seg
            outputs['log_var_cls'] = self.log_var_cls
        
        return outputs
    
    def get_task_weights(self):
        """Get current task weights based on learned uncertainty."""
        if self.use_uncertainty_weights:
            # Weight is inverse of variance
            weight_seg = torch.exp(-self.log_var_seg)
            weight_cls = torch.exp(-self.log_var_cls)
            return weight_seg.item(), weight_cls.item()
        return config.SEGMENTATION_WEIGHT, config.CLASSIFICATION_WEIGHT


class TaskWeightingModule(nn.Module):
    """
    Module for different task weighting strategies.
    
    Strategies:
    - fixed: Fixed weights from config
    - uncertainty: Homoscedastic uncertainty (Kendall et al., 2018)
    - gradnorm: Gradient normalization (Chen et al., 2018)
    - dynamic: Dynamic weight averaging (Liu et al., 2019)
    """
    
    def __init__(self, strategy='uncertainty', num_tasks=2):
        super().__init__()
        
        self.strategy = strategy
        self.num_tasks = num_tasks
        
        if strategy == 'uncertainty':
            # Learnable log variances
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        elif strategy == 'gradnorm':
            # Learnable loss weights
            self.weights = nn.Parameter(torch.ones(num_tasks))
            self.initial_losses = None
        elif strategy == 'dynamic':
            # Running average of loss ratios
            self.register_buffer('loss_history', torch.ones(num_tasks, 10))
            self.history_ptr = 0
    
    def get_weights(self, losses=None, model=None, shared_params=None):
        """
        Get task weights based on strategy.
        
        Args:
            losses: List of task losses (for gradnorm/dynamic)
            model: Model for gradient computation (for gradnorm)
            shared_params: Shared encoder parameters (for gradnorm)
        """
        if self.strategy == 'fixed':
            return torch.tensor([config.SEGMENTATION_WEIGHT, config.CLASSIFICATION_WEIGHT])
        
        elif self.strategy == 'uncertainty':
            # weight_i = 1 / (2 * var_i)
            weights = 0.5 * torch.exp(-self.log_vars)
            return weights
        
        elif self.strategy == 'gradnorm':
            # Normalize weights to sum to num_tasks
            weights = F.softmax(self.weights, dim=0) * self.num_tasks
            return weights
        
        elif self.strategy == 'dynamic':
            if losses is not None:
                # Compute relative loss decrease rate
                with torch.no_grad():
                    losses_tensor = torch.stack(losses)
                    # Update history
                    self.loss_history[:, self.history_ptr % 10] = losses_tensor
                    self.history_ptr += 1
                    
                    # Compute relative inverse training rate
                    if self.history_ptr >= 2:
                        recent = self.loss_history[:, (self.history_ptr - 1) % 10]
                        older = self.loss_history[:, (self.history_ptr - 2) % 10]
                        rates = recent / (older + 1e-8)
                        weights = rates / rates.sum() * self.num_tasks
                        return weights
            
            return torch.ones(self.num_tasks)
        
        return torch.ones(self.num_tasks)
    
    def get_regularization(self):
        """Get regularization term for uncertainty weighting."""
        if self.strategy == 'uncertainty':
            # Regularization: sum of log variances
            return self.log_vars.sum()
        return torch.tensor(0.0)


def build_model(model_type='multitask', weighting_strategy='uncertainty', pretrained=True):
    """
    Factory function to build models.
    
    Args:
        model_type: 'classification', 'segmentation', or 'multitask'
        weighting_strategy: Task weighting strategy for multitask
        pretrained: Use ImageNet-pretrained encoder (recommended for small datasets)
    """
    if model_type == 'classification':
        model = ClassificationOnlyModel(pretrained=pretrained)
    elif model_type == 'segmentation':
        model = SegmentationOnlyModel(pretrained=pretrained)
    elif model_type == 'multitask':
        use_uncertainty = weighting_strategy == 'uncertainty'
        model = MultiTaskModel(use_uncertainty_weights=use_uncertainty, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_model_summary(model, input_size=(1, 3, 512, 512)):
    """Get model parameter count summary."""
    x = torch.randn(*input_size)
    
    total_params = count_parameters(model)
    
    # Count by component
    components = {}
    for name, module in model.named_children():
        components[name] = count_parameters(module)
    
    return {
        'total_params': total_params,
        'components': components
    }


def print_model_comparison():
    """Print parameter comparison table for all models."""
    models = {
        'Classification-only': build_model('classification'),
        'Segmentation-only (U-Net)': build_model('segmentation'),
        'Multi-task (Ours)': build_model('multitask')
    }
    
    print("\n" + "=" * 70)
    print("MODEL PARAMETER COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'Total Params':>15} {'Size (MB)':>12}")
    print("-" * 70)
    
    for name, model in models.items():
        total = count_parameters(model)
        size_mb = total * 4 / (1024 ** 2)  # Assuming float32
        print(f"{name:<30} {total:>15,} {size_mb:>12.2f}")
    
    print("-" * 70)
    
    # Detailed breakdown for multi-task model
    print("\nMulti-task Model Component Breakdown:")
    print("-" * 50)
    mtl_model = models['Multi-task (Ours)']
    for name, module in mtl_model.named_children():
        params = count_parameters(module)
        print(f"  {name:<25} {params:>15,}")
    
    return models


if __name__ == "__main__":
    # Test models
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # Test classification model
    cls_model = build_model('classification')
    cls_out = cls_model(x)
    print(f"Classification model output: {cls_out['classification'].shape}")
    
    # Test segmentation model
    seg_model = build_model('segmentation')
    seg_out = seg_model(x)
    print(f"Segmentation model output: {seg_out['segmentation'].shape}")
    
    # Test multi-task model
    mtl_model = build_model('multitask')
    mtl_out = mtl_model(x)
    print(f"Multi-task classification output: {mtl_out['classification'].shape}")
    print(f"Multi-task segmentation output: {mtl_out['segmentation'].shape}")
    
    # Print comparison
    print_model_comparison()
