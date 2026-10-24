"""
Loss Functions for Multi-Task DR Learning.

Implements:
- Dice Loss for segmentation
- Focal Loss for classification (handling class imbalance)
- Combined multi-task loss with various weighting strategies

References:
- Milletari et al. (2016) "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth=None):
        super().__init__()
        self.smooth = smooth if smooth is not None else config.DICE_SMOOTH
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) predicted masks (logits)
            target: (B, C, H, W) ground truth masks
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        # Dice coefficient
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation."""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in DR grading.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma=None, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma if gamma is not None else config.FOCAL_GAMMA
        self.alpha = alpha  # Class weights (optional)
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C) predicted logits
            target: (B,) ground truth labels
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning with various weighting strategies.
    
    Strategies:
    - fixed: Fixed weights from config
    - uncertainty: Homoscedastic uncertainty (Kendall et al., 2018)
    - gradnorm: Gradient normalization (Chen et al., 2018)
    """
    
    def __init__(self, strategy='fixed', class_weights=None):
        super().__init__()
        
        self.strategy = strategy
        
        # Individual losses with class weights for classification
        self.seg_loss = DiceBCELoss()
        self.cls_loss = FocalLoss(alpha=class_weights)
        
        # Learnable parameters for different strategies
        if strategy == 'uncertainty':
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_cls = nn.Parameter(torch.zeros(1))
        elif strategy == 'gradnorm':
            self.weights = nn.Parameter(torch.ones(2))
            self.register_buffer('initial_losses', None)
    
    def forward(self, outputs, targets):
        """
        Compute combined multi-task loss.
        
        Args:
            outputs: Dict with 'segmentation' and 'classification' predictions
            targets: Dict with 'mask' and 'grade' ground truths
        """
        # Individual losses
        loss_seg = self.seg_loss(outputs['segmentation'], targets['mask'])
        loss_cls = self.cls_loss(outputs['classification'], targets['grade'])
        
        # Apply weighting strategy
        if self.strategy == 'fixed':
            total_loss = (config.SEGMENTATION_WEIGHT * loss_seg + 
                         config.CLASSIFICATION_WEIGHT * loss_cls)
            weights = (config.SEGMENTATION_WEIGHT, config.CLASSIFICATION_WEIGHT)
            
        elif self.strategy == 'uncertainty':
            # L = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
            precision_seg = torch.exp(-self.log_var_seg)
            precision_cls = torch.exp(-self.log_var_cls)
            
            total_loss = (precision_seg * loss_seg + 0.5 * self.log_var_seg +
                         precision_cls * loss_cls + 0.5 * self.log_var_cls)
            
            weights = (precision_seg.item(), precision_cls.item())
            
        elif self.strategy == 'gradnorm':
            # Normalize weights
            weights_normalized = F.softmax(self.weights, dim=0)
            total_loss = (weights_normalized[0] * loss_seg + 
                         weights_normalized[1] * loss_cls)
            weights = (weights_normalized[0].item(), weights_normalized[1].item())
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return {
            'total': total_loss,
            'segmentation': loss_seg,
            'classification': loss_cls,
            'weights': weights
        }
    
    def get_weights(self):
        """Get current task weights."""
        if self.strategy == 'fixed':
            return config.SEGMENTATION_WEIGHT, config.CLASSIFICATION_WEIGHT
        elif self.strategy == 'uncertainty':
            return torch.exp(-self.log_var_seg).item(), torch.exp(-self.log_var_cls).item()
        elif self.strategy == 'gradnorm':
            weights = F.softmax(self.weights, dim=0)
            return weights[0].item(), weights[1].item()


def compute_gradnorm_loss(model, criterion, shared_params, alpha=1.5):
    """
    Compute GradNorm loss for gradient balancing.
    
    Args:
        model: Multi-task model
        criterion: MultiTaskLoss with gradnorm strategy
        shared_params: Parameters of shared encoder
        alpha: Restoring force coefficient
    """
    if not hasattr(criterion, 'initial_losses') or criterion.initial_losses is None:
        return torch.tensor(0.0)
    
    # This is a placeholder - full implementation requires gradient computation
    # during backward pass which is complex
    return torch.tensor(0.0)


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    
    # Test segmentation losses
    pred_seg = torch.randn(batch_size, config.NUM_LESION_TYPES, 
                           config.IMAGE_SIZE, config.IMAGE_SIZE)
    target_seg = torch.randint(0, 2, (batch_size, config.NUM_LESION_TYPES,
                                       config.IMAGE_SIZE, config.IMAGE_SIZE)).float()
    
    dice_loss = DiceLoss()
    print(f"Dice loss: {dice_loss(pred_seg, target_seg):.4f}")
    
    dice_bce_loss = DiceBCELoss()
    print(f"Dice+BCE loss: {dice_bce_loss(pred_seg, target_seg):.4f}")
    
    # Test classification losses
    pred_cls = torch.randn(batch_size, config.NUM_CLASSES)
    target_cls = torch.randint(0, config.NUM_CLASSES, (batch_size,))
    
    focal_loss = FocalLoss()
    print(f"Focal loss: {focal_loss(pred_cls, target_cls):.4f}")
    
    # Test multi-task loss
    outputs = {'segmentation': pred_seg, 'classification': pred_cls}
    targets = {'mask': target_seg, 'grade': target_cls}
    
    for strategy in ['fixed', 'uncertainty']:
        mtl_loss = MultiTaskLoss(strategy=strategy)
        loss_dict = mtl_loss(outputs, targets)
        print(f"\nMulti-task loss ({strategy}):")
        print(f"  Total: {loss_dict['total']:.4f}")
        print(f"  Segmentation: {loss_dict['segmentation']:.4f}")
        print(f"  Classification: {loss_dict['classification']:.4f}")
        print(f"  Weights: {loss_dict['weights']}")
