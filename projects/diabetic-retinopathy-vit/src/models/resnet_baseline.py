"""
ResNet Baseline for Diabetic Retinopathy Grading.

Reference:
- He et al. (2016): "Deep Residual Learning for Image Recognition"
- Gulshan et al. (2016): "Development of a Deep Learning Algorithm for Detection of DR"
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any


class ResNetDR(nn.Module):
    """
    ResNet baseline for DR classification.
    
    Architecture choices based on literature:
    - ResNet-50 is common baseline in DR detection
    - Pre-trained on ImageNet provides good initialization
    - Custom classifier head for 5-class DR grading
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        variant: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize ResNet DR model.
        
        Args:
            num_classes: Number of DR grades (5 for 0-4)
            variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate for classifier
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.variant = variant
        
        # Load pretrained ResNet
        if variant == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = 512
        elif variant == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = 512
        elif variant == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = 2048
        elif variant == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            in_features = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        # Store feature dimension for compatibility
        self.hidden_dim = in_features
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return (logits, features)
            
        Returns:
            logits: Classification logits [B, num_classes]
            features: (optional) Feature embeddings [B, hidden_dim]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_cam_layers(self) -> nn.Module:
        """Return the layer to use for Grad-CAM."""
        return self.backbone.layer4
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        classifier = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'backbone': backbone,
            'classifier': classifier,
        }


def create_resnet_baseline(
    num_classes: int = 5,
    variant: str = 'resnet50',
    pretrained: bool = True,
    **kwargs,
) -> ResNetDR:
    """
    Factory function to create ResNet baseline.
    
    Args:
        num_classes: Number of classes
        variant: ResNet variant
        pretrained: Use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        ResNetDR model
    """
    return ResNetDR(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        **kwargs,
    )
