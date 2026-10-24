"""
CNN Baseline Model (ResNet) for Diabetic Retinopathy Classification.

Uses pretrained ResNet as backbone with custom classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class CNNClassifier(nn.Module):
    """
    CNN-based classifier using ResNet backbone.
    
    Architecture:
    - ResNet backbone (pretrained on ImageNet)
    - Global average pooling
    - Fully connected classification head
    
    Transfer learning is standard for medical imaging where
    datasets are smaller than ImageNet.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        """
        Initialize CNN classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: ResNet variant (resnet18, resnet34, resnet50)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights
            dropout: Dropout rate before classifier
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.feature_dim = feature_dim
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input images
            
        Returns:
            Features of shape (batch, feature_dim)
        """
        return self.backbone(x)
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        backbone_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "classifier" not in name
        )
        classifier_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "classifier" in name
        )
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "backbone": backbone_params,
            "classifier": classifier_params,
            "trainable": trainable_params,
            "total": total_params,
        }


class FeatureExtractor(nn.Module):
    """
    CNN feature extractor (ResNet backbone only).
    
    Used for hybrid models where features are passed to
    a separate classifier (e.g., KAN).
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        """
        Initialize feature extractor.
        
        Args:
            backbone: ResNet variant
            pretrained: Use pretrained weights
            freeze: Freeze all weights
        """
        super().__init__()
        
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classifier
        self.model.fc = nn.Identity()
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        return self.model(x)


if __name__ == "__main__":
    print("Testing CNN Classifier...")
    
    batch_size = 4
    num_classes = 5
    
    model = CNNClassifier(
        num_classes=num_classes,
        backbone="resnet18",
        pretrained=False,  # Don't download for testing
    )
    
    x = torch.randn(batch_size, 3, 224, 224)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nParameter counts: {model.count_parameters()}")
    
    # Test feature extraction
    print("\n" + "=" * 50)
    print("Testing Feature Extractor...")
    
    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    features = extractor(x)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature dim: {extractor.feature_dim}")
