"""
Hybrid CNN + KAN Model for Diabetic Retinopathy Classification.

Combines the spatial feature extraction capabilities of CNNs with
the parameter-efficient learnable activations of KAN.

Architecture:
- CNN backbone (ResNet): Extracts spatial features from fundus images
- KAN classifier: Classifies features using learnable B-spline activations

Rationale:
- CNNs excel at hierarchical spatial feature extraction
- KAN provides more expressive classification with fewer parameters
- Combined approach leverages strengths of both architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional

from .kan_layer import KANLayer
from .kan_model import KANClassifier
from .cnn_model import FeatureExtractor


class HybridCNNKAN(nn.Module):
    """
    Hybrid model combining CNN feature extraction with KAN classification.
    
    This architecture leverages:
    1. CNN (ResNet): Efficient spatial feature extraction through 
       convolutions, pooling, and skip connections
    2. KAN: Parameter-efficient classification with learnable 
       B-spline activation functions
    
    Compared to standard CNN + MLP:
    - Similar accuracy (empirically)
    - Fewer parameters in classifier head
    - More interpretable activations (can visualize learned functions)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        kan_hidden_dims: List[int] = [128, 64],
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize hybrid CNN-KAN model.
        
        Args:
            num_classes: Number of output classes (5 for DR grading)
            backbone: CNN backbone (resnet18, resnet34, resnet50)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Whether to freeze CNN weights
            kan_hidden_dims: Hidden dimensions for KAN classifier
            grid_size: B-spline grid size
            spline_order: B-spline order
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # CNN Feature Extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        feature_dim = self.feature_extractor.feature_dim
        
        # KAN Classifier
        self.kan_classifier = KANClassifier(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dims=kan_hidden_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.kan_hidden_dims = kan_hidden_dims
        self.grid_size = grid_size
        self.spline_order = spline_order
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Extract features with CNN
        features = self.feature_extractor(x)
        
        # Classify with KAN
        logits = self.kan_classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features without classification."""
        return self.feature_extractor(x)
    
    def count_parameters(self) -> dict:
        """
        Count parameters by component.
        
        Useful for comparing parameter efficiency between architectures.
        """
        cnn_params = sum(p.numel() for p in self.feature_extractor.parameters())
        kan_params = sum(p.numel() for p in self.kan_classifier.parameters())
        
        cnn_trainable = sum(
            p.numel() for p in self.feature_extractor.parameters() 
            if p.requires_grad
        )
        kan_trainable = sum(
            p.numel() for p in self.kan_classifier.parameters() 
            if p.requires_grad
        )
        
        total_params = cnn_params + kan_params
        total_trainable = cnn_trainable + kan_trainable
        
        return {
            "cnn_backbone": cnn_params,
            "kan_classifier": kan_params,
            "cnn_trainable": cnn_trainable,
            "kan_trainable": kan_trainable,
            "total": total_params,
            "total_trainable": total_trainable,
        }


class HybridCNNMLP(nn.Module):
    """
    Hybrid CNN + MLP model for comparison with CNN + KAN.
    
    Uses same CNN backbone but standard MLP classifier.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        mlp_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
    ):
        """Initialize hybrid CNN-MLP model."""
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN Feature Extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        feature_dim = self.feature_extractor.feature_dim
        
        # MLP Classifier
        layers = []
        dims = [feature_dim] + mlp_hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(dims[-1], num_classes))
        self.classifier = nn.Sequential(*layers)
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        cnn_params = sum(p.numel() for p in self.feature_extractor.parameters())
        mlp_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = cnn_params + mlp_params
        
        return {
            "cnn_backbone": cnn_params,
            "mlp_classifier": mlp_params,
            "total": total_params,
        }


def compare_hybrid_models() -> dict:
    """
    Compare parameter counts between CNN+KAN and CNN+MLP.
    
    Returns:
        Dictionary with comparison statistics
    """
    # CNN + KAN
    cnn_kan = HybridCNNKAN(
        num_classes=5,
        backbone="resnet18",
        pretrained=False,
        kan_hidden_dims=[128, 64],
        grid_size=5,
        spline_order=3,
    )
    
    # CNN + MLP (equivalent capacity)
    cnn_mlp = HybridCNNMLP(
        num_classes=5,
        backbone="resnet18",
        pretrained=False,
        mlp_hidden_dims=[256, 128],
    )
    
    kan_params = cnn_kan.count_parameters()
    mlp_params = cnn_mlp.count_parameters()
    
    return {
        "cnn_kan": kan_params,
        "cnn_mlp": mlp_params,
        "classifier_reduction": mlp_params["mlp_classifier"] / kan_params["kan_classifier"],
    }


if __name__ == "__main__":
    print("Testing Hybrid CNN-KAN Model...")
    
    batch_size = 4
    num_classes = 5
    
    model = HybridCNNKAN(
        num_classes=num_classes,
        backbone="resnet18",
        pretrained=False,
        kan_hidden_dims=[128, 64],
    )
    
    x = torch.randn(batch_size, 3, 224, 224)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nParameter counts: {model.count_parameters()}")
    
    # Compare models
    print("\n" + "=" * 50)
    print("Model Comparison:")
    comparison = compare_hybrid_models()
    
    print("\nCNN + KAN:")
    for k, v in comparison["cnn_kan"].items():
        print(f"  {k}: {v:,}")
    
    print("\nCNN + MLP:")
    for k, v in comparison["cnn_mlp"].items():
        print(f"  {k}: {v:,}")
    
    print(f"\nClassifier parameter reduction (MLP/KAN): {comparison['classifier_reduction']:.2f}x")
