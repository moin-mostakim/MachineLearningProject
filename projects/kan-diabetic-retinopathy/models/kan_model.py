"""
KAN Classifier Model for Diabetic Retinopathy.

Implements a full Kolmogorov-Arnold Network for classification,
composed of multiple KAN layers.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .kan_layer import KANLayer, KANLinear


class KANClassifier(nn.Module):
    """
    KAN-based classifier for DR severity grading.
    
    Architecture:
    - Input layer: Flatten image or take features
    - Multiple KAN hidden layers with learnable activations
    - Output layer: Logits for each class
    
    Key advantage: Parameter efficiency through learnable B-spline
    activation functions that can model complex non-linearities
    with fewer parameters than equivalent MLPs.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: List[int] = [64, 32],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: str = "silu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Initialize KAN classifier.
        
        Args:
            in_features: Input dimension (e.g., 512 for CNN features)
            num_classes: Number of output classes (5 for DR grading)
            hidden_dims: List of hidden layer dimensions
            grid_size: B-spline grid size (default 5)
            spline_order: B-spline order (default 3 = cubic)
            scale_noise: Initialization noise scale
            scale_base: Scale for base activation path
            scale_spline: Scale for spline path
            base_activation: Base activation function
            dropout: Dropout rate between layers
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        dims = [in_features] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(
                KANLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            )
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer (use simpler linear for final classification)
        # KAN is used for feature transformation, linear for final mapping
        self.classifier = nn.Linear(hidden_dims[-1] if hidden_dims else in_features, num_classes)
        
        # Store config for parameter counting
        self.grid_size = grid_size
        self.spline_order = spline_order
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, in_features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Handle 4D input (from images)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        # Pass through KAN layers
        x = self.layers(x)
        
        # Classify
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self) -> dict:
        """
        Count parameters by component.
        
        Returns:
            Dictionary with parameter counts
        """
        kan_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if "classifier" not in name
        )
        classifier_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if "classifier" in name
        )
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "kan_layers": kan_params,
            "classifier": classifier_params,
            "total": total_params,
        }


class KANImageClassifier(nn.Module):
    """
    KAN classifier that works directly on flattened images.
    
    Note: This is primarily for benchmarking. In practice,
    using a CNN backbone + KAN classifier (Hybrid) works better
    because CNNs are optimized for spatial feature extraction.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 5,
        hidden_dims: List[int] = [256, 64, 32],
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize image KAN classifier.
        
        Args:
            image_size: Input image size (square)
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of classes
            hidden_dims: Hidden dimensions
            grid_size: B-spline grid size
            spline_order: B-spline order
            dropout: Dropout rate
        """
        super().__init__()
        
        in_features = image_size * image_size * in_channels
        
        # Dimensionality reduction layer (images are too large for direct KAN)
        self.input_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # KAN classifier
        self.kan_classifier = KANClassifier(
            in_features=512,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        return self.kan_classifier(x)


if __name__ == "__main__":
    print("Testing KAN Classifier...")
    
    # Test basic KAN classifier
    batch_size = 4
    in_features = 512  # CNN features
    num_classes = 5
    
    model = KANClassifier(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=[64, 32],
        grid_size=5,
        spline_order=3,
    )
    
    x = torch.randn(batch_size, in_features)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nParameter counts: {model.count_parameters()}")
    
    # Test image classifier
    print("\n" + "=" * 50)
    print("Testing KAN Image Classifier...")
    
    img_model = KANImageClassifier(
        image_size=224,
        in_channels=3,
        num_classes=5,
        hidden_dims=[128, 64],
    )
    
    images = torch.randn(batch_size, 3, 224, 224)
    logits = img_model(images)
    
    print(f"Image input shape: {images.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in img_model.parameters()):,}")
