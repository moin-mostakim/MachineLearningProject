"""
MLP Baseline Model for Diabetic Retinopathy Classification.

Standard Multi-Layer Perceptron for comparison with KAN.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPClassifier(nn.Module):
    """
    MLP-based classifier for DR severity grading.
    
    Standard MLP architecture with:
    - Multiple fully connected layers
    - Fixed activation functions (ReLU/GELU)
    - Batch normalization
    - Dropout for regularization
    
    Used as baseline for comparison with KAN.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """
        Initialize MLP classifier.
        
        Args:
            in_features: Input dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function (relu, gelu, silu)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        dims = [in_features] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Batch norm
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.features = nn.Sequential(*layers)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1] if hidden_dims else in_features, num_classes)
    
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
        
        x = self.features(x)
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        feature_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "classifier" not in name
        )
        classifier_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "classifier" in name
        )
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "feature_layers": feature_params,
            "classifier": classifier_params,
            "total": total_params,
        }


class MLPImageClassifier(nn.Module):
    """
    MLP classifier that works directly on flattened images.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 5,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.3,
    ):
        """
        Initialize image MLP classifier.
        
        Args:
            image_size: Input image size
            in_channels: Number of channels
            num_classes: Number of classes
            hidden_dims: Hidden dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        in_features = image_size * image_size * in_channels
        
        # Input layer with dimensionality reduction
        self.input_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # MLP classifier
        self.mlp = MLPClassifier(
            in_features=512,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_layer(x)
        return self.mlp(x)


if __name__ == "__main__":
    print("Testing MLP Classifier...")
    
    batch_size = 4
    in_features = 512
    num_classes = 5
    
    model = MLPClassifier(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
    )
    
    x = torch.randn(batch_size, in_features)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nParameter counts: {model.count_parameters()}")
