"""MLP architecture for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for baseline experiments.
    
    Simple architecture for MNIST/FEMNIST style datasets.
    Parameters: ~200K (for 784 input, 200-200-10 hidden)
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        input_dim: int = 784,
        hidden_dims: list = None,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [200, 200]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding before classifier."""
        x = x.view(x.size(0), -1)
        return self.features(x)
    
    def get_layer_names(self):
        """Return layer names for personalization."""
        return ["features", "classifier"]
