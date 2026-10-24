"""Models for federated learning."""

import torch
import torch.nn as nn
from typing import List
import copy


class SimpleCNN(nn.Module):
    """Lightweight CNN for image classification."""
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_model(config) -> nn.Module:
    """Factory function to create model."""
    return SimpleCNN(config.model.num_classes, config.model.dropout)


def get_model_params(model: nn.Module) -> List[torch.Tensor]:
    """Get model parameters as a list of tensors."""
    return [p.data.clone() for p in model.parameters()]


def set_model_params(model: nn.Module, params: List[torch.Tensor]):
    """Set model parameters from a list of tensors."""
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def average_params(params_list: List[List[torch.Tensor]], weights: List[float] = None) -> List[torch.Tensor]:
    """Compute weighted average of model parameters."""
    if weights is None:
        weights = [1.0 / len(params_list)] * len(params_list)
    
    weights = [w / sum(weights) for w in weights]
    
    avg_params = []
    for param_idx in range(len(params_list[0])):
        avg_param = sum(
            w * params_list[client_idx][param_idx]
            for client_idx, w in enumerate(weights)
        )
        avg_params.append(avg_param)
    
    return avg_params
