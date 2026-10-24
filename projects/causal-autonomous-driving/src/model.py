"""Causal representation learning models for autonomous driving."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Lightweight CNN backbone for driving images."""
    
    def __init__(self, feature_dim: int = 512, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ERMClassifier(nn.Module):
    """Standard Empirical Risk Minimization baseline."""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class IRMClassifier(nn.Module):
    """
    Invariant Risk Minimization model.
    
    Reference: Arjovsky et al. (2019) "Invariant Risk Minimization"
    
    Key idea: Learn features that are predictive across all environments,
    encouraging invariance to spurious correlations.
    """
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        # Learnable scale for IRM penalty computation
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @staticmethod
    def compute_irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute IRM penalty: gradient of loss w.r.t. dummy classifier should be zero.
        """
        scale = torch.ones(1, requires_grad=True, device=logits.device)
        loss = F.cross_entropy(logits * scale, labels)
        grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
        return grad.pow(2).sum()


class CausalFeatureExtractor(nn.Module):
    """
    Causal feature extractor with domain-invariant representations.
    
    Uses adversarial training to learn features that are:
    1. Predictive of steering action
    2. Invariant across environments (weather conditions)
    """
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 4, 
                 num_environments: int = 3, dropout: float = 0.3):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # Domain discriminator (for adversarial invariance)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_environments)
        )
        
        self.num_environments = num_environments
    
    def forward(self, x: torch.Tensor, return_domain: bool = False):
        features = self.backbone(x)
        task_logits = self.task_classifier(features)
        
        if return_domain:
            domain_logits = self.domain_discriminator(features)
            return task_logits, domain_logits
        return task_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DataAugmentationBaseline(nn.Module):
    """Baseline with strong data augmentation instead of causal methods."""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply random augmentations during training
        if self.training:
            x = self._augment(x)
        features = self.backbone(x)
        return self.classifier(features)
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random brightness, contrast, and noise."""
        # Random brightness
        brightness = torch.rand(1).item() * 0.4 + 0.8  # 0.8 to 1.2
        x = x * brightness
        
        # Random noise
        noise = torch.randn_like(x) * 0.05
        x = x + noise
        
        return torch.clamp(x, 0, 1)
