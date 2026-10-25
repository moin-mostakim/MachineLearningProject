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
    """Standard Empirical Risk Minimization baseline - Regression version."""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 1, dropout: float = 0.3, regression: bool = True):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.regression = regression
        output_dim = 1 if regression else num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.classifier(features)
        if self.regression:
            return out.squeeze(-1)  # Return scalar for regression
        return out
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class IRMClassifier(nn.Module):
    """
    Invariant Risk Minimization model - Regression version.
    
    Reference: Arjovsky et al. (2019) "Invariant Risk Minimization"
    
    Key idea: Learn features that are predictive across all environments,
    encouraging invariance to spurious correlations.
    """
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 1, dropout: float = 0.3, regression: bool = True):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.regression = regression
        output_dim = 1 if regression else num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.classifier(features)
        if self.regression:
            return out.squeeze(-1)
        return out
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @staticmethod
    def compute_irm_penalty(preds: torch.Tensor, targets: torch.Tensor, regression: bool = True) -> torch.Tensor:
        """
        Compute IRM penalty: gradient of loss w.r.t. dummy scale should be zero.
        For regression: uses MSE loss instead of cross-entropy.
        """
        scale = torch.ones(1, requires_grad=True, device=preds.device)
        if regression:
            loss = F.mse_loss(preds * scale, targets)
        else:
            loss = F.cross_entropy(preds * scale, targets)
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
    """Baseline with strong data augmentation - Regression version."""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 1, dropout: float = 0.3, regression: bool = True):
        super().__init__()
        self.backbone = SimpleCNN(feature_dim)
        self.regression = regression
        output_dim = 1 if regression else num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self._augment(x)
        features = self.backbone(x)
        out = self.classifier(features)
        if self.regression:
            return out.squeeze(-1)
        return out
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random brightness, contrast, and noise."""
        brightness = torch.rand(1).item() * 0.4 + 0.8
        x = x * brightness
        noise = torch.randn_like(x) * 0.05
        x = x + noise
        return torch.clamp(x, 0, 1)
