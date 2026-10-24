"""Multimodal fusion model for neurodegenerative disease detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """Lightweight CNN encoder for brain images."""
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 512,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClinicalEncoder(nn.Module):
    """MLP encoder for clinical/tabular features."""
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list = [64, 32],
        output_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion module."""
    
    def __init__(
        self,
        image_dim: int = 128,
        clinical_dim: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.modality_weights = nn.Parameter(torch.ones(2))
    
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_proj = self.image_proj(image_features).unsqueeze(1)
        clinical_proj = self.clinical_proj(clinical_features).unsqueeze(1)
        
        modality_tokens = torch.cat([image_proj, clinical_proj], dim=1)
        
        attended, attn_weights = self.attention(
            modality_tokens, modality_tokens, modality_tokens
        )
        
        weights = F.softmax(self.modality_weights, dim=0)
        weighted_image = weights[0] * attended[:, 0, :]
        weighted_clinical = weights[1] * attended[:, 1, :]
        
        combined = torch.cat([weighted_image, weighted_clinical], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused, attn_weights


class MultimodalClassifier(nn.Module):
    """Complete multimodal classifier for neurodegenerative disease detection."""
    
    def __init__(self, config):
        super().__init__()
        
        self.image_encoder = ImageEncoder(
            in_channels=1,
            hidden_dim=config.image_encoder.hidden_dim,
            output_dim=config.fusion.hidden_dim
        )
        
        self.clinical_encoder = ClinicalEncoder(
            input_dim=config.clinical_encoder.input_dim,
            hidden_dims=config.clinical_encoder.hidden_dims,
            output_dim=config.clinical_encoder.output_dim,
            dropout=config.clinical_encoder.dropout
        )
        
        self.fusion = AttentionFusion(
            image_dim=config.fusion.hidden_dim,
            clinical_dim=config.clinical_encoder.output_dim,
            hidden_dim=config.fusion.hidden_dim,
            num_heads=config.fusion.num_heads,
            dropout=config.fusion.dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, config.num_classes)
        )
    
    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical)
        
        fused, attn_weights = self.fusion(image_features, clinical_features)
        
        logits = self.classifier(fused)
        
        return logits, attn_weights


class ImageOnlyClassifier(nn.Module):
    """Baseline: Image-only classifier."""
    
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            in_channels=1,
            hidden_dim=config.image_encoder.hidden_dim,
            output_dim=config.fusion.hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, config.num_classes)
        )
    
    def forward(self, image: torch.Tensor, clinical: torch.Tensor = None) -> Tuple[torch.Tensor, None]:
        features = self.image_encoder(image)
        return self.classifier(features), None


class ClinicalOnlyClassifier(nn.Module):
    """Baseline: Clinical-only classifier."""
    
    def __init__(self, config):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(
            input_dim=config.clinical_encoder.input_dim,
            hidden_dims=config.clinical_encoder.hidden_dims,
            output_dim=config.clinical_encoder.output_dim,
            dropout=config.clinical_encoder.dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.clinical_encoder.output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, config.num_classes)
        )
    
    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> Tuple[torch.Tensor, None]:
        features = self.clinical_encoder(clinical)
        return self.classifier(features), None


class ConcatFusionClassifier(nn.Module):
    """Baseline: Simple concatenation fusion."""
    
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            in_channels=1,
            hidden_dim=config.image_encoder.hidden_dim,
            output_dim=config.fusion.hidden_dim
        )
        self.clinical_encoder = ClinicalEncoder(
            input_dim=config.clinical_encoder.input_dim,
            hidden_dims=config.clinical_encoder.hidden_dims,
            output_dim=config.clinical_encoder.output_dim,
            dropout=config.clinical_encoder.dropout
        )
        concat_dim = config.fusion.hidden_dim + config.clinical_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, config.num_classes)
        )
    
    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> Tuple[torch.Tensor, None]:
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical)
        concat = torch.cat([image_features, clinical_features], dim=-1)
        return self.classifier(concat), None
