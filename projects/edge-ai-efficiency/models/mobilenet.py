"""
MobileNetV2 implementation for edge AI efficiency experiments.

Reference:
Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
https://arxiv.org/abs/1801.04381

Key features:
- Inverted residuals: expand → depthwise conv → project
- Linear bottlenecks: no activation after projection
- Residual connections when stride=1 and in_channels == out_channels
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU6 block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2).
    
    Structure:
    1. Expansion: 1x1 conv to expand channels (skip if t=1)
    2. Depthwise: 3x3 depthwise conv
    3. Projection: 1x1 conv to project back (linear, no activation)
    
    Residual connection when stride=1 and in_channels == out_channels
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
        kernel_size: int = 3
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise convolution
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                      stride=stride, groups=hidden_dim),
            # Projection (linear, no activation)
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 for CIFAR-10/ImageNette classification.
    
    Modified from original ImageNet architecture:
    - Adjustable input size (32x32 for CIFAR-10, 224x224 for ImageNette)
    - Width multiplier for model scaling
    - Configurable number of classes
    
    Architecture:
    - Initial conv: 3 → 32
    - 7 stages of inverted residuals
    - Final conv: exp → 1280
    - Global average pooling + classifier
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        input_size: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Standard MobileNetV2 configuration
        # [expansion, channels, num_blocks, stride]
        if input_size <= 32:
            # Modified for CIFAR-10 (smaller strides)
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # No stride for small images
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            first_stride = 1
        else:
            # Original for ImageNet-sized images
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            first_stride = 2
        
        # Build first conv layer
        input_channel = int(32 * width_mult)
        self.features = [ConvBNReLU(3, input_channel, stride=first_stride)]
        
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # Build last conv layer
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x


class MobileNetV2Small(nn.Module):
    """
    Smaller MobileNetV2 variant for edge deployment.
    
    Modifications:
    - Reduced number of stages
    - Lower expansion ratios
    - Fewer channels
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 0.5,
        input_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Reduced configuration
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [4, 24, 2, 1],
            [4, 32, 2, 2],
            [4, 64, 2, 2],
            [4, 96, 2, 1],
        ]
        
        first_stride = 1 if input_size <= 32 else 2
        
        input_channel = int(16 * width_mult)
        self.features = [ConvBNReLU(3, input_channel, stride=first_stride)]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        last_channel = 256
        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenetv2(num_classes: int = 10, width_mult: float = 1.0, 
                input_size: int = 32, **kwargs) -> MobileNetV2:
    """Factory function for MobileNetV2."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, 
                       input_size=input_size, **kwargs)


def mobilenetv2_small(num_classes: int = 10, width_mult: float = 0.5,
                      input_size: int = 32, **kwargs) -> MobileNetV2Small:
    """Factory function for small MobileNetV2."""
    return MobileNetV2Small(num_classes=num_classes, width_mult=width_mult,
                           input_size=input_size, **kwargs)
