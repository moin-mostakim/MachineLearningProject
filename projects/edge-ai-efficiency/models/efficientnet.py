"""
EfficientNet-Lite implementation for edge AI efficiency experiments.

Reference:
Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
https://arxiv.org/abs/1905.11946

EfficientNet-Lite modifications for edge deployment:
- Replace Swish with ReLU6 (more efficient on mobile)
- Remove squeeze-and-excitation (optional, for max efficiency)
- Fixed batch normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Recalibrates channel-wise feature responses by modeling 
    interdependencies between channels.
    """
    
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed_channels, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(squeezed_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv block (MBConv).
    
    Similar to MobileNetV2 inverted residual but with:
    - Optional Squeeze-and-Excitation
    - Configurable kernel size
    - Drop connect for regularization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
        use_se: bool = True
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if use_se and se_ratio > 0:
            layers.append(SqueezeExcitation(hidden_dim, se_ratio))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        """Drop connect regularization."""
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            (x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device
        )
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            out = self._drop_connect(out)
            out = out + x
        return out


class EfficientNetLite(nn.Module):
    """
    EfficientNet-Lite for edge deployment.
    
    Lite modifications:
    - ReLU6 instead of Swish
    - Optional SE removal
    - Reduced complexity
    
    Compound scaling:
    - depth = α^φ
    - width = β^φ  
    - resolution = γ^φ
    where α=1.2, β=1.1, γ=1.15 and α·β²·γ² ≈ 2
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        input_size: int = 32,
        dropout: float = 0.2,
        drop_connect_rate: float = 0.2,
        use_se: bool = True
    ):
        super().__init__()
        
        # Base configuration (EfficientNet-B0 style)
        # [expand_ratio, channels, num_blocks, kernel_size, stride]
        if input_size <= 32:
            # Modified for CIFAR-10
            base_config = [
                [1, 16, 1, 3, 1],
                [6, 24, 2, 3, 1],
                [6, 40, 2, 5, 2],
                [6, 80, 3, 3, 2],
                [6, 112, 3, 5, 1],
                [6, 192, 4, 5, 2],
                [6, 320, 1, 3, 1],
            ]
            first_stride = 1
        else:
            # Original for larger images
            base_config = [
                [1, 16, 1, 3, 1],
                [6, 24, 2, 3, 2],
                [6, 40, 2, 5, 2],
                [6, 80, 3, 3, 2],
                [6, 112, 3, 5, 1],
                [6, 192, 4, 5, 2],
                [6, 320, 1, 3, 1],
            ]
            first_stride = 2
        
        # Scale configuration
        def round_filters(filters: int) -> int:
            """Scale number of filters based on width multiplier."""
            filters *= width_mult
            new_filters = max(8, int(filters + 4) // 8 * 8)
            if new_filters < 0.9 * filters:
                new_filters += 8
            return int(new_filters)
        
        def round_repeats(repeats: int) -> int:
            """Scale number of layers based on depth multiplier."""
            return int(math.ceil(depth_mult * repeats))
        
        # Build stem
        stem_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, first_stride, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Build blocks
        blocks = []
        in_channels = stem_channels
        total_blocks = sum(round_repeats(cfg[2]) for cfg in base_config)
        block_idx = 0
        
        for expand_ratio, channels, num_blocks, kernel_size, stride in base_config:
            out_channels = round_filters(channels)
            num_blocks = round_repeats(num_blocks)
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                    drop_connect_rate=drop_rate,
                    use_se=use_se
                ))
                
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Build head
        head_channels = round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_channels, num_classes)
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def efficientnet_lite0(num_classes: int = 10, input_size: int = 32, 
                       **kwargs) -> EfficientNetLite:
    """EfficientNet-Lite0 (baseline)."""
    return EfficientNetLite(
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.0,
        input_size=input_size,
        **kwargs
    )


def efficientnet_lite_small(num_classes: int = 10, input_size: int = 32,
                            **kwargs) -> EfficientNetLite:
    """Smaller EfficientNet-Lite for edge deployment."""
    return EfficientNetLite(
        num_classes=num_classes,
        width_mult=0.5,
        depth_mult=0.5,
        input_size=input_size,
        use_se=False,  # Disable SE for max efficiency
        **kwargs
    )
