"""
Shared Encoder Module for Multi-Task DR Learning.

Implements a ResNet-style encoder that extracts multi-scale features
for both classification and segmentation tasks.

References:
- He et al. (2016) "Deep Residual Learning for Image Recognition"
- Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SharedEncoder(nn.Module):
    """
    Shared encoder for multi-task learning.
    
    Extracts hierarchical features at multiple scales for use by
    both classification and segmentation decoder heads.
    """
    
    def __init__(self, in_channels=3, channels=None):
        super().__init__()
        
        if channels is None:
            channels = config.ENCODER_CHANNELS
        
        self.channels = channels
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(channels[0], channels[0], num_blocks=2, stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks=2, stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks=2, stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], num_blocks=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(channels[3], config.BOTTLENECK_DIM),
            ConvBlock(config.BOTTLENECK_DIM, config.BOTTLENECK_DIM)
        )
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass returning multi-scale features.
        
        Returns:
            features: List of feature maps at different scales
            bottleneck: Final bottleneck features for classification
        """
        # Initial downsampling
        x = self.stem(x)  # H/4
        
        # Encoder stages with skip connections
        f1 = self.stage1(x)   # H/4
        f2 = self.stage2(f1)  # H/8
        f3 = self.stage3(f2)  # H/16
        f4 = self.stage4(f3)  # H/32
        
        # Bottleneck
        bottleneck = self.bottleneck(f4)
        
        features = [f1, f2, f3, f4]
        
        return features, bottleneck


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test encoder
    encoder = SharedEncoder()
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    features, bottleneck = encoder(x)
    
    print("Encoder output shapes:")
    for i, f in enumerate(features):
        print(f"  Feature {i+1}: {f.shape}")
    print(f"  Bottleneck: {bottleneck.shape}")
    print(f"\nTotal parameters: {count_parameters(encoder):,}")
