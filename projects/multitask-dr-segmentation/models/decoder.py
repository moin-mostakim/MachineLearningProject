"""
Decoder Modules for Multi-Task DR Learning.

Implements:
- U-Net style segmentation decoder with attention gates
- Classification head with global pooling

References:
- Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Oktay et al. (2018) "Attention U-Net: Learning Where to Look for the Pancreas"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AttentionGate(nn.Module):
    """Attention gate for focusing decoder on relevant encoder features."""
    
    def __init__(self, gate_channels, feat_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = feat_channels // 2
        
        self.W_g = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.W_x = nn.Conv2d(feat_channels, inter_channels, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal (from decoder)
            x: Skip connection features (from encoder)
        """
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and optional attention."""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = AttentionGate(in_channels, skip_channels)
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder block
            skip: Skip connection from encoder
        """
        if self.use_attention:
            skip = self.attention(x, skip)
        
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class SegmentationDecoder(nn.Module):
    """U-Net style decoder for lesion segmentation."""
    
    def __init__(self, encoder_channels=None, decoder_channels=None, 
                 num_classes=None, use_attention=True, bottleneck_dim=None):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = config.ENCODER_CHANNELS
        if decoder_channels is None:
            decoder_channels = config.DECODER_CHANNELS
        if num_classes is None:
            num_classes = config.NUM_LESION_TYPES
        if bottleneck_dim is None:
            bottleneck_dim = config.BOTTLENECK_DIM
        
        # Decoder blocks (from bottleneck to full resolution)
        self.dec4 = DecoderBlock(bottleneck_dim, encoder_channels[3], decoder_channels[0], use_attention)
        self.dec3 = DecoderBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1], use_attention)
        self.dec2 = DecoderBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2], use_attention)
        self.dec1 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3], use_attention)
        
        # Final upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], 4, stride=4),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Output head
        self.output = nn.Conv2d(decoder_channels[3], num_classes, 1)
    
    def forward(self, features, bottleneck):
        """
        Args:
            features: List of encoder features [f1, f2, f3, f4]
            bottleneck: Bottleneck features
        """
        f1, f2, f3, f4 = features
        
        x = self.dec4(bottleneck, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        
        x = self.final_upsample(x)
        x = self.output(x)
        
        return x


class ClassificationHead(nn.Module):
    """Classification head for DR grading."""
    
    def __init__(self, in_channels=None, hidden_dim=None, num_classes=None, dropout=None):
        super().__init__()
        
        if in_channels is None:
            in_channels = config.BOTTLENECK_DIM
        if hidden_dim is None:
            hidden_dim = config.CLASSIFIER_HIDDEN
        if num_classes is None:
            num_classes = config.NUM_CLASSES
        if dropout is None:
            dropout = config.DROPOUT_RATE
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, bottleneck):
        """
        Args:
            bottleneck: Bottleneck features from encoder
        """
        x = self.global_pool(bottleneck)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test decoder
    batch_size = 2
    h, w = config.IMAGE_SIZE // 4, config.IMAGE_SIZE // 4
    
    # Simulate encoder outputs
    features = [
        torch.randn(batch_size, config.ENCODER_CHANNELS[0], h, w),
        torch.randn(batch_size, config.ENCODER_CHANNELS[1], h // 2, w // 2),
        torch.randn(batch_size, config.ENCODER_CHANNELS[2], h // 4, w // 4),
        torch.randn(batch_size, config.ENCODER_CHANNELS[3], h // 8, w // 8),
    ]
    bottleneck = torch.randn(batch_size, config.BOTTLENECK_DIM, h // 8, w // 8)
    
    # Test segmentation decoder
    seg_decoder = SegmentationDecoder()
    seg_out = seg_decoder(features, bottleneck)
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Segmentation decoder parameters: {count_parameters(seg_decoder):,}")
    
    # Test classification head
    cls_head = ClassificationHead()
    cls_out = cls_head(bottleneck)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Classification head parameters: {count_parameters(cls_head):,}")
