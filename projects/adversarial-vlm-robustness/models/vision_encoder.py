"""Vision encoders for VLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-10 (smaller input size)."""
    
    def __init__(self, block, num_blocks: list, num_classes: int = 10, output_dim: int = 512):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18(output_dim: int = 512) -> ResNet:
    """ResNet-18 for CIFAR."""
    return ResNet(BasicBlock, [2, 2, 2, 2], output_dim=output_dim)


def resnet34(output_dim: int = 512) -> ResNet:
    """ResNet-34 for CIFAR."""
    return ResNet(BasicBlock, [3, 4, 6, 3], output_dim=output_dim)


class VisionEncoder(nn.Module):
    """Vision encoder wrapper with projection head."""
    
    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 512,
        projection_dim: int = 256,
        use_projection: bool = True
    ):
        super().__init__()
        
        if backbone == "resnet18":
            self.backbone = resnet18(output_dim=output_dim)
        elif backbone == "resnet34":
            self.backbone = resnet34(output_dim=output_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, projection_dim)
            )
        
        self.output_dim = projection_dim if use_projection else output_dim
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)
        
        if self.use_projection:
            projected = self.projection(features)
            if return_features:
                return projected, features
            return projected
        
        return features


def get_vision_encoder(config) -> VisionEncoder:
    """Factory function to create vision encoder from config."""
    return VisionEncoder(
        backbone=config.model.vision_encoder,
        output_dim=config.model.vision_dim,
        projection_dim=config.model.projection_dim,
        use_projection=True
    )


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


if __name__ == "__main__":
    model = VisionEncoder(backbone="resnet18", output_dim=512, projection_dim=256)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    params = count_parameters(model)
    print(f"Parameters: {params}")
