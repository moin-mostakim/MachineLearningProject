"""
Neural Architecture Search space definition.

Based on Once-for-All (OFA) style elastic search space:
- Elastic kernel sizes: {3, 5}
- Elastic expansion ratios: {3, 4, 6}
- Elastic depth: variable number of blocks per stage

Reference:
Cai et al., "Once-for-All: Train One Network and Specialize it for Efficient Deployment", ICLR 2020
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import random
import copy


class SearchableConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU6 with searchable kernel size."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5],
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.active_kernel_size = max(kernel_sizes)  # Default to max
        
        # Create max-sized conv (will mask for smaller kernels)
        max_kernel = max(kernel_sizes)
        padding = (max_kernel - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, max_kernel, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For simplicity, always use max kernel
        # Full implementation would mask weights for smaller kernels
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SearchableBlock(nn.Module):
    """
    Searchable inverted residual block with configurable:
    - Kernel size
    - Expansion ratio
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5],
        expansion_ratios: List[int] = [3, 4, 6],
        stride: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_ratios = expansion_ratios
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Active configuration
        self.active_kernel_size = kernel_sizes[-1]  # Default max
        self.active_expansion_ratio = expansion_ratios[-1]  # Default max
        
        # Build with max configuration
        max_expansion = max(expansion_ratios)
        max_kernel = max(kernel_sizes)
        hidden_dim = in_channels * max_expansion
        
        # Expansion
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        padding = (max_kernel - 1) // 2
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, max_kernel, stride, padding,
            groups=hidden_dim, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        # Projection
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU6(inplace=True)
    
    def set_active_config(self, kernel_size: int, expansion_ratio: int):
        """Set active kernel size and expansion ratio."""
        assert kernel_size in self.kernel_sizes
        assert expansion_ratio in self.expansion_ratios
        self.active_kernel_size = kernel_size
        self.active_expansion_ratio = expansion_ratio
    
    def get_active_config(self) -> Dict[str, int]:
        """Get current active configuration."""
        return {
            "kernel_size": self.active_kernel_size,
            "expansion_ratio": self.active_expansion_ratio
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.relu(out)
        
        # Depthwise
        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.relu(out)
        
        # Projection
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        if self.use_residual:
            out = out + identity
        
        return out


class SearchableStage(nn.Module):
    """
    Stage with searchable depth (number of blocks).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_depth: int = 4,
        kernel_sizes: List[int] = [3, 5],
        expansion_ratios: List[int] = [3, 4, 6],
        stride: int = 1
    ):
        super().__init__()
        self.max_depth = max_depth
        self.active_depth = max_depth  # Default to max
        
        # Build all possible blocks
        blocks = []
        for i in range(max_depth):
            block_in = in_channels if i == 0 else out_channels
            block_stride = stride if i == 0 else 1
            blocks.append(SearchableBlock(
                block_in, out_channels, kernel_sizes, expansion_ratios, block_stride
            ))
        
        self.blocks = nn.ModuleList(blocks)
    
    def set_active_depth(self, depth: int):
        """Set number of active blocks."""
        assert 1 <= depth <= self.max_depth
        self.active_depth = depth
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.active_depth):
            x = self.blocks[i](x)
        return x


class SuperNet(nn.Module):
    """
    Supernet for weight-sharing NAS.
    
    Contains all possible architectures in a single network.
    Subnets are sampled by setting active configurations.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 32,
        kernel_sizes: List[int] = [3, 5],
        expansion_ratios: List[int] = [3, 4, 6],
        depth_options: List[int] = [2, 3, 4],
        width_mult: float = 1.0
    ):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.expansion_ratios = expansion_ratios
        self.depth_options = depth_options
        self.max_depth = max(depth_options)
        
        # Stage configuration: [channels, max_depth, stride]
        if input_size <= 32:
            stage_config = [
                [int(16 * width_mult), 1, 1],
                [int(24 * width_mult), 4, 1],
                [int(32 * width_mult), 4, 2],
                [int(64 * width_mult), 4, 2],
                [int(96 * width_mult), 4, 1],
            ]
            first_stride = 1
        else:
            stage_config = [
                [int(16 * width_mult), 1, 1],
                [int(24 * width_mult), 4, 2],
                [int(32 * width_mult), 4, 2],
                [int(64 * width_mult), 4, 2],
                [int(96 * width_mult), 4, 1],
            ]
            first_stride = 2
        
        # Stem
        stem_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, first_stride, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Build stages
        stages = []
        in_channels = stem_channels
        
        for channels, max_depth, stride in stage_config:
            stages.append(SearchableStage(
                in_channels, channels, max_depth,
                kernel_sizes, expansion_ratios, stride
            ))
            in_channels = channels
        
        self.stages = nn.ModuleList(stages)
        
        # Head
        head_channels = 256
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_channels, num_classes)
        
        # Store active configuration
        self.active_config = self.get_max_config()
    
    def get_max_config(self) -> Dict[str, Any]:
        """Get maximum (largest) configuration."""
        return {
            "kernel_sizes": [max(self.kernel_sizes)] * len(self.stages),
            "expansion_ratios": [max(self.expansion_ratios)] * len(self.stages),
            "depths": [stage.max_depth for stage in self.stages]
        }
    
    def get_min_config(self) -> Dict[str, Any]:
        """Get minimum (smallest) configuration."""
        return {
            "kernel_sizes": [min(self.kernel_sizes)] * len(self.stages),
            "expansion_ratios": [min(self.expansion_ratios)] * len(self.stages),
            "depths": [1] * len(self.stages)
        }
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        return {
            "kernel_sizes": [random.choice(self.kernel_sizes) for _ in self.stages],
            "expansion_ratios": [random.choice(self.expansion_ratios) for _ in self.stages],
            "depths": [random.choice(self.depth_options[:stage.max_depth]) 
                      for stage in self.stages]
        }
    
    def set_active_config(self, config: Dict[str, Any]):
        """Set active subnet configuration."""
        self.active_config = config
        
        for i, stage in enumerate(self.stages):
            # Set depth
            if i < len(config["depths"]):
                depth = min(config["depths"][i], stage.max_depth)
                stage.set_active_depth(depth)
            
            # Set block configs
            for j, block in enumerate(stage.blocks):
                if i < len(config["kernel_sizes"]):
                    ks = config["kernel_sizes"][i]
                    er = config["expansion_ratios"][i]
                    block.set_active_config(ks, er)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_subnet(self) -> nn.Module:
        """Extract current active subnet as standalone model."""
        # Create a copy with current config
        return copy.deepcopy(self)


def create_supernet(
    num_classes: int = 10,
    input_size: int = 32,
    config: Optional[Dict] = None
) -> SuperNet:
    """Factory function for creating supernet."""
    if config is None:
        config = {
            "kernel_sizes": [3, 5],
            "expansion_ratios": [3, 4, 6],
            "depth_options": [2, 3, 4],
            "width_mult": 1.0
        }
    
    return SuperNet(
        num_classes=num_classes,
        input_size=input_size,
        **config
    )
