"""ResNet architecture for federated learning."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """
    ResNet18 adapted for CIFAR-10/100.
    
    Modifications from ImageNet version:
    - First conv: 3x3 instead of 7x7
    - No max pooling after first conv
    - Smaller input size (32x32 vs 224x224)
    
    Parameters: ~11.2M
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False, **kwargs):
        super().__init__()
        
        # Load base ResNet18
        self.model = models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        
        # Modify first conv for small images (CIFAR)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove max pooling
        self.model.maxpool = nn.Identity()
        
        # Modify final FC layer
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding before final FC."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_layer_names(self):
        """Return layer names for personalization."""
        return ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
