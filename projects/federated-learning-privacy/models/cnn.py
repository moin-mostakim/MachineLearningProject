"""CNN architectures for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for FEMNIST (28x28 grayscale images).
    Architecture from McMahan et al. 2017 (Communication-Efficient Learning).
    
    Parameters: ~100K (suitable for FL with limited communication)
    """
    
    def __init__(self, num_classes: int = 62, dropout: float = 0.2, **kwargs):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # After two pooling: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def get_layer_names(self):
        """Return layer names for personalization."""
        return ["conv1", "conv2", "fc1", "fc2"]


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 (32x32 RGB images).
    Slightly deeper architecture for color images.
    
    Architecture follows common FL baselines with:
    - 2 conv blocks with batch norm
    - 2 fully connected layers
    
    Parameters: ~1.2M
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.2, **kwargs):
        super().__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # After two pooling: 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding before classifier."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x
    
    def get_layer_names(self):
        """Return layer names for personalization."""
        return ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2", "classifier"]
