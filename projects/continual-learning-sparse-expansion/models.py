"""
Model architectures for Continual Learning experiments.

Implements:
- Base CNN for CIFAR-100
- Base MLP for Permuted MNIST
- Dynamic Sparse Expansion layers
- Multi-head output for task-incremental learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import config


class SparseLinear(nn.Module):
    """
    Linear layer with learnable sparse mask.
    
    Implements dynamic sparse expansion where new connections
    can be added sparsely for new tasks.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = None,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity or config.SPARSITY_RATIO
        
        # Dense weight matrix
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Binary mask for sparsity (not a parameter - updated manually)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
        # Track which weights belong to which task
        self.register_buffer('task_mask', torch.zeros(out_features, in_features))
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sparse mask during forward pass
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)
    
    def expand_sparse(self, task_id: int, num_new_connections: int = None):
        """
        Add new sparse connections for a new task.
        
        Uses gradient-based growth if enabled, otherwise random selection.
        """
        if num_new_connections is None:
            # Default: add sparsity_ratio fraction of total connections
            total_connections = self.in_features * self.out_features
            num_new_connections = int(total_connections * self.sparsity)
        
        # Find currently inactive (masked out) connections
        inactive_mask = (self.mask == 0)
        inactive_indices = torch.nonzero(inactive_mask, as_tuple=False)
        
        if len(inactive_indices) == 0:
            return 0  # No room for expansion
        
        # Randomly select connections to activate
        num_to_add = min(num_new_connections, len(inactive_indices))
        selected = torch.randperm(len(inactive_indices))[:num_to_add]
        
        for idx in selected:
            i, j = inactive_indices[idx]
            self.mask[i, j] = 1
            self.task_mask[i, j] = task_id
            # Initialize new weights with small values
            self.weight.data[i, j] = torch.randn(1).item() * config.SPARSE_INIT_STD
        
        return num_to_add
    
    def get_task_params(self, task_id: int) -> torch.Tensor:
        """Get weights associated with a specific task."""
        task_weights = self.weight * (self.task_mask == task_id).float()
        return task_weights
    
    def freeze_task(self, task_id: int):
        """Freeze weights associated with a specific task (for inference)."""
        pass  # Implemented via gradient masking in training
    
    def get_sparsity(self) -> float:
        """Return current sparsity level (fraction of zero weights)."""
        return 1.0 - (self.mask.sum() / self.mask.numel()).item()
    
    def get_num_active_params(self) -> int:
        """Return number of active (non-masked) parameters."""
        return int(self.mask.sum().item())


class DynamicSparseExpansionBlock(nn.Module):
    """
    A block that can expand sparsely for new tasks.
    
    Contains:
    - Sparse linear layer
    - Batch normalization
    - Activation
    - Optional task-specific adapters
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adapter: bool = None,
        adapter_dim: int = None
    ):
        super().__init__()
        self.use_adapter = use_adapter if use_adapter is not None else config.USE_TASK_ADAPTERS
        self.adapter_dim = adapter_dim or config.ADAPTER_BOTTLENECK_DIM
        
        self.sparse_linear = SparseLinear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # Task-specific adapters (lightweight per-task parameters)
        if self.use_adapter:
            self.adapters = nn.ModuleDict()
    
    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        out = self.sparse_linear(x)
        out = self.bn(out)
        out = self.activation(out)
        
        # Apply task adapter if available
        if self.use_adapter and task_id is not None:
            adapter_key = f"task_{task_id}"
            if adapter_key in self.adapters:
                out = out + self.adapters[adapter_key](out)
        
        out = self.dropout(out)
        return out
    
    def add_task_adapter(self, task_id: int):
        """Add a new adapter for a task."""
        if not self.use_adapter:
            return
        
        adapter_key = f"task_{task_id}"
        if adapter_key not in self.adapters:
            in_features = self.sparse_linear.out_features
            self.adapters[adapter_key] = nn.Sequential(
                nn.Linear(in_features, self.adapter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.adapter_dim, in_features)
            )
            # Initialize adapter output to zero (residual start)
            nn.init.zeros_(self.adapters[adapter_key][-1].weight)
            nn.init.zeros_(self.adapters[adapter_key][-1].bias)
    
    def expand_for_task(self, task_id: int) -> int:
        """Expand sparse connections and add adapter for new task."""
        num_expanded = self.sparse_linear.expand_sparse(task_id)
        self.add_task_adapter(task_id)
        return num_expanded


class MultiHeadOutput(nn.Module):
    """
    Multi-head output layer for task-incremental learning.
    
    Each task has its own output head to prevent interference.
    """
    
    def __init__(self, in_features: int, num_classes_per_task: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes_per_task = num_classes_per_task
        self.heads = nn.ModuleDict()
    
    def add_head(self, task_id: int):
        """Add a new output head for a task."""
        head_key = f"head_{task_id}"
        if head_key not in self.heads:
            self.heads[head_key] = nn.Linear(
                self.in_features, self.num_classes_per_task
            )
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        head_key = f"head_{task_id}"
        if head_key not in self.heads:
            raise ValueError(f"No head for task {task_id}")
        return self.heads[head_key](x)
    
    def get_all_outputs(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get outputs from all heads."""
        outputs = {}
        for head_key, head in self.heads.items():
            task_id = int(head_key.split("_")[1])
            outputs[task_id] = head(x)
        return outputs


class ContinualCNN(nn.Module):
    """
    CNN backbone for CIFAR-100 continual learning.
    
    Supports:
    - Standard training (fine-tuning baseline)
    - EWC regularization
    - Dynamic sparse expansion
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes_per_task: int = None,
        use_sparse_expansion: bool = False
    ):
        super().__init__()
        self.use_sparse_expansion = use_sparse_expansion
        self.num_classes_per_task = num_classes_per_task or config.CLASSES_PER_TASK
        
        # CNN backbone (shared across tasks)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Compute feature size after conv layers
        self.feature_size = 128 * 4 * 4  # For 32x32 input
        
        # Classifier with optional sparse expansion
        if use_sparse_expansion:
            self.classifier = nn.Sequential(
                DynamicSparseExpansionBlock(self.feature_size, 256),
                DynamicSparseExpansionBlock(256, 256),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
        
        # Multi-head output
        self.output = MultiHeadOutput(256, self.num_classes_per_task)
        
        # Track current task
        self.current_task = 0
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: int = None
    ) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task
        
        # Feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Classification with task-specific processing
        if self.use_sparse_expansion:
            for block in self.classifier:
                features = block(features, task_id)
        else:
            features = self.classifier(features)
        
        # Task-specific output head
        output = self.output(features, task_id)
        return output
    
    def add_task(self, task_id: int):
        """Prepare model for a new task."""
        self.current_task = task_id
        self.output.add_head(task_id)
        
        if self.use_sparse_expansion:
            total_expanded = 0
            for block in self.classifier:
                if isinstance(block, DynamicSparseExpansionBlock):
                    total_expanded += block.expand_for_task(task_id)
            return total_expanded
        return 0
    
    def get_feature_extractor(self) -> nn.Module:
        """Return feature extraction layers."""
        return self.features
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            "features": sum(p.numel() for p in self.features.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "output_heads": sum(p.numel() for p in self.output.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }
        
        if self.use_sparse_expansion:
            active_params = 0
            for block in self.classifier:
                if isinstance(block, DynamicSparseExpansionBlock):
                    active_params += block.sparse_linear.get_num_active_params()
            counts["active_sparse"] = active_params
        
        return counts


class ContinualMLP(nn.Module):
    """
    MLP for Permuted MNIST continual learning.
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = None,
        num_classes: int = 10,
        use_sparse_expansion: bool = False
    ):
        super().__init__()
        self.use_sparse_expansion = use_sparse_expansion
        hidden_sizes = hidden_sizes or config.BASE_HIDDEN_SIZES
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            if use_sparse_expansion:
                layers.append(DynamicSparseExpansionBlock(prev_size, hidden_size))
            else:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(config.DROPOUT_RATE))
            prev_size = hidden_size
        
        self.layers = nn.Sequential(*layers) if not use_sparse_expansion else nn.ModuleList(layers)
        self.output = nn.Linear(hidden_sizes[-1], num_classes)
        self.current_task = 0
    
    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task
        
        x = x.view(x.size(0), -1)
        
        if self.use_sparse_expansion:
            for layer in self.layers:
                x = layer(x, task_id)
        else:
            x = self.layers(x)
        
        return self.output(x)
    
    def add_task(self, task_id: int):
        """Prepare model for a new task."""
        self.current_task = task_id
        
        if self.use_sparse_expansion:
            total_expanded = 0
            for layer in self.layers:
                if isinstance(layer, DynamicSparseExpansionBlock):
                    total_expanded += layer.expand_for_task(task_id)
            return total_expanded
        return 0
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        return {
            "layers": sum(p.numel() for p in self.layers.parameters()),
            "output": sum(p.numel() for p in self.output.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }


def get_model(
    dataset_name: str = None,
    method: str = "finetune",
    input_size: Tuple = None,
    num_classes: int = None
) -> nn.Module:
    """
    Factory function to create appropriate model.
    
    Args:
        dataset_name: "split_cifar100" or "permuted_mnist"
        method: "finetune", "ewc", or "sparse_expansion"
        input_size: Input dimensions
        num_classes: Number of output classes per task
    """
    dataset_name = dataset_name or config.DATASET
    use_sparse = (method == "sparse_expansion")
    
    if dataset_name == "split_cifar100":
        model = ContinualCNN(
            input_channels=input_size[0] if input_size else 3,
            num_classes_per_task=num_classes or config.CLASSES_PER_TASK,
            use_sparse_expansion=use_sparse
        )
    elif dataset_name == "permuted_mnist":
        model = ContinualMLP(
            input_size=np.prod(input_size) if input_size else 784,
            num_classes=num_classes or 10,
            use_sparse_expansion=use_sparse
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return model.to(config.DEVICE)


if __name__ == "__main__":
    # Test model creation
    print("Testing ContinualCNN...")
    model = ContinualCNN(use_sparse_expansion=True)
    model.add_task(0)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, task_id=0)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {model.count_parameters()}")
    
    print("\nTesting ContinualMLP...")
    mlp = ContinualMLP(use_sparse_expansion=True)
    mlp.add_task(0)
    x = torch.randn(2, 1, 28, 28)
    out = mlp(x, task_id=0)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {mlp.count_parameters()}")
