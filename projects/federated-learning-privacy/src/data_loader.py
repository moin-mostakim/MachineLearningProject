"""Data loading with non-IID partitioning for federated learning."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticCIFAR10(Dataset):
    """Synthetic CIFAR-10-like dataset for FL experiments."""
    
    def __init__(self, num_samples: int = 5000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.num_classes = 10
        
        # Generate synthetic 32x32x3 images
        self.data = np.random.randn(num_samples, 3, 32, 32).astype(np.float32) * 0.2 + 0.5
        self.targets = np.random.randint(0, 10, num_samples)
        
        # Add class-specific patterns
        for i in range(num_samples):
            label = self.targets[i]
            # Add class-specific bias to make it learnable
            self.data[i, label % 3, :, :] += 0.1 * (label / 10)
        
        self.data = np.clip(self.data, 0, 1)
        logger.info(f"Created synthetic dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.data[idx]), int(self.targets[idx])


def create_non_iid_partitions(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Create non-IID data partitions using Dirichlet distribution.
    
    Reference: Hsu et al. (2019) "Measuring the Effects of Non-Identical Data Distribution"
    """
    np.random.seed(seed)
    
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(targets))
    
    # Dirichlet distribution for label proportions
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Partition indices
    client_indices = {i: [] for i in range(num_clients)}
    
    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        
        # Split class indices according to Dirichlet proportions
        proportions = label_distribution[class_idx]
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        current = 0
        for client_idx, num_samples in enumerate(splits):
            client_indices[client_idx].extend(
                class_indices[current:current + num_samples].tolist()
            )
            current += num_samples
    
    # Log distribution
    for client_idx, indices in client_indices.items():
        client_labels = [targets[i] for i in indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        logger.debug(f"Client {client_idx}: {len(indices)} samples, labels: {label_counts}")
    
    return client_indices


def create_client_dataloaders(
    config,
    seed: int = 42
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """Create dataloaders for each client and a test loader."""
    
    total_samples = config.federated.num_clients * config.data.samples_per_client
    train_dataset = SyntheticCIFAR10(int(total_samples * 0.8), seed)
    test_dataset = SyntheticCIFAR10(int(total_samples * 0.2), seed + 1000)
    
    if config.data.iid:
        # IID: random split
        indices = list(range(len(train_dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        samples_per_client = len(indices) // config.federated.num_clients
        client_indices = {
            i: indices[i*samples_per_client:(i+1)*samples_per_client]
            for i in range(config.federated.num_clients)
        }
    else:
        # Non-IID: Dirichlet partitioning
        client_indices = create_non_iid_partitions(
            train_dataset, config.federated.num_clients, 
            config.data.alpha, seed
        )
    
    client_loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(train_dataset, indices)
        client_loaders[client_id] = DataLoader(
            subset, batch_size=config.training.batch_size, shuffle=True
        )
    
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    return client_loaders, test_loader
