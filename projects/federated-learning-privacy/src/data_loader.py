"""
Data loading with non-IID partitioning for federated learning.

Dataset: CIFAR-10
- Source: torchvision (official) or HuggingFace Hub (uoft-cs/cifar10)
- URL: https://huggingface.co/datasets/uoft-cs/cifar10
- Official: https://www.cs.toronto.edu/~kriz/cifar.html
- License: MIT
- Size: 163MB (50K train + 10K test, 10 classes, 32x32 RGB)
- Reference: Krizhevsky (2009): "Learning Multiple Layers of Features from Tiny Images"
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple
import logging
import os

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CIFAR-10 statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class HuggingFaceCIFAR10(Dataset):
    """CIFAR-10 from HuggingFace Hub as fallback."""
    
    def __init__(self, split: str = "train", transform=None, max_samples=None):
        self.transform = transform
        logger.info(f"Loading CIFAR-10 from HuggingFace Hub ({split})...")
        ds = load_dataset("uoft-cs/cifar10", split=split)
        
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        
        self.data = []
        self.targets = []
        for sample in ds:
            self.data.append(np.array(sample["img"]))
            self.targets.append(sample["label"])
        
        self.data = np.stack(self.data)
        self.targets = np.array(self.targets)
        logger.info(f"Loaded {len(self.data)} CIFAR-10 samples from HuggingFace")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, int(self.targets[idx])


def get_cifar10_dataset(train: bool = True, data_dir: str = "./data", max_samples=None):
    """
    Load CIFAR-10 from torchvision with HuggingFace fallback.
    
    ALWAYS uses real CIFAR-10 data (no synthetic data).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
        logger.info(f"Loaded CIFAR-10 {'train' if train else 'test'} ({len(dataset)} samples) from torchvision")
        
        if max_samples and max_samples < len(dataset):
            indices = list(range(max_samples))
            dataset = Subset(dataset, indices)
        
        return dataset
        
    except Exception as e:
        logger.warning(f"torchvision CIFAR-10 failed: {e}")
        if HAS_DATASETS:
            split = "train" if train else "test"
            return HuggingFaceCIFAR10(split=split, transform=transform, max_samples=max_samples)
        raise RuntimeError(
            "Cannot load CIFAR-10. Install datasets library: pip install datasets"
        )
    
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
    """
    Create dataloaders for each client and a test loader.
    
    Uses REAL CIFAR-10 data from torchvision or HuggingFace.
    """
    # Load real CIFAR-10 data
    train_dataset = get_cifar10_dataset(train=True, data_dir="./data")
    test_dataset = get_cifar10_dataset(train=False, data_dir="./data")
    
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
