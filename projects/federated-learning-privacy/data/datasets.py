"""Dataset loading utilities for federated learning."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Dict
import numpy as np


class FederatedDataset:
    """
    Wrapper for federated dataset with client data splits.
    
    Supports non-IID partitioning for heterogeneous FL experiments.
    """
    
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        client_indices: Dict[int, List[int]],
        num_clients: int
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_indices = client_indices
        self.num_clients = num_clients
        
    def get_client_dataloader(
        self, 
        client_id: int, 
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 2
    ) -> DataLoader:
        """Get DataLoader for a specific client."""
        if client_id not in self.client_indices:
            raise ValueError(f"Client {client_id} not found. Available: {list(self.client_indices.keys())}")
        
        indices = self.client_indices[client_id]
        client_dataset = Subset(self.train_dataset, indices)
        
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(
        self,
        batch_size: int = 128,
        num_workers: int = 2
    ) -> DataLoader:
        """Get DataLoader for test set (global evaluation)."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_client_data_sizes(self) -> Dict[int, int]:
        """Get number of samples per client."""
        return {cid: len(indices) for cid, indices in self.client_indices.items()}
    
    def get_client_label_distribution(self) -> Dict[int, Dict[int, int]]:
        """Get label distribution per client for analysis."""
        distributions = {}
        
        # Get all labels
        if hasattr(self.train_dataset, 'targets'):
            all_labels = np.array(self.train_dataset.targets)
        else:
            all_labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        for client_id, indices in self.client_indices.items():
            client_labels = all_labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            distributions[client_id] = dict(zip(unique.tolist(), counts.tolist()))
        
        return distributions


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms for CIFAR-10."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    return train_transform, test_transform


def get_femnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms for FEMNIST."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data"
) -> Tuple[Dataset, Dataset]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of dataset ("cifar10", "mnist", "femnist")
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        train_transform, test_transform = get_cifar10_transforms()
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
    elif dataset_name == "mnist":
        train_transform, test_transform = get_femnist_transforms()
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
    elif dataset_name == "mnist_small":
        # Small MNIST subset for fast debugging
        train_transform, test_transform = get_femnist_transforms()
        
        full_train = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        full_test = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
        # Take only first 3000 train and 500 test samples
        from torch.utils.data import Subset
        train_dataset = Subset(full_train, list(range(3000)))
        test_dataset = Subset(full_test, list(range(500)))
        
    elif dataset_name == "femnist":
        # FEMNIST is EMNIST with federated split
        # We use EMNIST ByClass (62 classes) as approximation
        train_transform, test_transform = get_femnist_transforms()
        
        train_dataset = torchvision.datasets.EMNIST(
            root=data_dir,
            split='byclass',
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.EMNIST(
            root=data_dir,
            split='byclass',
            train=False,
            download=True,
            transform=test_transform
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset
