"""Dataset loading and preprocessing."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np


def get_cifar10_transforms(train: bool = True, normalize: bool = True):
    """Get CIFAR-10 transforms."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    if train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.ToTensor(),
        ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(transform_list)


class SyntheticCIFAR10(Dataset):
    """Synthetic CIFAR-10-like dataset for fast testing."""
    
    def __init__(self, num_samples: int = 1000, train: bool = True):
        self.num_samples = num_samples
        self.train = train
        # Generate synthetic data (CIFAR-10 like: 32x32x3, 10 classes)
        self.data = torch.randn(num_samples, 3, 32, 32) * 0.5 + 0.5
        self.data = self.data.clamp(0, 1)
        self.targets = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = False,  # Don't normalize for adversarial training
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize images
        subset_size: If specified, use only this many samples (for debugging)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform = get_cifar10_transforms(train=True, normalize=normalize)
    test_transform = get_cifar10_transforms(train=False, normalize=normalize)
    
    # Use synthetic CIFAR-10 for faster testing (real download is 170MB and very slow)
    print("Using synthetic CIFAR-10-like dataset for fast testing")
    train_size = subset_size if subset_size else 5000
    test_size = train_size // 5
    train_dataset = SyntheticCIFAR10(train_size, train=True)
    test_dataset = SyntheticCIFAR10(test_size, train=False)
    
    # Subset for debugging
    if subset_size is not None:
        train_indices = np.random.choice(len(train_dataset), min(subset_size, len(train_dataset)), replace=False)
        test_indices = np.random.choice(len(test_dataset), min(subset_size // 5, len(test_dataset)), replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


class CIFAR10WithTextLabels(Dataset):
    """CIFAR-10 dataset with text labels for VLM training."""
    
    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform=None,
        download: bool = True
    ):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        self.class_names = self.CLASSES
    
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        text_label = self.class_names[label]
        return image, label, text_label


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders from config."""
    if config.data.dataset == "cifar10":
        subset_size = None
        if config.training.debug:
            subset_size = 1000  # Small subset for debugging
        
        return get_cifar10_loaders(
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            normalize=False,  # Keep in [0, 1] for adversarial attacks
            subset_size=subset_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        subset_size=1000
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
