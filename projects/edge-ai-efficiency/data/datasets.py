"""
Dataset loading utilities for edge AI efficiency experiments.

Datasets:
---------
CIFAR-10 (Primary):
- Source: torchvision / HuggingFace Hub (uoft-cs/cifar10)
- URL: https://huggingface.co/datasets/uoft-cs/cifar10
- Official: https://www.cs.toronto.edu/~kriz/cifar.html
- License: MIT
- Size: 163MB (50K train + 10K test, 10 classes, 32x32 RGB)

ImageNette (ImageNet Subset):
- Source: fast.ai
- URL: https://github.com/fastai/imagenette
- License: Apache 2.0
- Size: 97MB (9,469 train + 3,925 val, 10 classes, 160x160)

References:
- Krizhevsky (2009): "Learning Multiple Layers of Features from Tiny Images"
- Sandler et al. (2018): "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import os
import random

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Dataset configuration
DATASET_CONFIG = {
    "cifar10": {
        "name": "CIFAR-10",
        "huggingface": "uoft-cs/cifar10",
        "url": "https://huggingface.co/datasets/uoft-cs/cifar10",
        "official": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "license": "MIT",
        "size": "163MB",
        "train_samples": 50000,
        "test_samples": 10000,
        "num_classes": 10,
        "image_size": "32x32",
    },
    "imagenette": {
        "name": "ImageNette",
        "url": "https://github.com/fastai/imagenette",
        "license": "Apache 2.0",
        "size": "97MB",
        "train_samples": 9469,
        "val_samples": 3925,
        "num_classes": 10,
        "image_size": "160x160",
    }
}


def get_cifar10_transforms(
    image_size: int = 32,
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, ...] = (0.2470, 0.2435, 0.2616),
    train: bool = True
) -> transforms.Compose:
    """
    Get CIFAR-10 transforms with data augmentation.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        train: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


def get_imagenette_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    train: bool = True
) -> transforms.Compose:
    """
    Get ImageNette transforms with data augmentation.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        train: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


def create_synthetic_cifar10(num_samples: int = 1000, num_classes: int = 10):
    """Create synthetic CIFAR-10-like data for fast testing (opt-in only)."""
    from torch.utils.data import TensorDataset
    
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


class HuggingFaceCIFAR10(torch.utils.data.Dataset):
    """CIFAR-10 loaded from HuggingFace Hub."""
    
    def __init__(self, split: str = "train", transform=None, max_samples: Optional[int] = None):
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        self.transform = transform
        print(f"Loading CIFAR-10 from HuggingFace Hub (split={split})...")
        
        dataset = load_dataset("uoft-cs/cifar10", split=split)
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        self.data = []
        self.targets = []
        
        for item in dataset:
            self.data.append(item['img'])
            self.targets.append(item['label'])
        
        print(f"Loaded {len(self.data)} CIFAR-10 samples from HuggingFace")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = "./data",
    subset_size: Optional[int] = None,
    use_synthetic: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Directory to store/load data
        subset_size: Optional subset size for debugging
        use_synthetic: Use synthetic data instead of real CIFAR-10
        
    Returns:
        Train and test data loaders
    """
    # Use synthetic data if requested or if CIFAR-10 download fails
    if use_synthetic:
        print("Using synthetic CIFAR-10-like data for fast testing")
        train_size = subset_size or 5000
        test_size = train_size // 5
        train_dataset = create_synthetic_cifar10(train_size)
        test_dataset = create_synthetic_cifar10(test_size)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # No workers for synthetic data
            pin_memory=pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )
        
        return train_loader, test_loader
    
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
    except Exception as e:
        print(f"Failed to download CIFAR-10: {e}")
        print("Falling back to synthetic data")
        return get_cifar10_loaders(batch_size, num_workers, pin_memory, data_dir, subset_size, use_synthetic=True)
    
    if subset_size is not None:
        # Random subset for debugging
        train_indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
        test_indices = random.sample(range(len(test_dataset)), min(subset_size // 5, len(test_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_imagenette_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = "./data",
    image_size: int = 224,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNette data loaders.
    Downloads ImageNette if not present.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Directory to store/load data
        image_size: Target image size
        subset_size: Optional subset size for debugging
        
    Returns:
        Train and validation data loaders
    """
    imagenette_dir = os.path.join(data_dir, "imagenette2-160")
    
    # Check if ImageNette exists, if not download
    if not os.path.exists(imagenette_dir):
        print("ImageNette not found. Using CIFAR-10 as fallback...")
        # Fall back to CIFAR-10 with resizing
        return get_cifar10_with_resize(batch_size, num_workers, pin_memory, 
                                       data_dir, image_size, subset_size)
    
    train_transform = get_imagenette_transforms(image_size=image_size, train=True)
    val_transform = get_imagenette_transforms(image_size=image_size, train=False)
    
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(imagenette_dir, "train"),
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(imagenette_dir, "val"),
        transform=val_transform
    )
    
    if subset_size is not None:
        train_indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(subset_size // 5, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_cifar10_with_resize(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = "./data",
    image_size: int = 224,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 with resizing to larger image size (for architecture compatibility).
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Directory to store/load data
        image_size: Target image size
        subset_size: Optional subset size for debugging
        
    Returns:
        Train and test data loaders
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    if subset_size is not None:
        train_indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
        test_indices = random.sample(range(len(test_dataset)), min(subset_size // 5, len(test_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_dataloaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = "./data",
    image_size: Optional[int] = None,
    subset_size: Optional[int] = None,
    use_synthetic: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders based on dataset choice.
    
    Args:
        dataset: Dataset name ("cifar10" or "imagenette")
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Directory to store/load data
        image_size: Optional target image size
        subset_size: Optional subset size for debugging
        
    Returns:
        Train and test/validation data loaders
    """
    if dataset.lower() == "cifar10":
        return get_cifar10_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_dir=data_dir,
            subset_size=subset_size,
            use_synthetic=use_synthetic
        )
    elif dataset.lower() == "imagenette":
        return get_imagenette_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_dir=data_dir,
            image_size=image_size or 224,
            subset_size=subset_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'cifar10' or 'imagenette'")


def get_calibration_loader(
    dataset: str = "cifar10",
    batch_size: int = 32,
    num_samples: int = 1000,
    data_dir: str = "./data",
    use_synthetic: bool = False
) -> DataLoader:
    """
    Get a calibration data loader for quantization.
    Uses training data without augmentation.
    
    Args:
        dataset: Dataset name
        batch_size: Batch size
        num_samples: Number of calibration samples
        data_dir: Directory to store/load data
        use_synthetic: Use synthetic data
        
    Returns:
        Calibration data loader
    """
    if use_synthetic:
        cal_dataset = create_synthetic_cifar10(num_samples)
        return DataLoader(
            cal_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    try:
        if dataset.lower() == "cifar10":
            transform = get_cifar10_transforms(train=False)
            cal_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform
            )
        else:
            transform = get_imagenette_transforms(train=False)
            imagenette_dir = os.path.join(data_dir, "imagenette2-160")
            if os.path.exists(imagenette_dir):
                cal_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(imagenette_dir, "train"),
                    transform=transform
                )
            else:
                transform = get_cifar10_transforms(train=False)
                cal_dataset = torchvision.datasets.CIFAR10(
                    root=data_dir, train=True, download=True, transform=transform
                )
        
        indices = random.sample(range(len(cal_dataset)), min(num_samples, len(cal_dataset)))
        cal_dataset = Subset(cal_dataset, indices)
    except Exception:
        cal_dataset = create_synthetic_cifar10(num_samples)
    
    return DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
