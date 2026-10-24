"""
Dataset loading and preprocessing for Adversarial VLM Robustness.

Datasets:
---------
CIFAR-10: Standard benchmark for image classification
- Source: torchvision (auto-download) or HuggingFace (streaming)
- URL: https://www.cs.toronto.edu/~kriz/cifar.html
- HuggingFace: https://huggingface.co/datasets/cifar10
- License: MIT-like (research use)
- Size: ~170MB (60,000 32x32 color images in 10 classes)
- Download: Automatic via torchvision.datasets.CIFAR10 or datasets.load_dataset()
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np
from PIL import Image

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Dataset configuration
DATASET_CONFIG = {
    "cifar10": {
        "name": "CIFAR-10",
        "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "huggingface": "https://huggingface.co/datasets/cifar10",
        "torchvision": "torchvision.datasets.CIFAR10",
        "license": "MIT-like (research use)",
        "size": "~170MB",
        "num_classes": 10,
        "image_size": 32,
        "train_samples": 50000,
        "test_samples": 10000,
    }
}


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
    """Synthetic CIFAR-10-like dataset for testing without download.
    
    Use only with --use-synthetic flag for quick testing.
    """
    
    def __init__(self, num_samples: int = 1000, train: bool = True):
        self.num_samples = num_samples
        self.train = train
        self.data = torch.randn(num_samples, 3, 32, 32) * 0.5 + 0.5
        self.data = self.data.clamp(0, 1)
        self.targets = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()


class HuggingFaceCIFAR10(Dataset):
    """CIFAR-10 dataset loaded from HuggingFace Hub.
    
    Supports streaming to avoid large upfront downloads.
    """
    
    def __init__(self, split: str = "train", transform=None, max_samples: Optional[int] = None):
        self.transform = transform
        print(f"Loading CIFAR-10 from HuggingFace Hub (split={split})...")
        
        if HF_AVAILABLE:
            dataset = load_dataset("uoft-cs/cifar10", split=split, trust_remote_code=True)
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            self.data = dataset
            print(f"  Loaded {len(self.data)} samples from HuggingFace")
        else:
            raise ImportError("datasets library not available. Install with: pip install datasets")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['img']
        label = item['label']
        
        if isinstance(image, Image.Image):
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
        
        return image, label


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = False,
    subset_size: Optional[int] = None,
    use_synthetic: bool = False,
    use_huggingface: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize images
        subset_size: If specified, use only this many samples (for debugging)
        use_synthetic: If True, use synthetic data instead of real CIFAR-10
        use_huggingface: If True, load from HuggingFace Hub (faster for small subsets)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform = get_cifar10_transforms(train=True, normalize=normalize)
    test_transform = get_cifar10_transforms(train=False, normalize=normalize)
    
    if use_synthetic:
        print("[SYNTHETIC MODE] Using synthetic CIFAR-10-like data (use --use-synthetic=False for real data)")
        train_size = subset_size if subset_size else 5000
        test_size = train_size // 5
        train_dataset = SyntheticCIFAR10(train_size, train=True)
        test_dataset = SyntheticCIFAR10(test_size, train=False)
    elif use_huggingface and HF_AVAILABLE:
        print(f"Loading real CIFAR-10 from HuggingFace Hub...")
        print(f"  HuggingFace URL: {DATASET_CONFIG['cifar10']['huggingface']}")
        
        max_train = subset_size if subset_size else None
        max_test = (subset_size // 5) if subset_size else None
        
        train_dataset = HuggingFaceCIFAR10(split="train", transform=train_transform, max_samples=max_train)
        test_dataset = HuggingFaceCIFAR10(split="test", transform=test_transform, max_samples=max_test)
    else:
        print(f"Downloading/loading real CIFAR-10 dataset from torchvision...")
        print(f"  Source: {DATASET_CONFIG['cifar10']['url']}")
        print(f"  Size: {DATASET_CONFIG['cifar10']['size']}")
        
        os.makedirs(data_dir, exist_ok=True)
        
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
        
        print(f"  Loaded {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        
        if subset_size is not None:
            train_indices = np.random.choice(len(train_dataset), min(subset_size, len(train_dataset)), replace=False)
            test_indices = np.random.choice(len(test_dataset), min(subset_size // 5, len(test_dataset)), replace=False)
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            print(f"  Using subset: {len(train_dataset)} train, {len(test_dataset)} test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
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
        
        use_synthetic = getattr(config.data, 'use_synthetic', False)
        
        return get_cifar10_loaders(
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            normalize=False,  # Keep in [0, 1] for adversarial attacks
            subset_size=subset_size,
            use_synthetic=use_synthetic
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
