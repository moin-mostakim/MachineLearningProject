"""
Dataset classes for Diabetic Retinopathy detection using KAN models.

Supports:
- HuggingFace APTOS 2019 dataset (real fundus images)
- Synthetic DR-like data for testing
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import random


DATASET_CONFIG = {
    "aptos": {
        "name": "APTOS 2019 Diabetic Retinopathy",
        "huggingface": "sngsfydy/aptos",
        "url": "https://huggingface.co/datasets/sngsfydy/aptos",
        "provenance": "Original data from Kaggle APTOS 2019 Blindness Detection competition",
        "license": "CC BY-SA 4.0",
        "samples": 3662,
        "classes": 5,
        "description": "Fundus images labeled with DR severity (0-4)",
    },
}


class HuggingFaceAPTOSDataset(Dataset):
    """
    APTOS 2019 dataset loaded from HuggingFace Hub.
    
    Uses sngsfydy/aptos dataset which mirrors the Kaggle APTOS 2019 data.
    No Kaggle account required. Uses lazy loading to reduce memory.
    """
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize dataset from HuggingFace Hub.
        
        Args:
            split: One of 'train', 'val', 'test'
            transform: Image transforms
            max_samples: Maximum samples to load (for testing/CPU training)
            seed: Random seed for reproducibility
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        self.split = split
        self.transform = transform
        self.seed = seed
        
        self.class_names = [
            "No DR",
            "Mild NPDR", 
            "Moderate NPDR",
            "Severe NPDR",
            "Proliferative DR",
        ]
        
        print(f"Loading APTOS dataset from HuggingFace Hub...")
        dataset = load_dataset("sngsfydy/aptos", split="train")
        
        all_labels = []
        for item in dataset:
            all_labels.append(item["label"])
        
        random.seed(seed)
        indices = list(range(len(all_labels)))
        random.shuffle(indices)
        
        n = len(indices)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        if split == "train":
            selected_indices = indices[:train_end]
        elif split == "val":
            selected_indices = indices[train_end:val_end]
        else:  # test
            selected_indices = indices[val_end:]
        
        if max_samples:
            selected_indices = selected_indices[:max_samples]
        
        self.dataset = dataset
        self.indices = selected_indices
        self.labels = torch.tensor([all_labels[i] for i in selected_indices], dtype=torch.long)
        
        print(f"Loaded {len(self.indices)} APTOS images for {split} split")
        print(f"Class distribution: {self._get_class_dist()}")
    
    def _get_class_dist(self) -> Dict[int, int]:
        """Get class distribution."""
        labels = self.labels.numpy()
        return {i: int((labels == i).sum()) for i in range(5)}
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image = self.dataset[real_idx]["image"]
        label = self.labels[idx].item()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency class weights."""
        labels_np = self.labels.numpy()
        class_counts = np.bincount(labels_np, minlength=5)
        total = len(labels_np)
        weights = total / (5 * class_counts + 1e-8)
        weights = weights / weights.sum() * 5
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        return self._get_class_dist()


class SyntheticDRDataset(Dataset):
    """Synthetic DR dataset for testing (fallback when HF unavailable)."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        self.transform = transform
        self.image_size = image_size
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        class_probs = [0.50, 0.10, 0.20, 0.08, 0.12]
        self.labels = torch.tensor(
            np.random.choice(5, size=num_samples, p=class_probs),
            dtype=torch.long
        )
        
        self.images = []
        for i in range(num_samples):
            img = self._create_fundus_image(self.labels[i].item())
            self.images.append(img)
        self.images = torch.stack(self.images)
        
        self.class_names = [
            "No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"
        ]
        
        print(f"Generated {num_samples} synthetic fundus images")
    
    def _create_fundus_image(self, severity: int) -> torch.Tensor:
        """Create synthetic fundus-like image."""
        size = self.image_size
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        
        r_base = 0.8 + 0.1 * np.random.randn()
        g_base = 0.4 + 0.1 * np.random.randn()
        b_base = 0.2 + 0.05 * np.random.randn()
        
        radius = np.sqrt(xx**2 + yy**2)
        mask = np.clip(1 - radius, 0, 1) ** 0.5
        
        r = np.ones((size, size)) * r_base * mask
        g = np.ones((size, size)) * g_base * mask
        b = np.ones((size, size)) * b_base * mask
        
        num_lesions = severity * 3 + np.random.randint(0, 3)
        lesion_intensity = 0.1 + severity * 0.05
        
        for _ in range(num_lesions):
            cx = np.random.uniform(-0.7, 0.7)
            cy = np.random.uniform(-0.7, 0.7)
            lesion_r = np.random.uniform(0.05, 0.15)
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            lesion = np.clip(1 - dist / lesion_r, 0, 1)
            r -= lesion * lesion_intensity * np.random.uniform(0.5, 1.0)
            g -= lesion * lesion_intensity * np.random.uniform(0.8, 1.0)
            b -= lesion * lesion_intensity * np.random.uniform(0.8, 1.0)
        
        r = np.clip(r + np.random.randn(size, size) * 0.02, 0, 1)
        g = np.clip(g + np.random.randn(size, size) * 0.02, 0, 1)
        b = np.clip(b + np.random.randn(size, size) * 0.02, 0, 1)
        
        img = np.stack([r, g, b], axis=0).astype(np.float32)
        return torch.from_numpy(img)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx].item()
        
        if self.transform:
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image = self.transform(image_pil)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency class weights."""
        labels_np = self.labels.numpy()
        class_counts = np.bincount(labels_np, minlength=5)
        total = len(labels_np)
        weights = total / (5 * class_counts + 1e-8)
        weights = weights / weights.sum() * 5
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        labels_np = self.labels.numpy()
        return {i: int((labels_np == i).sum()) for i in range(5)}


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_data_loaders(
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    use_synthetic: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        image_size: Image size
        batch_size: Batch size
        num_workers: DataLoader workers
        max_samples: Max samples per split (for CPU training)
        use_synthetic: Force synthetic data (for testing)
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)
    
    if use_synthetic:
        print("Using synthetic dataset (explicit opt-in)")
        total = (max_samples or 500) * 3
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        
        train_dataset = SyntheticDRDataset(
            num_samples=train_size,
            image_size=image_size,
            transform=train_transform,
            seed=seed,
        )
        val_dataset = SyntheticDRDataset(
            num_samples=val_size,
            image_size=image_size,
            transform=val_transform,
            seed=seed + 1,
        )
        test_dataset = SyntheticDRDataset(
            num_samples=total - train_size - val_size,
            image_size=image_size,
            transform=val_transform,
            seed=seed + 2,
        )
    else:
        print(f"Loading real APTOS dataset from HuggingFace")
        train_dataset = HuggingFaceAPTOSDataset(
            split="train",
            transform=train_transform,
            max_samples=max_samples,
            seed=seed,
        )
        val_dataset = HuggingFaceAPTOSDataset(
            split="val",
            transform=val_transform,
            max_samples=max_samples // 5 if max_samples else None,
            seed=seed,
        )
        test_dataset = HuggingFaceAPTOSDataset(
            split="test",
            transform=val_transform,
            max_samples=max_samples // 5 if max_samples else None,
            seed=seed,
        )
    
    class_weights = train_dataset.get_class_weights()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Testing APTOS dataset loading from HuggingFace...")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        image_size=224,
        batch_size=16,
        max_samples=100,
        num_workers=0,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {class_weights}")
    
    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
