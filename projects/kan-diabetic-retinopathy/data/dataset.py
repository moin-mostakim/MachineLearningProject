"""
Dataset classes for Diabetic Retinopathy detection.

Supports:
- APTOS 2019 dataset (if available)
- Synthetic DR-like data for testing
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
import random


class DRDataset(Dataset):
    """
    Diabetic Retinopathy Dataset.
    
    Supports loading from:
    - APTOS 2019 format (images in folder, labels in CSV)
    - Synthetic generated data
    
    DR Severity Levels:
    0 - No DR
    1 - Mild NPDR (Non-Proliferative)
    2 - Moderate NPDR
    3 - Severe NPDR
    4 - Proliferative DR
    """
    
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: Tensor of shape (N,) with class indices
            transform: Optional torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        self.class_names = [
            "No DR",
            "Mild NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Proliferative DR",
        ]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx].item()
        
        if self.transform:
            # Convert to PIL for transforms
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image = self.transform(image_pil)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate inverse frequency class weights for imbalanced data.
        
        Returns:
            Tensor of shape (num_classes,) with class weights
        """
        labels_np = self.labels.numpy()
        class_counts = np.bincount(labels_np, minlength=5)
        total = len(labels_np)
        
        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_synthetic_dataset(
    num_samples: int = 1000,
    image_size: int = 224,
    num_classes: int = 5,
    class_imbalance: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic DR-like dataset for testing.
    
    Generates images with patterns that loosely simulate fundus images:
    - Background gradient (simulating fundus coloring)
    - Random circular patterns (simulating blood vessels/lesions)
    - Class-dependent features
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of square images
        num_classes: Number of classes (5 for standard DR grading)
        class_imbalance: Whether to simulate APTOS-like class imbalance
        seed: Random seed
    
    Returns:
        Tuple of (images, labels) tensors
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Class distribution (APTOS-like imbalance)
    if class_imbalance:
        # APTOS 2019: ~50% class 0, ~10% class 1, ~20% class 2, ~8% class 3, ~12% class 4
        class_probs = [0.50, 0.10, 0.20, 0.08, 0.12]
    else:
        class_probs = [0.2] * 5
    
    labels = np.random.choice(num_classes, size=num_samples, p=class_probs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    images = []
    
    for i in range(num_samples):
        label = labels[i].item()
        
        # Create base fundus-like image
        img = create_fundus_like_image(image_size, label)
        images.append(img)
    
    images = torch.stack(images)
    
    return images, labels


def create_fundus_like_image(image_size: int, severity: int) -> torch.Tensor:
    """
    Create a synthetic fundus-like image.
    
    Features vary by severity level to create learnable patterns.
    
    Args:
        image_size: Size of square image
        severity: DR severity (0-4)
    
    Returns:
        Tensor of shape (3, H, W)
    """
    # Create coordinate grids
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xx, yy = np.meshgrid(x, y)
    
    # Base fundus color (orange-red gradient)
    r_base = 0.8 + 0.1 * np.random.randn()
    g_base = 0.4 + 0.1 * np.random.randn()
    b_base = 0.2 + 0.05 * np.random.randn()
    
    # Radial gradient for fundus shape
    radius = np.sqrt(xx**2 + yy**2)
    mask = np.clip(1 - radius, 0, 1) ** 0.5
    
    # Create channels
    r = np.ones((image_size, image_size)) * r_base * mask
    g = np.ones((image_size, image_size)) * g_base * mask
    b = np.ones((image_size, image_size)) * b_base * mask
    
    # Add severity-dependent features
    num_lesions = severity * 3 + np.random.randint(0, 3)
    lesion_intensity = 0.1 + severity * 0.05
    
    for _ in range(num_lesions):
        # Random lesion position
        cx = np.random.uniform(-0.7, 0.7)
        cy = np.random.uniform(-0.7, 0.7)
        lesion_r = np.random.uniform(0.05, 0.15)
        
        # Lesion mask
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        lesion = np.clip(1 - dist / lesion_r, 0, 1)
        
        # Add dark/red spots (hemorrhages)
        r -= lesion * lesion_intensity * np.random.uniform(0.5, 1.0)
        g -= lesion * lesion_intensity * np.random.uniform(0.8, 1.0)
        b -= lesion * lesion_intensity * np.random.uniform(0.8, 1.0)
    
    # Add blood vessel patterns (more visible in higher severity)
    if severity >= 2:
        num_vessels = severity
        for _ in range(num_vessels):
            angle = np.random.uniform(0, 2 * np.pi)
            vessel_x = np.cos(angle) * np.linspace(0, 0.8, 100)
            vessel_y = np.sin(angle) * np.linspace(0, 0.8, 100)
            
            for vx, vy in zip(vessel_x, vessel_y):
                ix = int((vx + 1) / 2 * (image_size - 1))
                iy = int((vy + 1) / 2 * (image_size - 1))
                if 0 <= ix < image_size and 0 <= iy < image_size:
                    r[iy, ix] *= 0.7
                    g[iy, ix] *= 0.5
                    b[iy, ix] *= 0.5
    
    # Clip and normalize
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    # Add noise
    noise = np.random.randn(image_size, image_size) * 0.02
    r = np.clip(r + noise, 0, 1)
    g = np.clip(g + noise, 0, 1)
    b = np.clip(b + noise, 0, 1)
    
    # Stack channels
    img = np.stack([r, g, b], axis=0).astype(np.float32)
    
    return torch.from_numpy(img)


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Based on common DR detection augmentation strategies.
    """
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
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_data_loaders(
    num_samples: int = 1000,
    image_size: int = 224,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        num_samples: Total number of samples
        image_size: Image size
        batch_size: Batch size
        train_split: Training set proportion
        val_split: Validation set proportion
        num_workers: DataLoader workers
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Generate synthetic data
    images, labels = create_synthetic_dataset(
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
    )
    
    # Create transforms
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)
    
    # Split data
    total = len(labels)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size
    
    indices = torch.randperm(total)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = DRDataset(
        images[train_indices],
        labels[train_indices],
        transform=train_transform,
    )
    val_dataset = DRDataset(
        images[val_indices],
        labels[val_indices],
        transform=val_transform,
    )
    test_dataset = DRDataset(
        images[test_indices],
        labels[test_indices],
        transform=val_transform,
    )
    
    # Calculate class weights from training set
    class_weights = train_dataset.get_class_weights()
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Testing synthetic dataset generation...")
    
    images, labels = create_synthetic_dataset(num_samples=100, image_size=224)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels)}")
    
    # Test data loaders
    print("\nTesting data loaders...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        num_samples=200,
        batch_size=16,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {class_weights}")
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
