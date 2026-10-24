"""
Dataset module for Diabetic Retinopathy grading.

Datasets:
---------
APTOS 2019 (Primary):
- Source: HuggingFace Hub (sngsfydy/aptos)
- URL: https://huggingface.co/datasets/sngsfydy/aptos
- Original: Kaggle APTOS 2019 Blindness Detection (provenance only)
- License: CC BY-SA 4.0
- Size: 3,662 fundus images with 5-class DR grading

Alternative Datasets:
- Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256 (2,750 samples)
- URL: https://huggingface.co/datasets/Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256

References:
- APTOS 2019 Kaggle Competition (original data source, provenance only)
- Gulshan et al. (2016): "Development of a Deep Learning Algorithm for Detection of DR"
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import random
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Dataset configuration
DATASET_CONFIG = {
    "aptos": {
        "name": "APTOS 2019 Diabetic Retinopathy",
        "huggingface": "sngsfydy/aptos",
        "url": "https://huggingface.co/datasets/sngsfydy/aptos",
        "provenance": "Kaggle APTOS 2019 Blindness Detection",
        "license": "CC BY-SA 4.0",
        "size": "~3GB",
        "samples": 3662,
        "classes": 5,
    },
    "dr_preprocessed": {
        "name": "Diabetic Retinopathy Preprocessed 256x256",
        "huggingface": "Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256",
        "url": "https://huggingface.co/datasets/Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256",
        "license": "CC BY-SA 4.0",
        "size": "~500MB",
        "samples": 2750,
        "classes": 5,
    }
}


class HuggingFaceAPTOSDataset(Dataset):
    """
    APTOS 2019 Diabetic Retinopathy dataset from HuggingFace Hub.
    
    DR Grades (5-class):
        0 - No DR
        1 - Mild NPDR
        2 - Moderate NPDR  
        3 - Severe NPDR
        4 - Proliferative DR
    """
    
    def __init__(
        self,
        split: str = 'train',
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        dataset_name: str = "sngsfydy/aptos",
    ):
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )
        
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Loading APTOS dataset from HuggingFace Hub...")
        full_dataset = load_dataset(dataset_name, split="train")
        
        total_samples = len(full_dataset)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        n_test = int(total_samples * test_split)
        n_val = int(total_samples * val_split)
        n_train = total_samples - n_test - n_val
        
        if split == 'train':
            selected_indices = indices[:n_train]
        elif split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]
        
        if max_samples and max_samples < len(selected_indices):
            selected_indices = selected_indices[:max_samples]
        
        self.images = []
        self.labels = []
        
        for idx in selected_indices:
            item = full_dataset[idx]
            self.images.append(item['image'])
            label = item['label']
            if isinstance(label, str):
                label = int(label)
            self.labels.append(label)
        
        print(f"Loaded {len(self.images)} APTOS images for {split} split")
        print(f"Class distribution: {self.get_class_distribution()}")
        
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _get_default_transforms(self) -> transforms.Compose:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_weights(self) -> torch.Tensor:
        class_counts = np.bincount(self.labels, minlength=5)
        total = len(self.labels)
        weights = total / (5 * class_counts + 1e-6)
        weights = weights / weights.sum() * 5
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict[int, int]:
        return {i: self.labels.count(i) for i in range(5)}


class DRDataset(Dataset):
    """
    Unified DR dataset that loads from HuggingFace by default.
    Falls back to synthetic only if explicitly requested.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        split: str = 'train',
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        use_synthetic: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.split = split
        self.image_size = image_size
        self.seed = seed
        
        if use_synthetic:
            print(f"Using synthetic DR data (opt-in) for {split}")
            self._dataset = SyntheticDRDataset(
                split=split,
                image_size=image_size,
                transform=transform,
                num_samples=max_samples or 500,
                seed=seed,
            )
        elif HAS_DATASETS:
            self._dataset = HuggingFaceAPTOSDataset(
                split=split,
                image_size=image_size,
                transform=transform,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            raise RuntimeError(
                "Cannot load DR data. Install datasets: pip install datasets"
            )
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        return self._dataset[idx]
    
    def get_class_weights(self):
        return self._dataset.get_class_weights()
    
    def get_class_distribution(self):
        return self._dataset.get_class_distribution()


class SyntheticDRDataset(Dataset):
    """Synthetic DR dataset for pipeline testing only."""
    
    def __init__(
        self,
        split: str = 'train',
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        num_samples: int = 500,
        seed: int = 42,
    ):
        self.split = split
        self.image_size = image_size
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        if split == 'train':
            n_samples = int(num_samples * 0.7)
        elif split == 'val':
            n_samples = int(num_samples * 0.15)
        else:
            n_samples = int(num_samples * 0.15)
        
        class_probs = [0.50, 0.15, 0.20, 0.10, 0.05]
        self.labels = list(np.random.choice(5, size=n_samples, p=class_probs))
        self.images = [self._generate_synthetic_image(self.labels[i], seed + i) 
                       for i in range(n_samples)]
        
        if transform is None:
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        self.transform = transform
        
        print(f"Generated {len(self.images)} synthetic DR images for {split}")
    
    def _generate_synthetic_image(self, grade: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        img = np.random.randint(100, 200, (self.image_size, self.image_size, 3), dtype=np.uint8)
        noise_level = grade * 10
        noise = np.random.randint(-noise_level, noise_level + 1, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_class_weights(self):
        class_counts = np.bincount(self.labels, minlength=5)
        weights = len(self.labels) / (5 * class_counts + 1e-6)
        return torch.FloatTensor(weights / weights.sum() * 5)
    
    def get_class_distribution(self):
        return {i: self.labels.count(i) for i in range(5)}


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    use_synthetic: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders using HuggingFace APTOS dataset.
    """
    train_dataset = DRDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        use_synthetic=use_synthetic,
        max_samples=max_samples,
        seed=seed,
    )
    
    val_dataset = DRDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        use_synthetic=use_synthetic,
        max_samples=max_samples // 5 if max_samples else None,
        seed=seed,
    )
    
    test_dataset = DRDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        use_synthetic=use_synthetic,
        max_samples=max_samples // 5 if max_samples else None,
        seed=seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
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
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing HuggingFace APTOS dataset loading...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        image_size=224,
        max_samples=100,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch[0].shape}")
    print(f"Labels: {batch[1]}")
