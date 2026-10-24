"""Data loading for multimodal neurodegenerative disease detection."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalAlzheimersDataset(Dataset):
    """
    Multimodal dataset for Alzheimer's disease detection.
    
    Uses synthetic data that mimics the structure of real datasets like ADNI.
    Real ADNI data requires a Data Use Agreement from USC.
    
    Modalities:
    - Brain MRI-like images (grayscale, simulated atrophy patterns)
    - Clinical features (age, MMSE score, CDR, etc.)
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 64,
        num_clinical_features: int = 10,
        num_classes: int = 3,
        seed: int = 42,
        split: str = "train"
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_clinical_features = num_clinical_features
        self.num_classes = num_classes
        self.split = split
        
        np.random.seed(seed + hash(split) % 1000)
        
        self.labels = np.random.randint(0, num_classes, num_samples)
        self.images = self._generate_synthetic_brain_images()
        self.clinical_features = self._generate_clinical_features()
        
        logger.info(f"Created {split} dataset with {num_samples} samples, "
                   f"{num_classes} classes, image size {image_size}x{image_size}")
    
    def _generate_synthetic_brain_images(self) -> np.ndarray:
        """Generate synthetic brain MRI-like images with class-specific patterns."""
        images = np.zeros((self.num_samples, 1, self.image_size, self.image_size), dtype=np.float32)
        
        for i, label in enumerate(self.labels):
            base_img = np.random.randn(self.image_size, self.image_size) * 0.1 + 0.5
            
            center = self.image_size // 2
            y, x = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            brain_mask = dist < self.image_size * 0.4
            
            base_img[brain_mask] += 0.3
            
            if label == 1:  # MCI - mild atrophy
                atrophy_region = dist < self.image_size * 0.15
                base_img[atrophy_region] -= 0.1
            elif label == 2:  # AD - significant atrophy
                atrophy_region = dist < self.image_size * 0.25
                base_img[atrophy_region] -= 0.2
                ventricular = dist < self.image_size * 0.1
                base_img[ventricular] += 0.15
            
            base_img = np.clip(base_img, 0, 1)
            images[i, 0] = base_img
        
        return images
    
    def _generate_clinical_features(self) -> np.ndarray:
        """Generate synthetic clinical features correlated with disease stage."""
        features = np.zeros((self.num_samples, self.num_clinical_features), dtype=np.float32)
        
        for i, label in enumerate(self.labels):
            if label == 0:  # Normal
                age = np.random.normal(70, 8)
                mmse = np.random.normal(29, 1)
                cdr = 0
            elif label == 1:  # MCI
                age = np.random.normal(73, 7)
                mmse = np.random.normal(26, 2)
                cdr = 0.5
            else:  # AD
                age = np.random.normal(76, 6)
                mmse = np.random.normal(21, 4)
                cdr = np.random.choice([1, 2])
            
            features[i, 0] = (age - 60) / 20  # Age normalized
            features[i, 1] = mmse / 30  # MMSE normalized
            features[i, 2] = cdr / 3  # CDR normalized
            features[i, 3] = np.random.uniform(0, 1)  # Education (normalized)
            features[i, 4] = np.random.choice([0, 1])  # Sex
            features[i, 5] = np.random.choice([0, 1, 2]) / 2  # APOE4 alleles
            features[i, 6:] = np.random.randn(self.num_clinical_features - 6) * 0.2 + label * 0.1
        
        return features
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(self.images[idx]),
            "clinical": torch.from_numpy(self.clinical_features[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_dataloaders(
    config,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_samples = int(config.data.max_samples * config.data.train_split)
    val_samples = int(config.data.max_samples * config.data.val_split)
    test_samples = config.data.max_samples - train_samples - val_samples
    
    train_dataset = MultimodalAlzheimersDataset(
        num_samples=train_samples,
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        num_classes=config.num_classes,
        seed=seed,
        split="train"
    )
    
    val_dataset = MultimodalAlzheimersDataset(
        num_samples=val_samples,
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        num_classes=config.num_classes,
        seed=seed + 1,
        split="validation"
    )
    
    test_dataset = MultimodalAlzheimersDataset(
        num_samples=test_samples,
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        num_classes=config.num_classes,
        seed=seed + 2,
        split="test"
    )
    
    return {
        "train": DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True),
        "validation": DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    }
