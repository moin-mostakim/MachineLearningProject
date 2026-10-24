"""Synthetic driving data with distribution shift simulation."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrivingDataset(Dataset):
    """
    Synthetic driving dataset with multiple environments (distribution shift).
    
    Simulates different conditions:
    - Environment 0: Clear weather (high contrast, bright)
    - Environment 1: Rain (low contrast, blurred edges)
    - Environment 2: Night (dark, low visibility)
    
    Labels: Steering actions based on road curvature
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 64,
        environment: int = 0,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.environment = environment
        self.num_classes = 4
        
        np.random.seed(seed + environment * 1000)
        
        self.images, self.labels = self._generate_data()
        logger.info(f"Created env {environment} dataset: {num_samples} samples")
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic driving images with environment-specific characteristics."""
        images = np.zeros((self.num_samples, 3, self.image_size, self.image_size), dtype=np.float32)
        labels = np.zeros(self.num_samples, dtype=np.int64)
        
        for i in range(self.num_samples):
            curvature = np.random.choice(4)  # 0: left, 1: slight-left, 2: slight-right, 3: right
            labels[i] = curvature
            
            img = self._generate_road_image(curvature)
            img = self._apply_environment_transform(img)
            images[i] = img
        
        return images, labels
    
    def _generate_road_image(self, curvature: int) -> np.ndarray:
        """Generate base road image with given curvature."""
        img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        
        # Sky (top half)
        img[:, :self.image_size//2, :] = 0.6
        
        # Ground (bottom half)
        img[1, self.image_size//2:, :] = 0.3  # Green-ish ground
        img[2, self.image_size//2:, :] = 0.2
        
        # Road (center stripe with curvature)
        center = self.image_size // 2
        road_width = self.image_size // 4
        
        curve_offset = (curvature - 1.5) * 5  # -7.5 to 7.5 offset
        
        for y in range(self.image_size // 2, self.image_size):
            progress = (y - self.image_size // 2) / (self.image_size // 2)
            x_center = int(center + curve_offset * progress)
            x_start = max(0, x_center - road_width // 2)
            x_end = min(self.image_size, x_center + road_width // 2)
            
            # Gray road
            img[0, y, x_start:x_end] = 0.4
            img[1, y, x_start:x_end] = 0.4
            img[2, y, x_start:x_end] = 0.4
            
            # Center line (yellow)
            if x_start < x_center < x_end:
                img[0, y, x_center-1:x_center+1] = 0.9
                img[1, y, x_center-1:x_center+1] = 0.9
                img[2, y, x_center-1:x_center+1] = 0.0
        
        return img
    
    def _apply_environment_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply environment-specific transformations."""
        if self.environment == 0:  # Clear
            img = img * 1.1
            noise = np.random.randn(*img.shape) * 0.02
        elif self.environment == 1:  # Rain
            img = img * 0.7
            noise = np.random.randn(*img.shape) * 0.1
        else:  # Night
            img = img * 0.4
            noise = np.random.randn(*img.shape) * 0.05
        
        img = img + noise
        return np.clip(img, 0, 1)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(self.images[idx]),
            "label": torch.tensor(self.labels[idx]),
            "environment": torch.tensor(self.environment)
        }


def create_multi_environment_loaders(
    config,
    seed: int = 42
) -> Dict[str, List[DataLoader]]:
    """Create dataloaders for multiple environments."""
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    train_samples = int(config.data.num_samples_per_env * config.data.train_split)
    val_samples = int(config.data.num_samples_per_env * config.data.val_split)
    test_samples = config.data.num_samples_per_env - train_samples - val_samples
    
    for env in range(config.training.num_environments):
        train_ds = DrivingDataset(train_samples, config.data.image_size, env, seed)
        val_ds = DrivingDataset(val_samples, config.data.image_size, env, seed + 100)
        test_ds = DrivingDataset(test_samples, config.data.image_size, env, seed + 200)
        
        train_loaders.append(DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False))
    
    return {"train": train_loaders, "validation": val_loaders, "test": test_loaders}
