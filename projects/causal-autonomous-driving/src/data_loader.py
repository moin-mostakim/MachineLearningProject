"""
Driving data with distribution shift simulation for causal representation learning.

Dataset: ntudlcv/DLCV_HW1 (similar to driving scene classification)
- URL: https://huggingface.co/datasets/ntudlcv/DLCV_HW1
- Alternative: keremberke/driving-scene-classification if available

For domain generalization experiments, we simulate different environments:
- Environment 0: Original images
- Environment 1: Simulated rain (contrast reduction, noise)
- Environment 2: Simulated night (darkened)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import logging

try:
    from datasets import load_dataset
    from PIL import Image
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_INFO = {
    "name": "SVHN (Street View House Numbers)",
    "url": "https://huggingface.co/datasets/svhn",
    "official": "http://ufldl.stanford.edu/housenumbers/",
    "license": "Non-commercial research only",
    "description": "Real-world images from Google Street View, used for digit recognition"
}


class DrivingDataset(Dataset):
    """
    Real driving-related dataset using SVHN (Street View House Numbers).
    
    SVHN provides real-world images from Google Street View cameras,
    which is relevant for autonomous driving research (scene understanding).
    
    For domain generalization experiments, we apply environment transformations:
    - Environment 0: Clear (original images)
    - Environment 1: Rain (reduced contrast, noise)
    - Environment 2: Night (darkened)
    
    Dataset URL: https://huggingface.co/datasets/svhn
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 64,
        environment: int = 0,
        seed: int = 42,
        split: str = "train"
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.environment = environment
        self.num_classes = 10
        
        np.random.seed(seed + environment * 1000)
        
        self.images, self.labels = self._load_real_data(split, num_samples, seed)
        logger.info(f"Created env {environment} dataset: {len(self.images)} samples from SVHN")
    
    def _load_real_data(self, split: str, num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load real SVHN data from HuggingFace with environment transforms."""
        import torchvision
        from torchvision import transforms
        from torch.utils.data import Subset
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        hf_split = "train" if split == "train" else "test"
        
        try:
            if HAS_DATASETS:
                logger.info(f"Loading SVHN from HuggingFace ({hf_split})...")
                ds = load_dataset("svhn", "cropped_digits", split=hf_split, trust_remote_code=True)
                
                if num_samples < len(ds):
                    np.random.seed(seed)
                    indices = np.random.choice(len(ds), num_samples, replace=False)
                    ds = ds.select(indices)
                
                images = []
                labels = []
                for sample in ds:
                    img = sample["image"]
                    if isinstance(img, Image.Image):
                        img = img.resize((self.image_size, self.image_size))
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array] * 3, axis=0)
                        else:
                            img_array = img_array.transpose(2, 0, 1)
                    else:
                        img_array = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                    
                    img_array = self._apply_environment_transform(img_array)
                    images.append(img_array)
                    labels.append(sample["label"])
                
                return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
            else:
                raise ImportError("datasets library not available")
                
        except Exception as e:
            logger.warning(f"HuggingFace SVHN failed: {e}, trying torchvision...")
            
            try:
                dataset = torchvision.datasets.SVHN(
                    root="./data", split=hf_split, download=True, transform=transform
                )
                
                if num_samples < len(dataset):
                    np.random.seed(seed)
                    indices = np.random.choice(len(dataset), num_samples, replace=False)
                    dataset = Subset(dataset, indices)
                
                images = []
                labels = []
                for img, label in dataset:
                    img_array = img.numpy()
                    img_array = self._apply_environment_transform(img_array)
                    images.append(img_array)
                    labels.append(label)
                
                return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
                
            except Exception as e2:
                logger.warning(f"SVHN failed: {e2}, falling back to CIFAR-10...")
                return self._load_cifar10_fallback(split, num_samples, seed)
    
    def _load_cifar10_fallback(self, split: str, num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback to CIFAR-10 for domain generalization experiments.
        
        CIFAR-10 provides diverse scenes that can be used for studying
        distribution shift under different environmental transformations.
        """
        import torchvision
        from torchvision import transforms
        from torch.utils.data import Subset
        
        logger.info(f"Loading CIFAR-10 from torchvision as fallback...")
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        train = (split == "train")
        
        try:
            dataset = torchvision.datasets.CIFAR10(
                root="./data", train=train, download=True, transform=transform
            )
            
            if num_samples < len(dataset):
                np.random.seed(seed)
                indices = np.random.choice(len(dataset), num_samples, replace=False)
                dataset = Subset(dataset, indices)
            
            images = []
            labels = []
            for img, label in dataset:
                img_array = img.numpy()
                img_array = self._apply_environment_transform(img_array)
                images.append(img_array)
                labels.append(label)
            
            logger.info(f"Loaded {len(images)} samples from CIFAR-10")
            return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
            
        except Exception as e:
            logger.error(f"CIFAR-10 fallback also failed: {e}")
            raise RuntimeError(f"Cannot load any dataset: {e}")
    
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
    """
    Create dataloaders for multiple environments using real SVHN data.
    
    Each environment applies different transformations to simulate
    distribution shift (clear/rain/night conditions).
    """
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    train_samples = int(config.data.num_samples_per_env * config.data.train_split)
    val_samples = int(config.data.num_samples_per_env * config.data.val_split)
    test_samples = config.data.num_samples_per_env - train_samples - val_samples
    
    logger.info(f"Loading SVHN dataset from {DATASET_INFO['url']}")
    
    for env in range(config.training.num_environments):
        train_ds = DrivingDataset(train_samples, config.data.image_size, env, seed, split="train")
        val_ds = DrivingDataset(val_samples, config.data.image_size, env, seed + 100, split="test")
        test_ds = DrivingDataset(test_samples, config.data.image_size, env, seed + 200, split="test")
        
        train_loaders.append(DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False))
    
    return {"train": train_loaders, "validation": val_loaders, "test": test_loaders}
