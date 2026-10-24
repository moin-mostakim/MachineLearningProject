"""
Data loading for multimodal neurodegenerative disease detection.

Dataset: Falah/Alzheimer_MRI from HuggingFace Hub
- URL: https://huggingface.co/datasets/Falah/Alzheimer_MRI
- License: MIT
- Classes: 4 (Mild Demented, Moderate Demented, Non Demented, Very Mild Demented)
- Images: Brain MRI scans (grayscale)
- Reference: Based on Alzheimer's MRI Dataset for classification

For multimodal experiments, clinical features are simulated based on class labels
since this dataset is image-only.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
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
    "name": "Falah/Alzheimer_MRI",
    "url": "https://huggingface.co/datasets/Falah/Alzheimer_MRI",
    "license": "MIT",
    "classes": ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"],
    "num_classes": 4,
}


class AlzheimerMRIDataset(Dataset):
    """
    Real Alzheimer's MRI dataset from HuggingFace (Falah/Alzheimer_MRI).
    
    URL: https://huggingface.co/datasets/Falah/Alzheimer_MRI
    Classes: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented
    
    Clinical features are simulated based on the class labels to support
    multimodal experiments (since the original dataset is image-only).
    """
    
    CLASS_MAP = {
        "Non_Demented": 0,
        "Very_Mild_Demented": 1, 
        "Mild_Demented": 2,
        "Moderate_Demented": 3
    }
    
    def __init__(
        self,
        split: str = "train",
        image_size: int = 64,
        num_clinical_features: int = 10,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__()
        self.image_size = image_size
        self.num_clinical_features = num_clinical_features
        self.split = split
        
        np.random.seed(seed + hash(split) % 1000)
        
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        logger.info(f"Loading Alzheimer MRI dataset from HuggingFace ({split})...")
        logger.info(f"Dataset URL: {DATASET_INFO['url']}")
        
        try:
            dataset = load_dataset("Falah/Alzheimer_MRI", split=split, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Split '{split}' not available, using 'train' with manual split")
            full_dataset = load_dataset("Falah/Alzheimer_MRI", split="train", trust_remote_code=True)
            n = len(full_dataset)
            if split == "train":
                dataset = full_dataset.select(range(int(n * 0.7)))
            elif split in ["validation", "val"]:
                dataset = full_dataset.select(range(int(n * 0.7), int(n * 0.85)))
            else:
                dataset = full_dataset.select(range(int(n * 0.85), n))
        
        if max_samples and max_samples < len(dataset):
            indices = list(range(max_samples))
            dataset = dataset.select(indices)
        
        self.images = []
        self.labels = []
        
        for sample in dataset:
            img = sample["image"]
            if isinstance(img, Image.Image):
                img = img.convert("L")
                img = img.resize((image_size, image_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
            else:
                img_array = np.array(img, dtype=np.float32)
                if len(img_array.shape) == 3:
                    img_array = img_array.mean(axis=2)
                img_array = img_array / 255.0 if img_array.max() > 1 else img_array
            
            self.images.append(img_array)
            label_str = sample["label"] if isinstance(sample["label"], str) else DATASET_INFO["classes"][sample["label"]]
            self.labels.append(self.CLASS_MAP.get(label_str, sample["label"]))
        
        self.images = np.array(self.images)[:, np.newaxis, :, :]
        self.labels = np.array(self.labels)
        self.clinical_features = self._generate_clinical_features()
        
        logger.info(f"Loaded {len(self.images)} samples from Falah/Alzheimer_MRI ({split})")
        logger.info(f"Classes: {DATASET_INFO['classes']}")
    
    def _generate_clinical_features(self) -> np.ndarray:
        """Generate clinical features correlated with disease stage (simulated)."""
        features = np.zeros((len(self.labels), self.num_clinical_features), dtype=np.float32)
        
        for i, label in enumerate(self.labels):
            if label == 0:  # Non Demented
                age = np.random.normal(70, 8)
                mmse = np.random.normal(29, 1)
                cdr = 0
            elif label == 1:  # Very Mild Demented
                age = np.random.normal(72, 7)
                mmse = np.random.normal(27, 2)
                cdr = 0.5
            elif label == 2:  # Mild Demented
                age = np.random.normal(74, 6)
                mmse = np.random.normal(24, 3)
                cdr = 1.0
            else:  # Moderate Demented
                age = np.random.normal(76, 5)
                mmse = np.random.normal(18, 4)
                cdr = 2.0
            
            features[i, 0] = (age - 60) / 20
            features[i, 1] = mmse / 30
            features[i, 2] = cdr / 3
            features[i, 3] = np.random.uniform(0, 1)
            features[i, 4] = np.random.choice([0, 1])
            features[i, 5] = np.random.choice([0, 1, 2]) / 2
            features[i, 6:] = np.random.randn(self.num_clinical_features - 6) * 0.2 + label * 0.1
        
        return features
    
    def __len__(self) -> int:
        return len(self.labels)
    
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
    """
    Create train, validation, and test dataloaders.
    
    Uses real Alzheimer MRI data from HuggingFace (Falah/Alzheimer_MRI).
    """
    train_dataset = AlzheimerMRIDataset(
        split="train",
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        max_samples=config.data.max_samples,
        seed=seed
    )
    
    val_dataset = AlzheimerMRIDataset(
        split="validation",
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        max_samples=config.data.max_samples // 5,
        seed=seed + 1
    )
    
    test_dataset = AlzheimerMRIDataset(
        split="test",
        image_size=config.data.image_size,
        num_clinical_features=config.clinical_encoder.input_dim,
        max_samples=config.data.max_samples // 5,
        seed=seed + 2
    )
    
    return {
        "train": DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True),
        "validation": DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    }
