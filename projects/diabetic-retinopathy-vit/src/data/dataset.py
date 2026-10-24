"""
Dataset module for Diabetic Retinopathy grading.

Supports:
- APTOS 2019 Blindness Detection dataset
- Messidor dataset (fallback)
- Synthetic fundus-like data generation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Optional, Tuple, List, Dict, Any
import random
from pathlib import Path


class DRDataset(Dataset):
    """
    Diabetic Retinopathy dataset supporting multiple data sources.
    
    DR Grades:
        0 - No DR
        1 - Mild NPDR
        2 - Moderate NPDR  
        3 - Severe NPDR
        4 - Proliferative DR
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        use_synthetic: bool = True,
        num_synthetic_samples: int = 500,
        seed: int = 42,
    ):
        """
        Initialize DR dataset.
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            image_size: Target image size
            transform: Optional torchvision transforms
            use_synthetic: Generate synthetic data if real data unavailable
            num_synthetic_samples: Number of synthetic samples to generate
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.use_synthetic = use_synthetic
        self.num_synthetic_samples = num_synthetic_samples
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.images = []
        self.labels = []
        
        # Try to load real data, fall back to synthetic
        if not self._load_aptos_data() and not self._load_messidor_data():
            if use_synthetic:
                print(f"Generating synthetic fundus data for {split} split...")
                self._generate_synthetic_data()
            else:
                raise ValueError("No dataset found and synthetic generation disabled")
        
        # Setup default transforms if not provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _load_aptos_data(self) -> bool:
        """Try to load APTOS 2019 dataset."""
        aptos_dir = self.data_dir / 'aptos2019'
        if not aptos_dir.exists():
            return False
        
        csv_path = aptos_dir / f'{self.split}.csv'
        images_dir = aptos_dir / f'{self.split}_images'
        
        if not csv_path.exists() or not images_dir.exists():
            return False
        
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = images_dir / f"{row['id_code']}.png"
                if img_path.exists():
                    self.images.append(str(img_path))
                    self.labels.append(int(row['diagnosis']))
        
        print(f"Loaded {len(self.images)} images from APTOS 2019 ({self.split})")
        return len(self.images) > 0
    
    def _load_messidor_data(self) -> bool:
        """Try to load Messidor dataset."""
        messidor_dir = self.data_dir / 'messidor'
        if not messidor_dir.exists():
            return False
        
        # Messidor uses different grading, would need mapping
        # 0: No DR, 1: Mild, 2: Moderate, 3: Severe
        # Map to 5-class: 0->0, 1->1, 2->2, 3->3|4
        
        print("Messidor dataset found but loading not implemented in this demo")
        return False
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic fundus-like images with DR features.
        
        This creates simplified representations of fundus images with:
        - Optic disc
        - Blood vessels
        - Lesion-like features (microaneurysms, hemorrhages, exudates)
        """
        # Determine number of samples per split
        if self.split == 'train':
            n_samples = int(self.num_synthetic_samples * 0.8)
        elif self.split == 'val':
            n_samples = int(self.num_synthetic_samples * 0.1)
        else:  # test
            n_samples = int(self.num_synthetic_samples * 0.1)
        
        # Class distribution (imbalanced like real DR data)
        # Based on APTOS 2019: ~50% class 0, ~20% class 2, ~15% class 1, ~10% class 3, ~5% class 4
        class_probs = [0.50, 0.15, 0.20, 0.10, 0.05]
        
        # Create synthetic data directory
        synth_dir = self.data_dir / 'synthetic' / self.split
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(n_samples):
            # Sample class based on realistic distribution
            label = np.random.choice(5, p=class_probs)
            
            # Generate synthetic fundus image
            img = self._create_synthetic_fundus(label, seed=self.seed + i)
            
            # Save image
            img_path = synth_dir / f'synth_{i:05d}.png'
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            self.images.append(str(img_path))
            self.labels.append(label)
        
        print(f"Generated {len(self.images)} synthetic fundus images for {self.split}")
    
    def _create_synthetic_fundus(self, grade: int, seed: int = 42) -> np.ndarray:
        """
        Create a synthetic fundus image with grade-appropriate features.
        
        Args:
            grade: DR grade (0-4)
            seed: Random seed
            
        Returns:
            RGB image as numpy array
        """
        np.random.seed(seed)
        
        size = self.image_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Background: dark red-brown (fundus color)
        bg_color = np.array([140, 80, 50]) + np.random.randint(-20, 20, 3)
        bg_color = np.clip(bg_color, 0, 255)
        img[:] = bg_color
        
        # Create circular fundus mask
        center = (size // 2, size // 2)
        radius = int(size * 0.45)
        y, x = np.ogrid[:size, :size]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Apply brighter fundus color inside circle
        fundus_color = np.array([180, 100, 60]) + np.random.randint(-15, 15, 3)
        fundus_color = np.clip(fundus_color, 0, 255)
        img[mask] = fundus_color
        
        # Add optic disc (bright yellowish circle)
        disc_center = (int(size * 0.65), int(size * 0.5))
        disc_radius = int(size * 0.08)
        cv2.circle(img, disc_center, disc_radius, (220, 200, 150), -1)
        
        # Add fovea (darker central region)
        fovea_center = (int(size * 0.45), int(size * 0.5))
        fovea_radius = int(size * 0.03)
        cv2.circle(img, fovea_center, fovea_radius, (120, 70, 40), -1)
        
        # Add blood vessels (branching pattern)
        self._add_blood_vessels(img, disc_center, radius)
        
        # Add DR lesions based on grade
        if grade >= 1:  # Mild: few microaneurysms
            self._add_microaneurysms(img, mask, count=np.random.randint(1, 5))
        
        if grade >= 2:  # Moderate: more microaneurysms, some hemorrhages
            self._add_microaneurysms(img, mask, count=np.random.randint(5, 15))
            self._add_hemorrhages(img, mask, count=np.random.randint(1, 5))
        
        if grade >= 3:  # Severe: many hemorrhages, exudates
            self._add_microaneurysms(img, mask, count=np.random.randint(15, 30))
            self._add_hemorrhages(img, mask, count=np.random.randint(5, 15))
            self._add_exudates(img, mask, count=np.random.randint(3, 10))
        
        if grade >= 4:  # Proliferative: neovascularization, large hemorrhages
            self._add_microaneurysms(img, mask, count=np.random.randint(30, 50))
            self._add_hemorrhages(img, mask, count=np.random.randint(15, 30))
            self._add_exudates(img, mask, count=np.random.randint(10, 20))
            self._add_neovascularization(img, mask)
        
        # Add some noise/texture
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def _add_blood_vessels(self, img: np.ndarray, disc_center: Tuple[int, int], 
                          radius: int):
        """Add simplified blood vessel pattern."""
        size = img.shape[0]
        vessel_color = (120, 50, 40)
        
        # Main vessels radiating from optic disc
        for angle in np.linspace(0, 2*np.pi, 8):
            if 0.3*np.pi < angle < 0.7*np.pi or 1.3*np.pi < angle < 1.7*np.pi:
                continue  # Skip horizontal directions (macula region)
            
            length = int(radius * 0.8)
            end_x = int(disc_center[0] + length * np.cos(angle))
            end_y = int(disc_center[1] + length * np.sin(angle))
            
            # Draw curved vessel
            points = []
            for t in np.linspace(0, 1, 20):
                x = int(disc_center[0] + t * length * np.cos(angle + 0.1*np.sin(t*5)))
                y = int(disc_center[1] + t * length * np.sin(angle + 0.1*np.sin(t*5)))
                if 0 <= x < size and 0 <= y < size:
                    points.append([x, y])
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                thickness = max(1, int(3 * (1 - t)))
                cv2.polylines(img, [points], False, vessel_color, thickness)
    
    def _add_microaneurysms(self, img: np.ndarray, mask: np.ndarray, count: int):
        """Add microaneurysm-like features (small red dots)."""
        size = img.shape[0]
        for _ in range(count):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            if mask[y, x]:
                radius = np.random.randint(1, 3)
                color = (150, 40, 40)  # Dark red
                cv2.circle(img, (x, y), radius, color, -1)
    
    def _add_hemorrhages(self, img: np.ndarray, mask: np.ndarray, count: int):
        """Add hemorrhage-like features (larger red blobs)."""
        size = img.shape[0]
        for _ in range(count):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            if mask[y, x]:
                radius = np.random.randint(3, 10)
                color = (130, 30, 30)  # Dark red
                cv2.circle(img, (x, y), radius, color, -1)
                cv2.GaussianBlur(img[max(0,y-radius):y+radius, max(0,x-radius):x+radius],
                               (5, 5), 2)
    
    def _add_exudates(self, img: np.ndarray, mask: np.ndarray, count: int):
        """Add exudate-like features (yellowish spots)."""
        size = img.shape[0]
        for _ in range(count):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            if mask[y, x]:
                radius = np.random.randint(2, 8)
                color = (220, 200, 150)  # Yellowish
                cv2.circle(img, (x, y), radius, color, -1)
    
    def _add_neovascularization(self, img: np.ndarray, mask: np.ndarray):
        """Add neovascularization-like features (abnormal vessel growth)."""
        size = img.shape[0]
        vessel_color = (140, 50, 40)
        
        # Random abnormal vessel clusters
        for _ in range(np.random.randint(2, 5)):
            center_x = np.random.randint(size//4, 3*size//4)
            center_y = np.random.randint(size//4, 3*size//4)
            
            if mask[center_y, center_x]:
                for _ in range(np.random.randint(3, 8)):
                    angle = np.random.uniform(0, 2*np.pi)
                    length = np.random.randint(5, 20)
                    end_x = int(center_x + length * np.cos(angle))
                    end_y = int(center_y + length * np.sin(angle))
                    cv2.line(img, (center_x, center_y), (end_x, end_y), vessel_color, 1)
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms based on split."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
        
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        class_counts = np.bincount(self.labels, minlength=5)
        total = len(self.labels)
        
        # Inverse frequency weighting
        weights = total / (5 * class_counts + 1e-6)
        weights = weights / weights.sum() * 5  # Normalize
        
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        return {i: self.labels.count(i) for i in range(5)}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    num_synthetic_samples: int = 500,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        num_synthetic_samples: Number of synthetic samples if real data unavailable
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = DRDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        num_synthetic_samples=num_synthetic_samples,
        seed=seed,
    )
    
    val_dataset = DRDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        num_synthetic_samples=num_synthetic_samples,
        seed=seed,
    )
    
    test_dataset = DRDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        num_synthetic_samples=num_synthetic_samples,
        seed=seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
    
    return train_loader, val_loader, test_loader
