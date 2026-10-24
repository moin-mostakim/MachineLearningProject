"""
PyTorch Dataset for Multi-Task DR Learning.

Handles loading of fundus images, segmentation masks, and DR grades.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DRMultiTaskDataset(Dataset):
    """Dataset for joint DR grading and lesion segmentation."""
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, split, "images")
        self.masks_dir = os.path.join(data_dir, split, "masks")
        
        # Load labels
        labels_path = os.path.join(data_dir, split, "labels.txt")
        self.samples = []
        
        with open(labels_path, 'r') as f:
            for line in f:
                img_id, grade = line.strip().split(',')
                self.samples.append((img_id, int(grade)))
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, grade = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        
        # Load masks for all lesion types
        masks = []
        for lesion_name in config.LESION_NAMES:
            mask_path = os.path.join(self.masks_dir, f"{img_id}_{lesion_name}.png")
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            masks.append(mask)
        
        masks = np.stack(masks, axis=0)  # (num_lesion_types, H, W)
        
        # Apply transforms
        if self.transform:
            image, masks = self.transform(image, masks)
        
        # Normalize image
        image = (image - np.array(config.MEAN)) / np.array(config.STD)
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (C, H, W)
        masks = torch.from_numpy(masks).float()  # (num_lesion_types, H, W)
        grade = torch.tensor(grade).long()
        
        return {
            'image': image,
            'mask': masks,
            'grade': grade,
            'img_id': img_id
        }


class RandomAugmentation:
    """Random augmentation for training."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, masks):
        # Random horizontal flip
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            masks = np.flip(masks, axis=2).copy()
        
        # Random vertical flip
        if np.random.random() < self.p:
            image = np.flipud(image).copy()
            masks = np.flip(masks, axis=1).copy()
        
        # Random rotation (90 degree increments)
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            masks = np.rot90(masks, k, axes=(1, 2)).copy()
        
        # Random brightness/contrast adjustment
        if np.random.random() < self.p:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            image = (image - 0.5) * contrast + 0.5 + (brightness - 1)
            image = np.clip(image, 0, 1)
        
        return image, masks


def get_dataloaders(data_dir=None):
    """Create train, validation, and test dataloaders."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    eff_config = config.get_effective_config()
    
    train_dataset = DRMultiTaskDataset(
        data_dir, split="train", 
        transform=RandomAugmentation(p=0.5)
    )
    val_dataset = DRMultiTaskDataset(data_dir, split="val")
    test_dataset = DRMultiTaskDataset(data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=eff_config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eff_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eff_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    from synthetic_data import create_synthetic_dataset
    
    data_dir = create_synthetic_dataset()
    train_loader, val_loader, test_loader = get_dataloaders(data_dir)
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Grade shape: {batch['grade'].shape}")
    print(f"  Grades: {batch['grade'].tolist()}")
