"""
PyTorch Dataset for Multi-Task DR Learning.

Supports:
1. Real IDRiD dataset from HuggingFace (MahsaTorki/IDRiD_Dataset)
2. Synthetic data generation for testing

Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- Source: HuggingFace Mirror of IEEE DataPort release
- URL: https://huggingface.co/datasets/MahsaTorki/IDRiD_Dataset
- License: CC BY 4.0
- Original: Porwal et al., "IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge"
- Samples: 54 train + 27 test with pixel-level lesion annotations
- Lesion Types: Microaneurysms (MA), Hemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE)
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import zipfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


DATASET_CONFIG = {
    "idrid": {
        "name": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
        "huggingface": "MahsaTorki/IDRiD_Dataset",
        "url": "https://huggingface.co/datasets/MahsaTorki/IDRiD_Dataset",
        "provenance": "IEEE DataPort IDRiD Challenge 2018",
        "license": "CC BY 4.0",
        "train_samples": 54,
        "test_samples": 27,
        "lesion_types": ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates"],
        "classes": 5,
    },
}


class HuggingFaceIDRiDDataset(Dataset):
    """Dataset for IDRiD from HuggingFace with segmentation masks."""
    
    LESION_DIRS = {
        "Microaneurysms": "1. Microaneurysms",
        "Hemorrhages": "2. Haemorrhages",
        "Soft_Exudates": "4. Soft Exudates",
        "Hard_Exudates": "3. Hard Exudates",
    }
    
    LESION_SUFFIXES = {
        "Microaneurysms": "_MA.tif",
        "Hemorrhages": "_HE.tif",
        "Soft_Exudates": "_SE.tif",
        "Hard_Exudates": "_EX.tif",
    }
    
    def __init__(self, split="train", transform=None, data_dir=None, download=True):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            data_dir: Directory to store/load data
            download: Whether to download from HuggingFace if not present
        """
        self.split = split
        self.transform = transform
        
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "idrid_hf")
        self.data_dir = data_dir
        
        if download:
            self._download_dataset()
        
        self._load_samples()
        
        print(f"HuggingFace IDRiD: Loaded {len(self.samples)} {split} samples")
    
    def _download_dataset(self):
        """Download IDRiD dataset from HuggingFace if not present."""
        seg_dir = os.path.join(self.data_dir, "A. Segmentation")
        grade_dir = os.path.join(self.data_dir, "B. Disease Grading")
        
        if os.path.exists(seg_dir) and os.path.exists(grade_dir):
            return
        
        print("Downloading IDRiD dataset from HuggingFace...")
        from huggingface_hub import hf_hub_download
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        for filename in ["A.Segmentation.zip", "B.Disease_Grading.zip"]:
            zip_path = hf_hub_download(
                repo_id="MahsaTorki/IDRiD_Dataset",
                filename=filename,
                repo_type="dataset",
                local_dir=self.data_dir
            )
            print(f"Downloaded: {filename}")
            
            with zipfile.ZipFile(os.path.join(self.data_dir, filename), 'r') as z:
                z.extractall(self.data_dir)
            print(f"Extracted: {filename}")
        
        print("IDRiD dataset ready!")
    
    def _load_samples(self):
        """Load sample list with image paths and labels."""
        self.samples = []
        
        if self.split in ["train", "val"]:
            set_name = "a. Training Set"
            labels_file = "a. IDRiD_Disease Grading_Training Labels.csv"
        else:
            set_name = "b. Testing Set"
            labels_file = "b. IDRiD_Disease Grading_Testing Labels.csv"
        
        images_dir = os.path.join(
            self.data_dir, "A. Segmentation", "1. Original Images", set_name
        )
        masks_base_dir = os.path.join(
            self.data_dir, "A. Segmentation", "2. All Segmentation Groundtruths", set_name
        )
        labels_path = os.path.join(
            self.data_dir, "B. Disease Grading", "2. Groundtruths", labels_file
        )
        
        labels_df = pd.read_csv(labels_path)
        labels_dict = dict(zip(labels_df['Image name'], labels_df['Retinopathy grade']))
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        for img_file in image_files:
            img_id = img_file.replace('.jpg', '')
            grading_id = self._seg_to_grading_id(img_id)
            
            grade = labels_dict.get(grading_id)
            if grade is None:
                continue
            
            img_path = os.path.join(images_dir, img_file)
            mask_paths = {}
            
            for lesion_name in config.LESION_NAMES:
                lesion_dir = self.LESION_DIRS.get(lesion_name)
                lesion_suffix = self.LESION_SUFFIXES.get(lesion_name)
                if lesion_dir and lesion_suffix:
                    mask_file = img_id + lesion_suffix
                    mask_path = os.path.join(masks_base_dir, lesion_dir, mask_file)
                    if os.path.exists(mask_path):
                        mask_paths[lesion_name] = mask_path
                    else:
                        mask_paths[lesion_name] = None
            
            self.samples.append({
                'img_id': img_id,
                'img_path': img_path,
                'mask_paths': mask_paths,
                'grade': int(grade),
            })
        
        if self.split == "val":
            n_val = len(self.samples) // 4
            self.samples = self.samples[-n_val:]
        elif self.split == "train":
            n_val = len(self.samples) // 4
            self.samples = self.samples[:-n_val] if n_val > 0 else self.samples
    
    def _seg_to_grading_id(self, seg_id):
        """Convert segmentation ID (IDRiD_01) to grading ID (IDRiD_001)."""
        parts = seg_id.split('_')
        if len(parts) == 2:
            num = int(parts[1])
            return f"IDRiD_{num:03d}"
        return seg_id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['img_path']).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        
        masks = []
        for lesion_name in config.LESION_NAMES:
            mask_path = sample['mask_paths'].get(lesion_name)
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
                mask = np.array(mask).astype(np.float32)
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
            masks.append(mask)
        
        masks = np.stack(masks, axis=0)
        
        if self.transform:
            image, masks = self.transform(image, masks)
        
        image = (image - np.array(config.MEAN)) / np.array(config.STD)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        masks = torch.from_numpy(masks).float()
        grade = torch.tensor(sample['grade']).long()
        
        return {
            'image': image,
            'mask': masks,
            'grade': grade,
            'img_id': sample['img_id']
        }


class DRMultiTaskDataset(Dataset):
    """Dataset for joint DR grading and lesion segmentation (synthetic data)."""
    
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
        
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        
        masks = []
        for lesion_name in config.LESION_NAMES:
            mask_path = os.path.join(self.masks_dir, f"{img_id}_{lesion_name}.png")
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            masks.append(mask)
        
        masks = np.stack(masks, axis=0)
        
        if self.transform:
            image, masks = self.transform(image, masks)
        
        image = (image - np.array(config.MEAN)) / np.array(config.STD)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        masks = torch.from_numpy(masks).float()
        grade = torch.tensor(grade).long()
        
        return {
            'image': image,
            'mask': masks,
            'grade': grade,
            'img_id': img_id
        }


class RandomAugmentation:
    """Enhanced random augmentation for medical imaging."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, masks):
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            masks = np.flip(masks, axis=2).copy()
        
        if np.random.random() < self.p:
            image = np.flipud(image).copy()
            masks = np.flip(masks, axis=1).copy()
        
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            masks = np.rot90(masks, k, axes=(1, 2)).copy()
        
        if np.random.random() < self.p:
            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.85, 1.15)
            image = (image - 0.5) * contrast + 0.5 + (brightness - 1)
            image = np.clip(image, 0, 1)
        
        if np.random.random() < 0.3:
            saturation = np.random.uniform(0.8, 1.2)
            gray = np.mean(image, axis=2, keepdims=True)
            image = gray + saturation * (image - gray)
            image = np.clip(image, 0, 1)
        
        return image, masks


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset."""
    grades = []
    for sample in dataset.samples:
        if isinstance(sample, dict):
            grades.append(sample['grade'])
        else:
            grades.append(sample[1])
    
    grades = np.array(grades)
    unique, counts = np.unique(grades, return_counts=True)
    
    total = len(grades)
    weights = total / (len(unique) * counts)
    
    weight_dict = dict(zip(unique, weights))
    full_weights = np.ones(config.NUM_CLASSES)
    for cls, w in weight_dict.items():
        if cls < len(full_weights):
            full_weights[cls] = w
    
    return torch.FloatTensor(full_weights)


def get_dataloaders(data_dir=None, use_huggingface=True):
    """Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory for data (auto-determined if None)
        use_huggingface: If True, use real IDRiD from HuggingFace; else use synthetic
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    eff_config = config.get_effective_config()
    
    if use_huggingface:
        print("\n=== Using Real IDRiD Dataset from HuggingFace ===")
        print(f"Source: {DATASET_CONFIG['idrid']['huggingface']}")
        print(f"License: {DATASET_CONFIG['idrid']['license']}")
        
        train_dataset = HuggingFaceIDRiDDataset(
            split="train",
            transform=RandomAugmentation(p=0.5),
            download=True
        )
        val_dataset = HuggingFaceIDRiDDataset(
            split="val",
            transform=None,
            download=True
        )
        test_dataset = HuggingFaceIDRiDDataset(
            split="test",
            transform=None,
            download=True
        )
        
        class_weights = compute_class_weights(train_dataset)
        print(f"Class weights: {class_weights.tolist()}")
    else:
        if data_dir is None:
            data_dir = config.DATA_DIR
        
        print("\n=== Using Synthetic Dataset ===")
        train_dataset = DRMultiTaskDataset(
            data_dir, split="train", 
            transform=RandomAugmentation(p=0.5)
        )
        val_dataset = DRMultiTaskDataset(data_dir, split="val")
        test_dataset = DRMultiTaskDataset(data_dir, split="test")
        class_weights = torch.ones(config.NUM_CLASSES)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=eff_config['batch_size'],
        shuffle=True,
        num_workers=0,
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
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Testing HuggingFace IDRiD dataset loading...")
    train_loader, val_loader, test_loader = get_dataloaders(use_huggingface=True)
    
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Grade shape: {batch['grade'].shape}")
    print(f"  Grades: {batch['grade'].tolist()}")
    print(f"  Image IDs: {batch['img_id']}")
