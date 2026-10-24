"""
Dataset utilities for Continual Learning experiments.

Implements:
- Split CIFAR-100: 100 classes divided into sequential tasks
- Permuted MNIST: Standard permutation-based continual learning benchmark
"""

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Optional
import config


class SplitCIFAR100:
    """
    Split CIFAR-100 into sequential tasks for continual learning.
    
    Each task contains a disjoint subset of classes.
    Default: 10 tasks with 10 classes each.
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = None,
        classes_per_task: int = None,
        download: bool = True
    ):
        self.num_tasks = num_tasks or config.NUM_TASKS
        self.classes_per_task = classes_per_task or config.CLASSES_PER_TASK
        self.root = root
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # No augmentation for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # Load full dataset
        self.train_data = torchvision.datasets.CIFAR100(
            root=root, train=True, download=download, transform=self.train_transform
        )
        self.test_data = torchvision.datasets.CIFAR100(
            root=root, train=False, download=download, transform=self.test_transform
        )
        
        # Create task splits
        self._create_task_splits()
    
    def _create_task_splits(self):
        """Create disjoint class splits for each task."""
        np.random.seed(config.SEED)
        
        # Shuffle class order for random task assignment
        all_classes = np.arange(100)
        np.random.shuffle(all_classes)
        
        self.task_classes = []
        for t in range(self.num_tasks):
            start_idx = t * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            self.task_classes.append(all_classes[start_idx:end_idx].tolist())
        
        # Pre-compute indices for each task
        self.train_indices = self._get_task_indices(self.train_data, self.task_classes)
        self.test_indices = self._get_task_indices(self.test_data, self.task_classes)
    
    def _get_task_indices(
        self,
        dataset: torchvision.datasets.CIFAR100,
        task_classes: List[List[int]]
    ) -> List[List[int]]:
        """Get sample indices for each task."""
        targets = np.array(dataset.targets)
        indices = []
        
        for classes in task_classes:
            task_idx = []
            for c in classes:
                task_idx.extend(np.where(targets == c)[0].tolist())
            indices.append(task_idx)
        
        return indices
    
    def get_task_dataloader(
        self,
        task_id: int,
        train: bool = True,
        batch_size: int = None
    ) -> data.DataLoader:
        """Get dataloader for a specific task."""
        batch_size = batch_size or config.BATCH_SIZE
        
        if train:
            indices = self.train_indices[task_id]
            dataset = data.Subset(self.train_data, indices)
            shuffle = True
        else:
            indices = self.test_indices[task_id]
            dataset = data.Subset(self.test_data, indices)
            shuffle = False
        
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
    
    def get_all_tasks_test_loader(
        self,
        batch_size: int = None
    ) -> List[data.DataLoader]:
        """Get test loaders for all tasks."""
        return [
            self.get_task_dataloader(t, train=False, batch_size=batch_size)
            for t in range(self.num_tasks)
        ]
    
    def get_task_info(self, task_id: int) -> dict:
        """Get information about a specific task."""
        return {
            "task_id": task_id,
            "classes": self.task_classes[task_id],
            "num_classes": len(self.task_classes[task_id]),
            "num_train_samples": len(self.train_indices[task_id]),
            "num_test_samples": len(self.test_indices[task_id])
        }


class PermutedMNIST:
    """
    Permuted MNIST for continual learning.
    
    Each task uses a different fixed permutation of the input pixels.
    The original MNIST is task 0 (identity permutation).
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = None,
        download: bool = True
    ):
        self.num_tasks = num_tasks or config.NUM_TASKS
        self.root = root
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full dataset
        self.train_data = torchvision.datasets.MNIST(
            root=root, train=True, download=download, transform=self.transform
        )
        self.test_data = torchvision.datasets.MNIST(
            root=root, train=False, download=download, transform=self.transform
        )
        
        # Generate permutations for each task
        self._generate_permutations()
    
    def _generate_permutations(self):
        """Generate fixed permutations for each task."""
        self.permutations = []
        input_size = 28 * 28
        
        for t in range(self.num_tasks):
            if t == 0:
                # First task: identity permutation (original MNIST)
                perm = np.arange(input_size)
            else:
                # Subsequent tasks: random permutation
                np.random.seed(config.PERMUTATION_SEED_START + t)
                perm = np.random.permutation(input_size)
            self.permutations.append(perm)
    
    def get_task_dataloader(
        self,
        task_id: int,
        train: bool = True,
        batch_size: int = None
    ) -> data.DataLoader:
        """Get dataloader for a specific task with permuted inputs."""
        batch_size = batch_size or config.BATCH_SIZE
        
        dataset = self.train_data if train else self.test_data
        permuted_dataset = PermutedDataset(
            dataset, self.permutations[task_id]
        )
        
        return data.DataLoader(
            permuted_dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def get_all_tasks_test_loader(
        self,
        batch_size: int = None
    ) -> List[data.DataLoader]:
        """Get test loaders for all tasks."""
        return [
            self.get_task_dataloader(t, train=False, batch_size=batch_size)
            for t in range(self.num_tasks)
        ]
    
    def get_task_info(self, task_id: int) -> dict:
        """Get information about a specific task."""
        return {
            "task_id": task_id,
            "permutation_seed": config.PERMUTATION_SEED_START + task_id if task_id > 0 else None,
            "is_identity": task_id == 0,
            "num_train_samples": len(self.train_data),
            "num_test_samples": len(self.test_data)
        }


class PermutedDataset(data.Dataset):
    """Wrapper dataset that applies a permutation to inputs."""
    
    def __init__(self, dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Flatten, permute, reshape
        img_flat = img.view(-1)
        img_permuted = img_flat[self.permutation]
        img_out = img_permuted.view(img.shape)
        
        return img_out, label


def get_dataset(dataset_name: str = None) -> Tuple:
    """
    Factory function to get the appropriate dataset.
    
    Returns:
        dataset: Dataset object with task dataloaders
        input_size: Input dimensions
        num_classes: Number of output classes
    """
    dataset_name = dataset_name or config.DATASET
    
    if dataset_name == "split_cifar100":
        dataset = SplitCIFAR100()
        input_size = (3, 32, 32)
        num_classes = config.CLASSES_PER_TASK  # Per-task output head
    elif dataset_name == "permuted_mnist":
        dataset = PermutedMNIST()
        input_size = (1, 28, 28)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, input_size, num_classes


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Split CIFAR-100...")
    cifar = SplitCIFAR100()
    for t in range(min(3, cifar.num_tasks)):
        info = cifar.get_task_info(t)
        print(f"Task {t}: {info}")
        loader = cifar.get_task_dataloader(t, train=True)
        batch = next(iter(loader))
        print(f"  Batch shape: {batch[0].shape}, Labels: {batch[1][:5]}")
    
    print("\nTesting Permuted MNIST...")
    mnist = PermutedMNIST()
    for t in range(min(3, mnist.num_tasks)):
        info = mnist.get_task_info(t)
        print(f"Task {t}: {info}")
        loader = mnist.get_task_dataloader(t, train=True)
        batch = next(iter(loader))
        print(f"  Batch shape: {batch[0].shape}")
