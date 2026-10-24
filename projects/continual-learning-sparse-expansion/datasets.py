"""
Dataset utilities for Continual Learning experiments.

Datasets:
---------
CIFAR-100 (Split):
- Source: torchvision / HuggingFace Hub (uoft-cs/cifar100)
- URL: https://huggingface.co/datasets/uoft-cs/cifar100
- Official: https://www.cs.toronto.edu/~kriz/cifar.html
- License: MIT
- Size: 170MB (50K train + 10K test, 100 classes, 32x32 RGB)

MNIST (Permuted):
- Source: torchvision / HuggingFace Hub (ylecun/mnist)
- URL: https://huggingface.co/datasets/ylecun/mnist
- Official: http://yann.lecun.com/exdb/mnist/
- License: CC BY-SA 3.0
- Size: 11MB (60K train + 10K test, 10 classes, 28x28 grayscale)

Implements:
- Split CIFAR-100: 100 classes divided into sequential tasks (default)
- Permuted MNIST: Standard permutation-based continual learning benchmark
- Synthetic datasets: Opt-in only for quick testing

References:
- Zenke et al. (2017): "Continual Learning Through Synaptic Intelligence"
- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks"
"""

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Optional
import config

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Dataset configuration
DATASET_CONFIG = {
    "cifar100": {
        "name": "CIFAR-100",
        "huggingface": "uoft-cs/cifar100",
        "url": "https://huggingface.co/datasets/uoft-cs/cifar100",
        "official": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "license": "MIT",
        "size": "170MB",
        "train_samples": 50000,
        "test_samples": 10000,
        "num_classes": 100,
        "image_size": "32x32",
    },
    "mnist": {
        "name": "MNIST",
        "huggingface": "ylecun/mnist",
        "url": "https://huggingface.co/datasets/ylecun/mnist",
        "official": "http://yann.lecun.com/exdb/mnist/",
        "license": "CC BY-SA 3.0",
        "size": "11MB",
        "train_samples": 60000,
        "test_samples": 10000,
        "num_classes": 10,
        "image_size": "28x28",
    }
}


class HuggingFaceCIFAR100Wrapper(data.Dataset):
    """Wrapper for CIFAR-100 loaded from HuggingFace Hub."""
    
    def __init__(self, hf_dataset, transform=None, max_samples: Optional[int] = None):
        self.transform = transform
        
        if max_samples and max_samples < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(max_samples))
        
        self.data = []
        self.targets = []
        
        for item in hf_dataset:
            self.data.append(np.array(item['img']))
            self.targets.append(item['fine_label'])
        
        self.data = np.array(self.data)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class HuggingFaceMNISTWrapper(data.Dataset):
    """Wrapper for MNIST loaded from HuggingFace Hub."""
    
    def __init__(self, hf_dataset, transform=None, max_samples: Optional[int] = None):
        self.transform = transform
        
        if max_samples and max_samples < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(max_samples))
        
        self.data = []
        self.targets = []
        
        for item in hf_dataset:
            self.data.append(np.array(item['image']))
            self.targets.append(item['label'])
        
        self.data = np.array(self.data)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx], mode='L')
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class SyntheticCIFAR100(data.Dataset):
    """Synthetic CIFAR-100-like dataset for fast testing (opt-in only)."""
    
    def __init__(self, num_samples: int = 5000, train: bool = True, transform=None):
        self.num_samples = num_samples
        self.train = train
        self.transform = transform
        # Generate synthetic 32x32x3 images
        self.data = np.random.randint(0, 256, (num_samples, 32, 32, 3), dtype=np.uint8)
        self.targets = np.random.randint(0, 100, num_samples).tolist()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class SplitCIFAR100:
    """
    Split CIFAR-100 into sequential tasks for continual learning.
    
    Each task contains a disjoint subset of classes.
    Default: 10 tasks with 10 classes each.
    
    By default loads real CIFAR-100 from torchvision/HuggingFace.
    Set use_synthetic=True for synthetic data (testing only).
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = None,
        classes_per_task: int = None,
        download: bool = True,
        use_synthetic: bool = False,
        max_samples: Optional[int] = None
    ):
        self.num_tasks = num_tasks or config.NUM_TASKS
        self.classes_per_task = classes_per_task or config.CLASSES_PER_TASK
        self.root = root
        self.use_synthetic = use_synthetic
        
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
        
        if use_synthetic:
            print("Using synthetic CIFAR-100-like dataset (opt-in)")
            self.train_data = SyntheticCIFAR100(
                num_samples=max_samples or 5000, train=True, transform=self.train_transform
            )
            self.test_data = SyntheticCIFAR100(
                num_samples=(max_samples or 5000) // 5, train=False, transform=self.test_transform
            )
        else:
            self._load_real_cifar100(download, max_samples)
        
        # Create task splits
        self._create_task_splits()
    
    def _load_real_cifar100(self, download: bool, max_samples: Optional[int]):
        """Load real CIFAR-100 from torchvision or HuggingFace."""
        try:
            print("Loading CIFAR-100 from torchvision...")
            self.train_data = torchvision.datasets.CIFAR100(
                root=self.root, train=True, download=download, transform=self.train_transform
            )
            self.test_data = torchvision.datasets.CIFAR100(
                root=self.root, train=False, download=download, transform=self.test_transform
            )
            
            if max_samples and max_samples < len(self.train_data):
                indices = list(range(max_samples))
                self.train_data = data.Subset(self.train_data, indices)
                self.train_data.targets = [self.train_data.dataset.targets[i] for i in indices]
            
            print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test samples from CIFAR-100")
            
        except Exception as e:
            print(f"torchvision CIFAR-100 download failed: {e}")
            if HAS_DATASETS:
                self._load_from_huggingface(max_samples)
            else:
                raise RuntimeError(
                    "Cannot load CIFAR-100. Options:\n"
                    "1. Ensure network connectivity for torchvision download\n"
                    "2. Install datasets library: pip install datasets\n"
                    "3. Use use_synthetic=True for synthetic data"
                )
    
    def _load_from_huggingface(self, max_samples: Optional[int]):
        """Load CIFAR-100 from HuggingFace Hub."""
        print("Loading CIFAR-100 from HuggingFace Hub...")
        hf_dataset = load_dataset("uoft-cs/cifar100")
        
        self.train_data = HuggingFaceCIFAR100Wrapper(
            hf_dataset["train"], transform=self.train_transform, max_samples=max_samples
        )
        self.test_data = HuggingFaceCIFAR100Wrapper(
            hf_dataset["test"], transform=self.test_transform
        )
        print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test from HuggingFace")
    
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


class SyntheticMNIST(data.Dataset):
    """Synthetic MNIST-like dataset for fast testing."""
    
    def __init__(self, num_samples: int = 6000, train: bool = True, transform=None):
        self.num_samples = num_samples
        self.train = train
        self.transform = transform
        # Generate synthetic 28x28 grayscale images
        self.data = np.random.randint(0, 256, (num_samples, 28, 28), dtype=np.uint8)
        self.targets = np.random.randint(0, 10, num_samples).tolist()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        from PIL import Image
        img = Image.fromarray(img, mode='L')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class PermutedMNIST:
    """
    Permuted MNIST for continual learning.
    
    Each task uses a different fixed permutation of the input pixels.
    The original MNIST is task 0 (identity permutation).
    
    By default loads real MNIST from torchvision/HuggingFace.
    Set use_synthetic=True for synthetic data (testing only).
    """
    
    def __init__(
        self,
        root: str = "./data",
        num_tasks: int = None,
        download: bool = True,
        use_synthetic: bool = False,
        max_samples: Optional[int] = None
    ):
        self.num_tasks = num_tasks or config.NUM_TASKS
        self.root = root
        self.use_synthetic = use_synthetic
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if use_synthetic:
            print("Using synthetic MNIST-like dataset (opt-in)")
            self.train_data = SyntheticMNIST(
                num_samples=max_samples or 6000, train=True, transform=self.transform
            )
            self.test_data = SyntheticMNIST(
                num_samples=(max_samples or 6000) // 6, train=False, transform=self.transform
            )
        else:
            self._load_real_mnist(download, max_samples)
        
        # Generate permutations for each task
        self._generate_permutations()
    
    def _load_real_mnist(self, download: bool, max_samples: Optional[int]):
        """Load real MNIST from torchvision or HuggingFace."""
        try:
            print("Loading MNIST from torchvision...")
            self.train_data = torchvision.datasets.MNIST(
                root=self.root, train=True, download=download, transform=self.transform
            )
            self.test_data = torchvision.datasets.MNIST(
                root=self.root, train=False, download=download, transform=self.transform
            )
            
            if max_samples and max_samples < len(self.train_data):
                indices = list(range(max_samples))
                self.train_data = data.Subset(self.train_data, indices)
            
            print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test samples from MNIST")
            
        except Exception as e:
            print(f"torchvision MNIST download failed: {e}")
            if HAS_DATASETS:
                self._load_from_huggingface(max_samples)
            else:
                raise RuntimeError(
                    "Cannot load MNIST. Options:\n"
                    "1. Ensure network connectivity for torchvision download\n"
                    "2. Install datasets library: pip install datasets\n"
                    "3. Use use_synthetic=True for synthetic data"
                )
    
    def _load_from_huggingface(self, max_samples: Optional[int]):
        """Load MNIST from HuggingFace Hub."""
        print("Loading MNIST from HuggingFace Hub...")
        hf_dataset = load_dataset("ylecun/mnist")
        
        self.train_data = HuggingFaceMNISTWrapper(
            hf_dataset["train"], transform=self.transform, max_samples=max_samples
        )
        self.test_data = HuggingFaceMNISTWrapper(
            hf_dataset["test"], transform=self.transform
        )
        print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test from HuggingFace")
    
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
