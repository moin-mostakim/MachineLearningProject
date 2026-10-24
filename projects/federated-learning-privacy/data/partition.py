"""Data partitioning strategies for federated learning."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from collections import defaultdict


class DirichletPartitioner:
    """
    Partition data using Dirichlet distribution for non-IID simulation.
    
    Based on:
    - Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
    - FedProx: Heterogeneous Federated Optimization (Li et al., 2020)
    
    The Dirichlet distribution parameter alpha controls heterogeneity:
    - alpha -> 0: Each client gets samples from single class (extremely non-IID)
    - alpha = 0.5: High heterogeneity (common in FL literature)
    - alpha = 1.0: Moderate heterogeneity
    - alpha -> inf: Each client gets uniform distribution (IID)
    """
    
    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def partition(
        self,
        labels: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Partition data indices among clients using Dirichlet distribution.
        
        Args:
            labels: Array of labels for all samples
            
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        num_samples = len(labels)
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Convert to numpy arrays
        for label in class_indices:
            class_indices[label] = np.array(class_indices[label])
        
        # Sample proportions from Dirichlet distribution
        client_indices = {i: [] for i in range(self.num_clients)}
        
        for label in range(num_classes):
            indices = class_indices[label]
            self.rng.shuffle(indices)
            
            # Sample proportions for this class
            proportions = self.rng.dirichlet([self.alpha] * self.num_clients)
            
            # Ensure minimum samples and normalize
            proportions = np.maximum(proportions, 1e-8)
            proportions = proportions / proportions.sum()
            
            # Split indices according to proportions
            splits = (proportions * len(indices)).astype(int)
            
            # Distribute remainder
            remainder = len(indices) - splits.sum()
            for i in range(remainder):
                splits[i % self.num_clients] += 1
            
            # Assign to clients
            start = 0
            for client_id, num_samples_client in enumerate(splits):
                client_indices[client_id].extend(
                    indices[start:start + num_samples_client].tolist()
                )
                start += num_samples_client
        
        # Convert to numpy arrays
        for client_id in client_indices:
            client_indices[client_id] = np.array(client_indices[client_id])
            self.rng.shuffle(client_indices[client_id])
            client_indices[client_id] = client_indices[client_id].tolist()
        
        return client_indices
    
    def get_partition_stats(
        self,
        labels: np.ndarray,
        client_indices: Dict[int, List[int]]
    ) -> Dict:
        """Get statistics about the partition."""
        stats = {
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "total_samples": len(labels),
            "samples_per_client": {},
            "classes_per_client": {},
            "label_distribution": {}
        }
        
        for client_id, indices in client_indices.items():
            client_labels = labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            
            stats["samples_per_client"][client_id] = len(indices)
            stats["classes_per_client"][client_id] = len(unique)
            stats["label_distribution"][client_id] = dict(zip(
                unique.tolist(), counts.tolist()
            ))
        
        return stats


def partition_data(
    train_dataset: Dataset,
    num_clients: int,
    method: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partition dataset among clients.
    
    Args:
        train_dataset: Training dataset
        num_clients: Number of clients
        method: Partitioning method ("dirichlet", "iid", "shard")
        alpha: Dirichlet concentration parameter
        seed: Random seed
        
    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    # Get labels
    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    
    if method == "dirichlet":
        partitioner = DirichletPartitioner(
            num_clients=num_clients,
            alpha=alpha,
            seed=seed
        )
        return partitioner.partition(labels)
    
    elif method == "iid":
        # IID: random uniform split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(train_dataset))
        rng.shuffle(indices)
        
        splits = np.array_split(indices, num_clients)
        return {i: splits[i].tolist() for i in range(num_clients)}
    
    elif method == "shard":
        # Shard: each client gets contiguous shards
        # Sort by label first
        sorted_indices = np.argsort(labels)
        num_shards = num_clients * 2  # 2 shards per client
        shard_size = len(labels) // num_shards
        
        shards = [
            sorted_indices[i * shard_size:(i + 1) * shard_size]
            for i in range(num_shards)
        ]
        
        # Assign 2 shards to each client
        rng = np.random.default_rng(seed)
        rng.shuffle(shards)
        
        client_indices = {}
        for i in range(num_clients):
            client_indices[i] = np.concatenate(
                [shards[2 * i], shards[2 * i + 1]]
            ).tolist()
        
        return client_indices
    
    else:
        raise ValueError(f"Unknown partitioning method: {method}")
