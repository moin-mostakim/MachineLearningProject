"""Data loading and partitioning for federated learning."""

from .datasets import get_dataset, FederatedDataset
from .partition import partition_data, DirichletPartitioner

__all__ = [
    "get_dataset",
    "FederatedDataset", 
    "partition_data",
    "DirichletPartitioner"
]
