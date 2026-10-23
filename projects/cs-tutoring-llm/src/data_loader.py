"""
Data loading and preprocessing for CS Tutoring LLM.

Loads CodeSearchNet dataset from HuggingFace and prepares it for training
the code explanation and tutoring models.

Reference:
- Husain et al., 2019. "CodeSearchNet Challenge: Evaluating the State of
  Semantic Code Search"
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeSample:
    """A single code-documentation pair."""
    code: str
    docstring: str
    func_name: str
    language: str
    repo: str
    path: str


class CodeSearchNetDataset(Dataset):
    """PyTorch Dataset for CodeSearchNet data."""
    
    def __init__(
        self,
        samples: List[CodeSample],
        tokenizer: PreTrainedTokenizer,
        max_code_length: int = 256,
        max_doc_length: int = 256,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_doc_length = max_doc_length
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize code
        code_encoding = self.tokenizer(
            sample.code,
            max_length=self.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize docstring (target)
        doc_encoding = self.tokenizer(
            sample.docstring,
            max_length=self.max_doc_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": code_encoding["input_ids"].squeeze(0),
            "attention_mask": code_encoding["attention_mask"].squeeze(0),
            "labels": doc_encoding["input_ids"].squeeze(0),
            "label_attention_mask": doc_encoding["attention_mask"].squeeze(0),
        }


class DataLoader:
    """Main data loading class for CodeSearchNet."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.tokenizer = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_tokenizer(self, model_name: Optional[str] = None) -> PreTrainedTokenizer:
        """Load tokenizer for the specified model."""
        model_name = model_name or self.config.model.base_model
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir
        )
        return self.tokenizer
    
    def load_dataset(self) -> Tuple[List[CodeSample], List[CodeSample], List[CodeSample]]:
        """Load CodeSearchNet dataset from HuggingFace."""
        logger.info(f"Loading dataset: {self.config.data.dataset_name}/{self.config.data.dataset_subset}")
        
        try:
            # Load from HuggingFace datasets
            dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_subset,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            logger.info("Creating synthetic dataset for demonstration...")
            return self._create_synthetic_dataset()
        
        # Process splits
        train_samples = self._process_split(
            dataset.get("train", dataset.get("train", [])),
            max_samples=self.config.data.max_samples_train
        )
        
        # Some datasets don't have explicit val/test splits
        if "validation" in dataset:
            val_samples = self._process_split(
                dataset["validation"],
                max_samples=self.config.data.max_samples_eval
            )
        else:
            # Split train data
            split_idx = int(len(train_samples) * 0.9)
            val_samples = train_samples[split_idx:]
            train_samples = train_samples[:split_idx]
        
        if "test" in dataset:
            test_samples = self._process_split(
                dataset["test"],
                max_samples=self.config.data.max_samples_eval
            )
        else:
            test_samples = val_samples[:len(val_samples)//2]
            val_samples = val_samples[len(val_samples)//2:]
        
        self.train_data = train_samples
        self.val_data = val_samples
        self.test_data = test_samples
        
        logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
        
        return train_samples, val_samples, test_samples
    
    def _process_split(
        self,
        split_data: HFDataset,
        max_samples: Optional[int] = None
    ) -> List[CodeSample]:
        """Process a dataset split into CodeSample objects."""
        samples = []
        
        for i, item in enumerate(split_data):
            if max_samples and i >= max_samples:
                break
            
            # Handle different dataset formats
            code = item.get("func_code_string", item.get("code", item.get("whole_func_string", "")))
            docstring = item.get("func_documentation_string", item.get("docstring", item.get("summary", "")))
            
            if not code or not docstring:
                continue
            
            sample = CodeSample(
                code=code.strip(),
                docstring=docstring.strip(),
                func_name=item.get("func_name", "unknown"),
                language=item.get("language", self.config.data.dataset_subset),
                repo=item.get("repository_name", item.get("repo", "unknown")),
                path=item.get("func_code_url", item.get("path", "unknown"))
            )
            samples.append(sample)
        
        return samples
    
    def _create_synthetic_dataset(self) -> Tuple[List[CodeSample], List[CodeSample], List[CodeSample]]:
        """Create synthetic dataset for demonstration when real data unavailable."""
        logger.info("Creating synthetic CS education dataset...")
        
        synthetic_samples = [
            CodeSample(
                code='''def binary_search(arr, target):
    """Search for target in sorted array using binary search."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
                docstring="Binary search algorithm that finds the position of a target value in a sorted array. Returns the index if found, -1 otherwise. Time complexity: O(log n), Space complexity: O(1).",
                func_name="binary_search",
                language="python",
                repo="algorithms",
                path="search/binary_search.py"
            ),
            CodeSample(
                code='''def quicksort(arr):
    """Sort array using quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',
                docstring="Quicksort implementation using list comprehensions. Divides array around a pivot element. Average time complexity: O(n log n), worst case O(n^2). Not in-place due to list creation.",
                func_name="quicksort",
                language="python",
                repo="algorithms",
                path="sort/quicksort.py"
            ),
            CodeSample(
                code='''class LinkedListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """Reverse a singly linked list."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev''',
                docstring="Reverses a singly linked list iteratively. Uses three pointers to reverse the direction of links. Time complexity: O(n), Space complexity: O(1).",
                func_name="reverse_linked_list",
                language="python",
                repo="data_structures",
                path="linked_list/reverse.py"
            ),
            CodeSample(
                code='''def fibonacci_dp(n):
    """Calculate nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]''',
                docstring="Computes the nth Fibonacci number using bottom-up dynamic programming. Stores intermediate results to avoid redundant calculations. Time complexity: O(n), Space complexity: O(n).",
                func_name="fibonacci_dp",
                language="python",
                repo="dynamic_programming",
                path="fibonacci/dp.py"
            ),
            CodeSample(
                code='''def bfs(graph, start):
    """Breadth-first search traversal of a graph."""
    from collections import deque
    visited = set()
    queue = deque([start])
    result = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            queue.extend(graph[vertex] - visited)
    return result''',
                docstring="Breadth-first search implementation using a queue. Explores all neighbors at current depth before moving deeper. Time complexity: O(V + E) where V is vertices and E is edges.",
                func_name="bfs",
                language="python",
                repo="algorithms",
                path="graph/bfs.py"
            ),
            CodeSample(
                code='''def merge_sort(arr):
    """Sort array using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
                docstring="Merge sort divides array into halves, recursively sorts each half, then merges sorted halves. Stable sort with guaranteed O(n log n) time complexity. Space complexity: O(n).",
                func_name="merge_sort",
                language="python",
                repo="algorithms",
                path="sort/merge_sort.py"
            ),
            CodeSample(
                code='''class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """Perform inorder traversal of binary tree."""
    result = []
    def traverse(node):
        if node:
            traverse(node.left)
            result.append(node.val)
            traverse(node.right)
    traverse(root)
    return result''',
                docstring="Inorder traversal visits left subtree, root, then right subtree. For binary search trees, produces sorted output. Time complexity: O(n), Space complexity: O(h) where h is tree height.",
                func_name="inorder_traversal",
                language="python",
                repo="data_structures",
                path="tree/inorder.py"
            ),
            CodeSample(
                code='''def dijkstra(graph, start):
    """Find shortest paths from start using Dijkstra's algorithm."""
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node].items():
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances''',
                docstring="Dijkstra's algorithm finds shortest paths from source to all vertices in weighted graph with non-negative edges. Uses min-heap for efficiency. Time complexity: O((V + E) log V).",
                func_name="dijkstra",
                language="python",
                repo="algorithms",
                path="graph/dijkstra.py"
            ),
        ]
        
        # Expand synthetic dataset by creating variations
        expanded_samples = synthetic_samples.copy()
        for sample in synthetic_samples:
            # Create question-answer pairs
            qa_sample = CodeSample(
                code=f"# Question: Explain the following code\n{sample.code}",
                docstring=f"This code implements {sample.func_name}. {sample.docstring}",
                func_name=sample.func_name,
                language=sample.language,
                repo=sample.repo,
                path=sample.path
            )
            expanded_samples.append(qa_sample)
        
        # Split into train/val/test
        np.random.seed(self.config.data.seed)
        indices = np.random.permutation(len(expanded_samples))
        
        train_end = int(len(indices) * 0.7)
        val_end = int(len(indices) * 0.85)
        
        train_samples = [expanded_samples[i] for i in indices[:train_end]]
        val_samples = [expanded_samples[i] for i in indices[train_end:val_end]]
        test_samples = [expanded_samples[i] for i in indices[val_end:]]
        
        self.train_data = train_samples
        self.val_data = val_samples
        self.test_data = test_samples
        
        logger.info(f"Created {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test synthetic samples")
        
        return train_samples, val_samples, test_samples
    
    def create_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """Create PyTorch DataLoaders for train/val/test splits."""
        if self.tokenizer is None:
            self.load_tokenizer()
        
        if self.train_data is None:
            self.load_dataset()
        
        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.num_workers
        
        # Create datasets
        train_dataset = CodeSearchNetDataset(
            self.train_data,
            self.tokenizer,
            max_code_length=self.config.data.max_code_length,
            max_doc_length=self.config.data.max_explanation_length
        )
        
        val_dataset = CodeSearchNetDataset(
            self.val_data,
            self.tokenizer,
            max_code_length=self.config.data.max_code_length,
            max_doc_length=self.config.data.max_explanation_length
        )
        
        test_dataset = CodeSearchNetDataset(
            self.test_data,
            self.tokenizer,
            max_code_length=self.config.data.max_code_length,
            max_doc_length=self.config.data.max_explanation_length
        )
        
        # Create dataloaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_raw_texts(self) -> Tuple[List[str], List[str]]:
        """Get raw code and docstring texts for BM25 indexing."""
        if self.train_data is None:
            self.load_dataset()
        
        all_samples = self.train_data + self.val_data + self.test_data
        codes = [s.code for s in all_samples]
        docstrings = [s.docstring for s in all_samples]
        
        return codes, docstrings


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    train, val, test = loader.load_dataset()
    
    print(f"\nSample code:\n{train[0].code[:200]}...")
    print(f"\nSample docstring:\n{train[0].docstring}")
