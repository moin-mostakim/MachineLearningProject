"""
Random Search for Neural Architecture Search.

Simple but effective baseline that samples random architectures
from the search space and evaluates them.

Reference:
Li & Talwalkar, "Random Search and Reproducibility for Neural Architecture Search", UAI 2019
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, Callable
import random
import copy
from dataclasses import dataclass

try:
    from models.search_space import SuperNet
except ImportError:
    from ..models.search_space import SuperNet


@dataclass
class Architecture:
    """Represents a sampled architecture."""
    config: Dict[str, Any]
    accuracy: float = 0.0
    latency_ms: float = 0.0
    params: int = 0
    flops: int = 0
    
    def fitness(self, latency_weight: float = 0.5) -> float:
        """Compute fitness score (higher is better)."""
        # Normalize latency to [0, 1] range (assume max 100ms)
        norm_latency = min(self.latency_ms / 100.0, 1.0)
        # Weighted combination
        return self.accuracy * (1 - latency_weight) + (1 - norm_latency) * latency_weight


class RandomSearch:
    """
    Random Search for NAS.
    
    Samples random architectures from the search space and
    evaluates their accuracy and latency.
    """
    
    def __init__(
        self,
        supernet: SuperNet,
        num_samples: int = 20,
        latency_constraint_ms: Optional[float] = None,
        size_constraint_mb: Optional[float] = None
    ):
        """
        Initialize random search.
        
        Args:
            supernet: Supernet to sample from
            num_samples: Number of random architectures to sample
            latency_constraint_ms: Optional latency constraint
            size_constraint_mb: Optional model size constraint
        """
        self.supernet = supernet
        self.num_samples = num_samples
        self.latency_constraint_ms = latency_constraint_ms
        self.size_constraint_mb = size_constraint_mb
        
        self.evaluated_architectures: List[Architecture] = []
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture configuration."""
        return self.supernet.sample_random_config()
    
    def evaluate_architecture(
        self,
        config: Dict[str, Any],
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu'
    ) -> Architecture:
        """
        Evaluate a single architecture.
        
        Args:
            config: Architecture configuration
            eval_fn: Evaluation function returning (accuracy, latency_ms, params, flops)
            device: Device to evaluate on
            
        Returns:
            Architecture with evaluation metrics
        """
        # Set supernet to this configuration
        self.supernet.set_active_config(config)
        self.supernet.to(device)
        
        # Evaluate
        accuracy, latency_ms, params, flops = eval_fn(self.supernet)
        
        arch = Architecture(
            config=copy.deepcopy(config),
            accuracy=accuracy,
            latency_ms=latency_ms,
            params=params,
            flops=flops
        )
        
        return arch
    
    def satisfies_constraints(self, arch: Architecture) -> bool:
        """Check if architecture satisfies constraints."""
        if self.latency_constraint_ms and arch.latency_ms > self.latency_constraint_ms:
            return False
        if self.size_constraint_mb:
            size_mb = arch.params * 4 / (1024 ** 2)  # Assuming float32
            if size_mb > self.size_constraint_mb:
                return False
        return True
    
    def search(
        self,
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu',
        verbose: bool = True
    ) -> List[Architecture]:
        """
        Run random search.
        
        Args:
            eval_fn: Evaluation function
            device: Device to use
            verbose: Print progress
            
        Returns:
            List of evaluated architectures sorted by accuracy
        """
        self.evaluated_architectures = []
        
        for i in range(self.num_samples):
            config = self.sample_architecture()
            arch = self.evaluate_architecture(config, eval_fn, device)
            
            if self.satisfies_constraints(arch):
                self.evaluated_architectures.append(arch)
                
                if verbose:
                    print(f"Sample {i+1}/{self.num_samples}: "
                          f"Acc={arch.accuracy:.2f}%, "
                          f"Latency={arch.latency_ms:.2f}ms, "
                          f"Params={arch.params/1e6:.2f}M")
            else:
                if verbose:
                    print(f"Sample {i+1}/{self.num_samples}: "
                          f"Constraint violation (Latency={arch.latency_ms:.2f}ms)")
        
        # Sort by accuracy (descending)
        self.evaluated_architectures.sort(key=lambda a: a.accuracy, reverse=True)
        
        return self.evaluated_architectures
    
    def get_best_architecture(self) -> Optional[Architecture]:
        """Get the best architecture found."""
        if not self.evaluated_architectures:
            return None
        return self.evaluated_architectures[0]
    
    def get_pareto_front(self) -> List[Architecture]:
        """
        Get Pareto-optimal architectures (accuracy vs latency trade-off).
        
        Returns:
            List of Pareto-optimal architectures
        """
        if not self.evaluated_architectures:
            return []
        
        pareto_front = []
        
        for arch in self.evaluated_architectures:
            is_dominated = False
            
            for other in self.evaluated_architectures:
                if other is arch:
                    continue
                # Check if other dominates arch
                if (other.accuracy >= arch.accuracy and 
                    other.latency_ms <= arch.latency_ms and
                    (other.accuracy > arch.accuracy or other.latency_ms < arch.latency_ms)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(arch)
        
        return pareto_front


def create_random_search(
    supernet: SuperNet,
    num_samples: int = 20,
    latency_constraint_ms: Optional[float] = None
) -> RandomSearch:
    """Factory function for random search."""
    return RandomSearch(
        supernet=supernet,
        num_samples=num_samples,
        latency_constraint_ms=latency_constraint_ms
    )
