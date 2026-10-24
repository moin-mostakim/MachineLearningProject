"""
Unified NAS Searcher interface.

Provides a common interface for different NAS algorithms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple, Optional, Callable
import time

try:
    from models.search_space import SuperNet
    from utils.metrics import count_parameters, count_flops, accuracy
    from utils.profiler import measure_latency
    from nas.random_search import RandomSearch, Architecture
    from nas.evolutionary import EvolutionarySearch
except ImportError:
    from ..models.search_space import SuperNet
    from ..utils.metrics import count_parameters, count_flops, accuracy
    from ..utils.profiler import measure_latency
    from .random_search import RandomSearch, Architecture
    from .evolutionary import EvolutionarySearch


class NASSearcher:
    """
    Unified Neural Architecture Search interface.
    
    Supports multiple search algorithms:
    - Random search
    - Evolutionary search
    """
    
    def __init__(
        self,
        supernet: SuperNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        algorithm: str = "evolutionary",
        num_samples: int = 20,
        num_generations: int = 10,
        population_size: int = 20,
        mutation_prob: float = 0.2,
        latency_constraint_ms: Optional[float] = None,
        size_constraint_mb: Optional[float] = None,
        device: str = 'cpu'
    ):
        """
        Initialize NAS searcher.
        
        Args:
            supernet: Supernet to search in
            train_loader: Training data loader
            val_loader: Validation data loader
            algorithm: Search algorithm ("random" or "evolutionary")
            num_samples: Number of samples for random search
            num_generations: Number of generations for evolutionary
            population_size: Population size for evolutionary
            mutation_prob: Mutation probability
            latency_constraint_ms: Optional latency constraint
            size_constraint_mb: Optional model size constraint
            device: Device to use
        """
        self.supernet = supernet
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.algorithm = algorithm
        self.device = device
        
        # Create searcher based on algorithm
        if algorithm == "random":
            self.searcher = RandomSearch(
                supernet=supernet,
                num_samples=num_samples,
                latency_constraint_ms=latency_constraint_ms,
                size_constraint_mb=size_constraint_mb
            )
        elif algorithm == "evolutionary":
            self.searcher = EvolutionarySearch(
                supernet=supernet,
                population_size=population_size,
                num_generations=num_generations,
                mutation_prob=mutation_prob,
                latency_constraint_ms=latency_constraint_ms,
                size_constraint_mb=size_constraint_mb
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.results: List[Architecture] = []
    
    def _quick_evaluate(self, model: nn.Module) -> Tuple[float, float, int, int]:
        """
        Quick evaluation of a model.
        
        Returns:
            (accuracy, latency_ms, params, flops)
        """
        model.eval()
        model.to(self.device)
        
        # Get a batch for evaluation
        images, labels = next(iter(self.val_loader))
        input_size = images.shape
        
        # Measure accuracy on subset
        correct = 0
        total = 0
        num_batches = min(5, len(self.val_loader))
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                if i >= num_batches:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100.0 * correct / total
        
        # Measure latency (on CPU for edge simulation)
        latency_stats = measure_latency(
            model, input_size, device='cpu', num_warmup=5, num_runs=20
        )
        
        # Count parameters and FLOPs
        params = count_parameters(model)
        flops = count_flops(model, input_size, device=self.device)
        
        return acc, latency_stats.mean_ms, params, flops
    
    def search(self, verbose: bool = True) -> List[Architecture]:
        """
        Run NAS search.
        
        Args:
            verbose: Print progress
            
        Returns:
            List of evaluated architectures
        """
        if verbose:
            print(f"Starting {self.algorithm} search...")
            print(f"Device: {self.device}")
        
        start_time = time.time()
        
        self.results = self.searcher.search(
            eval_fn=self._quick_evaluate,
            device=self.device,
            verbose=verbose
        )
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSearch completed in {elapsed:.2f}s")
            print(f"Evaluated {len(self.results)} architectures")
            
            best = self.get_best_architecture()
            if best:
                print(f"\nBest architecture:")
                print(f"  Accuracy: {best.accuracy:.2f}%")
                print(f"  Latency: {best.latency_ms:.2f}ms")
                print(f"  Params: {best.params / 1e6:.2f}M")
                print(f"  FLOPs: {best.flops / 1e6:.2f}M")
        
        return self.results
    
    def get_best_architecture(self) -> Optional[Architecture]:
        """Get the best architecture found."""
        return self.searcher.get_best_architecture()
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration of the best architecture."""
        best = self.get_best_architecture()
        return best.config if best else None
    
    def create_best_model(self) -> Optional[SuperNet]:
        """Create a model with the best configuration."""
        best_config = self.get_best_config()
        if best_config is None:
            return None
        
        self.supernet.set_active_config(best_config)
        return self.supernet


def create_nas_searcher(
    supernet: SuperNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> NASSearcher:
    """Factory function for NAS searcher."""
    return NASSearcher(
        supernet=supernet,
        train_loader=train_loader,
        val_loader=val_loader,
        algorithm=config.get("algorithm", "evolutionary"),
        num_samples=config.get("num_samples", 20),
        num_generations=config.get("num_generations", 10),
        population_size=config.get("population_size", 20),
        mutation_prob=config.get("mutation_prob", 0.2),
        latency_constraint_ms=config.get("latency_constraint_ms"),
        size_constraint_mb=config.get("size_constraint_mb"),
        device=device
    )
