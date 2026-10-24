"""
Evolutionary Search for Neural Architecture Search.

Implements a simple evolutionary algorithm with:
- Tournament selection
- Crossover
- Mutation

Reference:
Real et al., "Regularized Evolution for Image Classifier Architecture Search", AAAI 2019
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, Callable
import random
import copy
from dataclasses import dataclass, field

try:
    from models.search_space import SuperNet
    from nas.random_search import Architecture
except ImportError:
    from ..models.search_space import SuperNet
    from .random_search import Architecture


class EvolutionarySearch:
    """
    Evolutionary Search for NAS.
    
    Uses regularized evolution with aging:
    - Maintains a population of architectures
    - Selects parents via tournament selection
    - Applies mutation and crossover
    - Removes oldest architectures
    """
    
    def __init__(
        self,
        supernet: SuperNet,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_prob: float = 0.2,
        crossover_prob: float = 0.5,
        tournament_size: int = 5,
        latency_constraint_ms: Optional[float] = None,
        size_constraint_mb: Optional[float] = None
    ):
        """
        Initialize evolutionary search.
        
        Args:
            supernet: Supernet to search in
            population_size: Size of population
            num_generations: Number of evolution generations
            mutation_prob: Probability of mutation per gene
            crossover_prob: Probability of crossover
            tournament_size: Size of tournament for selection
            latency_constraint_ms: Optional latency constraint
            size_constraint_mb: Optional model size constraint
        """
        self.supernet = supernet
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.latency_constraint_ms = latency_constraint_ms
        self.size_constraint_mb = size_constraint_mb
        
        self.population: List[Architecture] = []
        self.history: List[Architecture] = []
        self.best_architecture: Optional[Architecture] = None
    
    def initialize_population(
        self,
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu',
        verbose: bool = True
    ):
        """Initialize random population."""
        if verbose:
            print("Initializing population...")
        
        self.population = []
        
        while len(self.population) < self.population_size:
            config = self.supernet.sample_random_config()
            arch = self._evaluate(config, eval_fn, device)
            
            if self._satisfies_constraints(arch):
                self.population.append(arch)
                self.history.append(arch)
                
                if verbose:
                    print(f"  Init {len(self.population)}/{self.population_size}: "
                          f"Acc={arch.accuracy:.2f}%")
    
    def _evaluate(
        self,
        config: Dict[str, Any],
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu'
    ) -> Architecture:
        """Evaluate an architecture configuration."""
        self.supernet.set_active_config(config)
        self.supernet.to(device)
        
        accuracy, latency_ms, params, flops = eval_fn(self.supernet)
        
        return Architecture(
            config=copy.deepcopy(config),
            accuracy=accuracy,
            latency_ms=latency_ms,
            params=params,
            flops=flops
        )
    
    def _satisfies_constraints(self, arch: Architecture) -> bool:
        """Check if architecture satisfies constraints."""
        if self.latency_constraint_ms and arch.latency_ms > self.latency_constraint_ms:
            return False
        if self.size_constraint_mb:
            size_mb = arch.params * 4 / (1024 ** 2)
            if size_mb > self.size_constraint_mb:
                return False
        return True
    
    def _tournament_select(self) -> Architecture:
        """Select parent via tournament selection."""
        candidates = random.sample(self.population, 
                                   min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda a: a.accuracy)
    
    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to a configuration."""
        new_config = copy.deepcopy(config)
        
        # Mutate kernel sizes
        if random.random() < self.mutation_prob:
            idx = random.randint(0, len(new_config["kernel_sizes"]) - 1)
            new_config["kernel_sizes"][idx] = random.choice(self.supernet.kernel_sizes)
        
        # Mutate expansion ratios
        if random.random() < self.mutation_prob:
            idx = random.randint(0, len(new_config["expansion_ratios"]) - 1)
            new_config["expansion_ratios"][idx] = random.choice(self.supernet.expansion_ratios)
        
        # Mutate depths
        if random.random() < self.mutation_prob:
            idx = random.randint(0, len(new_config["depths"]) - 1)
            max_depth = self.supernet.stages[idx].max_depth
            new_config["depths"][idx] = random.choice(
                [d for d in self.supernet.depth_options if d <= max_depth]
            )
        
        return new_config
    
    def _crossover(
        self, 
        config1: Dict[str, Any], 
        config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply crossover between two configurations."""
        if random.random() > self.crossover_prob:
            return copy.deepcopy(config1)
        
        new_config = {
            "kernel_sizes": [],
            "expansion_ratios": [],
            "depths": []
        }
        
        for key in new_config:
            for i in range(len(config1[key])):
                if random.random() < 0.5:
                    new_config[key].append(config1[key][i])
                else:
                    new_config[key].append(config2[key][i])
        
        return new_config
    
    def evolve_generation(
        self,
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu'
    ) -> Architecture:
        """Evolve one generation and return best offspring."""
        # Select two parents
        parent1 = self._tournament_select()
        parent2 = self._tournament_select()
        
        # Crossover
        child_config = self._crossover(parent1.config, parent2.config)
        
        # Mutate
        child_config = self._mutate(child_config)
        
        # Evaluate
        child = self._evaluate(child_config, eval_fn, device)
        
        # Add to population if satisfies constraints
        if self._satisfies_constraints(child):
            self.population.append(child)
            self.history.append(child)
            
            # Remove oldest (regularized evolution)
            if len(self.population) > self.population_size:
                self.population.pop(0)
        
        return child
    
    def search(
        self,
        eval_fn: Callable[[nn.Module], Tuple[float, float, int, int]],
        device: str = 'cpu',
        verbose: bool = True
    ) -> List[Architecture]:
        """
        Run evolutionary search.
        
        Args:
            eval_fn: Evaluation function
            device: Device to use
            verbose: Print progress
            
        Returns:
            List of all evaluated architectures
        """
        # Initialize population
        self.initialize_population(eval_fn, device, verbose)
        
        # Update best
        self.best_architecture = max(self.population, key=lambda a: a.accuracy)
        
        if verbose:
            print(f"\nInitial best: Acc={self.best_architecture.accuracy:.2f}%")
        
        # Evolution loop
        for gen in range(self.num_generations):
            if verbose:
                print(f"\nGeneration {gen + 1}/{self.num_generations}")
            
            # Evolve population
            for i in range(self.population_size // 2):
                child = self.evolve_generation(eval_fn, device)
                
                if child.accuracy > self.best_architecture.accuracy:
                    self.best_architecture = child
                    if verbose:
                        print(f"  New best! Acc={child.accuracy:.2f}%, "
                              f"Latency={child.latency_ms:.2f}ms")
            
            # Report generation stats
            if verbose:
                accs = [a.accuracy for a in self.population]
                print(f"  Population: mean={sum(accs)/len(accs):.2f}%, "
                      f"max={max(accs):.2f}%, min={min(accs):.2f}%")
        
        return self.history
    
    def get_best_architecture(self) -> Optional[Architecture]:
        """Get the best architecture found."""
        return self.best_architecture
    
    def get_top_k(self, k: int = 5) -> List[Architecture]:
        """Get top-k architectures by accuracy."""
        sorted_archs = sorted(self.history, key=lambda a: a.accuracy, reverse=True)
        return sorted_archs[:k]


def create_evolutionary_search(
    supernet: SuperNet,
    population_size: int = 20,
    num_generations: int = 10,
    mutation_prob: float = 0.2,
    latency_constraint_ms: Optional[float] = None
) -> EvolutionarySearch:
    """Factory function for evolutionary search."""
    return EvolutionarySearch(
        supernet=supernet,
        population_size=population_size,
        num_generations=num_generations,
        mutation_prob=mutation_prob,
        latency_constraint_ms=latency_constraint_ms
    )
