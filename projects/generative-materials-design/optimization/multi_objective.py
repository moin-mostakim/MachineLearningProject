"""
Multi-Objective Optimization for Molecular Design.

Implements:
- NSGA-II style Pareto optimization in latent space
- Gradient-based latent optimization
- Random search baseline

References:
----------
- NSGA-II (Deb et al., 2002): Non-dominated sorting genetic algorithm
- Pareto optimization for molecular design
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """Container for a Pareto-optimal solution."""
    latent: torch.Tensor
    objectives: torch.Tensor
    rank: int = 0
    crowding_distance: float = 0.0


class RandomSearchBaseline:
    """Random search baseline for comparison."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.latent_dim = config.model.latent_dim
        self.target_indices = config.optimization.target_properties
    
    @torch.no_grad()
    def search(
        self,
        num_samples: int = 1000,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Random search in latent space.
        
        Args:
            num_samples: Number of random samples
            device: Device to use
        
        Returns:
            Dictionary with latent vectors and predicted properties
        """
        self.model.eval()
        self.model.to(device)
        
        # Sample random latent vectors
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Predict properties
        properties = self.model.property_predictor(z)
        
        # Extract target properties
        target_props = properties[:, self.target_indices]
        
        return {
            'latent': z,
            'properties': properties,
            'target_properties': target_props
        }


class LatentOptimizer:
    """Gradient-based latent space optimization."""
    
    def __init__(
        self,
        model,
        config,
        target_properties: torch.Tensor,
        maximize: List[bool] = None
    ):
        """
        Initialize latent optimizer.
        
        Args:
            model: VAE model
            config: Configuration
            target_properties: Target property values
            maximize: List of booleans indicating which objectives to maximize
        """
        self.model = model
        self.config = config
        self.target_properties = target_properties
        self.target_indices = config.optimization.target_properties
        self.maximize = maximize or config.optimization.maximize
        
        self.latent_dim = config.model.latent_dim
        self.num_steps = config.optimization.latent_optimization_steps
        self.lr = config.optimization.latent_lr
    
    def optimize(
        self,
        num_samples: int = 100,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize latent vectors for target properties.
        
        Args:
            num_samples: Number of solutions to optimize
            device: Device to use
        
        Returns:
            Optimized latent vectors and properties
        """
        self.model.eval()
        self.model.to(device)
        target = self.target_properties.to(device)
        
        # Initialize latent vectors
        z = torch.randn(num_samples, self.latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=self.lr)
        
        losses = []
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # Predict properties
            props = self.model.property_predictor(z)
            target_props = props[:, self.target_indices]
            
            # Compute loss (minimize distance to target, accounting for max/min)
            loss = 0.0
            for i, (prop_idx, do_max) in enumerate(zip(range(len(self.target_indices)), self.maximize)):
                target_expanded = target[i].expand(target_props.size(0))
                if do_max:
                    # Maximize: minimize negative
                    loss += F.mse_loss(target_props[:, i], target_expanded)
                else:
                    # Minimize: minimize distance
                    loss += F.mse_loss(target_props[:, i], target_expanded)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Final predictions
        with torch.no_grad():
            final_props = self.model.property_predictor(z)
        
        return {
            'latent': z.detach(),
            'properties': final_props,
            'target_properties': final_props[:, self.target_indices],
            'optimization_losses': losses
        }


class ParetoOptimizer:
    """NSGA-II style multi-objective optimization in latent space."""
    
    def __init__(self, model, config):
        """
        Initialize Pareto optimizer.
        
        Args:
            model: VAE model
            config: Configuration
        """
        self.model = model
        self.config = config
        self.latent_dim = config.model.latent_dim
        self.target_indices = config.optimization.target_properties
        self.maximize = config.optimization.maximize
        
        self.pop_size = config.optimization.population_size
        self.num_generations = config.optimization.num_generations
        self.crossover_prob = config.optimization.crossover_prob
        self.mutation_prob = config.optimization.mutation_prob
    
    def _non_dominated_sort(
        self, 
        objectives: np.ndarray
    ) -> List[List[int]]:
        """
        Non-dominated sorting for Pareto ranking.
        
        Args:
            objectives: Objective values [pop_size, num_objectives]
        
        Returns:
            List of fronts (each front is a list of indices)
        """
        n = len(objectives)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        # Adjust for maximization
        adj_objectives = objectives.copy()
        for i, do_max in enumerate(self.maximize):
            if do_max:
                adj_objectives[:, i] = -adj_objectives[:, i]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Check if i dominates j
                if np.all(adj_objectives[i] <= adj_objectives[j]) and \
                   np.any(adj_objectives[i] < adj_objectives[j]):
                    dominated_solutions[i].append(j)
                elif np.all(adj_objectives[j] <= adj_objectives[i]) and \
                     np.any(adj_objectives[j] < adj_objectives[i]):
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _crowding_distance(
        self, 
        objectives: np.ndarray, 
        front: List[int]
    ) -> np.ndarray:
        """Compute crowding distance for solutions in a front."""
        if len(front) <= 2:
            return np.full(len(front), float('inf'))
        
        distances = np.zeros(len(front))
        num_objectives = objectives.shape[1]
        
        for m in range(num_objectives):
            # Sort by objective m
            sorted_idx = np.argsort(objectives[front, m])
            
            # Boundary points have infinite distance
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # Normalize
            obj_range = objectives[front[sorted_idx[-1]], m] - objectives[front[sorted_idx[0]], m]
            if obj_range < 1e-10:
                continue
            
            for i in range(1, len(front) - 1):
                distances[sorted_idx[i]] += (
                    objectives[front[sorted_idx[i+1]], m] - 
                    objectives[front[sorted_idx[i-1]], m]
                ) / obj_range
        
        return distances
    
    def _selection(
        self, 
        population: torch.Tensor, 
        objectives: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Tournament selection based on rank and crowding distance."""
        fronts = self._non_dominated_sort(objectives)
        
        # Assign ranks and crowding distances
        ranks = np.zeros(len(population))
        crowding = np.zeros(len(population))
        
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            cd = self._crowding_distance(objectives, front)
            for i, idx in enumerate(front):
                crowding[idx] = cd[i]
        
        # Select by tournament
        selected_indices = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(population), 2, replace=False)
            
            # Prefer lower rank, then higher crowding distance
            if ranks[i] < ranks[j]:
                selected_indices.append(i)
            elif ranks[i] > ranks[j]:
                selected_indices.append(j)
            elif crowding[i] > crowding[j]:
                selected_indices.append(i)
            else:
                selected_indices.append(j)
        
        return population[selected_indices], objectives[selected_indices]
    
    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Simulated binary crossover (SBX)."""
        if np.random.random() > self.crossover_prob:
            return parent1.clone()
        
        eta = 2.0  # Distribution index
        u = np.random.random(len(parent1))
        
        beta = np.where(
            u <= 0.5,
            (2 * u) ** (1 / (eta + 1)),
            (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        )
        beta = torch.tensor(beta, dtype=parent1.dtype, device=parent1.device)
        
        child = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        return child
    
    def _mutate(self, individual: torch.Tensor) -> torch.Tensor:
        """Polynomial mutation."""
        if np.random.random() > self.mutation_prob:
            return individual
        
        eta = 20.0  # Distribution index
        mutated = individual.clone()
        
        for i in range(len(individual)):
            if np.random.random() < 0.1:  # Per-gene mutation probability
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutated[i] += delta
        
        return mutated
    
    @torch.no_grad()
    def optimize(self, device: str = "cpu") -> Dict[str, any]:
        """
        Run multi-objective optimization.
        
        Returns:
            Dictionary with Pareto front and optimization history
        """
        self.model.eval()
        self.model.to(device)
        
        # Initialize population
        population = torch.randn(self.pop_size, self.latent_dim, device=device)
        
        # Evaluate initial population
        props = self.model.property_predictor(population)
        objectives = props[:, self.target_indices].cpu().numpy()
        
        history = {'generation': [], 'hypervolume': [], 'num_pareto': []}
        
        for gen in range(self.num_generations):
            # Selection
            selected_pop, selected_obj = self._selection(population, objectives)
            
            # Create offspring
            offspring = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected_pop[i]
                parent2 = selected_pop[min(i + 1, self.pop_size - 1)]
                
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            offspring = torch.stack(offspring[:self.pop_size])
            
            # Combine and evaluate
            combined = torch.cat([population, offspring])
            combined_props = self.model.property_predictor(combined)
            combined_objectives = combined_props[:, self.target_indices].cpu().numpy()
            
            # Select next generation
            population, objectives = self._selection(combined, combined_objectives)
            
            # Get Pareto front
            fronts = self._non_dominated_sort(objectives)
            pareto_front = [population[i] for i in fronts[0]]
            
            # Log progress
            history['generation'].append(gen)
            history['num_pareto'].append(len(fronts[0]))
            
            if gen % 10 == 0:
                logger.info(f"Generation {gen}: {len(fronts[0])} Pareto solutions")
        
        # Final Pareto front
        fronts = self._non_dominated_sort(objectives)
        pareto_indices = fronts[0]
        pareto_latent = population[pareto_indices]
        pareto_objectives = objectives[pareto_indices]
        
        # Generate molecules for Pareto front
        recon = self.model.decode(pareto_latent)
        
        return {
            'pareto_latent': pareto_latent,
            'pareto_objectives': torch.tensor(pareto_objectives),
            'pareto_molecules': recon,
            'history': history,
            'final_population': population,
            'final_objectives': objectives
        }


def compare_baselines(
    model,
    config,
    device: str = "cpu"
) -> Dict[str, Dict]:
    """
    Compare multi-objective optimization with baselines.
    
    Returns comparison of:
    - Random search
    - Single-objective optimization
    - Multi-objective (Pareto) optimization
    """
    results = {}
    
    logger.info("Running random search baseline...")
    random_baseline = RandomSearchBaseline(model, config)
    results['random_search'] = random_baseline.search(
        num_samples=config.optimization.random_search_samples,
        device=device
    )
    
    logger.info("Running multi-objective optimization...")
    pareto_opt = ParetoOptimizer(model, config)
    results['multi_objective'] = pareto_opt.optimize(device=device)
    
    logger.info("Running single-objective optimization (HOMO)...")
    # Target: high HOMO energy (first target property)
    target = torch.zeros(len(config.optimization.target_properties))
    target[0] = 1.0  # Target high HOMO
    
    latent_opt = LatentOptimizer(
        model, config, target,
        maximize=[True] * len(config.optimization.target_properties)
    )
    results['single_objective_homo'] = latent_opt.optimize(
        num_samples=100,
        device=device
    )
    
    return results


if __name__ == "__main__":
    # Test optimization
    import sys
    sys.path.append("..")
    from config import get_config
    from models.vae import build_model
    
    config = get_config(debug=True)
    model = build_model(config)
    
    # Test random search
    random_search = RandomSearchBaseline(model, config)
    results = random_search.search(num_samples=100)
    print(f"\nRandom search results:")
    print(f"  Latent shape: {results['latent'].shape}")
    print(f"  Target properties shape: {results['target_properties'].shape}")
    
    # Test Pareto optimization (quick test)
    config.optimization.num_generations = 5
    config.optimization.population_size = 20
    pareto_opt = ParetoOptimizer(model, config)
    results = pareto_opt.optimize()
    print(f"\nPareto optimization results:")
    print(f"  Pareto front size: {len(results['pareto_latent'])}")
