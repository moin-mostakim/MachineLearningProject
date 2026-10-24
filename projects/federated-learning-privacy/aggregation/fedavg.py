"""FedAvg aggregation strategy.

Implementation of Federated Averaging from:
McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy


class FedAvgAggregator:
    """
    Federated Averaging (FedAvg) aggregator.
    
    FedAvg aggregates client updates using weighted average based on
    the number of samples each client has:
    
    w_{t+1} = sum_k (n_k / n) * w_k^t
    
    where n_k is number of samples for client k, n is total samples.
    
    Reference:
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (AISTATS 2017)
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize FedAvg aggregator.
        
        Args:
            device: Device for aggregation
        """
        self.device = device
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        
    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
        global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using weighted average.
        
        Args:
            client_updates: List of (model_update, num_samples) tuples
            global_model: Optional global model to update in-place
            
        Returns:
            Aggregated model state dict
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in client_updates)
        
        # Get the structure from first update
        first_update, _ = client_updates[0]
        aggregated_state = {}
        
        # Initialize with zeros
        for key in first_update:
            aggregated_state[key] = torch.zeros_like(first_update[key])
        
        # Weighted average of updates
        for update, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in update:
                aggregated_state[key] += weight * update[key]
        
        # If we have a global model, add the aggregated update to it
        if global_model is not None:
            global_state = global_model.state_dict()
            for key in aggregated_state:
                aggregated_state[key] = global_state[key] + aggregated_state[key]
        elif self.global_model_state is not None:
            for key in aggregated_state:
                aggregated_state[key] = self.global_model_state[key] + aggregated_state[key]
        
        self.global_model_state = aggregated_state
        return aggregated_state
    
    def aggregate_full_models(
        self,
        client_models: List[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate full client models (not updates).
        
        Args:
            client_models: List of (model_state_dict, num_samples) tuples
            
        Returns:
            Aggregated model state dict
        """
        if not client_models:
            raise ValueError("No client models to aggregate")
        
        total_samples = sum(num_samples for _, num_samples in client_models)
        
        first_state, _ = client_models[0]
        aggregated_state = {}
        
        for key in first_state:
            aggregated_state[key] = torch.zeros_like(first_state[key])
        
        for state, num_samples in client_models:
            weight = num_samples / total_samples
            for key in state:
                aggregated_state[key] += weight * state[key]
        
        self.global_model_state = aggregated_state
        return aggregated_state
    
    def set_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Set the global model state."""
        self.global_model_state = copy.deepcopy(model_state)
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get the current global model state."""
        if self.global_model_state is None:
            raise ValueError("Global model state not set")
        return copy.deepcopy(self.global_model_state)


def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[Tuple[nn.Module, int]],
    in_place: bool = True
) -> nn.Module:
    """
    Functional interface for FedAvg aggregation.
    
    Args:
        global_model: Global model to update
        client_models: List of (client_model, num_samples) tuples
        in_place: Whether to update global_model in place
        
    Returns:
        Aggregated model
    """
    if not in_place:
        global_model = copy.deepcopy(global_model)
    
    # Get states and sample counts
    states_and_samples = [
        (model.state_dict(), num_samples)
        for model, num_samples in client_models
    ]
    
    # Aggregate
    aggregator = FedAvgAggregator()
    aggregated_state = aggregator.aggregate_full_models(states_and_samples)
    
    # Load aggregated state
    global_model.load_state_dict(aggregated_state)
    
    return global_model
