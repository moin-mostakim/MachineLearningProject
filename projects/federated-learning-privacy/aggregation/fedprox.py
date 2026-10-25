"""FedProx aggregation and training strategy.

Implementation of FedProx from:
Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy


class FedProxAggregator:
    """
    FedProx aggregator with proximal term regularization.
    
    FedProx extends FedAvg by adding a proximal term to handle
    systems and statistical heterogeneity:
    
    min_w F_k(w) + (mu/2) ||w - w^t||^2
    
    where:
    - F_k(w) is the local objective
    - w^t is the global model at round t
    - mu is the proximal term coefficient
    
    Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient
            device: Device for aggregation
        """
        self.mu = mu
        self.device = device
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        
    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
        global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using weighted average.
        
        Note: FedProx uses the same aggregation as FedAvg.
        The difference is in the local training (proximal term).
        
        Args:
            client_updates: List of (model_update, num_samples) tuples
            global_model: Optional global model to update
            
        Returns:
            Aggregated model state dict
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        total_samples = sum(num_samples for _, num_samples in client_updates)
        
        first_update, _ = client_updates[0]
        aggregated_state = {}
        
        for key in first_update:
            aggregated_state[key] = torch.zeros_like(first_update[key])
        
        for update, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in update:
                aggregated_state[key] += weight * update[key]
        
        if global_model is not None:
            global_state = global_model.state_dict()
            for key in aggregated_state:
                aggregated_state[key] = global_state[key] + aggregated_state[key]
        elif self.global_model_state is not None:
            for key in aggregated_state:
                aggregated_state[key] = self.global_model_state[key] + aggregated_state[key]
        
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


class FedProxClient:
    """
    FedProx client trainer with proximal term regularization.
    
    The key difference from standard local training is the proximal term:
    loss = task_loss + (mu/2) * ||w - w_global||^2
    
    This helps handle:
    1. Systems heterogeneity: variable local updates
    2. Statistical heterogeneity: non-IID data
    """
    
    def __init__(
        self,
        model: nn.Module,
        mu: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize FedProx client.
        
        Args:
            model: Local model
            mu: Proximal term coefficient
            device: Training device
        """
        self.model = model
        self.mu = mu
        self.device = device
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        
    def set_global_model(self, global_state: Dict[str, torch.Tensor]):
        """Set reference global model for proximal term."""
        self.global_model_state = {
            k: v.clone().to(self.device)
            for k, v in global_state.items()
        }
        self.model.load_state_dict(global_state)
        self.model.to(self.device)
    
    def compute_proximal_term(self) -> torch.Tensor:
        """
        Compute proximal term: (mu/2) * ||w - w_global||^2
        
        Returns:
            Proximal loss term
        """
        if self.global_model_state is None:
            return torch.tensor(0.0, device=self.device)
        
        proximal_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.global_model_state:
                global_param = self.global_model_state[name]
                proximal_loss += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2) * proximal_loss
    
    def train(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 5
    ) -> Dict[str, float]:
        """
        Train with FedProx proximal term.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Task loss function
            num_epochs: Number of local epochs
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_prox_loss = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                task_loss = criterion(output, target)
                
                # Proximal term
                prox_loss = self.compute_proximal_term()
                
                # Total loss
                loss = task_loss + prox_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_task_loss += task_loss.item() * len(data)
                total_prox_loss += prox_loss.item() * len(data)
                total_samples += len(data)
        
        return {
            "loss": total_loss / total_samples,
            "task_loss": total_task_loss / total_samples,
            "prox_loss": total_prox_loss / total_samples,
            "samples": total_samples
        }
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get model update (delta) from training."""
        if self.global_model_state is None:
            raise ValueError("Global model state not set")
        
        current_state = self.model.state_dict()
        update = {}
        
        for key in current_state:
            update[key] = current_state[key] - self.global_model_state[key]
        
        return update
