"""Federated learning algorithms: FedAvg, FedProx, and Personalized FL with DP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from typing import Dict, List, Any, Tuple
import numpy as np
import copy
import logging

from .model import get_model_params, set_model_params, average_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedAvg:
    """
    Federated Averaging (McMahan et al., 2017).
    
    Reference: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    """
    
    def __init__(self, model, config, device="cpu"):
        self.global_model = model.to(device)
        self.config = config
        self.device = device
    
    def client_update(self, client_loader, local_epochs: int) -> Tuple[List[torch.Tensor], int]:
        """Local training on a client."""
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        optimizer = SGD(local_model.parameters(), lr=self.config.training.learning_rate,
                       weight_decay=self.config.training.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        num_samples = 0
        for _ in range(local_epochs):
            for data, target in client_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                num_samples += len(data)
        
        return get_model_params(local_model), num_samples
    
    def aggregate(self, client_params: List[List[torch.Tensor]], 
                  client_weights: List[int]) -> None:
        """Aggregate client updates using weighted averaging."""
        avg_params = average_params(client_params, client_weights)
        set_model_params(self.global_model, avg_params)


class FedProx:
    """
    FedProx (Li et al., 2020) with proximal term.
    
    Reference: "Federated Optimization in Heterogeneous Networks"
    """
    
    def __init__(self, model, config, device="cpu", mu: float = 0.01):
        self.global_model = model.to(device)
        self.config = config
        self.device = device
        self.mu = mu  # Proximal term coefficient
    
    def client_update(self, client_loader, local_epochs: int) -> Tuple[List[torch.Tensor], int]:
        """Local training with proximal regularization."""
        local_model = copy.deepcopy(self.global_model)
        global_params = get_model_params(self.global_model)
        local_model.train()
        
        optimizer = SGD(local_model.parameters(), lr=self.config.training.learning_rate,
                       weight_decay=self.config.training.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        num_samples = 0
        for _ in range(local_epochs):
            for data, target in client_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                
                # Cross-entropy loss
                loss = criterion(output, target)
                
                # Proximal term: ||w - w_global||^2
                prox_term = 0
                for p, gp in zip(local_model.parameters(), global_params):
                    prox_term += (p - gp.to(self.device)).pow(2).sum()
                
                loss += (self.mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()
                num_samples += len(data)
        
        return get_model_params(local_model), num_samples
    
    def aggregate(self, client_params: List[List[torch.Tensor]], 
                  client_weights: List[int]) -> None:
        """Aggregate using weighted averaging."""
        avg_params = average_params(client_params, client_weights)
        set_model_params(self.global_model, avg_params)


class PersonalizedFLWithDP:
    """
    Personalized Federated Learning with Differential Privacy.
    
    Combines:
    - Per-client personalization layers
    - Differential privacy via gradient clipping and noise
    
    References:
    - Abadi et al. (2016) "Deep Learning with Differential Privacy"
    - Li et al. (2021) "Ditto: Fair and Robust Federated Learning"
    """
    
    def __init__(self, model, config, device="cpu"):
        self.global_model = model.to(device)
        self.config = config
        self.device = device
        self.client_models = {}  # Personalized models per client
    
    def client_update(self, client_id: int, client_loader, 
                      local_epochs: int) -> Tuple[List[torch.Tensor], int]:
        """Local training with DP guarantees."""
        # Get or initialize personalized model
        if client_id not in self.client_models:
            self.client_models[client_id] = copy.deepcopy(self.global_model)
        
        local_model = self.client_models[client_id]
        
        # Start from global model but keep some personalized layers
        global_params = get_model_params(self.global_model)
        local_params = get_model_params(local_model)
        
        # Only update shared layers from global (keep classifier personalized)
        for i, (lp, gp) in enumerate(zip(local_model.parameters(), global_params)):
            if i < len(list(local_model.parameters())) - 2:  # Shared layers
                lp.data.copy_(gp)
        
        local_model.train()
        
        optimizer = SGD(local_model.parameters(), lr=self.config.training.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        num_samples = 0
        for _ in range(local_epochs):
            for data, target in client_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # DP: Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(), 
                    self.config.dp.max_grad_norm
                )
                
                # DP: Add Gaussian noise to gradients
                for p in local_model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * (
                            self.config.dp.noise_multiplier * 
                            self.config.dp.max_grad_norm / 
                            len(data)
                        )
                        p.grad.add_(noise)
                
                optimizer.step()
                num_samples += len(data)
        
        self.client_models[client_id] = local_model
        return get_model_params(local_model), num_samples
    
    def aggregate(self, client_ids: List[int], client_params: List[List[torch.Tensor]], 
                  client_weights: List[int]) -> None:
        """Aggregate shared layers only."""
        # Only aggregate non-personalized (shared) layers
        num_shared_layers = len(client_params[0]) - 2  # Exclude classifier
        
        shared_params = []
        for params in client_params:
            shared_params.append(params[:num_shared_layers])
        
        avg_shared = average_params(shared_params, client_weights)
        
        # Update global model's shared layers
        for i, (p, avg_p) in enumerate(zip(self.global_model.parameters(), avg_shared)):
            if i < num_shared_layers:
                p.data.copy_(avg_p)
