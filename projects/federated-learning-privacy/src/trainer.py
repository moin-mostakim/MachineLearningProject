"""Federated training orchestration."""

import torch
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import logging
import copy

from .algorithms import FedAvg, FedProx, PersonalizedFLWithDP
from .model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedTrainer:
    """Orchestrates federated learning experiments."""
    
    def __init__(self, config, client_loaders, test_loader, device="cpu"):
        self.config = config
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
    
    def train_fedavg(self) -> Dict[str, Any]:
        """Train using FedAvg."""
        logger.info("Training with FedAvg...")
        model = create_model(self.config).to(self.device)
        algorithm = FedAvg(model, self.config, self.device)
        
        return self._train_loop(algorithm, "FedAvg")
    
    def train_fedprox(self, mu: float = 0.01) -> Dict[str, Any]:
        """Train using FedProx."""
        logger.info(f"Training with FedProx (μ={mu})...")
        model = create_model(self.config).to(self.device)
        algorithm = FedProx(model, self.config, self.device, mu)
        
        return self._train_loop(algorithm, "FedProx")
    
    def train_personalized_dp(self) -> Dict[str, Any]:
        """Train using Personalized FL with DP."""
        logger.info(f"Training with Personalized FL + DP (ε={self.config.dp.epsilon})...")
        model = create_model(self.config).to(self.device)
        algorithm = PersonalizedFLWithDP(model, self.config, self.device)
        
        return self._train_loop_personalized(algorithm, "PersonalizedFL+DP")
    
    def _train_loop(self, algorithm, name: str) -> Dict[str, Any]:
        """Standard FL training loop."""
        accuracies = []
        
        for round_idx in tqdm(range(self.config.federated.num_rounds), desc=name):
            # Select clients
            client_ids = np.random.choice(
                list(self.client_loaders.keys()),
                self.config.federated.clients_per_round,
                replace=False
            )
            
            client_params = []
            client_weights = []
            
            for client_id in client_ids:
                params, num_samples = algorithm.client_update(
                    self.client_loaders[client_id],
                    self.config.federated.local_epochs
                )
                client_params.append(params)
                client_weights.append(num_samples)
            
            algorithm.aggregate(client_params, client_weights)
            
            # Evaluate
            acc = self._evaluate(algorithm.global_model)
            accuracies.append(acc)
            
            if (round_idx + 1) % 5 == 0:
                logger.info(f"{name} Round {round_idx+1}: Accuracy = {acc:.4f}")
        
        return {
            "accuracies": accuracies,
            "final_accuracy": accuracies[-1],
            "model": algorithm.global_model
        }
    
    def _train_loop_personalized(self, algorithm, name: str) -> Dict[str, Any]:
        """Personalized FL training loop."""
        accuracies = []
        
        for round_idx in tqdm(range(self.config.federated.num_rounds), desc=name):
            client_ids = np.random.choice(
                list(self.client_loaders.keys()),
                self.config.federated.clients_per_round,
                replace=False
            )
            
            client_params = []
            client_weights = []
            
            for client_id in client_ids:
                params, num_samples = algorithm.client_update(
                    client_id,
                    self.client_loaders[client_id],
                    self.config.federated.local_epochs
                )
                client_params.append(params)
                client_weights.append(num_samples)
            
            algorithm.aggregate(client_ids.tolist(), client_params, client_weights)
            
            acc = self._evaluate(algorithm.global_model)
            accuracies.append(acc)
            
            if (round_idx + 1) % 5 == 0:
                logger.info(f"{name} Round {round_idx+1}: Accuracy = {acc:.4f}")
        
        return {
            "accuracies": accuracies,
            "final_accuracy": accuracies[-1],
            "model": algorithm.global_model,
            "privacy_budget": self.config.dp.epsilon
        }
    
    @torch.no_grad()
    def _evaluate(self, model) -> float:
        """Evaluate model on test set."""
        model.eval()
        correct, total = 0, 0
        
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
        
        model.train()
        return correct / total if total > 0 else 0
