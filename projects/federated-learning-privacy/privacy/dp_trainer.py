"""Differential privacy trainer for federated learning clients."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import copy

from .dp_mechanism import clip_model_gradients, add_noise_to_model_gradients, GaussianMechanism
from .privacy_accountant import RDPAccountant


class DPClientTrainer:
    """
    Client trainer with differential privacy guarantees.
    
    Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
    from Abadi et al. 2016.
    
    Key components:
    1. Per-sample gradient clipping
    2. Gaussian noise addition
    3. Privacy accounting
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        device: str = "cpu"
    ):
        """
        Initialize DP trainer.
        
        Args:
            model: PyTorch model
            noise_multiplier: Noise multiplier (sigma/C)
            max_grad_norm: Gradient clipping bound (C)
            delta: Target delta for DP
            device: Training device
        """
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.device = device
        
        self.mechanism = GaussianMechanism(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            device=device
        )
        
        self.accountant = RDPAccountant()
        
    def train_step(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """
        Train with DP for specified epochs.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of local epochs
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        total_grad_norms = []
        
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        sampling_rate = batch_size / dataset_size
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                grad_norm = clip_model_gradients(
                    self.model,
                    self.max_grad_norm
                )
                total_grad_norms.append(grad_norm)
                
                # Add noise to gradients
                add_noise_to_model_gradients(
                    self.model,
                    self.noise_multiplier,
                    self.max_grad_norm,
                    num_samples=len(data)
                )
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_samples += len(data)
                
                # Accumulate privacy
                self.accountant.accumulate(
                    noise_multiplier=self.noise_multiplier,
                    sampling_rate=sampling_rate,
                    num_steps=1
                )
        
        return {
            "loss": total_loss / total_samples,
            "avg_grad_norm": sum(total_grad_norms) / len(total_grad_norms) if total_grad_norms else 0,
            "samples": total_samples,
            "epsilon": self.accountant.get_epsilon(self.delta)
        }
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        epsilon = self.accountant.get_epsilon(self.delta)
        return epsilon, self.delta
    
    def get_model_update(
        self,
        initial_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get model update (delta) from training.
        
        Args:
            initial_state: Initial model state dict
            
        Returns:
            Dictionary of parameter updates
        """
        current_state = self.model.state_dict()
        
        update = {}
        for key in current_state:
            update[key] = current_state[key] - initial_state[key]
        
        return update
    
    def reset_privacy_accountant(self):
        """Reset privacy accountant."""
        self.accountant.reset()


class DPFederatedClient:
    """
    Federated learning client with differential privacy.
    
    Wraps model training with DP guarantees and handles
    communication with the server.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        dp_config: dict,
        training_config: dict,
        device: str = "cpu"
    ):
        """
        Initialize DP federated client.
        
        Args:
            client_id: Unique client identifier
            model: Local model
            dataloader: Client's data loader
            dp_config: DP configuration dict
            training_config: Training configuration dict
            device: Training device
        """
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # DP configuration
        self.dp_enabled = dp_config.get("enabled", True)
        self.noise_multiplier = dp_config.get("noise_multiplier", 1.0)
        self.max_grad_norm = dp_config.get("max_grad_norm", 1.0)
        self.delta = dp_config.get("delta", 1e-5)
        
        # Training configuration
        self.local_epochs = training_config.get("local_epochs", 5)
        self.lr = training_config.get("learning_rate", 0.01)
        self.momentum = training_config.get("momentum", 0.9)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        
        # Initialize DP trainer if enabled
        if self.dp_enabled:
            self.dp_trainer = DPClientTrainer(
                model=self.model,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                delta=self.delta,
                device=device
            )
        else:
            self.dp_trainer = None
        
        # Privacy accounting
        self.total_epsilon = 0.0
        
    def receive_global_model(self, global_state: Dict[str, torch.Tensor]):
        """
        Receive and load global model from server.
        
        Args:
            global_state: Global model state dict
        """
        self.model.load_state_dict(global_state)
        self.model.to(self.device)
    
    def local_train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training and return model update.
        
        Returns:
            Tuple of (model_update, training_metrics)
        """
        # Save initial state
        initial_state = copy.deepcopy(self.model.state_dict())
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        if self.dp_enabled and self.dp_trainer is not None:
            # Train with DP
            metrics = self.dp_trainer.train_step(
                dataloader=self.dataloader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=self.local_epochs
            )
            
            self.total_epsilon = metrics["epsilon"]
        else:
            # Train without DP
            metrics = self._train_without_dp(
                optimizer=optimizer,
                criterion=criterion
            )
        
        # Compute model update
        model_update = {}
        current_state = self.model.state_dict()
        for key in current_state:
            model_update[key] = current_state[key] - initial_state[key]
        
        metrics["client_id"] = self.client_id
        metrics["num_samples"] = len(self.dataloader.dataset)
        
        return model_update, metrics
    
    def _train_without_dp(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train without differential privacy."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_samples += len(data)
        
        return {
            "loss": total_loss / total_samples,
            "samples": total_samples,
            "epsilon": 0.0  # No privacy spent
        }
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy budget spent."""
        return self.total_epsilon, self.delta
