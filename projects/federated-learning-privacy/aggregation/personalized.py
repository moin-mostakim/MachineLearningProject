"""Personalized aggregation for heterogeneous federated learning.

Implements personalized FL with differential privacy, combining:
1. Weighted aggregation based on client similarity
2. Layer-wise personalization (some layers remain local)
3. Differential privacy guarantees
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Set
import copy
import numpy as np


class PersonalizedAggregator:
    """
    Personalized aggregator for heterogeneous federated learning.
    
    Key features:
    1. Weighted aggregation: Clients with similar data get more weight
    2. Layer-wise personalization: Keep some layers local
    3. DP integration: Add noise to aggregated updates
    
    This approach helps handle non-IID data by allowing clients
    to maintain personalized components while still benefiting
    from collaborative learning.
    """
    
    def __init__(
        self,
        personalized_layers: List[str] = None,
        mixing_weight: float = 0.5,
        dp_enabled: bool = True,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize personalized aggregator.
        
        Args:
            personalized_layers: Layer names to keep personalized
            mixing_weight: Weight for local model in mixture (alpha)
            dp_enabled: Whether to apply DP to aggregation
            noise_multiplier: DP noise multiplier
            max_grad_norm: DP gradient clipping norm
            device: Computation device
        """
        self.personalized_layers = set(personalized_layers or ["fc", "classifier"])
        self.mixing_weight = mixing_weight
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.client_personalized_states: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def is_personalized_layer(self, layer_name: str) -> bool:
        """Check if a layer should be personalized."""
        for personal_layer in self.personalized_layers:
            if personal_layer in layer_name:
                return True
        return False
    
    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
        client_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with personalization.
        
        Args:
            client_updates: List of (client_id, model_update, num_samples) tuples
            client_weights: Optional custom weights per client
            
        Returns:
            Aggregated global model state (excluding personalized layers)
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate weights
        total_samples = sum(num_samples for _, _, num_samples in client_updates)
        
        if client_weights is None:
            client_weights = {
                client_id: num_samples / total_samples
                for client_id, _, num_samples in client_updates
            }
        
        # Separate global and personalized updates
        first_update = client_updates[0][1]
        global_update = {}
        
        # Initialize global update (only non-personalized layers)
        for key in first_update:
            if not self.is_personalized_layer(key):
                global_update[key] = torch.zeros_like(first_update[key])
        
        # Aggregate non-personalized layers
        for client_id, update, num_samples in client_updates:
            weight = client_weights.get(client_id, num_samples / total_samples)
            
            for key in update:
                if not self.is_personalized_layer(key):
                    global_update[key] += weight * update[key]
                else:
                    # Store personalized layers per client
                    if client_id not in self.client_personalized_states:
                        self.client_personalized_states[client_id] = {}
                    
                    if self.global_model_state is not None and key in self.global_model_state:
                        # Store the full state (global + update)
                        self.client_personalized_states[client_id][key] = (
                            self.global_model_state[key] + update[key]
                        )
                    else:
                        self.client_personalized_states[client_id][key] = update[key]
        
        # Apply DP noise to aggregated update
        if self.dp_enabled:
            global_update = self._add_dp_noise(global_update)
        
        # Update global model state
        if self.global_model_state is not None:
            new_global_state = {}
            for key in self.global_model_state:
                if key in global_update:
                    new_global_state[key] = self.global_model_state[key] + global_update[key]
                else:
                    # Keep personalized layers from global (as default)
                    new_global_state[key] = self.global_model_state[key]
            self.global_model_state = new_global_state
        else:
            self.global_model_state = global_update
        
        return self.global_model_state
    
    def _add_dp_noise(
        self,
        update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise for DP."""
        noisy_update = {}
        
        noise_std = self.noise_multiplier * self.max_grad_norm
        
        for key, tensor in update.items():
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype
            )
            noisy_update[key] = tensor + noise
        
        return noisy_update
    
    def get_personalized_model(
        self,
        client_id: int,
        base_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get personalized model for a specific client.
        
        Combines global layers with client's personalized layers.
        
        Args:
            client_id: Client identifier
            base_state: Base model state (global model)
            
        Returns:
            Personalized model state for the client
        """
        personalized_state = copy.deepcopy(base_state)
        
        # Mix global and local for personalized layers
        if client_id in self.client_personalized_states:
            for key, local_value in self.client_personalized_states[client_id].items():
                if key in personalized_state:
                    # Mixture: alpha * local + (1 - alpha) * global
                    personalized_state[key] = (
                        self.mixing_weight * local_value +
                        (1 - self.mixing_weight) * personalized_state[key]
                    )
        
        return personalized_state
    
    def set_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Set the global model state."""
        self.global_model_state = copy.deepcopy(model_state)
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get the current global model state."""
        if self.global_model_state is None:
            raise ValueError("Global model state not set")
        return copy.deepcopy(self.global_model_state)


class PersonalizedDPClient:
    """
    Federated learning client with personalization and differential privacy.
    
    Combines:
    1. Local training with DP-SGD
    2. Personalized layers that stay local
    3. Adaptive mixing of global and local models
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        personalized_layers: List[str] = None,
        mixing_weight: float = 0.5,
        dp_config: dict = None,
        training_config: dict = None,
        device: str = "cpu"
    ):
        """
        Initialize personalized DP client.
        
        Args:
            client_id: Client identifier
            model: Local model
            dataloader: Client data loader
            personalized_layers: Layers to keep personalized
            mixing_weight: Weight for local model
            dp_config: DP configuration
            training_config: Training configuration
            device: Training device
        """
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # Personalization config
        self.personalized_layers = set(personalized_layers or ["fc", "classifier"])
        self.mixing_weight = mixing_weight
        
        # DP config
        dp_config = dp_config or {}
        self.dp_enabled = dp_config.get("enabled", True)
        self.noise_multiplier = dp_config.get("noise_multiplier", 1.0)
        self.max_grad_norm = dp_config.get("max_grad_norm", 1.0)
        self.delta = dp_config.get("delta", 1e-5)
        
        # Training config
        training_config = training_config or {}
        self.local_epochs = training_config.get("local_epochs", 5)
        self.lr = training_config.get("learning_rate", 0.01)
        self.momentum = training_config.get("momentum", 0.9)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.adaptation_steps = training_config.get("adaptation_steps", 5)
        
        # State tracking
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.personalized_state: Dict[str, torch.Tensor] = {}
        self.total_epsilon = 0.0
        
    def is_personalized_layer(self, layer_name: str) -> bool:
        """Check if layer should be personalized."""
        for personal_layer in self.personalized_layers:
            if personal_layer in layer_name:
                return True
        return False
    
    def receive_global_model(self, global_state: Dict[str, torch.Tensor]):
        """
        Receive global model and merge with personalized layers.
        
        Args:
            global_state: Global model state dict
        """
        self.global_model_state = copy.deepcopy(global_state)
        
        # Load model with mixing
        mixed_state = copy.deepcopy(global_state)
        
        for key in mixed_state:
            if self.is_personalized_layer(key) and key in self.personalized_state:
                # Mix local and global
                mixed_state[key] = (
                    self.mixing_weight * self.personalized_state[key] +
                    (1 - self.mixing_weight) * global_state[key]
                )
        
        self.model.load_state_dict(mixed_state)
        self.model.to(self.device)
    
    def local_train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training with DP and personalization.
        
        Returns:
            Tuple of (model_update, training_metrics)
        """
        # Save initial state
        initial_state = copy.deepcopy(self.model.state_dict())
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        grad_norms = []
        
        dataset_size = len(self.dataloader.dataset)
        batch_size = self.dataloader.batch_size
        sampling_rate = batch_size / dataset_size
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if self.dp_enabled:
                    # Clip gradients
                    grad_norm = self._clip_gradients()
                    grad_norms.append(grad_norm)
                    
                    # Add noise
                    self._add_gradient_noise(len(data))
                
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_samples += len(data)
        
        # Compute update
        current_state = self.model.state_dict()
        model_update = {}
        
        for key in current_state:
            model_update[key] = current_state[key] - initial_state[key]
            
            # Store personalized layers
            if self.is_personalized_layer(key):
                self.personalized_state[key] = current_state[key].clone()
        
        # Estimate privacy spent
        if self.dp_enabled:
            num_steps = self.local_epochs * len(self.dataloader)
            self.total_epsilon = self._estimate_epsilon(
                num_steps, sampling_rate
            )
        
        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / total_samples,
            "num_samples": len(self.dataloader.dataset),
            "avg_grad_norm": np.mean(grad_norms) if grad_norms else 0,
            "epsilon": self.total_epsilon
        }
        
        return model_update, metrics
    
    def _clip_gradients(self) -> float:
        """Clip gradients to max norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_factor = max(1.0, total_norm / self.max_grad_norm)
        
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.div_(clip_factor)
        
        return total_norm
    
    def _add_gradient_noise(self, batch_size: int):
        """Add Gaussian noise to gradients."""
        noise_std = self.noise_multiplier * self.max_grad_norm / batch_size
        
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=p.grad.shape,
                    device=p.grad.device,
                    dtype=p.grad.dtype
                )
                p.grad.data.add_(noise)
    
    def _estimate_epsilon(
        self,
        num_steps: int,
        sampling_rate: float
    ) -> float:
        """Estimate privacy spent using RDP."""
        # Simplified RDP accounting
        # For more accurate accounting, use privacy_accountant
        alpha = 2  # Order of Rényi divergence
        rdp = num_steps * sampling_rate**2 * alpha / (2 * self.noise_multiplier**2)
        epsilon = rdp + np.log(1 / self.delta) / (alpha - 1)
        return epsilon
    
    def local_adaptation(self, num_steps: int = None):
        """
        Perform local adaptation after receiving global model.
        
        This fine-tunes personalized layers on local data.
        """
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Only train personalized layers
        for name, param in self.model.named_parameters():
            param.requires_grad = self.is_personalized_layer(name)
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr * 0.1  # Lower LR for fine-tuning
        )
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        step = 0
        for data, target in self.dataloader:
            if step >= num_steps:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            step += 1
        
        # Re-enable all gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Update personalized state
        current_state = self.model.state_dict()
        for key in current_state:
            if self.is_personalized_layer(key):
                self.personalized_state[key] = current_state[key].clone()
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy budget spent."""
        return self.total_epsilon, self.delta
