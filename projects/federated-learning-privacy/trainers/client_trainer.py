"""Client-side training for federated learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import copy


class ClientTrainer:
    """
    Generic client trainer for federated learning.
    
    Supports multiple training modes:
    - Standard SGD (FedAvg)
    - FedProx with proximal term
    - DP-SGD with differential privacy
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
        mode: str = "fedavg",
        config: dict = None
    ):
        """
        Initialize client trainer.
        
        Args:
            client_id: Client identifier
            model: PyTorch model
            dataloader: Client's data loader
            device: Training device
            mode: Training mode ("fedavg", "fedprox", "dp")
            config: Configuration dictionary
        """
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.mode = mode
        self.config = config or {}
        
        # Training parameters
        self.lr = self.config.get("learning_rate", 0.01)
        self.momentum = self.config.get("momentum", 0.9)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.local_epochs = self.config.get("local_epochs", 5)
        
        # FedProx parameters
        self.mu = self.config.get("mu", 0.01)
        
        # DP parameters
        self.noise_multiplier = self.config.get("noise_multiplier", 1.0)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        
        # State
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        
    def set_global_model(self, global_state: Dict[str, torch.Tensor]):
        """Set reference global model."""
        self.global_model_state = {
            k: v.clone().to(self.device)
            for k, v in global_state.items()
        }
        self.model.load_state_dict(global_state)
        self.model.to(self.device)
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training.
        
        Returns:
            Tuple of (model_update, metrics)
        """
        if self.mode == "fedavg":
            return self._train_fedavg()
        elif self.mode == "fedprox":
            return self._train_fedprox()
        elif self.mode == "dp":
            return self._train_dp()
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")
    
    def _train_fedavg(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Standard FedAvg training."""
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
        correct = 0
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
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        # Compute update
        current_state = self.model.state_dict()
        update = {k: current_state[k] - initial_state[k] for k in current_state}
        
        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * correct / total_samples,
            "num_samples": len(self.dataloader.dataset)
        }
        
        return update, metrics
    
    def _train_fedprox(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """FedProx training with proximal term."""
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
        total_prox_loss = 0.0
        correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                task_loss = criterion(output, target)
                
                # Proximal term
                prox_loss = self._compute_proximal_term()
                
                loss = task_loss + prox_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_prox_loss += prox_loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        current_state = self.model.state_dict()
        update = {k: current_state[k] - initial_state[k] for k in current_state}
        
        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / total_samples,
            "prox_loss": total_prox_loss / total_samples,
            "accuracy": 100.0 * correct / total_samples,
            "num_samples": len(self.dataloader.dataset)
        }
        
        return update, metrics
    
    def _train_dp(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """DP-SGD training."""
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
        correct = 0
        total_samples = 0
        grad_norms = []
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Clip gradients
                grad_norm = self._clip_gradients()
                grad_norms.append(grad_norm)
                
                # Add noise
                self._add_gradient_noise(len(data))
                
                optimizer.step()
                
                total_loss += loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        current_state = self.model.state_dict()
        update = {k: current_state[k] - initial_state[k] for k in current_state}
        
        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * correct / total_samples,
            "num_samples": len(self.dataloader.dataset),
            "avg_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0
        }
        
        return update, metrics
    
    def _compute_proximal_term(self) -> torch.Tensor:
        """Compute FedProx proximal term."""
        if self.global_model_state is None:
            return torch.tensor(0.0, device=self.device)
        
        prox_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.global_model_state:
                global_param = self.global_model_state[name]
                prox_loss += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2) * prox_loss
    
    def _clip_gradients(self) -> float:
        """Clip gradients for DP."""
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
        """Add Gaussian noise for DP."""
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
    
    def evaluate(self, dataloader: DataLoader = None) -> Dict[str, float]:
        """Evaluate model on data."""
        if dataloader is None:
            dataloader = self.dataloader
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0,
            "accuracy": 100.0 * correct / total_samples if total_samples > 0 else 0,
            "num_samples": total_samples
        }
