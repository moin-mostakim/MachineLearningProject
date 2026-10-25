"""
Distributionally Robust Optimization (DRO) Training.

Based on:
- Sinha et al. (2018): "Certifiable Distributional Robustness with Principled Adversarial Training"
- Ben-Tal et al. (2013): "Robust Optimization"

Key idea: Instead of robust to worst-case perturbations, be robust to worst-case
distributions in a Wasserstein ball around the empirical distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base import BaseTrainer


class DROTrainer(BaseTrainer):
    """
    Wasserstein Distributionally Robust Optimization.
    
    Reference: Sinha et al. (2018)
    
    Objective: min_θ sup_{Q: W_c(Q, P_n) ≤ ρ} E_Q[L(θ; ξ)]
    
    Where W_c is the Wasserstein distance and ρ is the robustness radius.
    
    Using Lagrangian relaxation:
    min_θ max_γ≥0 {E_P_n[sup_z L(θ; z) - γ * c(z, ξ)] + γρ}
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.rho = self.config.dro.rho
        self.gamma = self.config.dro.gamma_init
        self.gamma_lr = self.config.dro.gamma_lr
        self.inner_steps = self.config.dro.inner_steps
        self.inner_lr = self.config.dro.inner_lr
        
        print(f"[DRO Training] rho={self.rho}, gamma_init={self.gamma}, "
              f"inner_steps={self.inner_steps}")
    
    def _inner_maximization(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Inner maximization: find worst-case distribution.
        
        Solve: max_z L(θ; z) - γ * ||z - x||²
        """
        # Initialize z at original samples
        z = images.clone().detach().requires_grad_(True)
        
        for _ in range(self.inner_steps):
            # Compute objective
            outputs = self.model(z)
            loss = self.criterion(outputs, labels)
            
            # Transport cost (squared L2 distance)
            transport_cost = ((z - images) ** 2).sum(dim=(1, 2, 3)).mean()
            
            # Objective to maximize
            objective = loss - self.gamma * transport_cost
            
            # Gradient ascent
            grad = torch.autograd.grad(objective, z)[0]
            z = z.detach() + self.inner_lr * grad
            z = torch.clamp(z, 0, 1)
            z.requires_grad_(True)
        
        return z.detach()
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """DRO training step."""
        # Inner maximization - find worst-case samples
        self.model.eval()
        z_worst = self._inner_maximization(images, labels)
        
        # Outer minimization - update model parameters
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(z_worst)
        loss = self.criterion(outputs, labels)
        
        # Transport cost for dual variable update
        transport_cost = ((z_worst - images) ** 2).sum(dim=(1, 2, 3)).mean()
        
        # Full DRO objective
        dro_loss = loss + self.gamma * (self.rho - transport_cost)
        
        dro_loss.backward()
        self.optimizer.step()
        
        # Update dual variable γ (gradient ascent on Lagrangian)
        with torch.no_grad():
            self.gamma = max(0, self.gamma + self.gamma_lr * (transport_cost.item() - self.rho))
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            
            # Clean accuracy
            clean_outputs = self.model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct = (clean_preds == labels).sum().item()
        
        return {
            "loss": loss.item(),
            "dro_loss": dro_loss.item(),
            "transport_cost": transport_cost.item(),
            "gamma": self.gamma,
            "correct": correct,
            "clean_correct": clean_correct
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train epoch with DRO-specific metrics."""
        self.model.train()
        total_loss = 0
        total_dro_loss = 0
        total_transport = 0
        correct = 0
        clean_correct = 0
        total = 0
        
        debug_mode = self.config.training.debug
        max_batches = self.config.training.debug_batches if debug_mode else float('inf')
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            metrics = self.train_step(images, labels)
            
            total_loss += metrics["loss"]
            total_dro_loss += metrics["dro_loss"]
            total_transport += metrics["transport_cost"]
            correct += metrics["correct"]
            clean_correct += metrics.get("clean_correct", metrics["correct"])
            total += labels.size(0)
            
            self.global_step += 1
            
            if batch_idx % self.config.log_interval == 0:
                print(f"  Batch {batch_idx}: Loss={metrics['loss']:.4f}, "
                      f"γ={metrics['gamma']:.4f}, T={metrics['transport_cost']:.4f}")
        
        n_batches = min(len(self.train_loader), max_batches)
        
        return {
            "loss": total_loss / n_batches,
            "dro_loss": total_dro_loss / n_batches,
            "transport_cost": total_transport / n_batches,
            "accuracy": correct / total,
            "clean_accuracy": clean_correct / total,
            "gamma": self.gamma
        }


class GroupDROTrainer(BaseTrainer):
    """
    Group Distributionally Robust Optimization.
    
    Reference: Sagawa et al. (2020) "Distributionally Robust Neural Networks"
    
    Optimizes for worst-case group performance rather than average performance.
    """
    
    def __init__(self, *args, num_groups: int = 10, adjustment: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_groups = num_groups
        self.adjustment = adjustment
        
        # Group weights (initialized uniformly)
        self.group_weights = torch.ones(num_groups) / num_groups
        self.group_counts = torch.zeros(num_groups)
        
        print(f"[Group DRO] num_groups={num_groups}, adjustment={adjustment}")
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Group DRO training step (using labels as groups for CIFAR-10)."""
        self.optimizer.zero_grad()
        
        outputs = self.model(images)
        
        # Compute per-sample losses
        losses = F.cross_entropy(outputs, labels, reduction='none')
        
        # Compute group losses (using class labels as groups)
        group_losses = torch.zeros(self.num_groups, device=images.device)
        group_counts = torch.zeros(self.num_groups, device=images.device)
        
        for g in range(self.num_groups):
            mask = labels == g
            if mask.sum() > 0:
                group_losses[g] = losses[mask].mean()
                group_counts[g] = mask.sum()
        
        # Update group weights based on losses
        with torch.no_grad():
            # Exponential weights based on loss
            self.group_weights = self.group_weights.to(images.device)
            active_groups = group_counts > 0
            if active_groups.sum() > 0:
                # Robust optimization: upweight groups with higher loss
                adjusted_losses = group_losses + self.adjustment
                self.group_weights[active_groups] = torch.softmax(
                    adjusted_losses[active_groups], dim=0
                )
        
        # Weighted loss
        weighted_loss = (self.group_weights * group_losses).sum()
        
        weighted_loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            
            # Worst group accuracy
            group_correct = torch.zeros(self.num_groups)
            for g in range(self.num_groups):
                mask = labels == g
                if mask.sum() > 0:
                    group_correct[g] = (preds[mask] == labels[mask]).float().mean()
            
            worst_group_acc = group_correct[group_counts.cpu() > 0].min().item()
        
        return {
            "loss": weighted_loss.item(),
            "correct": correct,
            "worst_group_acc": worst_group_acc
        }
