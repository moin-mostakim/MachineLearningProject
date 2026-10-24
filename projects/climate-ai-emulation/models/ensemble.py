"""
Deep Ensemble implementation for uncertainty quantification.

Deep ensembles train multiple independently initialized models and
aggregate their predictions for:
- Improved accuracy (ensemble averaging)
- Uncertainty quantification (prediction variance)

References:
- Lakshminarayanan et al. (2017) Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles
- Fort et al. (2019) Deep Ensembles: A Loss Landscape Perspective
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable
import copy


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains and manages multiple independent model instances.
    Provides uncertainty through disagreement between models.
    
    Key properties:
    - Models must be trained independently (different random seeds)
    - 5 members is typically sufficient (diminishing returns beyond)
    - Better calibrated than MC Dropout for many tasks
    - Higher computational cost (linear in ensemble size)
    """
    
    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        ensemble_size: int = 5,
        aggregation: str = "mean"
    ):
        """
        Initialize deep ensemble.
        
        Args:
            base_model_fn: Function that returns a new model instance
            ensemble_size: Number of ensemble members
            aggregation: How to combine predictions ("mean", "median")
        """
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.aggregation = aggregation
        
        # Create ensemble members
        self.members = nn.ModuleList([
            base_model_fn() for _ in range(ensemble_size)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - aggregated prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Aggregated prediction
        """
        predictions = []
        for member in self.members:
            pred = member(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (E, B, C, H, W)
        
        if self.aggregation == "mean":
            return predictions.mean(dim=0)
        elif self.aggregation == "median":
            return predictions.median(dim=0).values
        else:
            return predictions.mean(dim=0)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty from ensemble disagreement.
        
        Args:
            x: Input tensor
            return_all: Return all member predictions
            
        Returns:
            Tuple of (mean prediction, variance)
            If return_all: also returns all predictions
        """
        predictions = []
        for member in self.members:
            member.eval()
            with torch.no_grad():
                pred = member(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (E, B, C, H, W)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        if return_all:
            return mean, variance, predictions
        return mean, variance
    
    def get_member(self, idx: int) -> nn.Module:
        """Get specific ensemble member."""
        return self.members[idx]
    
    def count_parameters(self) -> int:
        """Count total parameters (all members)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters_per_member(self) -> int:
        """Count parameters per member."""
        return sum(p.numel() for p in self.members[0].parameters() if p.requires_grad)


class EnsembleTrainer:
    """
    Trainer for deep ensembles.
    
    Manages independent training of each ensemble member.
    """
    
    def __init__(
        self,
        ensemble: DeepEnsemble,
        optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
        loss_fn: nn.Module,
        device: str = "cuda"
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble: DeepEnsemble model
            optimizer_fn: Function to create optimizer for a member
            loss_fn: Loss function
            device: Device to train on
        """
        self.ensemble = ensemble
        self.loss_fn = loss_fn
        self.device = device
        
        # Create optimizer for each member
        self.optimizers = [
            optimizer_fn(member) for member in ensemble.members
        ]
        
    def train_step(
        self,
        batch: dict,
        member_idx: Optional[int] = None
    ) -> dict:
        """
        Training step for one or all members.
        
        Args:
            batch: Data batch with 'input' and 'target'
            member_idx: Train specific member (None = all)
            
        Returns:
            Dictionary of losses
        """
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        
        losses = {}
        
        if member_idx is not None:
            members_to_train = [(member_idx, self.ensemble.members[member_idx])]
        else:
            members_to_train = list(enumerate(self.ensemble.members))
        
        for idx, member in members_to_train:
            member.train()
            self.optimizers[idx].zero_grad()
            
            pred = member(inputs)
            loss = self.loss_fn(pred, targets)
            
            loss.backward()
            self.optimizers[idx].step()
            
            losses[f'member_{idx}'] = loss.item()
        
        losses['mean'] = sum(losses.values()) / len(losses)
        return losses
    
    def validate(self, val_loader) -> dict:
        """
        Validate ensemble.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.ensemble.eval()
        
        total_loss = 0
        total_variance = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                mean, variance = self.ensemble.predict_with_uncertainty(inputs)
                loss = self.loss_fn(mean, targets)
                
                total_loss += loss.item()
                total_variance += variance.mean().item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'mean_variance': total_variance / num_batches
        }


class NegativeLogLikelihoodLoss(nn.Module):
    """
    Negative log-likelihood loss for heteroscedastic models.
    
    For model predicting mean μ and variance σ²:
    NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
    
    This loss encourages both accurate predictions and
    appropriate uncertainty estimates.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL loss.
        
        Args:
            mean: Predicted mean
            variance: Predicted variance
            target: Ground truth
            
        Returns:
            NLL loss value
        """
        variance = variance + self.eps
        
        nll = 0.5 * torch.log(variance) + 0.5 * (target - mean)**2 / variance
        
        return nll.mean()


class CRPS(nn.Module):
    """
    Continuous Ranked Probability Score.
    
    Proper scoring rule for probabilistic predictions.
    Measures calibration and sharpness simultaneously.
    
    For Gaussian predictions:
    CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    
    Reference: Gneiting & Raftery (2007)
    """
    
    def __init__(self):
        super().__init__()
        self.sqrt_pi = torch.sqrt(torch.tensor(torch.pi))
        
    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CRPS assuming Gaussian predictions.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: Ground truth
            
        Returns:
            CRPS value (lower is better)
        """
        z = (target - mean) / (std + 1e-8)
        
        # Standard normal CDF and PDF
        cdf = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device))))
        pdf = torch.exp(-0.5 * z**2) / torch.sqrt(torch.tensor(2 * torch.pi, device=z.device))
        
        # CRPS for Gaussian
        crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / self.sqrt_pi.to(z.device))
        
        return crps.mean()


if __name__ == "__main__":
    # Test deep ensemble
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config
    from models.unet import UNet
    
    config = get_config(debug=True)
    
    # Create base model function
    def create_unet():
        return UNet(
            in_channels=config.data.input_time_steps * config.model.input_channels,
            out_channels=config.model.output_channels,
            base_channels=16,  # Smaller for testing
            depth=2,
            dropout=config.model.dropout_rate
        )
    
    # Create ensemble
    ensemble = DeepEnsemble(
        base_model_fn=create_unet,
        ensemble_size=config.uncertainty.ensemble_size
    )
    
    print(f"Ensemble size: {ensemble.ensemble_size}")
    print(f"Parameters per member: {ensemble.count_parameters_per_member():,}")
    print(f"Total parameters: {ensemble.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(
        batch_size,
        config.data.input_time_steps,
        config.model.input_channels,
        config.data.lat_resolution,
        config.data.lon_resolution
    )
    
    print(f"\nInput shape: {x.shape}")
    
    out = ensemble(x)
    print(f"Output shape: {out.shape}")
    
    # Test uncertainty
    mean, var = ensemble.predict_with_uncertainty(x)
    print(f"Uncertainty - Mean: {mean.shape}, Variance: {var.shape}")
    
    # Test CRPS
    crps = CRPS()
    std = torch.sqrt(var + 1e-8)
    target = torch.randn_like(mean)
    crps_val = crps(mean, std, target)
    print(f"\nCRPS value: {crps_val.item():.4f}")
    
    print("\nEnsemble tests passed!")
