"""Differential privacy mechanisms for gradient perturbation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class DPMechanism(ABC):
    """Abstract base class for DP mechanisms."""
    
    @abstractmethod
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add noise to tensor for DP guarantee."""
        pass
    
    @abstractmethod
    def clip_gradient(self, gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clip gradient to bound sensitivity."""
        pass


class GaussianMechanism(DPMechanism):
    """
    Gaussian mechanism for differential privacy.
    
    Based on:
    - Deep Learning with Differential Privacy (Abadi et al., 2016)
    - The Algorithmic Foundations of Differential Privacy (Dwork & Roth, 2014)
    
    For (epsilon, delta)-DP with L2 sensitivity S:
    sigma >= sqrt(2 * ln(1.25/delta)) * S / epsilon
    
    In practice, we use noise_multiplier = sigma / S
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize Gaussian mechanism.
        
        Args:
            noise_multiplier: Ratio of noise std to clipping norm (sigma/C)
            max_grad_norm: Maximum gradient L2 norm (clipping bound C)
            device: Device for noise generation
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Noise std = noise_multiplier * max_grad_norm
        self.noise_std = noise_multiplier * max_grad_norm
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor (typically aggregated gradients)
            
        Returns:
            Noisy tensor with same shape
        """
        noise = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype
        )
        return tensor + noise
    
    def clip_gradient(
        self,
        gradient: torch.Tensor,
        max_norm: Optional[float] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Clip gradient to bound L2 norm.
        
        Args:
            gradient: Input gradient tensor
            max_norm: Maximum L2 norm (uses self.max_grad_norm if None)
            
        Returns:
            Tuple of (clipped gradient, original norm)
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
            
        grad_norm = gradient.norm(2)
        clip_factor = max(1.0, grad_norm / max_norm)
        
        return gradient / clip_factor, grad_norm.item()


def clip_model_gradients(
    model: nn.Module,
    max_grad_norm: float
) -> float:
    """
    Clip all gradients in model to bound total L2 norm.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum total L2 norm
        
    Returns:
        Original gradient norm before clipping
    """
    # Compute total gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip
    clip_factor = max(1.0, total_norm / max_grad_norm)
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(clip_factor)
    
    return total_norm


def add_noise_to_model_gradients(
    model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
    num_samples: int
) -> None:
    """
    Add Gaussian noise to all gradients in model.
    
    Args:
        model: PyTorch model
        noise_multiplier: Noise multiplier (sigma/C)
        max_grad_norm: Gradient clipping norm (C)
        num_samples: Number of samples (for scaling)
    """
    noise_std = noise_multiplier * max_grad_norm / num_samples
    
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=p.grad.shape,
                device=p.grad.device,
                dtype=p.grad.dtype
            )
            p.grad.data.add_(noise)


def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    num_steps: int,
    sampling_rate: float,
    accountant: str = "rdp"
) -> float:
    """
    Compute noise multiplier for target (epsilon, delta)-DP.
    
    Uses binary search to find minimum noise_multiplier.
    
    Args:
        epsilon: Target privacy budget
        delta: Target delta
        num_steps: Total number of training steps
        sampling_rate: Batch sampling rate (batch_size / dataset_size)
        accountant: Type of privacy accountant ("rdp" or "simple")
        
    Returns:
        Noise multiplier achieving target privacy
    """
    # Simple analytical bound for Gaussian mechanism
    # This is a conservative estimate; RDP gives tighter bounds
    
    # For subsampled Gaussian mechanism with T steps:
    # epsilon ~= sqrt(2 * T * q^2 * ln(1/delta)) / sigma
    # where q = sampling_rate, sigma = noise_multiplier
    
    # Solving for sigma:
    # sigma = sqrt(2 * T * q^2 * ln(1/delta)) / epsilon
    
    if accountant == "simple":
        sigma = np.sqrt(2 * num_steps * sampling_rate**2 * np.log(1/delta)) / epsilon
        return max(sigma, 0.1)  # Minimum noise
    
    # Binary search for RDP accountant
    low, high = 0.1, 100.0
    
    while high - low > 0.01:
        mid = (low + high) / 2
        # Import here to avoid circular dependency
        from .privacy_accountant import RDPAccountant
        
        accountant = RDPAccountant()
        spent_eps = accountant.compute_epsilon(
            noise_multiplier=mid,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
            delta=delta
        )
        
        if spent_eps < epsilon:
            high = mid
        else:
            low = mid
    
    return high
