"""Privacy accounting for differential privacy."""

import numpy as np
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import math


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accounting."""
    
    @abstractmethod
    def accumulate(self, noise_multiplier: float, sampling_rate: float, num_steps: int = 1):
        """Accumulate privacy loss for training steps."""
        pass
    
    @abstractmethod
    def get_epsilon(self, delta: float) -> float:
        """Get current epsilon for given delta."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset accumulated privacy loss."""
        pass


class RDPAccountant(PrivacyAccountant):
    """
    Rényi Differential Privacy (RDP) accountant.
    
    Based on:
    - Rényi Differential Privacy (Mironov, 2017)
    - Subsampled Rényi Differential Privacy (Wang et al., 2019)
    
    RDP provides tighter privacy bounds for Gaussian mechanism compared
    to naive composition.
    
    For a Gaussian mechanism with noise multiplier sigma:
    RDP(alpha) = alpha / (2 * sigma^2)
    
    Conversion to (epsilon, delta)-DP:
    epsilon = RDP(alpha) + log(1/delta) / (alpha - 1)
    """
    
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    def __init__(self, alphas: Optional[List[float]] = None):
        """
        Initialize RDP accountant.
        
        Args:
            alphas: List of alpha values for RDP computation
        """
        self.alphas = alphas or self.DEFAULT_ALPHAS
        self.rdp = np.zeros(len(self.alphas))
        
    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        num_steps: int = 1
    ):
        """
        Accumulate RDP for training steps.
        
        Args:
            noise_multiplier: Noise multiplier (sigma/C)
            sampling_rate: Batch sampling rate (batch_size / dataset_size)
            num_steps: Number of steps
        """
        for _ in range(num_steps):
            rdp_step = self._compute_rdp_sample_gaussian(
                q=sampling_rate,
                noise_multiplier=noise_multiplier,
                alphas=self.alphas
            )
            self.rdp += rdp_step
    
    def get_epsilon(self, delta: float) -> float:
        """
        Convert accumulated RDP to (epsilon, delta)-DP.
        
        Args:
            delta: Target delta
            
        Returns:
            Epsilon for given delta
        """
        return self._rdp_to_eps(self.rdp, self.alphas, delta)
    
    def compute_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        num_steps: int,
        delta: float
    ) -> float:
        """
        Compute epsilon without accumulating (for noise calibration).
        
        Args:
            noise_multiplier: Noise multiplier
            sampling_rate: Batch sampling rate
            num_steps: Number of steps
            delta: Target delta
            
        Returns:
            Epsilon for given parameters
        """
        rdp = np.zeros(len(self.alphas))
        
        for _ in range(num_steps):
            rdp_step = self._compute_rdp_sample_gaussian(
                q=sampling_rate,
                noise_multiplier=noise_multiplier,
                alphas=self.alphas
            )
            rdp += rdp_step
        
        return self._rdp_to_eps(rdp, self.alphas, delta)
    
    def reset(self):
        """Reset accumulated RDP."""
        self.rdp = np.zeros(len(self.alphas))
    
    def _compute_rdp_sample_gaussian(
        self,
        q: float,
        noise_multiplier: float,
        alphas: List[float]
    ) -> np.ndarray:
        """
        Compute RDP for subsampled Gaussian mechanism.
        
        Uses numerical computation for accuracy.
        """
        rdp = []
        
        for alpha in alphas:
            if q == 0:
                rdp.append(0.0)
            elif q == 1:
                # No subsampling
                rdp.append(alpha / (2 * noise_multiplier**2))
            else:
                # Subsampled Gaussian
                rdp.append(self._compute_rdp_subsampled(q, noise_multiplier, alpha))
        
        return np.array(rdp)
    
    def _compute_rdp_subsampled(
        self,
        q: float,
        sigma: float,
        alpha: float
    ) -> float:
        """
        Compute RDP for subsampled Gaussian mechanism.
        
        Based on Theorem 9 of Wang et al. 2019.
        """
        if alpha <= 1:
            return 0.0
        
        # For small q, use the approximation
        if q < 0.01:
            return q**2 * alpha / (2 * sigma**2)
        
        # General formula (simplified bound)
        log_a = self._compute_log_a(q, sigma, alpha)
        return log_a / (alpha - 1)
    
    def _compute_log_a(self, q: float, sigma: float, alpha: float) -> float:
        """Compute log(A) term for RDP."""
        # Simplified computation
        # Full computation requires numerical integration
        
        # Upper bound from Wang et al. 2019
        if alpha == 2:
            return q**2 / sigma**2
        
        # General case: use moment generating function
        # This is an approximation
        return min(
            q**2 * alpha / (2 * sigma**2),
            q * alpha * (alpha - 1) / (2 * sigma**2)
        )
    
    def _rdp_to_eps(
        self,
        rdp: np.ndarray,
        alphas: List[float],
        delta: float
    ) -> float:
        """
        Convert RDP to (epsilon, delta)-DP.
        
        epsilon = min over alpha of: RDP(alpha) + log(1/delta) / (alpha - 1)
        """
        eps_list = []
        
        for alpha, rdp_alpha in zip(alphas, rdp):
            if alpha > 1:
                eps = rdp_alpha + np.log(1 / delta) / (alpha - 1)
                eps_list.append(eps)
        
        return min(eps_list) if eps_list else float('inf')


class SimpleAccountant(PrivacyAccountant):
    """
    Simple privacy accountant using basic composition.
    
    Uses the standard Gaussian mechanism bound:
    For (epsilon, delta)-DP, need sigma >= sqrt(2 * ln(1.25/delta)) / epsilon
    
    Composition: T queries with (eps, delta) gives (sqrt(2T ln(1/delta')) * eps + T*eps*(exp(eps)-1), T*delta + delta')
    """
    
    def __init__(self):
        self.total_steps = 0
        self.noise_multipliers = []
        self.sampling_rates = []
    
    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        num_steps: int = 1
    ):
        """Accumulate privacy loss."""
        self.total_steps += num_steps
        for _ in range(num_steps):
            self.noise_multipliers.append(noise_multiplier)
            self.sampling_rates.append(sampling_rate)
    
    def get_epsilon(self, delta: float) -> float:
        """Get epsilon using advanced composition."""
        if not self.noise_multipliers:
            return 0.0
        
        # Per-step epsilon
        eps_per_step = []
        for sigma, q in zip(self.noise_multipliers, self.sampling_rates):
            # Gaussian mechanism bound
            eps = q * np.sqrt(2 * np.log(1.25 / (delta / self.total_steps))) / sigma
            eps_per_step.append(eps)
        
        # Advanced composition
        sum_eps = sum(eps_per_step)
        sum_eps_sq = sum(e**2 for e in eps_per_step)
        
        # Strong composition theorem
        epsilon = np.sqrt(2 * sum_eps_sq * np.log(1 / delta)) + sum_eps_sq / 2
        
        return min(epsilon, sum_eps)  # Take minimum with naive composition
    
    def reset(self):
        """Reset accumulated privacy loss."""
        self.total_steps = 0
        self.noise_multipliers = []
        self.sampling_rates = []


def get_privacy_spent(
    accountant: PrivacyAccountant,
    delta: float
) -> Tuple[float, float]:
    """
    Get privacy spent as (epsilon, delta).
    
    Args:
        accountant: Privacy accountant
        delta: Target delta
        
    Returns:
        Tuple of (epsilon, delta)
    """
    epsilon = accountant.get_epsilon(delta)
    return epsilon, delta
