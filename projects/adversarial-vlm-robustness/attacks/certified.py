"""
Certified Defenses via Randomized Smoothing.

Based on Cohen et al. (2019): "Certified Adversarial Robustness via Randomized Smoothing"
https://arxiv.org/abs/1902.02918
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import numpy as np
from typing import Tuple, Optional
import math


class RandomizedSmoothing:
    """
    Randomized smoothing for certified adversarial robustness.
    
    Reference: Cohen et al. (2019)
    
    Key idea: By adding Gaussian noise to inputs and taking majority vote,
    we can certify a radius around each input where the prediction is guaranteed
    to be constant under any L2 perturbation.
    
    Certified radius: r = sigma * Φ^(-1)(p_A)
    where p_A is the probability of the most likely class under Gaussian noise.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        sigma: float = 0.25,
        device: str = "cuda"
    ):
        """
        Args:
            model: Base classifier
            num_classes: Number of output classes
            sigma: Gaussian noise standard deviation
            device: Device for computation
        """
        self.model = model
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
    
    def _sample_noise(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Generate n noisy copies of input x."""
        batch_size = x.shape[0]
        # Repeat input n times
        x_repeated = x.repeat(n, 1, 1, 1)
        noise = torch.randn_like(x_repeated) * self.sigma
        return torch.clamp(x_repeated + noise, 0, 1)
    
    def predict(
        self,
        x: torch.Tensor,
        n: int = 100,
        batch_size: int = 100
    ) -> torch.Tensor:
        """
        Monte Carlo prediction using randomized smoothing.
        
        Args:
            x: Input images (B, C, H, W)
            n: Number of noise samples
            batch_size: Batch size for forward passes
            
        Returns:
            Predicted classes (B,)
        """
        self.model.eval()
        input_batch_size = x.shape[0]
        
        # Accumulate class votes
        votes = torch.zeros(input_batch_size, self.num_classes, device=self.device)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                this_batch_size = min(batch_size, n - i)
                
                # Sample noisy inputs
                noisy_inputs = []
                for j in range(input_batch_size):
                    noise = torch.randn(this_batch_size, *x.shape[1:], device=self.device) * self.sigma
                    noisy = torch.clamp(x[j:j+1].expand(this_batch_size, -1, -1, -1) + noise, 0, 1)
                    noisy_inputs.append(noisy)
                
                noisy_batch = torch.cat(noisy_inputs, dim=0)
                
                # Get predictions
                outputs = self.model(noisy_batch)
                preds = outputs.argmax(dim=1)
                
                # Count votes
                for j in range(input_batch_size):
                    start_idx = j * this_batch_size
                    end_idx = start_idx + this_batch_size
                    batch_preds = preds[start_idx:end_idx]
                    for pred in batch_preds:
                        votes[j, pred] += 1
        
        return votes.argmax(dim=1)
    
    def certify(
        self,
        x: torch.Tensor,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Certify input with confidence bound.
        
        Args:
            x: Single input image (1, C, H, W)
            n0: Number of samples for initial prediction
            n: Number of samples for certification
            alpha: Confidence level (failure probability)
            batch_size: Batch size for forward passes
            
        Returns:
            Tuple of (predicted_class, certified_radius)
            Returns (-1, 0.0) for abstained predictions
        """
        self.model.eval()
        assert x.shape[0] == 1, "Certify one image at a time"
        
        # Step 1: Initial prediction
        votes_0 = torch.zeros(self.num_classes, device=self.device)
        
        with torch.no_grad():
            for i in range(0, n0, batch_size):
                this_batch_size = min(batch_size, n0 - i)
                noise = torch.randn(this_batch_size, *x.shape[1:], device=self.device) * self.sigma
                noisy = torch.clamp(x.expand(this_batch_size, -1, -1, -1) + noise, 0, 1)
                outputs = self.model(noisy)
                preds = outputs.argmax(dim=1)
                for pred in preds:
                    votes_0[pred] += 1
        
        c_hat = votes_0.argmax().item()
        
        # Step 2: Certification with more samples
        votes = torch.zeros(self.num_classes, device=self.device)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                this_batch_size = min(batch_size, n - i)
                noise = torch.randn(this_batch_size, *x.shape[1:], device=self.device) * self.sigma
                noisy = torch.clamp(x.expand(this_batch_size, -1, -1, -1) + noise, 0, 1)
                outputs = self.model(noisy)
                preds = outputs.argmax(dim=1)
                for pred in preds:
                    votes[pred] += 1
        
        n_A = int(votes[c_hat].item())
        
        # Compute lower bound on p_A using binomial confidence interval
        p_A_lower = self._lower_confidence_bound(n_A, n, alpha)
        
        if p_A_lower < 0.5:
            # Cannot certify - abstain
            return torch.tensor(-1), torch.tensor(0.0)
        
        # Certified radius
        radius = self.sigma * stats.norm.ppf(p_A_lower)
        
        return torch.tensor(c_hat), torch.tensor(radius)
    
    def _lower_confidence_bound(self, k: int, n: int, alpha: float) -> float:
        """
        Compute Clopper-Pearson lower confidence bound.
        
        Args:
            k: Number of successes
            n: Number of trials
            alpha: Confidence level
            
        Returns:
            Lower bound on true probability
        """
        return stats.beta.ppf(alpha, k, n - k + 1)


def certify_radius(
    model: nn.Module,
    x: torch.Tensor,
    sigma: float = 0.25,
    n_samples: int = 1000,
    alpha: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda"
) -> Tuple[int, float]:
    """
    Functional interface for certification.
    
    Args:
        model: Classifier
        x: Single input (1, C, H, W)
        sigma: Noise standard deviation
        n_samples: Number of Monte Carlo samples
        alpha: Confidence level
        num_classes: Number of classes
        device: Computation device
        
    Returns:
        Tuple of (predicted_class, certified_radius)
    """
    smoother = RandomizedSmoothing(model, num_classes, sigma, device)
    pred, radius = smoother.certify(x, n0=100, n=n_samples, alpha=alpha)
    return pred.item(), radius.item()


def evaluate_certified_accuracy(
    model: nn.Module,
    dataloader,
    sigma: float = 0.25,
    radii: list = [0.0, 0.25, 0.5, 0.75, 1.0],
    n_samples: int = 1000,
    alpha: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> dict:
    """
    Evaluate certified accuracy at different radii.
    
    Args:
        model: Classifier
        dataloader: Test data
        sigma: Noise standard deviation
        radii: List of radii to evaluate
        n_samples: Samples for certification
        alpha: Confidence level
        num_classes: Number of classes
        device: Computation device
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with certified accuracy at each radius
    """
    model.eval()
    smoother = RandomizedSmoothing(model, num_classes, sigma, device)
    
    correct_at_radius = {r: 0 for r in radii}
    total = 0
    
    for images, labels in dataloader:
        for i in range(images.shape[0]):
            if max_samples is not None and total >= max_samples:
                break
            
            image = images[i:i+1].to(device)
            label = labels[i].item()
            
            pred, radius = smoother.certify(image, n=n_samples, alpha=alpha)
            
            if pred.item() == label:
                for r in radii:
                    if radius.item() >= r:
                        correct_at_radius[r] += 1
            
            total += 1
        
        if max_samples is not None and total >= max_samples:
            break
    
    return {
        f"certified_acc_r{r}": correct_at_radius[r] / total
        for r in radii
    }


class SmoothLoss(nn.Module):
    """
    Loss function for training with randomized smoothing.
    
    Uses noise augmentation during training to improve smoothed classifier accuracy.
    """
    
    def __init__(self, sigma: float = 0.25, num_samples: int = 1):
        super().__init__()
        self.sigma = sigma
        self.num_samples = num_samples
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with noise augmentation."""
        total_loss = 0
        
        for _ in range(self.num_samples):
            noise = torch.randn_like(images) * self.sigma
            noisy_images = torch.clamp(images + noise, 0, 1)
            outputs = model(noisy_images)
            total_loss += self.criterion(outputs, labels)
        
        return total_loss / self.num_samples


if __name__ == "__main__":
    # Test randomized smoothing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    device = "cpu"
    model = DummyModel().to(device)
    
    x = torch.randn(1, 3, 32, 32).to(device)
    
    # Test certification
    pred, radius = certify_radius(
        model, x, sigma=0.25, n_samples=100, 
        num_classes=10, device=device
    )
    print(f"Predicted class: {pred}")
    print(f"Certified radius: {radius:.4f}")
