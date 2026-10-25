"""
Projected Gradient Descent (PGD) Attack.

Based on Madry et al. (2018): "Towards Deep Learning Models Resistant to Adversarial Attacks"
https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class PGDAttack:
    """
    PGD Attack for generating adversarial examples.
    
    Reference: Madry et al. (2018)
    
    Key parameters:
    - epsilon: Maximum L-inf perturbation (8/255 standard for CIFAR-10)
    - alpha: Step size (typically epsilon/4)
    - num_steps: Number of gradient ascent steps (7-20 typical)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 7,
        random_start: bool = True,
        targeted: bool = False,
        loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.targeted = targeted
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
    
    def perturb(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        Args:
            images: Clean images (B, C, H, W)
            labels: True labels
            target_labels: Target labels for targeted attack
            
        Returns:
            Adversarial images
        """
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        # Initialize perturbation
        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(images + delta, 0, 1) - images
        else:
            delta = torch.zeros_like(images)
        
        delta.requires_grad = True
        
        for _ in range(self.num_steps):
            adv_images = images + delta
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Compute loss
            if self.targeted and target_labels is not None:
                loss = -self.loss_fn(outputs, target_labels)
            else:
                loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta_update = self.alpha * grad.sign()
            delta = delta.detach() + delta_update
            
            # Project to epsilon ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = torch.clamp(images + delta, 0, 1) - images
            delta.requires_grad = True
        
        return (images + delta.detach()).clamp(0, 1)
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return self.perturb(images, labels, **kwargs)


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8/255,
    alpha: float = 2/255,
    num_steps: int = 7,
    random_start: bool = True
) -> torch.Tensor:
    """
    Functional interface for PGD attack.
    
    Args:
        model: Target model
        images: Clean images
        labels: True labels
        epsilon: L-inf perturbation bound
        alpha: Step size
        num_steps: Number of PGD steps
        random_start: Random initialization within epsilon ball
        
    Returns:
        Adversarial images
    """
    attacker = PGDAttack(
        model=model,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        random_start=random_start
    )
    return attacker(images, labels)


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) - single-step attack.
    
    Reference: Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples"
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255
    ):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()
    
    def perturb(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        images = images.clone().detach().requires_grad_(True)
        
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        
        perturbation = self.epsilon * images.grad.sign()
        adv_images = images + perturbation
        
        return adv_images.clamp(0, 1).detach()
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.perturb(images, labels)


class AutoAttack:
    """
    Simplified Auto-Attack for robust evaluation.
    
    Combines multiple attacks: APGD-CE, APGD-DLR, FAB, Square
    
    Reference: Croce & Hein (2020) "Reliable evaluation of adversarial robustness 
    with an ensemble of diverse parameter-free attacks"
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        norm: str = "Linf"
    ):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        
        # Use PGD with different configurations as approximation
        self.attacks = [
            PGDAttack(model, epsilon, epsilon/4, num_steps=20, random_start=True),
            PGDAttack(model, epsilon, epsilon/8, num_steps=40, random_start=True),
        ]
    
    def perturb(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Find strongest adversarial examples across multiple attacks."""
        best_adv = images.clone()
        best_loss = torch.full((images.shape[0],), float('-inf'), device=images.device)
        
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        for attack in self.attacks:
            adv = attack(images, labels)
            with torch.no_grad():
                outputs = self.model(adv)
                loss = loss_fn(outputs, labels)
            
            # Update best adversarial examples
            improved = loss > best_loss
            best_adv[improved] = adv[improved]
            best_loss[improved] = loss[improved]
        
        return best_adv


def evaluate_robustness(
    model: nn.Module,
    dataloader,
    epsilon: float = 8/255,
    num_steps: int = 20,
    device: str = "cuda"
) -> dict:
    """
    Evaluate model robustness under PGD attack.
    
    Returns:
        Dictionary with clean and adversarial accuracy
    """
    model.eval()
    attacker = PGDAttack(model, epsilon=epsilon, num_steps=num_steps)
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()
        
        # Adversarial accuracy
        adv_images = attacker(images, labels)
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_correct += (adv_preds == labels).sum().item()
        
        total += labels.size(0)
    
    return {
        "clean_accuracy": clean_correct / total,
        "adversarial_accuracy": adv_correct / total,
        "epsilon": epsilon,
        "num_steps": num_steps
    }


if __name__ == "__main__":
    # Test PGD attack
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 10, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)
    
    model = DummyModel()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    
    attacker = PGDAttack(model, epsilon=8/255, num_steps=7)
    adv_images = attacker(images, labels)
    
    perturbation = (adv_images - images).abs().max()
    print(f"Max perturbation: {perturbation.item():.4f}")
    print(f"Within epsilon: {perturbation.item() <= 8/255 + 1e-6}")
