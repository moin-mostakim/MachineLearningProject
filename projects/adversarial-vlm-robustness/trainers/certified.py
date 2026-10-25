"""
Certified Training via Randomized Smoothing.

Based on Cohen et al. (2019): "Certified Adversarial Robustness via Randomized Smoothing"
https://arxiv.org/abs/1902.02918
"""

import torch
import torch.nn as nn
from typing import Dict

from .base import BaseTrainer
from attacks.certified import RandomizedSmoothing


class CertifiedTrainer(BaseTrainer):
    """
    Certified training using Gaussian noise augmentation.
    
    Reference: Cohen et al. (2019)
    
    Key idea: Train with Gaussian noise added to inputs, which improves
    the performance of the smoothed classifier and enables certification.
    
    The smoothed classifier g(x) = argmax_c P(f(x+η) = c) where η ~ N(0, σ²I)
    is provably robust: if p_A > 0.5, then g(x+δ) = g(x) for all ||δ||_2 ≤ r
    where r = σ * Φ^(-1)(p_A)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sigma = self.config.certified.sigma
        self.num_noise_samples = 1  # Samples per input during training
        
        print(f"[Certified Training] sigma={self.sigma}")
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Certified training step with noise augmentation."""
        self.optimizer.zero_grad()
        
        # Add Gaussian noise
        noise = torch.randn_like(images) * self.sigma
        noisy_images = torch.clamp(images + noise, 0, 1)
        
        if self.config.training.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(noisy_images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        # Compute accuracy (on noisy images)
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            
            # Also compute clean accuracy
            clean_outputs = self.model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct = (clean_preds == labels).sum().item()
        
        return {
            "loss": loss.item(),
            "correct": correct,
            "clean_correct": clean_correct
        }
    
    def validate_with_smoothing(self, n_samples: int = 100) -> Dict[str, float]:
        """Validate using Monte Carlo smoothing."""
        self.model.eval()
        correct = 0
        total = 0
        
        debug_mode = self.config.training.debug
        max_batches = self.config.training.debug_batches if debug_mode else float('inf')
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch_size = images.shape[0]
                votes = torch.zeros(batch_size, 10, device=self.device)
                
                # Monte Carlo sampling
                for _ in range(n_samples):
                    noise = torch.randn_like(images) * self.sigma
                    noisy_images = torch.clamp(images + noise, 0, 1)
                    outputs = self.model(noisy_images)
                    preds = outputs.argmax(dim=1)
                    
                    for i in range(batch_size):
                        votes[i, preds[i]] += 1
                
                final_preds = votes.argmax(dim=1)
                correct += (final_preds == labels).sum().item()
                total += batch_size
        
        return {"smoothed_accuracy": correct / total}


class SmoothAdvTrainer(CertifiedTrainer):
    """
    Combined smoothing and adversarial training.
    
    Applies adversarial perturbations to noisy images for stronger robustness.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from attacks.pgd import PGDAttack
        self.attacker = PGDAttack(
            model=self.model,
            epsilon=self.config.attack.epsilon,
            alpha=self.config.attack.alpha,
            num_steps=self.config.attack.num_steps // 2  # Fewer steps for efficiency
        )
        
        print(f"[Smooth+Adv Training] sigma={self.sigma}, epsilon={self.config.attack.epsilon}")
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Training with both noise and adversarial perturbations."""
        # Add Gaussian noise
        noise = torch.randn_like(images) * self.sigma
        noisy_images = torch.clamp(images + noise, 0, 1)
        
        # Generate adversarial perturbation on noisy images
        self.model.eval()
        adv_noisy_images = self.attacker(noisy_images, labels)
        
        # Train
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(adv_noisy_images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
        
        return {
            "loss": loss.item(),
            "correct": correct
        }


class ConsistencyRegularizedTrainer(CertifiedTrainer):
    """
    Certified training with consistency regularization.
    
    Adds regularization to encourage consistent predictions under noise.
    """
    
    def __init__(self, *args, consistency_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_weight = consistency_weight
        print(f"[Consistency Certified Training] weight={consistency_weight}")
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Training with consistency regularization."""
        self.optimizer.zero_grad()
        
        # Two noisy versions
        noise1 = torch.randn_like(images) * self.sigma
        noise2 = torch.randn_like(images) * self.sigma
        noisy1 = torch.clamp(images + noise1, 0, 1)
        noisy2 = torch.clamp(images + noise2, 0, 1)
        
        # Forward passes
        outputs1 = self.model(noisy1)
        outputs2 = self.model(noisy2)
        
        # Classification loss
        cls_loss = self.criterion(outputs1, labels)
        
        # Consistency loss (KL divergence between two noisy predictions)
        cons_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(outputs1, dim=1),
            torch.softmax(outputs2.detach(), dim=1)
        )
        
        loss = cls_loss + self.consistency_weight * cons_loss
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            preds = outputs1.argmax(dim=1)
            correct = (preds == labels).sum().item()
        
        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "cons_loss": cons_loss.item(),
            "correct": correct
        }
