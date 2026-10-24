"""Standard training (baseline)."""

import torch
import torch.nn as nn
from typing import Dict

from .base import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard training baseline without adversarial examples.
    
    This serves as the baseline for comparing adversarial training methods.
    """
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Standard training step."""
        self.optimizer.zero_grad()
        
        if self.config.training.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
        
        return {
            "loss": loss.item(),
            "correct": correct
        }
