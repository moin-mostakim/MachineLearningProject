"""Training modules for causal representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Any
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERMTrainer:
    """Standard ERM training across all environments."""
    
    def __init__(self, model, train_loaders, val_loaders, config, device="cpu"):
        self.model = model.to(device)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_accuracies = []
    
    def train(self) -> Dict[str, Any]:
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Combine all environments for ERM
            for env_loader in self.train_loaders:
                for batch in env_loader:
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.train_losses.append(avg_loss)
            
            val_acc = self.evaluate()
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                       f"Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return {"train_losses": self.train_losses, "val_accuracies": self.val_accuracies}
    
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        
        for env_loader in self.val_loaders:
            for batch in env_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0


class IRMTrainer:
    """IRM training with invariance penalty."""
    
    def __init__(self, model, train_loaders, val_loaders, config, device="cpu"):
        self.model = model.to(device)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_accuracies = []
        self.irm_penalties = []
    
    def train(self) -> Dict[str, Any]:
        global_step = 0
        
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            total_loss = 0
            total_penalty = 0
            num_batches = 0
            
            # Get iterators for all environments
            env_iters = [iter(loader) for loader in self.train_loaders]
            
            while True:
                # Get batch from each environment
                batches = []
                try:
                    for env_iter in env_iters:
                        batches.append(next(env_iter))
                except StopIteration:
                    break
                
                env_losses = []
                env_penalties = []
                
                for batch in batches:
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    penalty = self.model.compute_irm_penalty(logits, labels)
                    
                    env_losses.append(loss)
                    env_penalties.append(penalty)
                
                # Average loss across environments
                mean_loss = torch.stack(env_losses).mean()
                mean_penalty = torch.stack(env_penalties).mean()
                
                # Anneal penalty weight
                penalty_weight = self.config.irm.penalty_weight if global_step >= self.config.irm.penalty_anneal_iters else 0.0
                
                total_objective = mean_loss + penalty_weight * mean_penalty
                
                self.optimizer.zero_grad()
                total_objective.backward()
                self.optimizer.step()
                
                total_loss += mean_loss.item()
                total_penalty += mean_penalty.item()
                num_batches += 1
                global_step += 1
            
            if num_batches > 0:
                self.train_losses.append(total_loss / num_batches)
                self.irm_penalties.append(total_penalty / num_batches)
            
            val_acc = self.evaluate()
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                       f"Loss: {self.train_losses[-1]:.4f}, Penalty: {self.irm_penalties[-1]:.6f}, "
                       f"Val Acc: {val_acc:.4f}")
        
        return {"train_losses": self.train_losses, "val_accuracies": self.val_accuracies,
                "irm_penalties": self.irm_penalties}
    
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        
        for env_loader in self.val_loaders:
            for batch in env_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0
