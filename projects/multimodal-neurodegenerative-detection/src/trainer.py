"""Training module for multimodal classifiers."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multimodal classifiers."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            images = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(images, clinical)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in loader:
            images = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            labels = batch["label"].to(self.device)
            
            logits, _ = self.model(images, clinical)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        return {
            "loss": total_loss / len(loader) if len(loader) > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0,
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        
        for epoch in range(self.config.training.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_metrics = self.evaluate(self.val_loader)
            self.val_losses.append(val_metrics["loss"])
            self.val_accuracies.append(val_metrics["accuracy"])
            
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": self.best_val_acc
        }
