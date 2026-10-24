"""Base trainer class with common functionality."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import time
import os
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer with common training utilities."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None
        
        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }
        
        # Resource monitoring
        self.resource_monitor = None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        return optim.SGD(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    @abstractmethod
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Single training step. Must be implemented by subclasses."""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        debug_mode = self.config.training.debug
        max_batches = self.config.training.debug_batches if debug_mode else float('inf')
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Training step
            metrics = self.train_step(images, labels)
            
            total_loss += metrics["loss"]
            correct += metrics.get("correct", 0)
            total += labels.size(0)
            
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={metrics['loss']:.4f}")
        
        avg_loss = total_loss / min(len(self.train_loader), max_batches)
        accuracy = correct / total if total > 0 else 0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
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
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / min(len(self.val_loader), max_batches)
        accuracy = correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Full training loop."""
        epochs = num_epochs or self.config.training.epochs
        
        if self.config.training.debug:
            epochs = self.config.training.debug_epochs
            print(f"[DEBUG MODE] Training for {epochs} epochs with {self.config.training.debug_batches} batches/epoch")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            
            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                print(f"New best validation accuracy: {self.best_val_acc:.4f}")
        
        total_time = time.time() - start_time
        
        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "total_time": total_time,
            "epochs_trained": epochs
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_acc": self.best_val_acc,
            "history": self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
