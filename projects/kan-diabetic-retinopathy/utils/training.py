"""
Training utilities for DR detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import time
import numpy as np

from .metrics import calculate_metrics, evaluate_model
from .resource_monitor import ResourceMonitor


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    log_interval: int = 10,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        gradient_clip: Max gradient norm
        log_interval: Print every N batches
        max_batches: Limit batches for debug mode
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    num_batches = len(train_loader)
    if max_batches:
        num_batches = min(num_batches, max_batches)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100. * correct / total:.2f}%"
            )
    
    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": correct / total,
        "qwk": calculate_metrics(y_true, y_pred)["qwk"],
    }
    
    return metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device
        max_batches: Limit batches for debug mode
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_proba = []
    
    num_batches = len(data_loader)
    if max_batches:
        num_batches = min(num_batches, max_batches)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            proba = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    
    # Get comprehensive metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = total_loss / num_batches
    
    return metrics


class Trainer:
    """
    Training manager for DR detection models.
    
    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Resource monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        config: Optional[Any] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: LR scheduler
            device: Device
            config: Configuration object
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Training state
        self.current_epoch = 0
        self.best_val_qwk = -float("inf")
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_qwk": [],
            "val_loss": [],
            "val_acc": [],
            "val_qwk": [],
            "val_auc": [],
            "lr": [],
        }
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor()
    
    def train(
        self,
        num_epochs: int,
        patience: int = 10,
        log_interval: int = 10,
        save_path: Optional[str] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs
            patience: Early stopping patience
            log_interval: Logging interval
            save_path: Path to save best model
            max_batches: Limit batches per epoch (debug mode)
        
        Returns:
            Training history and final metrics
        """
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        if max_batches:
            print(f"Max batches per epoch (debug): {max_batches}")
        print(f"{'='*60}\n")
        
        self.resource_monitor.start()
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                gradient_clip=1.0,
                log_interval=log_interval,
                max_batches=max_batches,
            )
            
            # Validate
            val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                max_batches=max_batches,
            )
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["qwk"])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_qwk"].append(train_metrics["qwk"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_qwk"].append(val_metrics["qwk"])
            self.history["val_auc"].append(val_metrics.get("auc_macro"))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | QWK: {train_metrics['qwk']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | QWK: {val_metrics['qwk']:.4f}")
            if val_metrics.get("auc_macro"):
                print(f"  Val AUC:    {val_metrics['auc_macro']:.4f}")
            print(f"  Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for improvement
            if val_metrics["qwk"] > self.best_val_qwk:
                self.best_val_qwk = val_metrics["qwk"]
                self.best_epoch = epoch
                no_improve_count = 0
                
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ New best model saved (QWK: {self.best_val_qwk:.4f})")
            else:
                no_improve_count += 1
                print(f"  No improvement for {no_improve_count} epochs")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Final resource stats
        resource_stats = self.resource_monitor.stop()
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Epoch: {self.best_epoch + 1}")
        print(f"Best Val QWK: {self.best_val_qwk:.4f}")
        print(f"Total Time: {resource_stats['total_time']:.1f}s")
        print(f"Peak Memory: {resource_stats['peak_memory_mb']:.1f} MB")
        
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_qwk": self.best_val_qwk,
            "resource_stats": resource_stats,
        }


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model
        optimizer_name: adamw, sgd, adam
        learning_rate: Learning rate
        weight_decay: Weight decay
    
    Returns:
        Optimizer
    """
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    num_epochs: int = 50,
    warmup_epochs: int = 5,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: cosine, step, plateau
        num_epochs: Total epochs
        warmup_epochs: Warmup epochs
    
    Returns:
        LR scheduler
    """
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
