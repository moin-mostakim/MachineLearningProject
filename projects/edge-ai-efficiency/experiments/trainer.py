"""
Training utilities for edge AI models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any, Callable
import time
import copy


class Trainer:
    """
    Model trainer with support for:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        num_epochs: int = 50,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 4e-5,
        use_amp: bool = True,
        device: str = 'cuda',
        label_smoothing: float = 0.1,
        gradient_clip_norm: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer (created if None)
            scheduler: LR scheduler (created if None)
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            use_amp: Use automatic mixed precision
            device: Device to train on
            label_smoothing: Label smoothing factor
            gradient_clip_norm: Gradient clipping norm
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp and device != 'cpu'
        self.gradient_clip_norm = gradient_clip_norm
        
        # Loss function with label smoothing
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = criterion
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        else:
            self.scheduler = scheduler
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history: List[Dict[str, float]] = []
        self.best_accuracy = 0.0
        self.best_model_state = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def train(self, verbose: bool = True) -> nn.Module:
        """
        Run full training.
        
        Args:
            verbose: Print progress
            
        Returns:
            Trained model
        """
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s): "
                      f"Train Loss={train_metrics['loss']:.4f}, "
                      f"Train Acc={train_metrics['accuracy']:.2f}%, "
                      f"Val Acc={val_metrics['accuracy']:.2f}%")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        return self.model
    
    def get_best_model(self) -> nn.Module:
        """Get the best model (by validation accuracy)."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history."""
        return self.history


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Trainer:
    """Factory function for trainer."""
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 50),
        learning_rate=config.get('learning_rate', 0.1),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 4e-5),
        use_amp=config.get('use_amp', True),
        device=device,
        label_smoothing=config.get('label_smoothing', 0.1),
        gradient_clip_norm=config.get('gradient_clip_norm', 1.0)
    )
