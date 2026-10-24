"""
Training module for Molecular VAE.

Implements:
- Standard VAE training loop
- Learning rate scheduling
- Early stopping
- Resource monitoring
- Checkpoint saving
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_losses: Dict[str, float]
    val_losses: Dict[str, float]
    learning_rate: float
    epoch_time: float
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class ResourceMonitor:
    """Monitor system resources during training."""
    
    def __init__(self, log_gpu: bool = True, log_memory: bool = True):
        self.log_gpu = log_gpu
        self.log_memory = log_memory
        self.history = []
    
    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        stats = {}
        
        if self.log_memory:
            import psutil
            process = psutil.Process()
            stats['memory_mb'] = process.memory_info().rss / 1024 / 1024
            stats['cpu_percent'] = psutil.cpu_percent()
        
        if self.log_gpu and torch.cuda.is_available():
            stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_memory_max_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return stats
    
    def log(self, step: int):
        """Log current resource usage."""
        stats = self.get_stats()
        stats['step'] = step
        self.history.append(stats)
        return stats


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Trainer for Molecular VAE."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        if config.training.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.training.min_lr
            )
        elif config.training.lr_scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(
            log_gpu=config.resources.log_gpu,
            log_memory=config.resources.log_memory
        )
        
        # Training history
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        
        # Checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            losses = self.criterion(outputs, batch)
            
            # Backward pass
            losses['loss'].backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            num_batches += 1
            
            # Resource monitoring
            if batch_idx % self.config.resources.log_interval == 0:
                self.resource_monitor.log(batch_idx)
        
        # Average losses
        return {k: v / num_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(batch)
            losses = self.criterion(outputs, batch)
            
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Learning rate step
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            # Get resource stats
            resources = self.resource_monitor.get_stats()
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_losses['loss'],
                val_loss=val_losses['loss'],
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rate=current_lr,
                epoch_time=time.time() - epoch_start,
                memory_mb=resources.get('memory_mb'),
                gpu_memory_mb=resources.get('gpu_memory_mb')
            )
            self.history.append(metrics)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_losses['loss']:.4f} | "
                f"Val Loss: {val_losses['loss']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {metrics.epoch_time:.1f}s"
            )
            
            # Save best model
            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_model.pt')
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Early stopping
            if self.early_stopping(val_losses['loss']):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model and training history
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': [vars(m) for m in self.history]
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.config.training.checkpoint_dir, 'training_history.json')
        history_dict = [vars(m) for m in self.history]
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2, default=str)
        
        logger.info(f"Saved training history to {path}")


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    train_loader,
    val_loader,
    config,
    device: str = "cpu"
) -> Tuple[nn.Module, List[TrainingMetrics]]:
    """Convenience function for training."""
    trainer = Trainer(model, criterion, config, device)
    history = trainer.train(train_loader, val_loader)
    return model, history


if __name__ == "__main__":
    # Test training setup
    import sys
    sys.path.append("..")
    from config import get_config
    from models.vae import build_model
    from training.losses import build_loss
    from data.dataset import get_dataloaders
    
    config = get_config(debug=True)
    
    # Build model and loss
    model = build_model(config)
    criterion = build_loss(config, "multi_objective")
    
    # Get data
    train_loader, val_loader, _ = get_dataloaders(config, debug_samples=100)
    
    # Quick training test
    trainer = Trainer(model, criterion, config, config.device)
    train_losses = trainer.train_epoch(train_loader)
    
    print("\nTrain losses after 1 epoch:")
    for k, v in train_losses.items():
        print(f"  {k}: {v:.4f}")
