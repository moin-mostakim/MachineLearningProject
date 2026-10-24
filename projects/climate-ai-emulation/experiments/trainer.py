"""
Training module for climate emulation models.

Provides unified training interface for:
- Baseline models
- Neural network emulators
- Deep ensembles

Features:
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Checkpoint management
- Resource monitoring
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.metrics import ClimateMetrics, rmse
from utils.resource_monitor import Timer, ResourceMonitor, ExperimentLogger


@dataclass
class TrainingState:
    """Tracks training progress."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    training_complete: bool = False


class Trainer:
    """
    Unified trainer for climate emulation models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader,
        val_loader,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training state
        self.state = TrainingState()
        
        # Loss function
        self.criterion = self._get_loss_fn()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision
        self.use_amp = config.training.use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics calculator
        self.metrics = ClimateMetrics()
        
        # Logging
        self.logger = ExperimentLogger(config.experiment.log_dir)
        self.timer = Timer()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(enable_gpu=(device == "cuda"))
        
    def _get_loss_fn(self) -> nn.Module:
        """Get loss function based on config."""
        loss_name = self.config.training.loss
        
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on config."""
        cfg = self.config.training
        
        if cfg.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2)
            )
        elif cfg.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2)
            )
        elif cfg.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay
            )
        else:
            return optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config."""
        cfg = self.config.training
        
        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.min_learning_rate
            )
        elif cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.epochs // 3,
                gamma=0.1
            )
        elif cfg.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.config.training.gradient_clip_val > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.state.global_step += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_rmse = 0.0
        total_acc = 0.0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            total_rmse += rmse(outputs, targets).item()
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            num_batches += 1
        
        # Compute aggregate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics.compute_deterministic(all_preds, all_targets)
        metrics['val_loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Override number of epochs
            
        Returns:
            Dictionary of training history
        """
        num_epochs = num_epochs or self.config.training.epochs
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch
                epoch_start = time.time()
                
                # Train
                with self.timer.time('train'):
                    train_metrics = self.train_epoch()
                
                # Validate
                with self.timer.time('validate'):
                    val_metrics = self.validate()
                
                epoch_time = time.time() - epoch_start
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Record history
                history['train_loss'].append(train_metrics['train_loss'])
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_rmse'].append(val_metrics['rmse'])
                history['val_acc'].append(val_metrics['acc'])
                history['learning_rate'].append(train_metrics['learning_rate'])
                
                # Log to logger
                self.logger.log_metric('train_loss', train_metrics['train_loss'])
                self.logger.log_metric('val_loss', val_metrics['val_loss'])
                self.logger.log_metric('val_rmse', val_metrics['rmse'])
                
                # Print progress
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f} | "
                      f"Val ACC: {val_metrics['acc']:.4f} | "
                      f"LR: {train_metrics['learning_rate']:.2e} | "
                      f"Time: {epoch_time:.1f}s")
                
                # Early stopping check
                if val_metrics['val_loss'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics['val_loss']
                    self.state.patience_counter = 0
                    # Save best model
                    self._save_checkpoint('best_model.pt')
                else:
                    self.state.patience_counter += 1
                    
                    if self.state.patience_counter >= self.config.training.early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            self.state.training_complete = True
            
        finally:
            # Stop monitoring
            self.resource_monitor.stop()
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        # Print summary
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.state.best_val_loss:.4f}")
        print(f"Total epochs: {self.state.epoch + 1}")
        
        # Resource summary
        resources = self.resource_monitor.get_peak_usage()
        print(f"\nResource Usage:")
        for key, value in resources.items():
            print(f"  {key}: {value:.2f}")
        
        timing = self.timer.get_all_stats()
        print(f"\nTiming:")
        for name, stats in timing.items():
            print(f"  {name}: {stats['total']:.1f}s total, {stats['mean']:.2f}s avg")
        
        return history
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.experiment.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.state.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state.epoch = checkpoint['epoch']
        self.state.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class EnsembleTrainer:
    """
    Trainer for deep ensembles.
    
    Trains each ensemble member independently with different random seeds.
    """
    
    def __init__(
        self,
        ensemble,
        config: Config,
        train_loader,
        val_loader,
        device: str = "cuda"
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble: DeepEnsemble model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
        """
        self.ensemble = ensemble.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create trainer for each member
        self.member_trainers = []
        for i, member in enumerate(ensemble.members):
            # Different random seed for each member
            torch.manual_seed(config.experiment.seed + i * 1000)
            
            trainer = Trainer(
                model=member,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            self.member_trainers.append(trainer)
    
    def train(self, num_epochs: Optional[int] = None) -> List[Dict]:
        """
        Train all ensemble members.
        
        Args:
            num_epochs: Override number of epochs
            
        Returns:
            List of training histories for each member
        """
        histories = []
        
        print(f"\n{'='*60}")
        print(f"Training Deep Ensemble ({len(self.member_trainers)} members)")
        print(f"{'='*60}\n")
        
        for i, trainer in enumerate(self.member_trainers):
            print(f"\n--- Training Member {i+1}/{len(self.member_trainers)} ---")
            history = trainer.train(num_epochs)
            histories.append(history)
        
        return histories


if __name__ == "__main__":
    # Test trainer
    from config import get_config
    from data.dataset import create_dataloaders
    from models.unet import UNet
    
    print("Testing trainer...")
    
    config = get_config(debug=True)
    
    # Create data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = UNet(
        in_channels=config.data.input_time_steps * config.model.input_channels,
        out_channels=config.model.output_channels,
        base_channels=16,  # Smaller for testing
        depth=2,
        dropout=config.model.dropout_rate
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train for few epochs
    history = trainer.train(num_epochs=2)
    
    print("\nTraining history:")
    for key, values in history.items():
        print(f"  {key}: {values}")
    
    print("\nTrainer tests passed!")
