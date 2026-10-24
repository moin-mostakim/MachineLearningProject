"""
Training utilities for CS Tutoring LLM models.

Implements training loops with:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Checkpoint management

References:
- Devlin et al., 2019. "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al., 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config
from src.resource_monitor import ResourceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Training state container."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    early_stop_counter: int = 0
    training_losses: List[float] = None
    validation_losses: List[float] = None
    
    def __post_init__(self):
        self.training_losses = self.training_losses or []
        self.validation_losses = self.validation_losses or []


class Trainer:
    """
    Model trainer with advanced training features.
    
    Features:
    - Mixed precision (FP16) training for memory efficiency
    - Gradient accumulation for larger effective batch sizes
    - Learning rate warmup and scheduling
    - Gradient clipping for stability
    - Early stopping based on validation loss
    - Checkpoint saving and resumption
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        self.config = config or get_config()
        self.model = model
        
        # Determine device
        if device is None:
            if self.config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.config.device)
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.state = TrainingState()
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor(self.config)
        
        self._setup_training()
    
    def _setup_training(self):
        """Initialize optimizer, scheduler, and scaler."""
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
        
        # Mixed precision scaler
        if self.config.training.fp16 and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Using FP16 mixed precision training")
        else:
            self.scaler = None
    
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=num_training_steps,
            pct_start=self.config.training.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        logger.info(f"Scheduler: OneCycleLR with {num_warmup_steps} warmup steps")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (defaults to config)
            
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        
        # Calculate total steps
        num_training_steps = (
            len(train_loader) * num_epochs // 
            self.config.training.gradient_accumulation_steps
        )
        self._setup_scheduler(num_training_steps)
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total optimization steps: {num_training_steps}")
        
        self.resource_monitor.start_monitoring()
        
        try:
            for epoch in range(self.state.epoch, num_epochs):
                self.state.epoch = epoch
                
                # Training epoch
                train_loss = self._train_epoch(train_loader, epoch)
                self.state.training_losses.append(train_loss)
                
                # Validation
                if val_loader is not None:
                    val_loss = self._validate(val_loader)
                    self.state.validation_losses.append(val_loss)
                    
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
                    
                    # Early stopping check
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self.state.early_stop_counter = 0
                        self._save_checkpoint("best_model.pt")
                    else:
                        self.state.early_stop_counter += 1
                        if self.state.early_stop_counter >= self.config.training.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % 1 == 0:  # Save every epoch
                    self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Return training history
        return {
            "training_losses": self.state.training_losses,
            "validation_losses": self.state.validation_losses,
            "best_val_loss": self.state.best_val_loss,
            "final_epoch": self.state.epoch + 1,
            "resource_stats": self.resource_monitor.get_summary(),
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            disable=not self.config.debug_mode
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.training.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if (step + 1) % self.config.training.logging_steps == 0:
                self.resource_monitor.log_current_usage()
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not self.config.debug_mode):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "training_losses": self.state.training_losses,
                "validation_losses": self.state.validation_losses,
            },
            "config": {
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.batch_size,
                "num_epochs": self.config.training.num_epochs,
            }
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        state = checkpoint.get("training_state", {})
        self.state.epoch = state.get("epoch", 0)
        self.state.global_step = state.get("global_step", 0)
        self.state.best_val_loss = state.get("best_val_loss", float('inf'))
        self.state.training_losses = state.get("training_losses", [])
        self.state.validation_losses = state.get("validation_losses", [])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


class GeneratorTrainer(Trainer):
    """Specialized trainer for sequence-to-sequence models."""
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch for seq2seq model."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            disable=not self.config.debug_mode
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Handle label masking for language modeling
            labels = batch["labels"].clone()
            labels[labels == 0] = -100  # Ignore padding in loss
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=labels
                    )
                    loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )
                loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1
            
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.training.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    # Test trainer
    from model import CodeExplanationModel
    from data_loader import DataLoader
    
    config = get_config()
    config.debug_mode = True
    config.training.num_epochs = 1
    config.data.max_samples_train = 100
    
    # Load data
    loader = DataLoader(config)
    loader.load_tokenizer(config.model.generation_model)
    loader.load_dataset()
    train_loader, val_loader, _ = loader.create_dataloaders(batch_size=2)
    
    # Create model
    model = CodeExplanationModel(config)
    
    # Train
    trainer = GeneratorTrainer(model, config)
    history = trainer.train(train_loader, val_loader, num_epochs=1)
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['training_losses'][-1]:.4f}")
    if history['validation_losses']:
        print(f"Final val loss: {history['validation_losses'][-1]:.4f}")
