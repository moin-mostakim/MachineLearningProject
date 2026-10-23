"""Training module for LoRA fine-tuning."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """Trainer for LoRA fine-tuning."""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        output_dir: str = "./checkpoints",
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 100
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        
        self.device = next(model.parameters()).device
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_dataloader) * num_epochs
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.global_step = 0
        
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.num_epochs}")
        logger.info(f"  Batch size: {self.train_dataloader.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            epoch_loss = self._train_epoch(epoch)
            
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best")
                
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                          f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                          f"Train Loss: {epoch_loss:.4f}")
        
        self._save_checkpoint("final")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_val_loss": best_val_loss if self.val_dataloader else None,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=True
        )
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                
                if self.global_step % self.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    self.train_losses.append(avg_loss)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}"
                    })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / name
        self.model.save_pretrained(checkpoint_path)
        
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "global_step": self.global_step
        }
        
        with open(checkpoint_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    import math
    return math.exp(loss) if loss < 20 else float("inf")
