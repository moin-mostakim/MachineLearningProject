"""Training modules for causal representation learning - Regression version."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Any
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERMTrainer:
    """Standard ERM training across all environments - Regression version."""
    
    def __init__(self, model, train_loaders, val_loaders, config, device="cpu"):
        self.model = model.to(device)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        self.regression = getattr(config.model, 'regression', True)
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
        self.criterion = nn.MSELoss() if self.regression else nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_metrics = []
    
    def train(self) -> Dict[str, Any]:
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for env_loader in self.train_loaders:
                for batch in env_loader:
                    images = batch["image"].to(self.device)
                    if self.regression:
                        targets = batch["steering_angle"].float().to(self.device)
                    else:
                        targets = batch["label"].to(self.device)
                    
                    self.optimizer.zero_grad()
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.train_losses.append(avg_loss)
            
            val_mae, val_rmse = self.evaluate()
            self.val_metrics.append({"mae": val_mae, "rmse": val_rmse})
            
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                       f"Loss: {avg_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        
        return {"train_losses": self.train_losses, "val_metrics": self.val_metrics}
    
    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate and return MAE and RMSE for steering angle regression."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for env_loader in self.val_loaders:
            for batch in env_loader:
                images = batch["image"].to(self.device)
                targets = batch["steering_angle"].float().to(self.device)
                
                preds = self.model(images)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        return mae, rmse


class IRMTrainer:
    """IRM training with invariance penalty - Regression version."""
    
    def __init__(self, model, train_loaders, val_loaders, config, device="cpu"):
        self.model = model.to(device)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        self.regression = getattr(config.model, 'regression', True)
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
        self.criterion = nn.MSELoss() if self.regression else nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_metrics = []
        self.irm_penalties = []
    
    def train(self) -> Dict[str, Any]:
        global_step = 0
        
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            total_loss = 0
            total_penalty = 0
            num_batches = 0
            
            env_iters = [iter(loader) for loader in self.train_loaders]
            
            while True:
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
                    if self.regression:
                        targets = batch["steering_angle"].float().to(self.device)
                    else:
                        targets = batch["label"].to(self.device)
                    
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                    penalty = self.model.compute_irm_penalty(preds, targets, regression=self.regression)
                    
                    env_losses.append(loss)
                    env_penalties.append(penalty)
                
                mean_loss = torch.stack(env_losses).mean()
                mean_penalty = torch.stack(env_penalties).mean()
                
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
            
            val_mae, val_rmse = self.evaluate()
            self.val_metrics.append({"mae": val_mae, "rmse": val_rmse})
            
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                       f"Loss: {self.train_losses[-1]:.4f}, Penalty: {self.irm_penalties[-1]:.6f}, "
                       f"Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        
        return {"train_losses": self.train_losses, "val_metrics": self.val_metrics,
                "irm_penalties": self.irm_penalties}
    
    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate and return MAE and RMSE for steering angle regression."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for env_loader in self.val_loaders:
            for batch in env_loader:
                images = batch["image"].to(self.device)
                targets = batch["steering_angle"].float().to(self.device)
                
                preds = self.model(images)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        return mae, rmse
