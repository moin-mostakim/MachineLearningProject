"""
Training utilities for Diabetic Retinopathy models.

Includes:
- Ordinal loss function
- Training loop
- Validation loop
- Early stopping
- Resource monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
import time
from tqdm import tqdm
import gc


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal-aware cross-entropy loss for DR grading.
    
    Combines standard CE with ordinal penalty that increases
    loss for predictions far from the true class.
    
    Reference: "Ordinal Regression with Multiple Output CNN for Age Estimation"
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        ordinal_weight: float = 0.3,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize ordinal CE loss.
        
        Args:
            num_classes: Number of classes
            ordinal_weight: Weight for ordinal penalty term
            label_smoothing: Label smoothing factor
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        
        # Precompute ordinal distance matrix
        self.register_buffer(
            'distance_matrix',
            torch.zeros(num_classes, num_classes),
        )
        for i in range(num_classes):
            for j in range(num_classes):
                self.distance_matrix[i, j] = abs(i - j) / (num_classes - 1)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ordinal-aware loss.
        
        Args:
            logits: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Combined loss
        """
        # Standard cross-entropy
        ce = self.ce_loss(logits, targets)
        
        # Ordinal penalty: penalize predictions based on distance from truth
        probs = F.softmax(logits, dim=-1)
        
        # Get distance weights for each sample based on true class
        distances = self.distance_matrix[targets]  # [B, num_classes]
        
        # Weighted penalty: sum(prob * distance)
        ordinal_penalty = (probs * distances).sum(dim=-1).mean()
        
        return ce + self.ordinal_weight * ordinal_penalty


class ResourceMonitor:
    """Monitor GPU memory and training time."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset monitoring."""
        self.gpu_memory_peak = 0
        self.gpu_memory_allocated = []
        self.batch_times = []
        self.epoch_times = []
        self.start_time = None
    
    def start_epoch(self):
        """Mark epoch start."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_epoch(self) -> Dict[str, float]:
        """
        Mark epoch end and return stats.
        
        Returns:
            Dictionary with resource stats
        """
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        stats = {
            'epoch_time_sec': epoch_time,
            'avg_batch_time_ms': np.mean(self.batch_times) * 1000 if self.batch_times else 0,
        }
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.gpu_memory_peak = max(self.gpu_memory_peak, peak_memory)
            stats['gpu_memory_peak_gb'] = peak_memory
            stats['gpu_memory_current_gb'] = torch.cuda.memory_allocated() / 1024**3
        
        self.batch_times = []
        return stats
    
    def log_batch(self, batch_time: float):
        """Log batch time."""
        self.batch_times.append(batch_time)
    
    def get_summary(self) -> Dict[str, float]:
        """Get overall resource summary."""
        return {
            'total_training_time_sec': sum(self.epoch_times),
            'avg_epoch_time_sec': np.mean(self.epoch_times) if self.epoch_times else 0,
            'gpu_memory_peak_gb': self.gpu_memory_peak,
        }


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            is_improvement = score > self.best_score + self.min_delta
        else:
            is_improvement = score < self.best_score - self.min_delta
        
        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    resource_monitor: Optional[ResourceMonitor] = None,
    debug_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        scaler: Optional gradient scaler for mixed precision
        max_grad_norm: Maximum gradient norm for clipping
        resource_monitor: Optional resource monitor
        debug_batches: If set, stop after this many batches
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        if debug_batches and batch_idx >= debug_batches:
            break
        
        batch_start = time.time()
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        num_batches += 1
        
        batch_time = time.time() - batch_start
        if resource_monitor:
            resource_monitor.log_batch(batch_time)
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    from .metrics import quadratic_weighted_kappa, accuracy_score
    
    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': accuracy_score(all_targets, all_preds),
        'qwk': quadratic_weighted_kappa(
            np.array(all_targets), np.array(all_preds)
        ),
    }
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    debug_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        debug_batches: If set, stop after this many batches
        
    Returns:
        metrics: Dictionary with validation metrics
        all_preds: All predictions
        all_targets: All targets
        all_probs: All predicted probabilities
    """
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch_idx, (images, targets) in enumerate(pbar):
            if debug_batches and batch_idx >= debug_batches:
                break
            
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            num_batches += 1
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute comprehensive metrics
    from .metrics import compute_all_metrics
    
    metrics = compute_all_metrics(all_targets, all_preds, all_probs)
    metrics['loss'] = total_loss / max(num_batches, 1)
    
    return metrics, all_preds, all_targets, all_probs


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with parameter groups.
    
    Applies lower weight decay to bias and normalization parameters.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        
    Returns:
        Optimizer
    """
    # Separate parameters for different weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    return AdamW(param_groups, lr=learning_rate, betas=betas)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 5,
    scheduler_type: str = 'cosine',
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        scheduler_type: Type of scheduler ('cosine' or 'linear')
        
    Returns:
        Learning rate scheduler
    """
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    
    # Main scheduler
    if scheduler_type == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6,
        )
    else:
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_epochs - warmup_epochs,
        )
    
    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        path: Save path
        scheduler: Optional scheduler state
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load into
        optimizer: Optional optimizer to load
        scheduler: Optional scheduler to load
        
    Returns:
        Checkpoint dictionary with epoch and metrics
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
    }
