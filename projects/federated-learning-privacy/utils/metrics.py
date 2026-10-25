"""Metrics computation for federated learning."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        num_classes: Number of classes (inferred if not provided)
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    # Convert logits to predictions if needed
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    # Move to CPU for numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1
    
    # Overall accuracy
    accuracy = (predictions == targets).mean() * 100
    
    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in range(num_classes):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return {
        "accuracy": accuracy,
        "precision_macro": np.mean(precision_per_class) * 100,
        "recall_macro": np.mean(recall_per_class) * 100,
        "f1_macro": np.mean(f1_per_class) * 100,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class
    }


class ConfusionMatrix:
    """Confusion matrix for multi-class classification."""
    
    def __init__(self, num_classes: int):
        """Initialize confusion matrix."""
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update matrix with batch of predictions."""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        for p, t in zip(predictions, targets):
            self.matrix[t, p] += 1
    
    def get_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.matrix
    
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        return np.diag(self.matrix).sum() / self.matrix.sum() * 100
    
    def get_class_accuracy(self) -> np.ndarray:
        """Get per-class accuracy."""
        return np.diag(self.matrix) / self.matrix.sum(axis=1) * 100
    
    def reset(self):
        """Reset matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


class FederatedMetrics:
    """Metrics tracking for federated learning experiments."""
    
    def __init__(self, num_clients: int, num_classes: int):
        """Initialize federated metrics tracker."""
        self.num_clients = num_clients
        self.num_classes = num_classes
        
        self.global_metrics = defaultdict(list)
        self.client_metrics = {i: defaultdict(list) for i in range(num_clients)}
        self.communication_rounds = 0
        
    def log_round(
        self,
        round_idx: int,
        train_loss: float,
        test_accuracy: float,
        test_loss: float,
        epsilon: float = 0.0,
        client_losses: List[float] = None
    ):
        """Log metrics for a training round."""
        self.global_metrics["round"].append(round_idx)
        self.global_metrics["train_loss"].append(train_loss)
        self.global_metrics["test_accuracy"].append(test_accuracy)
        self.global_metrics["test_loss"].append(test_loss)
        self.global_metrics["epsilon"].append(epsilon)
        
        if client_losses:
            for i, loss in enumerate(client_losses):
                self.client_metrics[i]["loss"].append(loss)
        
        self.communication_rounds = round_idx + 1
    
    def log_client(self, client_id: int, metrics: Dict[str, float]):
        """Log metrics for a specific client."""
        for key, value in metrics.items():
            self.client_metrics[client_id][key].append(value)
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "communication_rounds": self.communication_rounds,
            "final_accuracy": self.global_metrics["test_accuracy"][-1] if self.global_metrics["test_accuracy"] else 0,
            "best_accuracy": max(self.global_metrics["test_accuracy"]) if self.global_metrics["test_accuracy"] else 0,
            "final_loss": self.global_metrics["test_loss"][-1] if self.global_metrics["test_loss"] else 0,
            "total_epsilon": self.global_metrics["epsilon"][-1] if self.global_metrics["epsilon"] else 0,
            "avg_client_final_loss": np.mean([
                metrics["loss"][-1] if metrics["loss"] else 0
                for metrics in self.client_metrics.values()
            ])
        }
    
    def to_dict(self) -> Dict:
        """Convert all metrics to dictionary."""
        return {
            "global": dict(self.global_metrics),
            "clients": {
                i: dict(metrics)
                for i, metrics in self.client_metrics.items()
            },
            "summary": self.get_summary()
        }
