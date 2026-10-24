"""
Metrics for Diabetic Retinopathy grading evaluation.

Primary metric: Quadratic Weighted Kappa (QWK)
- Standard metric for DR grading (APTOS, EyePACS competitions)
- Measures agreement while accounting for ordinal nature of DR grades
- Penalizes disagreements quadratically based on distance

References:
- Cohen (1968): "Weighted kappa: nominal scale agreement"
- APTOS 2019 competition scoring
"""

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple, Optional, Union
import warnings


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> float:
    """
    Compute Quadratic Weighted Kappa.
    
    QWK measures agreement between raters/predictions accounting for:
    1. Chance agreement
    2. Ordinal nature of ratings (larger disagreements penalized more)
    
    Formula: kappa = 1 - (sum(w * O) / sum(w * E))
    where w[i,j] = (i-j)^2 / (N-1)^2
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        QWK score in [-1, 1], higher is better
    """
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Build weight matrix (quadratic weights)
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    
    # Compute expected matrix
    hist_true = np.bincount(y_true, minlength=num_classes)
    hist_pred = np.bincount(y_pred, minlength=num_classes)
    expected = np.outer(hist_true, hist_pred).astype(float)
    expected /= expected.sum()
    
    # Normalize observed matrix
    observed = cm.astype(float)
    observed /= observed.sum()
    
    # Compute kappa
    num = np.sum(weights * observed)
    den = np.sum(weights * expected)
    
    if den == 0:
        return 1.0  # Perfect agreement
    
    kappa = 1 - (num / den)
    return kappa


def optimize_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = 5,
    metric: str = 'qwk',
) -> Tuple[np.ndarray, float]:
    """
    Optimize classification thresholds for ordinal data.
    
    Instead of using argmax, find optimal thresholds that maximize QWK.
    This is a common post-processing technique for DR grading.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities [N, num_classes]
        num_classes: Number of classes
        metric: Metric to optimize ('qwk' or 'accuracy')
        
    Returns:
        optimal_thresholds: Array of thresholds
        best_score: Best metric score achieved
    """
    # Convert probs to cumulative for ordinal regression style
    # Threshold approach: predict class based on cumsum exceeding threshold
    
    # Simple approach: use probability-based continuous prediction
    # Then optimize binning thresholds
    
    # Compute expected class as weighted sum
    class_indices = np.arange(num_classes)
    y_continuous = np.sum(y_probs * class_indices, axis=1)
    
    # Grid search for thresholds
    best_thresholds = np.array([0.5, 1.5, 2.5, 3.5])  # Default
    best_score = -1
    
    # Search space
    from itertools import product
    
    threshold_candidates = [
        np.arange(0.3, 0.8, 0.1),
        np.arange(1.2, 1.8, 0.1),
        np.arange(2.2, 2.8, 0.1),
        np.arange(3.2, 3.8, 0.1),
    ]
    
    # Coarse grid search
    for t0 in threshold_candidates[0]:
        for t1 in threshold_candidates[1]:
            for t2 in threshold_candidates[2]:
                for t3 in threshold_candidates[3]:
                    if not (t0 < t1 < t2 < t3):
                        continue
                    
                    thresholds = np.array([t0, t1, t2, t3])
                    y_pred = np.digitize(y_continuous, thresholds)
                    
                    if metric == 'qwk':
                        score = quadratic_weighted_kappa(y_true, y_pred, num_classes)
                    else:
                        score = accuracy_score(y_true, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_thresholds = thresholds
    
    return best_thresholds, best_score


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for DR grading.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Optional predicted probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Primary metric: Quadratic Weighted Kappa
    metrics['quadratic_weighted_kappa'] = quadratic_weighted_kappa(
        y_true, y_pred, num_classes
    )
    
    # Also compute linear weighted kappa for comparison
    metrics['linear_weighted_kappa'] = cohen_kappa_score(
        y_true, y_pred, weights='linear'
    )
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class accuracy (sensitivity per class)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
    for i in range(num_classes):
        metrics[f'accuracy_class_{i}'] = per_class_acc[i]
    
    # Sensitivity/Specificity for referable DR (grade >= 2)
    y_true_binary = (y_true >= 2).astype(int)
    y_pred_binary = (y_pred >= 2).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary, y_pred_binary, labels=[0, 1]
    ).ravel()
    
    metrics['sensitivity_referable'] = tp / (tp + fn + 1e-6)
    metrics['specificity_referable'] = tn / (tn + fp + 1e-6)
    
    # AUC scores (if probabilities available)
    if y_probs is not None:
        try:
            # One-vs-rest AUC
            y_true_onehot = np.eye(num_classes)[y_true]
            metrics['auc_macro'] = roc_auc_score(
                y_true_onehot, y_probs, average='macro', multi_class='ovr'
            )
            metrics['auc_weighted'] = roc_auc_score(
                y_true_onehot, y_probs, average='weighted', multi_class='ovr'
            )
            
            # Binary AUC for referable DR
            probs_referable = y_probs[:, 2:].sum(axis=1)
            metrics['auc_referable'] = roc_auc_score(y_true_binary, probs_referable)
            
        except Exception as e:
            warnings.warn(f"Could not compute AUC: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
            metrics['auc_referable'] = 0.0
    
    return metrics


def get_confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict:
    """
    Get confusion matrix as dictionary for analysis.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with confusion matrix info
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    class_names = [
        'No DR',
        'Mild NPDR',
        'Moderate NPDR',
        'Severe NPDR',
        'Proliferative DR',
    ]
    
    return {
        'matrix': cm.tolist(),
        'class_names': class_names,
        'total_samples': len(y_true),
        'per_class_samples': cm.sum(axis=1).tolist(),
    }


class MetricTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self, metrics: List[str]):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.values = {m: [] for m in self.metrics}
        self.counts = {m: 0 for m in self.metrics}
    
    def update(self, metric_dict: Dict[str, float], n: int = 1):
        """
        Update with new values.
        
        Args:
            metric_dict: Dictionary of metric values
            n: Number of samples
        """
        for name, value in metric_dict.items():
            if name in self.values:
                self.values[name].append(value * n)
                self.counts[name] += n
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values."""
        return {
            name: sum(values) / max(self.counts[name], 1)
            for name, values in self.values.items()
        }
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value and index for a metric.
        
        Args:
            metric: Metric name
            mode: 'max' or 'min'
            
        Returns:
            best_value, best_index
        """
        values = self.values.get(metric, [])
        if not values:
            return 0.0, -1
        
        if mode == 'max':
            idx = np.argmax(values)
        else:
            idx = np.argmin(values)
        
        return values[idx], idx


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics
        title: Table title
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    # Group metrics
    primary = ['quadratic_weighted_kappa']
    secondary = ['accuracy', 'auc_macro', 'f1_macro']
    clinical = ['sensitivity_referable', 'specificity_referable', 'auc_referable']
    
    print("\nPrimary Metric:")
    for m in primary:
        if m in metrics:
            print(f"  {m}: {metrics[m]:.4f}")
    
    print("\nSecondary Metrics:")
    for m in secondary:
        if m in metrics:
            print(f"  {m}: {metrics[m]:.4f}")
    
    print("\nClinical Metrics (Referable DR: grade >= 2):")
    for m in clinical:
        if m in metrics:
            print(f"  {m}: {metrics[m]:.4f}")
    
    print("\nPer-Class Accuracy:")
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    for i, name in enumerate(class_names):
        key = f'accuracy_class_{i}'
        if key in metrics:
            print(f"  {name}: {metrics[key]:.4f}")
    
    print(f"{'='*50}\n")
