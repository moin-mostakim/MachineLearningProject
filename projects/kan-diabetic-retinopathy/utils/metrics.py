"""
Evaluation metrics for Diabetic Retinopathy classification.

Key metrics:
- Quadratic Weighted Kappa (QWK): Primary metric for ordinal classification
- Accuracy: Overall correctness
- AUC: Multi-class ROC AUC
- Per-class metrics: Precision, Recall, F1
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)
from typing import Dict, Any, Optional


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK).
    
    QWK is the standard metric for ordinal classification in DR detection.
    It measures agreement between predictions and true labels, accounting
    for the ordinal nature of DR severity levels.
    
    The quadratic weighting penalizes predictions that are further from
    the true label more heavily than near-misses.
    
    Formula:
    w_{i,j} = (i - j)^2 / (N - 1)^2
    
    where i, j are the true and predicted classes, N is number of classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        QWK score in range [-1, 1] (1 = perfect agreement)
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_proba: Prediction probabilities (N, num_classes) for AUC
        num_classes: Number of classes
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Quadratic Weighted Kappa
    metrics["qwk"] = quadratic_weighted_kappa(y_true, y_pred)
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )
    
    metrics["precision_per_class"] = precision.tolist()
    metrics["recall_per_class"] = recall.tolist()
    metrics["f1_per_class"] = f1.tolist()
    metrics["support_per_class"] = support.tolist()
    
    # Macro averages
    metrics["precision_macro"] = np.mean(precision)
    metrics["recall_macro"] = np.mean(recall)
    metrics["f1_macro"] = np.mean(f1)
    
    # Weighted averages (accounting for class imbalance)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["precision_weighted"] = precision_w
    metrics["recall_weighted"] = recall_w
    metrics["f1_weighted"] = f1_w
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=range(num_classes)
    ).tolist()
    
    # AUC (if probabilities provided)
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            # One-vs-rest AUC
            y_true_onehot = np.eye(num_classes)[y_true]
            metrics["auc_macro"] = roc_auc_score(
                y_true_onehot, y_proba, multi_class="ovr", average="macro"
            )
            metrics["auc_weighted"] = roc_auc_score(
                y_true_onehot, y_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            # AUC can fail if some classes have no samples
            metrics["auc_macro"] = None
            metrics["auc_weighted"] = None
    
    return metrics


def print_metrics(metrics: Dict[str, Any], class_names: Optional[list] = None):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from calculate_metrics
        class_names: Optional class names
    """
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\n📊 Primary Metric:")
    print(f"   Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"   F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"   F1 (weighted):     {metrics['f1_weighted']:.4f}")
    
    if metrics.get("auc_macro") is not None:
        print(f"   AUC (macro):       {metrics['auc_macro']:.4f}")
        print(f"   AUC (weighted):    {metrics['auc_weighted']:.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    print(f"   {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"   {'-'*59}")
    
    for i, name in enumerate(class_names):
        print(
            f"   {name:<15} "
            f"{metrics['precision_per_class'][i]:.4f}       "
            f"{metrics['recall_per_class'][i]:.4f}       "
            f"{metrics['f1_per_class'][i]:.4f}       "
            f"{metrics['support_per_class'][i]:<10}"
        )
    
    print("\n" + "=" * 60)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate a model on a data loader.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with (images, labels)
        device: Device to evaluate on
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_proba = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            proba = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_proba.append(proba.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)
    
    return calculate_metrics(y_true, y_pred, y_proba, num_classes)


if __name__ == "__main__":
    # Test metrics calculation
    np.random.seed(42)
    
    y_true = np.random.randint(0, 5, size=100)
    y_pred = np.random.randint(0, 5, size=100)
    y_proba = np.random.rand(100, 5)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics)
