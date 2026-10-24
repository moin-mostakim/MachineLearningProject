"""
Evaluation Metrics for Multi-Task DR Learning.

Classification Metrics:
- Accuracy
- Quadratic Weighted Kappa (standard for DR grading)
- F1 Score (macro)
- AUC-ROC

Segmentation Metrics:
- Dice Coefficient / F1
- IoU (Intersection over Union)
- Precision / Recall

References:
- Cohen, J. (1968) "Weighted kappa: Nominal scale agreement provision for scaled disagreement"
- Porwal et al. (2018) "IDRiD Challenge evaluation metrics"
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_fscore_support
)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ClassificationMetrics:
    """Compute classification metrics for DR grading."""
    
    def __init__(self, num_classes=None):
        self.num_classes = num_classes or config.NUM_CLASSES
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, pred, target, prob=None):
        """
        Update with batch predictions.
        
        Args:
            pred: (B,) predicted class indices
            target: (B,) ground truth class indices
            prob: (B, C) class probabilities (optional)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        self.predictions.extend(pred.tolist())
        self.targets.extend(target.tolist())
        
        if prob is not None:
            if isinstance(prob, torch.Tensor):
                prob = prob.detach().cpu().numpy()
            self.probabilities.extend(prob.tolist())
    
    def compute(self):
        """Compute all classification metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # Quadratic Weighted Kappa (standard for DR)
        metrics['kappa'] = cohen_kappa_score(targets, preds, weights='quadratic')
        
        # F1 Score (macro average)
        metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        
        for i, name in enumerate(config.CLASS_NAMES):
            if i < len(precision):
                metrics[f'precision_{name}'] = precision[i]
                metrics[f'recall_{name}'] = recall[i]
                metrics[f'f1_{name}'] = f1[i]
        
        # AUC (if probabilities available)
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            try:
                # One-vs-rest AUC
                targets_onehot = np.eye(self.num_classes)[targets]
                metrics['auc'] = roc_auc_score(targets_onehot, probs, 
                                               average='macro', multi_class='ovr')
            except Exception:
                metrics['auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, preds, 
                                                        labels=list(range(self.num_classes)))
        
        return metrics


class SegmentationMetrics:
    """Compute segmentation metrics for lesion detection."""
    
    def __init__(self, num_classes=None, threshold=0.5):
        self.num_classes = num_classes or config.NUM_LESION_TYPES
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.dice_scores = {i: [] for i in range(self.num_classes)}
        self.iou_scores = {i: [] for i in range(self.num_classes)}
        self.precision_scores = {i: [] for i in range(self.num_classes)}
        self.recall_scores = {i: [] for i in range(self.num_classes)}
    
    def update(self, pred, target):
        """
        Update with batch predictions.
        
        Args:
            pred: (B, C, H, W) predicted masks (logits or probabilities)
            target: (B, C, H, W) ground truth masks
        """
        if isinstance(pred, torch.Tensor):
            pred = torch.sigmoid(pred).detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Binarize predictions
        pred_binary = (pred > self.threshold).astype(np.float32)
        
        batch_size = pred.shape[0]
        
        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_mask = pred_binary[b, c].flatten()
                target_mask = target[b, c].flatten()
                
                # Dice coefficient
                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()
                
                if union > 0:
                    dice = (2 * intersection) / (union + 1e-8)
                else:
                    dice = 1.0  # Both empty
                self.dice_scores[c].append(dice)
                
                # IoU
                union_iou = pred_mask.sum() + target_mask.sum() - intersection
                if union_iou > 0:
                    iou = intersection / (union_iou + 1e-8)
                else:
                    iou = 1.0
                self.iou_scores[c].append(iou)
                
                # Precision
                if pred_mask.sum() > 0:
                    precision = intersection / pred_mask.sum()
                else:
                    precision = 1.0 if target_mask.sum() == 0 else 0.0
                self.precision_scores[c].append(precision)
                
                # Recall
                if target_mask.sum() > 0:
                    recall = intersection / target_mask.sum()
                else:
                    recall = 1.0
                self.recall_scores[c].append(recall)
    
    def compute(self):
        """Compute all segmentation metrics."""
        metrics = {}
        
        # Per-class metrics
        dice_per_class = []
        iou_per_class = []
        
        for c in range(self.num_classes):
            lesion_name = config.LESION_NAMES[c] if c < len(config.LESION_NAMES) else f"class_{c}"
            
            dice = np.mean(self.dice_scores[c]) if self.dice_scores[c] else 0.0
            iou = np.mean(self.iou_scores[c]) if self.iou_scores[c] else 0.0
            precision = np.mean(self.precision_scores[c]) if self.precision_scores[c] else 0.0
            recall = np.mean(self.recall_scores[c]) if self.recall_scores[c] else 0.0
            
            metrics[f'dice_{lesion_name}'] = dice
            metrics[f'iou_{lesion_name}'] = iou
            metrics[f'precision_{lesion_name}'] = precision
            metrics[f'recall_{lesion_name}'] = recall
            
            dice_per_class.append(dice)
            iou_per_class.append(iou)
        
        # Mean metrics
        metrics['dice_mean'] = np.mean(dice_per_class)
        metrics['iou_mean'] = np.mean(iou_per_class)
        metrics['precision_mean'] = np.mean([np.mean(self.precision_scores[c]) 
                                             for c in range(self.num_classes)])
        metrics['recall_mean'] = np.mean([np.mean(self.recall_scores[c]) 
                                          for c in range(self.num_classes)])
        
        return metrics


class MultiTaskMetrics:
    """Combined metrics for multi-task evaluation."""
    
    def __init__(self):
        self.cls_metrics = ClassificationMetrics()
        self.seg_metrics = SegmentationMetrics()
    
    def reset(self):
        self.cls_metrics.reset()
        self.seg_metrics.reset()
    
    def update(self, outputs, targets):
        """
        Update with batch predictions.
        
        Args:
            outputs: Dict with 'segmentation' and 'classification' predictions
            targets: Dict with 'mask' and 'grade' ground truths
        """
        # Classification
        cls_pred = outputs['classification'].argmax(dim=1)
        cls_prob = torch.softmax(outputs['classification'], dim=1)
        self.cls_metrics.update(cls_pred, targets['grade'], cls_prob)
        
        # Segmentation
        self.seg_metrics.update(outputs['segmentation'], targets['mask'])
    
    def compute(self):
        """Compute all metrics."""
        cls_metrics = self.cls_metrics.compute()
        seg_metrics = self.seg_metrics.compute()
        
        # Combine
        metrics = {f'cls_{k}': v for k, v in cls_metrics.items() if k != 'confusion_matrix'}
        metrics.update({f'seg_{k}': v for k, v in seg_metrics.items()})
        metrics['confusion_matrix'] = cls_metrics.get('confusion_matrix')
        
        return metrics


def print_metrics_summary(metrics, prefix=""):
    """Print formatted metrics summary."""
    print(f"\n{prefix}METRICS SUMMARY")
    print("=" * 60)
    
    # Classification metrics
    if 'cls_accuracy' in metrics:
        print("\nClassification Metrics:")
        print("-" * 40)
        print(f"  Accuracy:     {metrics.get('cls_accuracy', 0):.4f}")
        print(f"  Kappa:        {metrics.get('cls_kappa', 0):.4f}")
        print(f"  F1 (macro):   {metrics.get('cls_f1_macro', 0):.4f}")
        if 'cls_auc' in metrics:
            print(f"  AUC:          {metrics.get('cls_auc', 0):.4f}")
    
    # Segmentation metrics
    if 'seg_dice_mean' in metrics:
        print("\nSegmentation Metrics:")
        print("-" * 40)
        print(f"  Dice (mean):  {metrics.get('seg_dice_mean', 0):.4f}")
        print(f"  IoU (mean):   {metrics.get('seg_iou_mean', 0):.4f}")
        
        print("\n  Per-lesion Dice:")
        for name in config.LESION_NAMES:
            dice = metrics.get(f'seg_dice_{name}', 0)
            print(f"    {name:<20} {dice:.4f}")


if __name__ == "__main__":
    # Test metrics
    batch_size = 4
    
    # Test classification metrics
    cls_metrics = ClassificationMetrics()
    for _ in range(3):
        pred = torch.randint(0, 5, (batch_size,))
        target = torch.randint(0, 5, (batch_size,))
        prob = torch.softmax(torch.randn(batch_size, 5), dim=1)
        cls_metrics.update(pred, target, prob)
    
    cls_results = cls_metrics.compute()
    print("Classification metrics:")
    for k, v in cls_results.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Test segmentation metrics
    seg_metrics = SegmentationMetrics()
    for _ in range(3):
        pred = torch.randn(batch_size, 4, 128, 128)
        target = torch.randint(0, 2, (batch_size, 4, 128, 128)).float()
        seg_metrics.update(pred, target)
    
    seg_results = seg_metrics.compute()
    print("\nSegmentation metrics:")
    for k, v in seg_results.items():
        print(f"  {k}: {v:.4f}")
