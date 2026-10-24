"""
Evaluation metrics for neurosymbolic reasoning.

Computes:
- Reasoning accuracy (overall and by question type)
- Hypothesis validity rate
- Symbolic rule quality metrics
- Knowledge graph consistency
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ReasoningMetrics:
    """Container for reasoning evaluation metrics."""
    
    # Core metrics
    accuracy: float = 0.0
    hypothesis_validity_rate: float = 0.0
    
    # Breakdown by reasoning type
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Symbolic reasoning metrics
    rule_activation_rate: float = 0.0
    avg_reasoning_depth: float = 0.0
    rule_confidence_mean: float = 0.0
    
    # Knowledge graph metrics
    kg_consistency: float = 0.0
    entity_coverage: float = 0.0
    
    # Additional statistics
    num_samples: int = 0
    num_correct: int = 0
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "accuracy": self.accuracy,
            "hypothesis_validity_rate": self.hypothesis_validity_rate,
            "accuracy_by_type": self.accuracy_by_type,
            "rule_activation_rate": self.rule_activation_rate,
            "avg_reasoning_depth": self.avg_reasoning_depth,
            "rule_confidence_mean": self.rule_confidence_mean,
            "kg_consistency": self.kg_consistency,
            "entity_coverage": self.entity_coverage,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "REASONING METRICS",
            "=" * 50,
            f"Accuracy: {self.accuracy:.4f} ({self.num_correct}/{self.num_samples})",
            f"Hypothesis Validity Rate: {self.hypothesis_validity_rate:.4f}",
            "",
            "Symbolic Reasoning:",
            f"  Rule Activation Rate: {self.rule_activation_rate:.4f}",
            f"  Avg Reasoning Depth: {self.avg_reasoning_depth:.2f}",
            f"  Rule Confidence Mean: {self.rule_confidence_mean:.4f}",
            "",
            "Knowledge Graph:",
            f"  KG Consistency: {self.kg_consistency:.4f}",
            f"  Entity Coverage: {self.entity_coverage:.4f}",
            "=" * 50
        ]
        
        if self.accuracy_by_type:
            lines.insert(5, "")
            lines.insert(6, "Accuracy by Type:")
            for t, acc in self.accuracy_by_type.items():
                lines.insert(7, f"  {t}: {acc:.4f}")
        
        return "\n".join(lines)


class MetricsTracker:
    """Tracks metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.predictions: List[int] = []
        self.labels: List[int] = []
        self.hypothesis_validities: List[float] = []
        self.question_types: List[str] = []
        
        # Symbolic reasoning tracking
        self.rule_activations: List[float] = []
        self.reasoning_depths: List[int] = []
        self.rule_confidences: List[float] = []
        
        # KG tracking
        self.kg_consistencies: List[float] = []
        self.entity_coverages: List[float] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        hypothesis_validity: Optional[torch.Tensor] = None,
        question_types: Optional[List[str]] = None,
        symbolic_output: Optional[Dict] = None,
        kg_output: Optional[Dict] = None
    ):
        """Update tracked metrics with batch results."""
        # Core predictions
        self.predictions.extend(predictions.cpu().tolist())
        self.labels.extend(labels.cpu().tolist())
        
        # Hypothesis validity
        if hypothesis_validity is not None:
            self.hypothesis_validities.extend(hypothesis_validity.cpu().tolist())
        
        # Question types
        if question_types is not None:
            self.question_types.extend(question_types)
        else:
            self.question_types.extend(["unknown"] * len(predictions))
        
        # Symbolic reasoning metrics
        if symbolic_output is not None:
            trace = symbolic_output.get("reasoning_trace", {})
            
            # Rule activations
            if "rule_activations" in trace:
                activations = trace["rule_activations"]
                if activations:
                    # Average activation across steps and rules
                    avg_activation = sum(
                        a.mean().item() for a in activations
                    ) / len(activations)
                    self.rule_activations.append(avg_activation)
            
            # Reasoning depth (number of steps)
            if "steps" in trace:
                self.reasoning_depths.append(len(trace["steps"]))
            
            # Rule confidences
            if "confidence_scores" in trace:
                confidences = trace["confidence_scores"]
                if confidences:
                    avg_conf = sum(c.mean().item() for c in confidences) / len(confidences)
                    self.rule_confidences.append(avg_conf)
        
        # KG metrics (placeholder - computed in evaluation)
        if kg_output is not None:
            self.kg_consistencies.append(1.0)  # Placeholder
            self.entity_coverages.append(1.0)  # Placeholder
    
    def compute_metrics(self) -> ReasoningMetrics:
        """Compute final metrics from tracked values."""
        metrics = ReasoningMetrics()
        
        if not self.predictions:
            return metrics
        
        # Convert to numpy
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Core accuracy
        metrics.num_samples = len(preds)
        metrics.num_correct = int((preds == labels).sum())
        metrics.accuracy = metrics.num_correct / metrics.num_samples
        
        # Hypothesis validity rate
        if self.hypothesis_validities:
            metrics.hypothesis_validity_rate = np.mean(self.hypothesis_validities)
        
        # Accuracy by question type
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        
        for pred, label, qtype in zip(preds, labels, self.question_types):
            type_total[qtype] += 1
            if pred == label:
                type_correct[qtype] += 1
        
        metrics.accuracy_by_type = {
            t: type_correct[t] / type_total[t]
            for t in type_total.keys()
        }
        
        # Symbolic reasoning metrics
        if self.rule_activations:
            metrics.rule_activation_rate = np.mean(self.rule_activations)
        
        if self.reasoning_depths:
            metrics.avg_reasoning_depth = np.mean(self.reasoning_depths)
        
        if self.rule_confidences:
            metrics.rule_confidence_mean = np.mean(self.rule_confidences)
        
        # KG metrics
        if self.kg_consistencies:
            metrics.kg_consistency = np.mean(self.kg_consistencies)
        
        if self.entity_coverages:
            metrics.entity_coverage = np.mean(self.entity_coverages)
        
        # Confusion matrix
        num_classes = max(max(preds), max(labels)) + 1
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for p, l in zip(preds, labels):
            cm[l, p] += 1
        metrics.confusion_matrix = cm
        
        return metrics


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    hypothesis_validity: Optional[torch.Tensor] = None,
    symbolic_output: Optional[Dict] = None
) -> ReasoningMetrics:
    """
    Compute metrics for a single batch or accumulated results.
    
    Args:
        predictions: (N,) predicted class indices
        labels: (N,) ground truth labels
        hypothesis_validity: Optional (N,) validity scores
        symbolic_output: Optional dict with symbolic reasoning info
        
    Returns:
        ReasoningMetrics instance
    """
    tracker = MetricsTracker()
    tracker.update(predictions, labels, hypothesis_validity, symbolic_output=symbolic_output)
    return tracker.compute_metrics()


def compute_reasoning_accuracy_breakdown(
    predictions: List[int],
    labels: List[int],
    reasoning_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy breakdown by reasoning type.
    
    Returns:
        Dict mapping reasoning type to accuracy metrics
    """
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for pred, label, rtype in zip(predictions, labels, reasoning_types):
        results[rtype]["total"] += 1
        if pred == label:
            results[rtype]["correct"] += 1
    
    return {
        rtype: {
            "accuracy": stats["correct"] / max(stats["total"], 1),
            "correct": stats["correct"],
            "total": stats["total"]
        }
        for rtype, stats in results.items()
    }


def compute_hypothesis_metrics(
    hypothesis_validity: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for hypothesis generation quality.
    
    Returns:
        Dict with hypothesis-related metrics
    """
    correct_mask = (predictions == labels).float()
    
    # Valid hypotheses for correct predictions
    valid_when_correct = (
        hypothesis_validity[correct_mask.bool()].mean().item()
        if correct_mask.sum() > 0 else 0.0
    )
    
    # Valid hypotheses for incorrect predictions
    valid_when_incorrect = (
        hypothesis_validity[~correct_mask.bool()].mean().item()
        if (~correct_mask.bool()).sum() > 0 else 0.0
    )
    
    # Correlation between validity and correctness
    correlation = float(np.corrcoef(
        hypothesis_validity.cpu().numpy(),
        correct_mask.cpu().numpy()
    )[0, 1]) if len(hypothesis_validity) > 1 else 0.0
    
    return {
        "overall_validity": hypothesis_validity.mean().item(),
        "validity_when_correct": valid_when_correct,
        "validity_when_incorrect": valid_when_incorrect,
        "validity_correctness_correlation": correlation
    }


def format_metrics_table(metrics: ReasoningMetrics) -> str:
    """Format metrics as a nice ASCII table."""
    lines = [
        "┌" + "─" * 48 + "┐",
        "│" + "EVALUATION METRICS".center(48) + "│",
        "├" + "─" * 24 + "┬" + "─" * 23 + "┤",
        f"│{'Metric':<24}│{'Value':>22} │",
        "├" + "─" * 24 + "┼" + "─" * 23 + "┤",
        f"│{'Accuracy':<24}│{metrics.accuracy:>21.4f} │",
        f"│{'Hypothesis Validity':<24}│{metrics.hypothesis_validity_rate:>21.4f} │",
        f"│{'Rule Activation Rate':<24}│{metrics.rule_activation_rate:>21.4f} │",
        f"│{'Avg Reasoning Depth':<24}│{metrics.avg_reasoning_depth:>21.2f} │",
        f"│{'KG Consistency':<24}│{metrics.kg_consistency:>21.4f} │",
        "├" + "─" * 24 + "┴" + "─" * 23 + "┤",
        f"│{'Samples: ' + str(metrics.num_samples):<48}│",
        f"│{'Correct: ' + str(metrics.num_correct):<48}│",
        "└" + "─" * 48 + "┘"
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics computation
    preds = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 0])
    labels = torch.tensor([0, 1, 2, 3, 1, 1, 2, 0, 0, 0])
    validity = torch.tensor([0.8, 0.9, 0.7, 0.85, 0.3, 0.95, 0.8, 0.4, 0.9, 0.85])
    
    metrics = compute_metrics(preds, labels, validity)
    print(metrics)
    print("\n" + format_metrics_table(metrics))
