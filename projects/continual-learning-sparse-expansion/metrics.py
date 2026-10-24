"""
Evaluation Metrics for Continual Learning.

Implements standard continual learning metrics:
- Average Accuracy: Mean accuracy across all tasks
- Forgetting Measure: How much performance degrades on previous tasks
- Backward Transfer: Change in performance on old tasks after learning new ones
- Forward Transfer: How learning previous tasks affects new task learning

References:
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (2017)
- Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (2019)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ContinualLearningMetrics:
    """Container for continual learning metrics."""
    
    # Accuracy matrix: R[i,j] = accuracy on task j after training on task i
    accuracy_matrix: np.ndarray
    
    # Derived metrics
    average_accuracy: float  # Average over all tasks at the end
    forgetting_measure: float  # Average forgetting across tasks
    backward_transfer: float  # Average backward transfer
    forward_transfer: Optional[float]  # Average forward transfer (if measured)
    
    # Per-task metrics
    per_task_accuracy: Dict[int, float]
    per_task_forgetting: Dict[int, float]
    
    def __repr__(self):
        return (
            f"ContinualLearningMetrics(\n"
            f"  average_accuracy={self.average_accuracy:.2f}%,\n"
            f"  forgetting_measure={self.forgetting_measure:.2f}%,\n"
            f"  backward_transfer={self.backward_transfer:.2f}%\n"
            f")"
        )


class MetricsTracker:
    """
    Track and compute continual learning metrics throughout training.
    
    Maintains an accuracy matrix where R[i,j] is the accuracy on task j
    after training on task i (0-indexed).
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        
        # R[i,j] = accuracy on task j after training through task i
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        # Track training history for each task
        self.training_history = {t: [] for t in range(num_tasks)}
    
    def record_accuracy(
        self,
        current_task: int,
        evaluated_task: int,
        accuracy: float
    ):
        """Record accuracy on evaluated_task after training on current_task."""
        self.accuracy_matrix[current_task, evaluated_task] = accuracy
    
    def record_all_accuracies(
        self,
        current_task: int,
        accuracies: Dict[int, float]
    ):
        """Record accuracies on all tasks after training on current_task."""
        for task_id, acc in accuracies.items():
            if isinstance(task_id, str) and task_id.startswith("task_"):
                task_id = int(task_id.split("_")[1])
            if task_id < self.num_tasks:
                self.accuracy_matrix[current_task, task_id] = acc
    
    def compute_average_accuracy(self) -> float:
        """
        Compute average accuracy across all tasks at the end of training.
        
        A = (1/T) * sum_{j=1}^{T} R_{T,j}
        
        where T is the total number of tasks.
        """
        T = self.num_tasks
        return np.mean(self.accuracy_matrix[T-1, :T])
    
    def compute_forgetting(self) -> Dict[int, float]:
        """
        Compute forgetting measure for each task.
        
        F_j = max_{l in {1,...,T-1}} (R_{l,j} - R_{T,j}) for task j < T
        
        Forgetting is the maximum decrease in accuracy compared to
        the best previous performance.
        """
        T = self.num_tasks
        forgetting = {}
        
        for j in range(T - 1):  # Forgetting only for tasks 0 to T-2
            # Find max accuracy on task j across all training stages
            max_acc_before_final = np.max(self.accuracy_matrix[j:T-1, j])
            final_acc = self.accuracy_matrix[T-1, j]
            
            # Forgetting is drop from best performance
            forgetting[j] = max(0, max_acc_before_final - final_acc)
        
        return forgetting
    
    def compute_average_forgetting(self) -> float:
        """
        Compute average forgetting across all previous tasks.
        
        F = (1/(T-1)) * sum_{j=0}^{T-2} F_j
        """
        forgetting = self.compute_forgetting()
        if len(forgetting) == 0:
            return 0.0
        return np.mean(list(forgetting.values()))
    
    def compute_backward_transfer(self) -> float:
        """
        Compute backward transfer.
        
        BWT = (1/(T-1)) * sum_{j=0}^{T-2} (R_{T,j} - R_{j,j})
        
        Positive BWT means learning new tasks improved old task performance.
        Negative BWT indicates forgetting.
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        
        bwt_sum = 0
        for j in range(T - 1):
            # Performance after all training vs immediately after task j training
            bwt_sum += self.accuracy_matrix[T-1, j] - self.accuracy_matrix[j, j]
        
        return bwt_sum / (T - 1)
    
    def compute_forward_transfer(
        self,
        random_baseline: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Compute forward transfer.
        
        FWT = (1/(T-1)) * sum_{j=1}^{T-1} (R_{j-1,j} - b_j)
        
        where b_j is the random baseline accuracy for task j.
        
        Positive FWT means previous task learning helped new task.
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        
        fwt_sum = 0
        for j in range(1, T):
            # Accuracy on task j before training on it (zero-shot)
            zero_shot_acc = self.accuracy_matrix[j-1, j] if j > 0 else 0
            
            baseline = random_baseline.get(j, 0) if random_baseline else 0
            fwt_sum += zero_shot_acc - baseline
        
        return fwt_sum / (T - 1)
    
    def get_metrics(self) -> ContinualLearningMetrics:
        """Compute and return all metrics."""
        forgetting_per_task = self.compute_forgetting()
        
        # Per-task final accuracy
        T = self.num_tasks
        per_task_accuracy = {
            t: self.accuracy_matrix[T-1, t] for t in range(T)
        }
        
        return ContinualLearningMetrics(
            accuracy_matrix=self.accuracy_matrix.copy(),
            average_accuracy=self.compute_average_accuracy(),
            forgetting_measure=self.compute_average_forgetting(),
            backward_transfer=self.compute_backward_transfer(),
            forward_transfer=self.compute_forward_transfer(),
            per_task_accuracy=per_task_accuracy,
            per_task_forgetting=forgetting_per_task
        )
    
    def print_accuracy_matrix(self):
        """Print formatted accuracy matrix."""
        print("\nAccuracy Matrix R[i,j] (task j accuracy after training task i):")
        print("-" * 60)
        
        # Header
        header = "       " + "".join([f"Task {j:2d} " for j in range(self.num_tasks)])
        print(header)
        print("-" * 60)
        
        for i in range(self.num_tasks):
            row = f"After {i}: "
            for j in range(self.num_tasks):
                if self.accuracy_matrix[i, j] > 0:
                    row += f"{self.accuracy_matrix[i, j]:6.2f}% "
                else:
                    row += "   --   "
            print(row)
        
        print("-" * 60)
    
    def print_summary(self):
        """Print summary of all metrics."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("CONTINUAL LEARNING METRICS SUMMARY")
        print("=" * 60)
        
        self.print_accuracy_matrix()
        
        print(f"\nAverage Accuracy: {metrics.average_accuracy:.2f}%")
        print(f"Average Forgetting: {metrics.forgetting_measure:.2f}%")
        print(f"Backward Transfer: {metrics.backward_transfer:+.2f}%")
        
        if metrics.forward_transfer is not None:
            print(f"Forward Transfer: {metrics.forward_transfer:+.2f}%")
        
        print("\nPer-Task Final Accuracy:")
        for t, acc in metrics.per_task_accuracy.items():
            forgetting = metrics.per_task_forgetting.get(t, 0)
            print(f"  Task {t}: {acc:.2f}% (forgetting: {forgetting:.2f}%)")
        
        print("=" * 60)


def compare_methods(
    method_metrics: Dict[str, MetricsTracker]
) -> Dict[str, Dict]:
    """
    Compare metrics across multiple methods.
    
    Args:
        method_metrics: Dictionary mapping method name to MetricsTracker
    
    Returns:
        Comparison dictionary with all metrics for each method
    """
    comparison = {}
    
    for method_name, tracker in method_metrics.items():
        metrics = tracker.get_metrics()
        comparison[method_name] = {
            "average_accuracy": metrics.average_accuracy,
            "forgetting": metrics.forgetting_measure,
            "backward_transfer": metrics.backward_transfer,
            "per_task_accuracy": metrics.per_task_accuracy
        }
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10}")
    print("-" * 80)
    
    for method_name, m in comparison.items():
        print(
            f"{method_name:<25} "
            f"{m['average_accuracy']:>9.2f}% "
            f"{m['forgetting']:>11.2f}% "
            f"{m['backward_transfer']:>+9.2f}%"
        )
    
    print("=" * 80)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    tracker = MetricsTracker(num_tasks=5)
    
    # Simulate training and record accuracies
    # After task 0: only task 0 evaluated
    tracker.record_accuracy(0, 0, 95.0)
    
    # After task 1: tasks 0 and 1 evaluated
    tracker.record_accuracy(1, 0, 85.0)  # Some forgetting
    tracker.record_accuracy(1, 1, 92.0)
    
    # After task 2
    tracker.record_accuracy(2, 0, 80.0)
    tracker.record_accuracy(2, 1, 88.0)
    tracker.record_accuracy(2, 2, 94.0)
    
    # After task 3
    tracker.record_accuracy(3, 0, 75.0)
    tracker.record_accuracy(3, 1, 85.0)
    tracker.record_accuracy(3, 2, 90.0)
    tracker.record_accuracy(3, 3, 93.0)
    
    # After task 4
    tracker.record_accuracy(4, 0, 70.0)
    tracker.record_accuracy(4, 1, 82.0)
    tracker.record_accuracy(4, 2, 87.0)
    tracker.record_accuracy(4, 3, 89.0)
    tracker.record_accuracy(4, 4, 91.0)
    
    tracker.print_summary()
