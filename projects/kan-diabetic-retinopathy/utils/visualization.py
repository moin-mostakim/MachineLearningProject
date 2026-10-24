"""
Visualization utilities for experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import os


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train")
    ax.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train")
    ax.plot(epochs, history["val_acc"], "r-", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # QWK
    ax = axes[1, 0]
    ax.plot(epochs, history["train_qwk"], "b-", label="Train")
    ax.plot(epochs, history["val_qwk"], "r-", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quadratic Weighted Kappa")
    ax.set_title("QWK Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    if history.get("lr"):
        ax.plot(epochs, history["lr"], "g-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = True,
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        show: Whether to display
        normalize: Normalize by row
    """
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    
    cm = np.array(confusion_matrix)
    
    if normalize:
        cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_normalized = cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm_normalized, cmap="Blues")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_normalized[i, j]:.2f}"
            if not normalize:
                text = f"{cm[i, j]}"
            color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot comparison of different models.
    
    Args:
        results: Dictionary mapping model names to their results
        save_path: Path to save figure
        show: Whether to display
    """
    model_names = list(results.keys())
    
    # Extract metrics
    qwk_scores = [results[m].get("qwk", 0) for m in model_names]
    accuracies = [results[m].get("accuracy", 0) for m in model_names]
    param_counts = [results[m].get("params", 0) / 1e6 for m in model_names]  # In millions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # QWK comparison
    ax = axes[0]
    bars = ax.bar(model_names, qwk_scores, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    ax.set_ylabel("Quadratic Weighted Kappa")
    ax.set_title("QWK Comparison")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, qwk_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score:.3f}", ha="center", va="bottom")
    ax.tick_params(axis="x", rotation=15)
    
    # Accuracy comparison
    ax = axes[1]
    bars = ax.bar(model_names, accuracies, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom")
    ax.tick_params(axis="x", rotation=15)
    
    # Parameter count comparison
    ax = axes[2]
    bars = ax.bar(model_names, param_counts, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    ax.set_ylabel("Parameters (Millions)")
    ax.set_title("Parameter Count Comparison")
    for bar, count in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{count:.2f}M", ha="center", va="bottom")
    ax.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved model comparison to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_results(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str = "./results",
    show: bool = False,
):
    """
    Generate all result visualizations.
    
    Args:
        all_results: Dictionary with all model results
        output_dir: Output directory
        show: Whether to display plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot model comparison
    plot_model_comparison(
        all_results,
        save_path=os.path.join(output_dir, "model_comparison.png"),
        show=show,
    )
    
    # Plot training curves for each model
    for model_name, results in all_results.items():
        if "history" in results:
            plot_training_curves(
                results["history"],
                save_path=os.path.join(output_dir, f"{model_name}_training.png"),
                show=show,
            )
        
        if "confusion_matrix" in results:
            plot_confusion_matrix(
                results["confusion_matrix"],
                save_path=os.path.join(output_dir, f"{model_name}_confusion.png"),
                show=show,
            )


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization utilities...")
    
    # Dummy history
    history = {
        "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
        "val_loss": [1.1, 0.9, 0.7, 0.6, 0.55],
        "train_acc": [0.3, 0.5, 0.6, 0.7, 0.75],
        "val_acc": [0.25, 0.45, 0.55, 0.65, 0.7],
        "train_qwk": [0.2, 0.4, 0.5, 0.6, 0.65],
        "val_qwk": [0.15, 0.35, 0.45, 0.55, 0.6],
        "lr": [0.001, 0.001, 0.0005, 0.0005, 0.00025],
    }
    
    plot_training_curves(history, save_path="/tmp/test_curves.png", show=False)
    print("Saved test training curves")
    
    # Dummy confusion matrix
    cm = np.array([
        [50, 5, 2, 1, 0],
        [3, 20, 5, 1, 1],
        [2, 4, 30, 3, 1],
        [1, 1, 3, 15, 2],
        [0, 1, 2, 3, 25],
    ])
    
    plot_confusion_matrix(cm, save_path="/tmp/test_confusion.png", show=False)
    print("Saved test confusion matrix")
