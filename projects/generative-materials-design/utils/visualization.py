"""
Visualization utilities for molecular generation.

Includes:
- Training curves
- Latent space visualization
- Pareto front plots
- Generated molecule visualization
"""

import os
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization will be limited.")


def plot_training_curves(
    history: List[Dict],
    save_path: str = "training_curves.png"
):
    """Plot training and validation loss curves."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot: matplotlib not available")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train')
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Individual losses
    if 'train_losses' in history[0]:
        recon_train = [h['train_losses'].get('recon_loss', 0) for h in history]
        kl_train = [h['train_losses'].get('kl_loss', 0) for h in history]
        prop_train = [h['train_losses'].get('property_loss', 0) for h in history]
        
        axes[0, 1].plot(epochs, recon_train, 'g-', label='Reconstruction')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, kl_train, 'm-', label='KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, prop_train, 'c-', label='Property')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Property Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved training curves to {save_path}")


def plot_latent_space(
    latent_vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: str = "latent_space.png",
    method: str = "pca"
):
    """
    Visualize latent space using dimensionality reduction.
    
    Args:
        latent_vectors: [N, latent_dim] array
        labels: Optional labels for coloring
        save_path: Path to save plot
        method: 'pca' or 'tsne'
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot: matplotlib not available")
        return
    
    from sklearn.decomposition import PCA
    
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(latent_vectors)
        explained_var = sum(reducer.explained_variance_ratio_) * 100
        title = f"Latent Space (PCA, {explained_var:.1f}% variance)"
    else:
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            coords = reducer.fit_transform(latent_vectors)
            title = "Latent Space (t-SNE)"
        except ImportError:
            logger.warning("t-SNE not available, using PCA")
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(latent_vectors)
            title = "Latent Space (PCA)"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, 
                           cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Property')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved latent space visualization to {save_path}")


def plot_pareto_front(
    objectives: np.ndarray,
    pareto_mask: np.ndarray,
    objective_names: List[str],
    save_path: str = "pareto_front.png"
):
    """
    Plot Pareto front for 2D or 3D objectives.
    
    Args:
        objectives: [N, num_objectives] array
        pareto_mask: Boolean mask indicating Pareto-optimal solutions
        objective_names: Names of objectives
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot: matplotlib not available")
        return
    
    num_obj = objectives.shape[1]
    
    if num_obj == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Non-Pareto points
        ax.scatter(objectives[~pareto_mask, 0], objectives[~pareto_mask, 1],
                  c='gray', alpha=0.3, s=20, label='Dominated')
        
        # Pareto points
        pareto_obj = objectives[pareto_mask]
        sorted_idx = np.argsort(pareto_obj[:, 0])
        pareto_obj = pareto_obj[sorted_idx]
        
        ax.scatter(pareto_obj[:, 0], pareto_obj[:, 1],
                  c='red', s=50, label='Pareto-optimal')
        ax.plot(pareto_obj[:, 0], pareto_obj[:, 1], 'r--', alpha=0.5)
        
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_title('Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif num_obj >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Non-Pareto points
        ax.scatter(objectives[~pareto_mask, 0], 
                  objectives[~pareto_mask, 1],
                  objectives[~pareto_mask, 2],
                  c='gray', alpha=0.2, s=10, label='Dominated')
        
        # Pareto points
        ax.scatter(objectives[pareto_mask, 0],
                  objectives[pareto_mask, 1],
                  objectives[pareto_mask, 2],
                  c='red', s=50, label='Pareto-optimal')
        
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_zlabel(objective_names[2] if len(objective_names) > 2 else 'Objective 3')
        ax.set_title('Pareto Front (3D)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Pareto front to {save_path}")


def plot_property_distributions(
    generated_props: np.ndarray,
    training_props: np.ndarray,
    property_names: List[str],
    save_path: str = "property_distributions.png"
):
    """
    Compare property distributions between generated and training molecules.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot: matplotlib not available")
        return
    
    num_props = len(property_names)
    num_cols = min(3, num_props)
    num_rows = (num_props + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = np.atleast_2d(axes)
    
    for i, name in enumerate(property_names):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[0, col]
        
        ax.hist(training_props[:, i], bins=30, alpha=0.5, label='Training', density=True)
        ax.hist(generated_props[:, i], bins=30, alpha=0.5, label='Generated', density=True)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_props, num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[0, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved property distributions to {save_path}")


def plot_optimization_comparison(
    results: Dict,
    save_path: str = "optimization_comparison.png"
):
    """
    Plot comparison between optimization methods.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot: matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = list(results.keys())
    
    # Compare mean objectives
    if 'target_properties' in results.get('random_search', {}):
        for i, method in enumerate(methods):
            if 'target_properties' in results[method]:
                props = results[method]['target_properties']
                if hasattr(props, 'numpy'):
                    props = props.numpy()
                
                mean_props = np.mean(props, axis=0)
                axes[0].bar(np.arange(len(mean_props)) + i * 0.25, 
                          mean_props, width=0.25, label=method)
        
        axes[0].set_xlabel('Property Index')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Property Comparison')
        axes[0].legend()
    
    # Pareto front size over generations
    if 'history' in results.get('multi_objective', {}):
        history = results['multi_objective']['history']
        if 'num_pareto' in history:
            axes[1].plot(history['generation'], history['num_pareto'])
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Pareto Front Size')
            axes[1].set_title('Pareto Front Evolution')
            axes[1].grid(True)
    
    # Objective scatter
    if 'pareto_objectives' in results.get('multi_objective', {}):
        pareto_obj = results['multi_objective']['pareto_objectives']
        if hasattr(pareto_obj, 'numpy'):
            pareto_obj = pareto_obj.numpy()
        
        if pareto_obj.shape[1] >= 2:
            axes[2].scatter(pareto_obj[:, 0], pareto_obj[:, 1], s=50, c='red')
            axes[2].set_xlabel('Objective 1')
            axes[2].set_ylabel('Objective 2')
            axes[2].set_title('Pareto Front')
            axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved optimization comparison to {save_path}")


def create_results_report(
    results: Dict,
    save_dir: str = "results"
):
    """Create comprehensive results report with all visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Training curves
    if 'training_history' in results:
        plot_training_curves(
            results['training_history'],
            os.path.join(save_dir, 'training_curves.png')
        )
    
    # Latent space
    if 'latent_vectors' in results:
        plot_latent_space(
            results['latent_vectors'],
            results.get('labels'),
            os.path.join(save_dir, 'latent_space.png')
        )
    
    # Pareto front
    if 'optimization' in results and 'pareto_objectives' in results['optimization'].get('multi_objective', {}):
        obj = results['optimization']['multi_objective']['pareto_objectives']
        if hasattr(obj, 'numpy'):
            obj = obj.numpy()
        
        pareto_mask = np.ones(len(obj), dtype=bool)
        plot_pareto_front(
            obj, pareto_mask,
            ['HOMO', 'LUMO', 'Gap'],
            os.path.join(save_dir, 'pareto_front.png')
        )
    
    # Optimization comparison
    if 'optimization' in results:
        plot_optimization_comparison(
            results['optimization'],
            os.path.join(save_dir, 'optimization_comparison.png')
        )
    
    logger.info(f"Created results report in {save_dir}")
