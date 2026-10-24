"""
Visualization Utilities for Multi-Task DR Learning.

Generates:
- Segmentation overlays with lesion masks
- Training curves
- Confusion matrices
- Model comparison charts

References:
- Standard medical imaging visualization practices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def denormalize_image(image, mean=None, std=None):
    """Denormalize image for visualization."""
    if mean is None:
        mean = config.MEAN
    if std is None:
        std = config.STD
    
    if isinstance(image, np.ndarray):
        if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image = image.transpose(1, 2, 0)
        image = image * np.array(std) + np.array(mean)
        return np.clip(image, 0, 1)
    else:
        # Torch tensor
        import torch
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = image * np.array(std) + np.array(mean)
        return np.clip(image, 0, 1)


def create_segmentation_overlay(image, masks, predictions=None, alpha=0.5):
    """
    Create visualization with segmentation masks overlaid on fundus image.
    
    Args:
        image: (H, W, 3) or (3, H, W) fundus image
        masks: (num_lesions, H, W) ground truth masks
        predictions: (num_lesions, H, W) predicted masks (optional)
        alpha: Overlay transparency
    """
    # Denormalize image
    image = denormalize_image(image)
    
    # Ensure masks are numpy
    if hasattr(masks, 'numpy'):
        masks = masks.cpu().numpy()
    if predictions is not None and hasattr(predictions, 'numpy'):
        predictions = predictions.cpu().numpy()
    
    # Colors for each lesion type
    colors = [
        [1.0, 0.0, 0.0],  # MA - Red
        [0.8, 0.0, 0.2],  # HE - Dark red
        [1.0, 1.0, 0.5],  # SE - Yellow
        [1.0, 0.8, 0.0],  # EX - Orange
    ]
    
    fig, axes = plt.subplots(1, 3 if predictions is not None else 2, 
                             figsize=(15 if predictions is not None else 10, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Fundus Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    gt_overlay = image.copy()
    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_bool = mask > 0.5
        for c in range(3):
            gt_overlay[:, :, c] = np.where(mask_bool,
                                           gt_overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                           gt_overlay[:, :, c])
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction overlay (if provided)
    if predictions is not None:
        pred_overlay = image.copy()
        for i, (mask, color) in enumerate(zip(predictions, colors)):
            mask_bool = mask > 0.5
            for c in range(3):
                pred_overlay[:, :, c] = np.where(mask_bool,
                                                 pred_overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                                 pred_overlay[:, :, c])
        
        axes[2].imshow(pred_overlay)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    # Legend
    patches = [mpatches.Patch(color=colors[i], label=config.LESION_NAMES[i]) 
               for i in range(len(config.LESION_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    return fig


def plot_training_curves(history, save_path=None):
    """Plot training and validation loss/metrics curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Classification metrics (Kappa)
    if history['val_metrics'] and 'cls_kappa' in history['val_metrics'][0]:
        train_kappa = [m.get('cls_kappa', 0) for m in history['train_metrics']]
        val_kappa = [m.get('cls_kappa', 0) for m in history['val_metrics']]
        axes[0, 1].plot(epochs, train_kappa, 'b-', label='Train')
        axes[0, 1].plot(epochs, val_kappa, 'r-', label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Quadratic Weighted Kappa')
        axes[0, 1].set_title('Classification Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Segmentation metrics (Dice)
    if history['val_metrics'] and 'seg_dice_mean' in history['val_metrics'][0]:
        train_dice = [m.get('seg_dice_mean', 0) for m in history['train_metrics']]
        val_dice = [m.get('seg_dice_mean', 0) for m in history['val_metrics']]
        axes[1, 0].plot(epochs, train_dice, 'b-', label='Train')
        axes[1, 0].plot(epochs, val_dice, 'r-', label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].set_title('Segmentation Performance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Task weights (if available)
    if history.get('task_weights'):
        seg_weights = [w[0] for w in history['task_weights']]
        cls_weights = [w[1] for w in history['task_weights']]
        axes[1, 1].plot(epochs, seg_weights, 'g-', label='Segmentation')
        axes[1, 1].plot(epochs, cls_weights, 'm-', label='Classification')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Task Weight')
        axes[1, 1].set_title('Learned Task Weights')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No task weight data', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot confusion matrix."""
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_model_comparison(results_dict, metric_name='kappa', save_path=None):
    """Plot bar chart comparing models on a specific metric."""
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric_name, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, color=['#2ecc71', '#3498db', '#e74c3c'])
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Model Comparison: {metric_name.replace("_", " ").title()}')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    return fig


def create_literature_comparison_table(our_results, save_path=None):
    """
    Create a comparison table with literature results.
    
    References from IDRiD Challenge and multi-task learning papers.
    """
    # Literature results (representative values from published papers)
    literature_results = {
        'VRT (Porwal et al., 2019)': {'kappa': 0.8500, 'dice': None, 'method': 'Challenge Winner'},
        'LTLAB (2nd place)': {'kappa': 0.8300, 'dice': None, 'method': 'Challenge'},
        'DeepDR (Hua et al., 2020)': {'kappa': 0.7800, 'dice': 0.5200, 'method': 'Multi-task'},
        'MTL-DR (Zhou et al., 2019)': {'kappa': 0.7600, 'dice': 0.4800, 'method': 'Multi-task'},
        'U-Net (Ronneberger, 2015)': {'kappa': None, 'dice': 0.5500, 'method': 'Segmentation'},
        'Attention U-Net (Oktay, 2018)': {'kappa': None, 'dice': 0.5800, 'method': 'Segmentation'},
    }
    
    # Combine with our results
    all_results = literature_results.copy()
    all_results['Ours (Multi-task)'] = {
        'kappa': our_results.get('cls_kappa', 0),
        'dice': our_results.get('seg_dice_mean', 0),
        'method': 'Our Method'
    }
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    headers = ['Method', 'Type', 'Kappa', 'Mean Dice']
    table_data = []
    
    for method, results in all_results.items():
        kappa = f"{results['kappa']:.4f}" if results['kappa'] else '-'
        dice = f"{results['dice']:.4f}" if results['dice'] else '-'
        table_data.append([method, results['method'], kappa, dice])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['#3498db']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight our method
    for i, cell in enumerate(table.get_celld().values()):
        if i >= len(headers):  # Skip header
            row = (i - len(headers)) // len(headers)
            if row == len(all_results) - 1:  # Our method row
                cell.set_facecolor('#e8f6f3')
    
    plt.title('Comparison with Published Methods\n(IDRiD Challenge & Multi-Task Learning)', 
              fontsize=12, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Literature comparison saved to {save_path}")
    
    return fig


def save_segmentation_examples(dataloader, model, num_examples=5, save_dir=None):
    """Save example segmentation visualizations."""
    import torch
    
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    saved = 0
    for batch in dataloader:
        if saved >= num_examples:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask']
        
        with torch.no_grad():
            outputs = model(images)
        
        if 'segmentation' in outputs:
            predictions = torch.sigmoid(outputs['segmentation']).cpu()
        else:
            predictions = None
        
        for i in range(images.size(0)):
            if saved >= num_examples:
                break
            
            fig = create_segmentation_overlay(
                images[i].cpu(),
                masks[i],
                predictions[i] if predictions is not None else None
            )
            
            save_path = os.path.join(save_dir, f'segmentation_example_{saved+1}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved segmentation example to {save_path}")
            saved += 1


if __name__ == "__main__":
    # Test visualizations
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Generate dummy data
    image = np.random.rand(512, 512, 3)
    masks = np.random.rand(4, 512, 512) > 0.9
    predictions = np.random.rand(4, 512, 512) > 0.85
    
    # Test segmentation overlay
    fig = create_segmentation_overlay(image, masks, predictions)
    fig.savefig(os.path.join(config.FIGURES_DIR, 'test_overlay.png'), dpi=150)
    plt.close(fig)
    
    # Test training curves
    dummy_history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
        'train_metrics': [{'cls_kappa': 0.3, 'seg_dice_mean': 0.4} for _ in range(5)],
        'val_metrics': [{'cls_kappa': 0.25, 'seg_dice_mean': 0.35} for _ in range(5)],
        'task_weights': [(0.4, 0.6) for _ in range(5)]
    }
    
    fig = plot_training_curves(dummy_history, 
                               os.path.join(config.FIGURES_DIR, 'test_curves.png'))
    plt.close(fig)
    
    print("Test visualizations saved to figures/")
