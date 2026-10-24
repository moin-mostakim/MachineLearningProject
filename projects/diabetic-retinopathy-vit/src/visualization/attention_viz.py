"""
Attention visualization and explainability tools.

Provides:
- Attention map visualization
- Lesion-specific attention heatmaps
- Grad-CAM visualization
- Integrated gradients saliency maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import cv2


def visualize_attention_maps(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    save_path: Optional[str] = None,
    head_fusion: str = 'mean',
    layer_indices: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Visualize attention maps from ViT model.
    
    Args:
        model: ViT model with attention output
        image: Preprocessed image tensor [1, C, H, W]
        original_image: Original image for overlay [H, W, C]
        save_path: Optional path to save figure
        head_fusion: How to combine heads ('mean', 'max', 'min')
        layer_indices: Which layers to visualize (default: all)
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Get attention maps
    with torch.no_grad():
        if hasattr(model, 'get_attention_maps'):
            info = model.get_attention_maps(image)
            attentions = info.get('attentions', [])
        else:
            _, _, attentions = model(image, return_attention=True)
    
    if attentions is None or len(attentions) == 0:
        print("No attention maps available")
        return None
    
    # Select layers
    if layer_indices is None:
        layer_indices = list(range(len(attentions)))
    
    # Process attention maps
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(2, max(num_layers, 1), figsize=(4 * num_layers, 8))
    
    if num_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for i, layer_idx in enumerate(layer_indices):
        if isinstance(attentions[layer_idx], dict):
            attn = attentions[layer_idx].get('standard', None)
            if attn is None:
                continue
        else:
            attn = attentions[layer_idx]
        
        # Get CLS token attention to all patches
        # attn shape: [B, num_heads, N, N]
        attn = attn[0]  # First (and only) batch
        
        # Fuse heads
        if head_fusion == 'mean':
            attn = attn.mean(dim=0)
        elif head_fusion == 'max':
            attn = attn.max(dim=0)[0]
        else:
            attn = attn.min(dim=0)[0]
        
        # Get attention from CLS token (first row)
        cls_attn = attn[0, 1:]  # Exclude self-attention
        
        # Reshape to spatial
        num_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size ** 2 == num_patches:
            attn_map = cls_attn.reshape(grid_size, grid_size)
        else:
            # Interpolate to square
            sqrt_approx = int(np.ceil(np.sqrt(num_patches)))
            attn_map = F.interpolate(
                cls_attn.view(1, 1, -1),
                size=sqrt_approx ** 2,
                mode='linear',
                align_corners=False,
            ).view(sqrt_approx, sqrt_approx)
        
        attn_map = attn_map.cpu().numpy()
        
        # Resize to original image size
        attn_map_resized = cv2.resize(
            attn_map,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        # Normalize
        attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (
            attn_map_resized.max() - attn_map_resized.min() + 1e-8
        )
        
        # Plot attention map
        axes[0, i].imshow(attn_map_resized, cmap='hot')
        axes[0, i].set_title(f'Layer {layer_idx + 1} Attention')
        axes[0, i].axis('off')
        
        # Plot overlay
        heatmap = cm.hot(attn_map_resized)[:, :, :3]
        overlay = 0.5 * original_image / 255.0 + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Layer {layer_idx + 1} Overlay')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    return fig


def visualize_lesion_attention(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize lesion-specific attention from Multi-Scale Lesion ViT.
    
    Args:
        model: MultiScaleLesionViT model
        image: Preprocessed image tensor [1, C, H, W]
        original_image: Original image for overlay
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    
    if not hasattr(model, 'get_lesion_heatmaps'):
        print("Model does not support lesion heatmaps")
        return None
    
    with torch.no_grad():
        # Get lesion heatmaps
        heatmaps = model.get_lesion_heatmaps(image)
        
        # Get lesion probabilities
        _, _, info = model(image, return_attention=True)
        lesion_probs = info.get('lesion_probs', None)
    
    if heatmaps is None:
        print("Could not generate lesion heatmaps")
        return None
    
    # Lesion type names
    lesion_names = [
        'Microaneurysms',
        'Hemorrhages',
        'Hard Exudates',
        'Soft Exudates',
    ]
    
    num_lesions = heatmaps.shape[1]
    fig, axes = plt.subplots(2, num_lesions, figsize=(4 * num_lesions, 8))
    
    for i in range(num_lesions):
        heatmap = heatmaps[0, i].cpu().numpy()
        
        # Resize to original image size
        heatmap_resized = cv2.resize(
            heatmap,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        # Normalize
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
            heatmap_resized.max() - heatmap_resized.min() + 1e-8
        )
        
        # Plot heatmap
        axes[0, i].imshow(heatmap_resized, cmap='hot')
        prob_str = f" ({lesion_probs[0, i]:.2f})" if lesion_probs is not None else ""
        axes[0, i].set_title(f'{lesion_names[i]}{prob_str}')
        axes[0, i].axis('off')
        
        # Plot overlay
        heatmap_color = cm.hot(heatmap_resized)[:, :, :3]
        overlay = 0.5 * original_image / 255.0 + 0.5 * heatmap_color
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'{lesion_names[i]} Overlay')
        axes[1, i].axis('off')
    
    plt.suptitle('Lesion-Aware Attention Maps', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved lesion attention to {save_path}")
    
    return fig


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM activation map.
    
    Reference: "Grad-CAM: Visual Explanations from Deep Networks"
    
    Args:
        model: Model to analyze
        image: Input image [1, C, H, W]
        target_class: Class to compute CAM for (default: predicted)
        target_layer: Layer to compute CAM from (default: last conv/block)
        
    Returns:
        CAM heatmap [H, W]
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    image.requires_grad = True
    
    # Get target layer
    if target_layer is None:
        if hasattr(model, 'get_cam_layers'):
            target_layer = model.get_cam_layers()
        elif hasattr(model, 'blocks'):
            target_layer = model.blocks[-1]
        elif hasattr(model, 'backbone'):
            target_layer = model.backbone.layer4
        else:
            raise ValueError("Could not determine target layer for Grad-CAM")
    
    # Hook to capture activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        output = model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        activation = activations[0]
        gradient = gradients[0]
        
        # For transformer: activation shape [B, N, D]
        if activation.dim() == 3:
            # Average gradients over tokens
            weights = gradient.mean(dim=1, keepdim=True)
            
            # Weighted combination
            cam = (weights * activation).sum(dim=-1)  # [B, N]
            
            # Reshape to spatial (excluding CLS token if present)
            num_tokens = cam.shape[1]
            if hasattr(model, 'patch_embed'):
                # Exclude CLS token
                cam = cam[:, 1:]
                num_tokens = cam.shape[1]
            
            grid_size = int(np.sqrt(num_tokens))
            if grid_size ** 2 == num_tokens:
                cam = cam.view(1, grid_size, grid_size)
            else:
                # Approximate
                sqrt_approx = int(np.ceil(np.sqrt(num_tokens)))
                cam = F.interpolate(
                    cam.unsqueeze(1),
                    size=(sqrt_approx, sqrt_approx),
                    mode='bilinear',
                    align_corners=False,
                )
        else:
            # For CNN: activation shape [B, C, H, W]
            weights = gradient.mean(dim=[2, 3], keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1))
        
        # Normalize
        cam = cam.cpu().numpy()[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
        
    finally:
        forward_handle.remove()
        backward_handle.remove()


def compute_integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution.
    
    Reference: "Axiomatic Attribution for Deep Networks"
    
    Args:
        model: Model to analyze
        image: Input image [1, C, H, W]
        target_class: Class to compute IG for
        steps: Number of integration steps
        baseline: Baseline image (default: black)
        
    Returns:
        Attribution map [H, W]
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(image)
    else:
        baseline = baseline.to(device)
    
    # Get target class
    if target_class is None:
        with torch.no_grad():
            output = model(image)
            target_class = output.argmax(dim=1).item()
    
    # Compute integrated gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (image - baseline)
        for i in range(1, steps + 1)
    ]
    
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        
        output = model(scaled_input)
        model.zero_grad()
        output[0, target_class].backward()
        
        gradients.append(scaled_input.grad.detach())
    
    # Average gradients
    avg_gradient = torch.stack(gradients).mean(dim=0)
    
    # Compute attribution
    attribution = (image - baseline) * avg_gradient
    
    # Sum over channels
    attribution = attribution.sum(dim=1).cpu().numpy()[0]
    
    # Take absolute value and normalize
    attribution = np.abs(attribution)
    attribution = (attribution - attribution.min()) / (
        attribution.max() - attribution.min() + 1e-8
    )
    
    return attribution


def create_explainability_report(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    true_label: int,
    class_names: List[str],
    save_path: str,
) -> plt.Figure:
    """
    Create comprehensive explainability report.
    
    Args:
        model: Model to explain
        image: Preprocessed image
        original_image: Original image
        true_label: Ground truth label
        class_names: List of class names
        save_path: Path to save report
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = output.argmax(dim=1).item()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Original image with prediction
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original_image)
    ax1.set_title(f'Prediction: {class_names[pred_class]}\n'
                  f'True: {class_names[true_label]}\n'
                  f'Confidence: {probs[pred_class]:.2%}')
    ax1.axis('off')
    
    # Probability distribution
    ax2 = fig.add_subplot(2, 3, 2)
    colors = ['green' if i == true_label else 'blue' for i in range(len(class_names))]
    colors[pred_class] = 'red' if pred_class != true_label else 'green'
    ax2.barh(class_names, probs, color=colors)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 1)
    
    # Grad-CAM
    ax3 = fig.add_subplot(2, 3, 3)
    try:
        gradcam = compute_gradcam(model, image_tensor, pred_class)
        gradcam_resized = cv2.resize(
            gradcam,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        ax3.imshow(original_image)
        ax3.imshow(gradcam_resized, cmap='jet', alpha=0.5)
        ax3.set_title('Grad-CAM')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Grad-CAM Error:\n{str(e)[:50]}',
                ha='center', va='center')
    ax3.axis('off')
    
    # Integrated Gradients
    ax4 = fig.add_subplot(2, 3, 4)
    try:
        ig = compute_integrated_gradients(model, image_tensor, pred_class, steps=20)
        ig_resized = cv2.resize(
            ig,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        ax4.imshow(original_image)
        ax4.imshow(ig_resized, cmap='hot', alpha=0.5)
        ax4.set_title('Integrated Gradients')
    except Exception as e:
        ax4.text(0.5, 0.5, f'IG Error:\n{str(e)[:50]}',
                ha='center', va='center')
    ax4.axis('off')
    
    # Lesion attention (if available)
    ax5 = fig.add_subplot(2, 3, 5)
    if hasattr(model, 'get_lesion_heatmaps'):
        try:
            with torch.no_grad():
                heatmaps = model.get_lesion_heatmaps(image_tensor)
            if heatmaps is not None:
                combined = heatmaps[0].mean(dim=0).cpu().numpy()
                combined_resized = cv2.resize(
                    combined,
                    (original_image.shape[1], original_image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                ax5.imshow(original_image)
                ax5.imshow(combined_resized, cmap='hot', alpha=0.5)
                ax5.set_title('Lesion Attention (Combined)')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Lesion Attn Error:\n{str(e)[:50]}',
                    ha='center', va='center')
    else:
        ax5.text(0.5, 0.5, 'Lesion attention\nnot available',
                ha='center', va='center')
    ax5.axis('off')
    
    # Scale weights (if available)
    ax6 = fig.add_subplot(2, 3, 6)
    if hasattr(model, 'get_attention_maps'):
        try:
            with torch.no_grad():
                info = model.get_attention_maps(image_tensor)
            if 'scale_weights' in info and info['scale_weights'] is not None:
                scale_weights = info['scale_weights'].cpu().numpy()[0]
                scale_names = ['8x8 (Fine)', '16x16 (Med)', '32x32 (Coarse)']
                ax6.bar(scale_names[:len(scale_weights)], scale_weights)
                ax6.set_ylabel('Weight')
                ax6.set_title('Multi-Scale Weights')
            else:
                ax6.text(0.5, 0.5, 'Scale weights\nnot available',
                        ha='center', va='center')
        except Exception:
            ax6.text(0.5, 0.5, 'Scale weights\nnot available',
                    ha='center', va='center')
    else:
        ax6.text(0.5, 0.5, 'Scale weights\nnot available',
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved explainability report to {save_path}")
    
    return fig


def save_attention_grid(
    attention_maps: List[np.ndarray],
    save_path: str,
    titles: Optional[List[str]] = None,
):
    """
    Save grid of attention maps.
    
    Args:
        attention_maps: List of attention map arrays
        save_path: Path to save
        titles: Optional list of titles
    """
    n = len(attention_maps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    
    for i, attn in enumerate(attention_maps):
        r, c = i // cols, i % cols
        axes[r, c].imshow(attn, cmap='hot')
        if titles and i < len(titles):
            axes[r, c].set_title(titles[i])
        axes[r, c].axis('off')
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
