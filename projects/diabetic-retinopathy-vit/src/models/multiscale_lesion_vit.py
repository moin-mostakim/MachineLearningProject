"""
Multi-Scale Vision Transformer with Lesion-Aware Attention for DR Grading.

This is our proposed method that combines:
1. Multi-scale processing: Capture lesions at different sizes
2. Lesion-aware attention: Specialized attention for DR lesion types
3. Explainability: Built-in attention visualization

References:
- Multi-scale ViT: Liu et al. (2021) "Swin Transformer"
- Medical attention: Schlemper et al. (2019) "Attention-Gated Networks"
- DR-specific: Voets et al. (2019) "Reproduction study using public dataset of DR detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class LesionAwareAttention(nn.Module):
    """
    Lesion-aware attention mechanism.
    
    This module learns to attend to different types of DR lesions:
    - Microaneurysms (small, punctate)
    - Hemorrhages (variable size, dark red)
    - Hard exudates (bright, well-defined)
    - Soft exudates/cotton wool spots (fluffy, white)
    
    Each lesion type gets dedicated attention heads.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_lesion_types: int = 4,
        lesion_attention_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        """
        Initialize lesion-aware attention.
        
        Args:
            dim: Input dimension
            num_heads: Total number of attention heads
            num_lesion_types: Number of lesion types to model
            lesion_attention_heads: Heads dedicated to lesion attention
            qkv_bias: Use bias in QKV projection
            attn_drop: Attention dropout
            proj_drop: Output projection dropout
        """
        super().__init__()
        
        assert lesion_attention_heads <= num_heads
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_lesion_types = num_lesion_types
        self.lesion_attention_heads = lesion_attention_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard QKV for general attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Lesion-specific query projections
        # Each lesion type learns its own query pattern
        self.lesion_queries = nn.ModuleList([
            nn.Linear(dim, self.head_dim * lesion_attention_heads, bias=qkv_bias)
            for _ in range(num_lesion_types)
        ])
        
        # Lesion type embedding (learnable)
        self.lesion_embeddings = nn.Parameter(
            torch.randn(num_lesion_types, self.head_dim)
        )
        
        # Gating mechanism for lesion attention
        self.lesion_gate = nn.Sequential(
            nn.Linear(dim, num_lesion_types),
            nn.Softmax(dim=-1),
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.lesion_attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with lesion-aware attention.
        
        Args:
            x: Input tensor [B, N, D]
            return_attention: Return attention weights
            
        Returns:
            out: Output tensor [B, N, D]
            attn_dict: Optional dict with attention weights
        """
        B, N, D = x.shape
        
        # Standard attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Lesion-aware attention for dedicated heads
        lesion_gate = self.lesion_gate(x[:, 0])  # Use CLS token for gating [B, num_lesion]
        
        lesion_attns = []
        for i, lesion_query_proj in enumerate(self.lesion_queries):
            # Lesion-specific queries
            lesion_q = lesion_query_proj(x)  # [B, N, lesion_heads * head_dim]
            lesion_q = lesion_q.reshape(B, N, self.lesion_attention_heads, self.head_dim)
            lesion_q = lesion_q.permute(0, 2, 1, 3)  # [B, lesion_heads, N, head_dim]
            
            # Add lesion type embedding to query
            lesion_q = lesion_q + self.lesion_embeddings[i].view(1, 1, 1, -1)
            
            # Use standard K, V for lesion attention
            k_lesion = k[:, :self.lesion_attention_heads]
            v_lesion = v[:, :self.lesion_attention_heads]
            
            lesion_attn = (lesion_q @ k_lesion.transpose(-2, -1)) * self.scale
            lesion_attn = lesion_attn.softmax(dim=-1)
            lesion_attns.append(lesion_attn * lesion_gate[:, i:i+1, None, None])
        
        # Combine lesion attentions
        combined_lesion_attn = sum(lesion_attns)
        
        # Blend standard and lesion attention for dedicated heads
        attn_modified = attn.clone()
        blend_weight = 0.5  # Learnable in full implementation
        attn_modified[:, :self.lesion_attention_heads] = (
            blend_weight * attn[:, :self.lesion_attention_heads] +
            (1 - blend_weight) * combined_lesion_attn
        )
        
        # Compute output
        out = (attn_modified @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Store for visualization
        self.attention_weights = attn.detach()
        self.lesion_attention_weights = {
            f'lesion_{i}': lesion_attns[i].detach()
            for i in range(self.num_lesion_types)
        }
        
        if return_attention:
            attn_dict = {
                'standard': attn,
                'lesion': self.lesion_attention_weights,
                'lesion_gates': lesion_gate,
            }
            return out, attn_dict
        
        return out, None


class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patch embedding for capturing lesions at different sizes.
    
    Uses multiple patch sizes to capture:
    - Small patches (8x8): Microaneurysms
    - Medium patches (16x16): Hemorrhages, exudates
    - Large patches (32x32): Overall structure, neovascularization
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: List[int] = [8, 16, 32],
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        
        # Create embedding for each scale
        self.patch_embeds = nn.ModuleList()
        self.num_patches_per_scale = []
        
        for ps in patch_sizes:
            num_patches = (img_size // ps) ** 2
            self.num_patches_per_scale.append(num_patches)
            
            self.patch_embeds.append(
                nn.Conv2d(
                    in_channels,
                    embed_dim // len(patch_sizes),
                    kernel_size=ps,
                    stride=ps,
                )
            )
        
        # Projection to combine scales
        self.scale_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scale-specific position embeddings
        self.scale_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_p + 1, embed_dim // len(patch_sizes)))
            for num_p in self.num_patches_per_scale
        ])
        
        # Scale attention: learn to weight different scales
        self.scale_attention = nn.Sequential(
            nn.Linear(embed_dim, len(patch_sizes)),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale embedding.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            embeddings: Combined embeddings [B, N, embed_dim]
            scale_weights: Attention weights for each scale [B, num_scales]
        """
        B = x.shape[0]
        scale_embeddings = []
        
        for i, (patch_embed, pos_embed) in enumerate(
            zip(self.patch_embeds, self.scale_pos_embeds)
        ):
            # Patch embedding at this scale
            patches = patch_embed(x)  # [B, D/S, H/P, W/P]
            patches = patches.flatten(2).transpose(1, 2)  # [B, N, D/S]
            
            # Add position embedding
            cls_token = torch.zeros(B, 1, patches.shape[-1], device=x.device)
            patches = torch.cat([cls_token, patches], dim=1)
            patches = patches + pos_embed
            
            scale_embeddings.append(patches)
        
        # Interpolate to common sequence length
        target_len = max(e.shape[1] for e in scale_embeddings)
        aligned_embeddings = []
        
        for emb in scale_embeddings:
            if emb.shape[1] < target_len:
                emb = F.interpolate(
                    emb.transpose(1, 2),
                    size=target_len,
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
            aligned_embeddings.append(emb)
        
        # Concatenate along feature dimension
        combined = torch.cat(aligned_embeddings, dim=-1)  # [B, N, D]
        combined = self.scale_proj(combined)
        
        # Compute scale attention weights (for explainability)
        global_feat = combined.mean(dim=1)
        scale_weights = self.scale_attention(global_feat)
        
        return combined, scale_weights


class MultiScaleTransformerBlock(nn.Module):
    """Transformer block with lesion-aware attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_lesion_types: int = 4,
        lesion_attention_heads: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LesionAwareAttention(
            dim,
            num_heads=num_heads,
            num_lesion_types=num_lesion_types,
            lesion_attention_heads=lesion_attention_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass."""
        attn_out, attn_dict = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, attn_dict


class DropPath(nn.Module):
    """Stochastic depth."""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep_prob) * random_tensor


class MultiScaleLesionViT(nn.Module):
    """
    Multi-Scale Vision Transformer with Lesion-Aware Attention.
    
    Our proposed method for DR grading combining:
    1. Multi-scale patch embedding for lesion size variation
    2. Lesion-aware attention mechanism
    3. Built-in explainability through attention visualization
    
    Key innovations:
    - Dedicated attention heads for different lesion types
    - Scale-aware processing for microaneurysms vs large hemorrhages
    - Interpretable lesion gates showing which lesion types are detected
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: List[int] = [8, 16, 32],
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        num_lesion_types: int = 4,
        lesion_attention_heads: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
    ):
        """
        Initialize Multi-Scale Lesion ViT.
        
        Args:
            img_size: Input image size
            patch_sizes: List of patch sizes for multi-scale
            in_channels: Number of input channels
            num_classes: Number of DR grades
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            num_lesion_types: Number of lesion types (default 4)
            lesion_attention_heads: Heads for lesion attention
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Use bias in QKV
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.num_lesion_types = num_lesion_types
        
        # Multi-scale patch embedding
        self.patch_embed = MultiScalePatchEmbedding(
            img_size=img_size,
            patch_sizes=patch_sizes,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding (learnable)
        max_patches = (img_size // min(patch_sizes)) ** 2 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks with lesion-aware attention
        self.blocks = nn.ModuleList([
            MultiScaleTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_lesion_types=num_lesion_types,
                lesion_attention_heads=lesion_attention_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head with lesion-type aware features
        self.lesion_head = nn.Linear(embed_dim, num_lesion_types)
        self.main_head = nn.Linear(embed_dim + num_lesion_types, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through features.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_attention: Return attention maps
            
        Returns:
            features: [B, embed_dim]
            info_dict: Dict with scale weights, lesion info, attentions
        """
        B = x.shape[0]
        
        # Multi-scale patch embedding
        x, scale_weights = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding (truncate/interpolate if needed)
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_embed
        
        x = self.pos_drop(x)
        
        # Transformer blocks
        attentions = []
        lesion_info = []
        
        for block in self.blocks:
            x, attn_dict = block(x, return_attention=return_attention)
            if return_attention and attn_dict is not None:
                attentions.append(attn_dict)
                if 'lesion_gates' in attn_dict:
                    lesion_info.append(attn_dict['lesion_gates'])
        
        x = self.norm(x)
        
        info_dict = {
            'scale_weights': scale_weights,
            'attentions': attentions if return_attention else None,
            'lesion_gates': lesion_info if lesion_info else None,
        }
        
        return x[:, 0], info_dict
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
        return_lesion_probs: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: Return (logits, features)
            return_attention: Return attention info
            return_lesion_probs: Return lesion type probabilities
            
        Returns:
            logits or tuple of (logits, features, info_dict)
        """
        features, info_dict = self.forward_features(x, return_attention)
        
        # Lesion-type predictions (auxiliary)
        lesion_probs = torch.sigmoid(self.lesion_head(features))
        
        # Combine features with lesion info for final classification
        combined_features = torch.cat([features, lesion_probs], dim=-1)
        logits = self.main_head(combined_features)
        
        if return_attention or return_lesion_probs:
            info_dict['lesion_probs'] = lesion_probs
        
        if return_attention:
            return logits, features, info_dict
        if return_features:
            return logits, features
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict:
        """
        Get attention maps for visualization.
        
        Returns dict with:
        - standard_attention: Standard attention weights
        - lesion_attention: Per-lesion-type attention
        - scale_weights: Multi-scale attention weights
        - lesion_probs: Predicted lesion type probabilities
        """
        _, _, info_dict = self.forward(x, return_attention=True)
        return info_dict
    
    def get_lesion_heatmaps(
        self,
        x: torch.Tensor,
        lesion_type: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get lesion-specific attention heatmaps.
        
        Args:
            x: Input tensor [B, C, H, W]
            lesion_type: Specific lesion type (0-3), or None for all
            
        Returns:
            heatmaps: [B, H, W] or [B, num_lesion_types, H, W]
        """
        info_dict = self.get_attention_maps(x)
        
        if info_dict['attentions'] is None or len(info_dict['attentions']) == 0:
            return None
        
        # Aggregate lesion attention from last layer
        last_attn = info_dict['attentions'][-1]
        
        if not isinstance(last_attn, dict) or 'lesion' not in last_attn:
            return None
        
        # Get spatial attention (exclude CLS token)
        B = x.shape[0]
        img_size = x.shape[-1]
        patch_size = 16  # Approximate
        grid_size = img_size // patch_size
        
        heatmaps = []
        for i in range(self.num_lesion_types):
            lesion_key = f'lesion_{i}'
            if lesion_key not in last_attn['lesion']:
                continue
                
            lesion_attn = last_attn['lesion'][lesion_key]
            
            # Average over heads, take CLS attention to patches
            attn_cls = lesion_attn.mean(dim=1)[:, 0, 1:]  # [B, N-1]
            
            # Reshape to spatial using interpolation to target size
            attn_map = F.interpolate(
                attn_cls.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, N-1]
                size=(grid_size, grid_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1).squeeze(1)  # [B, grid_size, grid_size]
            
            heatmaps.append(attn_map)
        
        if not heatmaps:
            return None
            
        heatmaps = torch.stack(heatmaps, dim=1)  # [B, num_lesion, H, W]
        
        if lesion_type is not None:
            return heatmaps[:, lesion_type]
        return heatmaps
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        patch_embed = sum(p.numel() for p in self.patch_embed.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        heads = sum(p.numel() for p in self.lesion_head.parameters())
        heads += sum(p.numel() for p in self.main_head.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'patch_embed': patch_embed,
            'blocks': blocks,
            'heads': heads,
        }


def create_multiscale_lesion_vit(
    img_size: int = 224,
    num_classes: int = 5,
    embed_dim: int = 384,
    depth: int = 8,
    num_heads: int = 6,
    num_lesion_types: int = 4,
    **kwargs,
) -> MultiScaleLesionViT:
    """
    Factory function to create Multi-Scale Lesion ViT.
    
    Args:
        img_size: Input image size
        num_classes: Number of classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        num_lesion_types: Number of lesion types
        **kwargs: Additional arguments
        
    Returns:
        MultiScaleLesionViT model
    """
    return MultiScaleLesionViT(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_lesion_types=num_lesion_types,
        **kwargs,
    )
