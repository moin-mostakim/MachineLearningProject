"""
Kolmogorov-Arnold Network (KAN) Layer Implementation.

Based on Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
arXiv:2404.19756

The key insight of KAN is to use learnable activation functions (B-splines)
on edges (weights) rather than fixed activations on nodes.

Mathematical foundation:
- Kolmogorov-Arnold representation theorem states that any multivariate
  continuous function f: [0,1]^n -> R can be written as:
  f(x) = Σ_{q=0}^{2n} Φ_q(Σ_{p=1}^n φ_{q,p}(x_p))
  
- KAN implements this with learnable univariate functions (B-splines)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class BSplineBasis(nn.Module):
    """
    B-spline basis functions for KAN.
    
    Uses a simplified approach with fixed-size output for stable computation.
    Each basis function is evaluated at the input points.
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        grid_eps: float = 0.02,
    ):
        """
        Initialize B-spline basis.
        
        Args:
            grid_size: Number of grid intervals (determines num_basis)
            spline_order: Order of B-spline (not used in simplified version)
            grid_range: Range for grid points
            grid_eps: Extension beyond grid_range
        """
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Number of basis functions equals grid_size
        self.num_basis = grid_size
        
        # Create grid centers for basis functions
        centers = torch.linspace(
            grid_range[0] - grid_eps,
            grid_range[1] + grid_eps,
            grid_size
        )
        self.register_buffer("centers", centers)
        
        # Width of each basis function
        self.width = (grid_range[1] - grid_range[0] + 2 * grid_eps) / (grid_size - 1) if grid_size > 1 else 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute basis values using Gaussian-like radial basis functions.
        
        This is a simplified version that provides smooth, differentiable
        basis functions suitable for gradient-based learning.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Basis values of shape (..., in_features, num_basis)
        """
        # x shape: (batch, in_features)
        # centers shape: (num_basis,)
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, num_basis)
        
        # Compute radial basis functions (Gaussian-like)
        # This provides smooth, differentiable basis functions
        distances = (x_expanded - centers) / self.width
        bases = torch.exp(-0.5 * distances ** 2)
        
        # Normalize so basis functions sum to approximately 1
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        
        return bases  # (batch, in_features, num_basis)


class KANLinear(nn.Module):
    """
    KAN Linear layer with learnable B-spline activations.
    
    Unlike standard linear layers where:
        y = W @ x + b (with activation applied after)
    
    KAN computes:
        y_j = Σ_i φ_{i,j}(x_i)
    
    where φ_{i,j} are learnable univariate functions implemented as B-splines.
    
    Each edge has its own learnable activation function, providing more
    expressiveness with fewer parameters than MLPs.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: str = "silu",
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        grid_eps: float = 0.02,
    ):
        """
        Initialize KAN linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            grid_size: Number of B-spline grid intervals
            spline_order: Order of B-splines
            scale_noise: Noise scale for weight initialization
            scale_base: Scale for base activation component
            scale_spline: Scale for spline component
            base_activation: Base activation function (silu, relu, tanh)
            grid_range: Range for B-spline grid
            grid_eps: Grid extension for boundaries
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        # B-spline basis
        self.spline_basis = BSplineBasis(
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
            grid_eps=grid_eps,
        )
        
        # Number of basis functions per input (equals grid_size in simplified version)
        num_basis = grid_size
        self.num_basis = num_basis
        
        # Learnable spline coefficients
        # Shape: (out_features, in_features, num_basis)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * scale_noise
        )
        
        # Base weight (for residual/base activation path)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
        )
        
        # Base activation
        if base_activation == "silu":
            self.base_activation = nn.SiLU()
        elif base_activation == "relu":
            self.base_activation = nn.ReLU()
        elif base_activation == "tanh":
            self.base_activation = nn.Tanh()
        elif base_activation == "gelu":
            self.base_activation = nn.GELU()
        else:
            self.base_activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Computes: y = scale_base * (base_weight @ base_act(x)) 
                    + scale_spline * (spline_weight @ spline_basis(x))
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Output tensor of shape (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Base activation path
        # Shape: (batch, in_features)
        base_output = self.base_activation(x)
        # Linear combination: (batch, out_features)
        base_output = F.linear(base_output, self.base_weight)
        
        # Spline path
        # Get B-spline basis values: (batch, in_features, num_basis)
        spline_basis = self.spline_basis(x)
        
        # Apply spline coefficients
        # spline_weight: (out_features, in_features, num_basis)
        # spline_basis: (batch, in_features, num_basis)
        # We want: (batch, out_features)
        
        # Einsum: sum over in_features and num_basis
        spline_output = torch.einsum(
            "oik,bik->bo",
            self.spline_weight,
            spline_basis
        )
        
        # Combine base and spline paths
        output = self.scale_base * base_output + self.scale_spline * spline_output
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )


class KANLayer(nn.Module):
    """
    Full KAN layer with normalization and optional dropout.
    
    Combines KANLinear with layer normalization for stable training.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: str = "silu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """Initialize KAN layer with normalization."""
        super().__init__()
        
        self.kan_linear = KANLinear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            **kwargs
        )
        
        self.layer_norm = nn.LayerNorm(out_features) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization and dropout."""
        x = self.kan_linear(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


def count_kan_parameters(in_features: int, out_features: int, grid_size: int, spline_order: int) -> int:
    """
    Count parameters in a KAN linear layer.
    
    Parameters:
    - Spline coefficients: out_features * in_features * grid_size
    - Base weights: out_features * in_features
    """
    num_basis = grid_size
    spline_params = out_features * in_features * num_basis
    base_params = out_features * in_features
    return spline_params + base_params


def count_mlp_parameters(in_features: int, out_features: int) -> int:
    """
    Count parameters in an MLP linear layer (weight + bias).
    """
    return out_features * in_features + out_features


if __name__ == "__main__":
    # Test KAN layer
    print("Testing KAN Layer...")
    
    batch_size = 4
    in_features = 512
    out_features = 64
    
    kan_layer = KANLinear(in_features, out_features, grid_size=5, spline_order=3)
    x = torch.randn(batch_size, in_features)
    y = kan_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"KAN parameters: {sum(p.numel() for p in kan_layer.parameters())}")
    
    # Compare with MLP
    mlp_layer = nn.Linear(in_features, out_features)
    print(f"MLP parameters: {sum(p.numel() for p in mlp_layer.parameters())}")
    
    # Parameter comparison
    print("\n=== Parameter Comparison ===")
    kan_params = count_kan_parameters(512, 64, 5, 3)
    mlp_params = count_mlp_parameters(512, 64)
    print(f"KAN layer (512->64): {kan_params:,} parameters")
    print(f"MLP layer (512->64): {mlp_params:,} parameters")
    print(f"Ratio (KAN/MLP): {kan_params/mlp_params:.2f}x")
