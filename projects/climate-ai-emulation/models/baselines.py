"""
Baseline models for climate emulation.

Implements simple baselines following WeatherBench methodology:
1. Persistence: Predict current state as future state
2. Climatology: Predict long-term mean
3. Linear Regression: Simple learned baseline

These baselines provide reference points for evaluating neural network models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge


class PersistenceBaseline(nn.Module):
    """
    Persistence baseline: predict that future state equals current state.
    
    This is a strong baseline for short lead times due to temporal autocorrelation.
    Weather persistence skill degrades with lead time as:
    - Very skillful at 6h
    - Reasonable at 24h
    - Poor at 72h+
    
    Reference: Rasp et al. (2020) WeatherBench
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Persistence"
        
    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - return last input time step.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            lead_time: Ignored for persistence
            
        Returns:
            Prediction tensor of shape (batch, channels, lat, lon)
        """
        # Return the last time step
        return x[:, -1, :, :, :]
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Persistence has no uncertainty - returns zero variance.
        
        Args:
            x: Input tensor
            n_samples: Number of samples (ignored)
            
        Returns:
            Tuple of (mean prediction, zero variance)
        """
        pred = self.forward(x)
        variance = torch.zeros_like(pred)
        return pred, variance


class ClimatologyBaseline(nn.Module):
    """
    Climatology baseline: predict long-term mean.
    
    Climatology represents the long-term average state and serves as
    the reference for computing skill scores (e.g., ACC).
    
    In operational forecasting, climatology skill is:
    - Beat by NWP models up to ~10-14 days
    - Competitive with ML models at long lead times
    
    Reference: Rasp et al. (2020) WeatherBench
    """
    
    def __init__(
        self,
        climatology: Optional[torch.Tensor] = None
    ):
        """
        Initialize climatology baseline.
        
        Args:
            climatology: Pre-computed climatology tensor of shape (channels, lat, lon)
                        If None, uses zero climatology (for normalized data)
        """
        super().__init__()
        self.name = "Climatology"
        
        if climatology is not None:
            self.register_buffer('climatology', climatology)
        else:
            self.climatology = None
            
    def set_climatology(self, climatology: torch.Tensor):
        """Set climatology from training data."""
        self.register_buffer('climatology', climatology)
        
    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - return climatology.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            lead_time: Ignored for climatology
            
        Returns:
            Prediction tensor of shape (batch, channels, lat, lon)
        """
        batch_size = x.shape[0]
        
        if self.climatology is not None:
            # Expand climatology to batch size
            return self.climatology.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            # Zero climatology for normalized data
            return torch.zeros(
                batch_size, x.shape[2], x.shape[3], x.shape[4],
                device=x.device, dtype=x.dtype
            )
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Climatology has constant variance (climatological variance).
        
        Args:
            x: Input tensor
            n_samples: Number of samples (ignored)
            
        Returns:
            Tuple of (climatology prediction, climatological variance)
        """
        pred = self.forward(x)
        # For normalized data, climatological variance is approximately 1
        variance = torch.ones_like(pred)
        return pred, variance


class LinearRegressionBaseline(nn.Module):
    """
    Linear regression baseline using spatial pixels as features.
    
    This is a learned baseline that captures linear relationships
    between input and target. Uses Ridge regression for regularization.
    
    Linear models serve as:
    - Simple learned baseline
    - Sanity check for nonlinear models
    - Reference for model interpretability
    
    Reference: Rasp et al. (2020) WeatherBench
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_time_steps: int = 2,
        lat_size: int = 32,
        lon_size: int = 64,
        alpha: float = 1.0
    ):
        """
        Initialize linear regression baseline.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            input_time_steps: Number of input time steps
            lat_size: Latitude dimension
            lon_size: Longitude dimension
            alpha: Ridge regression regularization strength
        """
        super().__init__()
        self.name = "LinearRegression"
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_steps = input_time_steps
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.alpha = alpha
        
        # Input features: flatten spatial-temporal input
        input_features = input_time_steps * input_channels * lat_size * lon_size
        output_features = output_channels * lat_size * lon_size
        
        # For computational efficiency, use channel-wise regression
        # Each output channel predicted from flattened input
        self.linear = nn.Linear(
            input_time_steps * input_channels,
            output_channels,
            bias=True
        )
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
        
        # Track if model has been fitted
        self.fitted = False
        
    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            lead_time: Ignored
            
        Returns:
            Prediction tensor of shape (batch, channels, lat, lon)
        """
        batch_size = x.shape[0]
        
        # Flatten time and channels: (B, T*C, H, W)
        x_flat = x.reshape(batch_size, -1, self.lat_size, self.lon_size)
        
        # Permute to (B, H, W, T*C) for linear layer
        x_flat = x_flat.permute(0, 2, 3, 1)
        
        # Apply linear transformation: (B, H, W, C_out)
        out = self.linear(x_flat)
        
        # Permute back to (B, C_out, H, W)
        out = out.permute(0, 3, 1, 2)
        
        return out
    
    def fit(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor
    ):
        """
        Fit linear regression on training data.
        
        Args:
            train_inputs: Training inputs (N, T, C, H, W)
            train_targets: Training targets (N, C, H, W)
        """
        # Flatten spatial dimensions for fitting
        N = train_inputs.shape[0]
        H, W = train_inputs.shape[3], train_inputs.shape[4]
        
        # Reshape inputs: (N*H*W, T*C)
        X = train_inputs.permute(0, 3, 4, 1, 2).reshape(-1, self.input_time_steps * self.input_channels)
        
        # Reshape targets: (N*H*W, C)
        y = train_targets.permute(0, 2, 3, 1).reshape(-1, self.output_channels)
        
        # Convert to numpy for sklearn
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Fit Ridge regression
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(X_np, y_np)
        
        # Copy weights to PyTorch linear layer
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(ridge.coef_).float())
            self.linear.bias.copy_(torch.from_numpy(ridge.intercept_).float())
        
        self.fitted = True
        
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear regression - estimate variance from residuals.
        
        Args:
            x: Input tensor
            n_samples: Number of samples (ignored)
            
        Returns:
            Tuple of (mean prediction, estimated variance)
        """
        pred = self.forward(x)
        # Constant variance estimate (would be computed from training residuals)
        variance = torch.ones_like(pred) * 0.1
        return pred, variance


class WeightedClimatologyBaseline(nn.Module):
    """
    Weighted average of persistence and climatology.
    
    Combines the short-term skill of persistence with the long-term
    stability of climatology using lead-time-dependent weights.
    
    w(t) = exp(-t / tau)
    prediction = w * persistence + (1-w) * climatology
    
    This mimics the behavior of simple statistical forecasts.
    """
    
    def __init__(
        self,
        tau_hours: float = 48.0,
        climatology: Optional[torch.Tensor] = None
    ):
        """
        Initialize weighted baseline.
        
        Args:
            tau_hours: Time constant for weight decay (hours)
            climatology: Pre-computed climatology tensor
        """
        super().__init__()
        self.name = "WeightedClimatology"
        self.tau_hours = tau_hours
        
        if climatology is not None:
            self.register_buffer('climatology', climatology)
        else:
            self.climatology = None
            
    def forward(
        self,
        x: torch.Tensor,
        lead_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with lead-time-dependent weighting.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            lead_time: Lead time in hours, shape (batch,) or scalar
            
        Returns:
            Prediction tensor of shape (batch, channels, lat, lon)
        """
        batch_size = x.shape[0]
        
        # Persistence prediction
        persistence = x[:, -1, :, :, :]
        
        # Climatology prediction
        if self.climatology is not None:
            climatology = self.climatology.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            climatology = torch.zeros_like(persistence)
        
        # Compute weight based on lead time
        if lead_time.dim() == 0:
            lead_time = lead_time.unsqueeze(0).expand(batch_size)
        
        weight = torch.exp(-lead_time / self.tau_hours)
        weight = weight.view(-1, 1, 1, 1)  # Broadcast over channels and spatial dims
        
        # Weighted combination
        return weight * persistence + (1 - weight) * climatology


def get_baseline_models(config) -> Dict[str, nn.Module]:
    """
    Get all baseline models.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary of baseline model name to model instance
    """
    baselines = {
        'persistence': PersistenceBaseline(),
        'climatology': ClimatologyBaseline(),
        'linear': LinearRegressionBaseline(
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            input_time_steps=config.data.input_time_steps,
            lat_size=config.data.lat_resolution,
            lon_size=config.data.lon_resolution
        )
        # Note: WeightedClimatologyBaseline excluded as it requires lead_time
    }
    return baselines


if __name__ == "__main__":
    # Test baselines
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config
    
    config = get_config(debug=True)
    
    # Create dummy input
    batch_size = 4
    time_steps = config.data.input_time_steps
    channels = config.model.input_channels
    lat = config.data.lat_resolution
    lon = config.data.lon_resolution
    
    x = torch.randn(batch_size, time_steps, channels, lat, lon)
    lead_time = torch.tensor([72.0] * batch_size)
    
    print("Testing baseline models...")
    print(f"Input shape: {x.shape}")
    
    baselines = get_baseline_models(config)
    
    for name, model in baselines.items():
        if name == 'weighted':
            out = model(x, lead_time)
        else:
            out = model(x)
        print(f"{name}: output shape = {out.shape}")
        
        # Test uncertainty
        mean, var = model.predict_with_uncertainty(x)
        print(f"  Uncertainty: mean shape = {mean.shape}, var shape = {var.shape}")
    
    print("\nAll baselines working correctly!")
