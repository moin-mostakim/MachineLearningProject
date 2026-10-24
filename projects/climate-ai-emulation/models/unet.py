"""
U-Net architecture for climate emulation.

Based on the encoder-decoder architecture with skip connections.
Adapted for weather/climate data with:
- Residual connections for learning anomalies
- Attention gates for focusing on important features
- Instance normalization for spatial data

References:
- Ronneberger et al. (2015) U-Net: original architecture
- Oktay et al. (2018) Attention U-Net
- Bi et al. (2023) Pangu-Weather: 3D U-Net variant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Double convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "gelu",
        normalization: str = "instance",
        dropout: float = 0.0
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU(inplace=True)
        
        # Normalization layer
        def get_norm(channels):
            if normalization == "batch":
                return nn.BatchNorm2d(channels)
            elif normalization == "instance":
                return nn.InstanceNorm2d(channels, affine=True)
            elif normalization == "group":
                return nn.GroupNorm(8, channels)
            elif normalization == "layer":
                return nn.GroupNorm(1, channels)
            else:
                return nn.Identity()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm1 = get_norm(out_channels)
        self.act1 = act_fn
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm2 = get_norm(out_channels)
        self.act2 = act_fn
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net skip connections.
    
    Helps the model focus on relevant features from the encoder
    when combining with decoder features.
    
    Reference: Oktay et al. (2018) Attention U-Net
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention gate.
        
        Args:
            g: Gating signal from decoder (lower resolution)
            x: Skip connection from encoder
            
        Returns:
            Attention-weighted skip features
        """
        # Upsample gate to match skip resolution if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block: convolution followed by downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        normalization: str = "instance",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv_block = ConvBlock(
            in_channels, out_channels,
            activation=activation,
            normalization=normalization,
            dropout=dropout
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (downsampled features, skip connection features)
        """
        features = self.conv_block(x)
        downsampled = self.pool(features)
        return downsampled, features


class DecoderBlock(nn.Module):
    """Decoder block: upsampling followed by convolution with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation: str = "gelu",
        normalization: str = "instance",
        dropout: float = 0.0,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(
                in_channels // 2, skip_channels, skip_channels // 2
            )
        
        self.conv_block = ConvBlock(
            in_channels // 2 + skip_channels, out_channels,
            activation=activation,
            normalization=normalization,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features from previous decoder layer
            skip: Skip connection from encoder
            
        Returns:
            Decoded features
        """
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        if self.use_attention:
            skip = self.attention(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    """
    U-Net for climate emulation.
    
    Architecture:
    - Encoder: Series of conv blocks with max pooling
    - Bottleneck: Deep conv block
    - Decoder: Series of upsampling + conv blocks with skip connections
    - Optional: Attention gates, residual connection, multi-scale output
    
    Key adaptations for weather/climate:
    - Residual learning: output = input + learned_anomaly
    - Instance normalization: better for spatial data
    - Attention gates: focus on dynamic regions
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        activation: str = "gelu",
        normalization: str = "instance",
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (variables * time steps)
            out_channels: Number of output channels
            base_channels: Number of channels in first layer (doubles each level)
            depth: Number of encoder/decoder levels
            activation: Activation function
            normalization: Normalization type
            dropout: Dropout rate
            use_attention: Use attention gates in decoder
            use_residual: Add residual connection (input + output)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.use_residual = use_residual
        
        # Initial convolution to project time steps
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        encoder_channels = [base_channels]
        
        for i in range(depth):
            in_ch = encoder_channels[-1]
            out_ch = base_channels * (2 ** (i + 1))
            self.encoders.append(
                EncoderBlock(
                    in_ch, out_ch,
                    activation=activation,
                    normalization=normalization,
                    dropout=dropout
                )
            )
            encoder_channels.append(out_ch)
        
        # Bottleneck
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = ConvBlock(
            bottleneck_channels, bottleneck_channels * 2,
            activation=activation,
            normalization=normalization,
            dropout=dropout
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        decoder_in = bottleneck_channels * 2
        
        for i in range(depth - 1, -1, -1):
            skip_ch = encoder_channels[i + 1]
            out_ch = base_channels * (2 ** i) if i > 0 else base_channels
            self.decoders.append(
                DecoderBlock(
                    decoder_in, skip_ch, out_ch,
                    activation=activation,
                    normalization=normalization,
                    dropout=dropout,
                    use_attention=use_attention
                )
            )
            decoder_in = out_ch
        
        # Output projection
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # For residual connection: project last input time step
        if use_residual:
            self.residual_proj = nn.Conv2d(
                out_channels, out_channels, kernel_size=1
            ) if in_channels != out_channels else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            
        Returns:
            Output tensor of shape (batch, channels, lat, lon)
        """
        batch_size = x.shape[0]
        
        # Flatten time and channels
        # (B, T, C, H, W) -> (B, T*C, H, W)
        x_flat = x.reshape(batch_size, -1, x.shape[-2], x.shape[-1])
        
        # Store last input for residual connection
        if self.use_residual:
            residual = x[:, -1, :, :, :]  # (B, C, H, W)
        
        # Initial projection
        h = self.input_conv(x_flat)
        
        # Encoder path
        skips = []
        for encoder in self.encoders:
            h, skip = encoder(h)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            h = decoder(h, skip)
        
        # Output projection
        out = self.output_conv(h)
        
        # Residual connection
        if self.use_residual:
            out = out + self.residual_proj(residual)
        
        return out
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes
            
        Returns:
            Tuple of (mean prediction, variance)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (S, B, C, H, W)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        self.eval()  # Disable dropout
        
        return mean, variance
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetWithUncertainty(UNet):
    """
    U-Net with heteroscedastic uncertainty output.
    
    Predicts both mean and variance (aleatoric uncertainty).
    Combined with MC Dropout gives epistemic + aleatoric uncertainty.
    
    Reference: Kendall & Gal (2017) "What Uncertainties Do We Need"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        activation: str = "gelu",
        normalization: str = "instance",
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True,
        min_variance: float = 1e-6
    ):
        # Double output channels for mean and log-variance
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Mean + log-variance
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual
        )
        
        self.actual_out_channels = out_channels
        self.min_variance = min_variance
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and variance.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        out = super().forward(x)
        
        # Split into mean and log-variance
        mean = out[:, :self.actual_out_channels]
        log_var = out[:, self.actual_out_channels:]
        
        # Convert log-variance to variance with minimum bound
        variance = F.softplus(log_var) + self.min_variance
        
        return mean, variance
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combined epistemic + aleatoric uncertainty.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean prediction, total variance)
        """
        self.train()  # Enable dropout
        
        means = []
        variances = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, var = self.forward(x)
                means.append(mean)
                variances.append(var)
        
        means = torch.stack(means, dim=0)
        variances = torch.stack(variances, dim=0)
        
        # Epistemic uncertainty: variance of means
        epistemic = means.var(dim=0)
        
        # Aleatoric uncertainty: mean of variances
        aleatoric = variances.mean(dim=0)
        
        # Total uncertainty
        total_variance = epistemic + aleatoric
        
        # Mean prediction
        mean_prediction = means.mean(dim=0)
        
        self.eval()
        
        return mean_prediction, total_variance


if __name__ == "__main__":
    # Test U-Net
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config
    
    config = get_config(debug=True)
    
    # Create model
    model = UNet(
        in_channels=config.data.input_time_steps * config.model.input_channels,
        out_channels=config.model.output_channels,
        base_channels=config.model.unet_base_channels,
        depth=config.model.unet_depth,
        dropout=config.model.dropout_rate,
        use_attention=config.model.unet_use_attention,
        use_residual=config.model.use_residual
    )
    
    print(f"U-Net parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(
        batch_size,
        config.data.input_time_steps,
        config.model.input_channels,
        config.data.lat_resolution,
        config.data.lon_resolution
    )
    
    print(f"Input shape: {x.shape}")
    
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Test uncertainty
    mean, var = model.predict_with_uncertainty(x, n_samples=5)
    print(f"Uncertainty - Mean: {mean.shape}, Variance: {var.shape}")
    
    # Test heteroscedastic version
    print("\nTesting UNetWithUncertainty...")
    model_uq = UNetWithUncertainty(
        in_channels=config.data.input_time_steps * config.model.input_channels,
        out_channels=config.model.output_channels,
        base_channels=config.model.unet_base_channels,
        depth=config.model.unet_depth,
        dropout=config.model.dropout_rate
    )
    
    print(f"UNetWithUncertainty parameters: {model_uq.count_parameters():,}")
    
    mean, var = model_uq(x)
    print(f"Heteroscedastic - Mean: {mean.shape}, Variance: {var.shape}")
    
    print("\nU-Net tests passed!")
