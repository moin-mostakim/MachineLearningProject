"""
ConvLSTM architecture for climate emulation.

Convolutional LSTM captures spatiotemporal dependencies by replacing
fully connected layers in LSTM with convolutional operations.

References:
- Shi et al. (2015) Convolutional LSTM Network for Precipitation Nowcasting
- Shi et al. (2017) Deep Learning for Precipitation Nowcasting (ConvGRU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.
    
    Implements the following equations:
    i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
    f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
    c_t = f_t * c_{t-1} + i_t * tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)
    o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
    h_t = o_t * tanh(c_t)
    
    where * denotes convolution.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3
    ):
        """
        Initialize ConvLSTM cell.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden state channels
            kernel_size: Kernel size for convolutions
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Combined convolution for input-to-hidden (all 4 gates)
        self.conv_ih = nn.Conv2d(
            input_channels, 4 * hidden_channels, kernel_size,
            padding=self.padding, bias=True
        )
        
        # Combined convolution for hidden-to-hidden (all 4 gates)
        self.conv_hh = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size,
            padding=self.padding, bias=False
        )
        
        # Initialize forget gate bias to 1 for better gradient flow
        nn.init.zeros_(self.conv_ih.bias)
        self.conv_ih.bias.data[hidden_channels:2*hidden_channels].fill_(1.0)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for single time step.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            state: Tuple of (h, c) hidden and cell states
            
        Returns:
            Tuple of (h, (h, c)) - output and new state
        """
        batch, _, height, width = x.shape
        
        # Initialize state if not provided
        if state is None:
            h = torch.zeros(
                batch, self.hidden_channels, height, width,
                device=x.device, dtype=x.dtype
            )
            c = torch.zeros(
                batch, self.hidden_channels, height, width,
                device=x.device, dtype=x.dtype
            )
        else:
            h, c = state
        
        # Combined input and hidden convolutions
        gates = self.conv_ih(x) + self.conv_hh(h)
        
        # Split gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c = f * c + i * g
        
        # Compute hidden state
        h = o * torch.tanh(c)
        
        return h, (h, c)


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for spatiotemporal prediction.
    
    Processes input sequence through stacked ConvLSTM layers
    and returns final hidden state or full sequence.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        num_layers: Optional[int] = None,
        dropout: float = 0.0,
        return_all_layers: bool = False
    ):
        """
        Initialize ConvLSTM.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: List of hidden channels per layer
            kernel_size: Kernel size for convolutions
            num_layers: Number of layers (alternative to len(hidden_channels))
            dropout: Dropout between layers
            return_all_layers: Return outputs from all layers
        """
        super().__init__()
        
        if num_layers is not None:
            if isinstance(hidden_channels, int):
                hidden_channels = [hidden_channels] * num_layers
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.return_all_layers = return_all_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        for i, hidden_ch in enumerate(hidden_channels):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            self.layers.append(
                ConvLSTMCell(in_ch, hidden_ch, kernel_size)
            )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through sequence.
        
        Args:
            x: Input tensor of shape (batch, time, channels, height, width)
            hidden_state: List of (h, c) tuples for each layer
            
        Returns:
            Tuple of (output, final_states)
            - output: (batch, time, hidden[-1], height, width) if return_all_layers=False
                     else list of outputs per layer
            - final_states: List of (h, c) tuples for each layer
        """
        batch, seq_len, _, height, width = x.shape
        
        # Initialize hidden states
        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        # Process sequence
        layer_outputs = []
        final_states = []
        
        current_input = x
        
        for layer_idx, layer in enumerate(self.layers):
            layer_output = []
            state = hidden_state[layer_idx]
            
            for t in range(seq_len):
                h, state = layer(current_input[:, t], state)
                layer_output.append(h)
            
            # Stack time steps
            layer_output = torch.stack(layer_output, dim=1)
            
            # Apply dropout between layers (not on last layer)
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output.reshape(-1, *layer_output.shape[2:]))
                layer_output = layer_output.reshape(batch, seq_len, *layer_output.shape[1:])
            
            layer_outputs.append(layer_output)
            final_states.append(state)
            current_input = layer_output
        
        if self.return_all_layers:
            return layer_outputs, final_states
        else:
            return layer_outputs[-1], final_states


class ConvLSTMPredictor(nn.Module):
    """
    ConvLSTM-based climate emulator.
    
    Architecture:
    - ConvLSTM encoder processes input sequence
    - ConvLSTM decoder autoregressively generates future states
    - Optional: scheduled sampling during training
    
    For climate emulation, typically used with:
    - Multi-step input (past states)
    - Single-step output (predicted state at lead time)
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize ConvLSTM predictor.
        
        Args:
            input_channels: Number of input channels per time step
            output_channels: Number of output channels
            hidden_channels: Hidden channels per layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            use_residual: Use residual connection
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_residual = use_residual
        
        # Encoder ConvLSTM
        self.encoder = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            return_all_layers=False
        )
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels[-1], output_channels, 1)
        )
        
        # Residual projection if needed
        if use_residual and input_channels != output_channels:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, 1)
        else:
            self.residual_proj = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, lat, lon)
            
        Returns:
            Output tensor of shape (batch, channels, lat, lon)
        """
        # Store last input for residual
        if self.use_residual:
            residual = x[:, -1, :self.output_channels]  # (B, C, H, W)
        
        # Encode sequence
        encoded, states = self.encoder(x)  # (B, T, H, H, W)
        
        # Use last hidden state
        last_hidden = encoded[:, -1]  # (B, H, H, W)
        
        # Project to output
        out = self.output_conv(last_hidden)
        
        # Add residual
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
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        self.eval()
        
        return mean, variance
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvGRUCell(nn.Module):
    """
    ConvGRU cell - simplified alternative to ConvLSTM.
    
    Fewer parameters than ConvLSTM but often similar performance.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Reset and update gates
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels, 2 * hidden_channels,
            kernel_size, padding=padding
        )
        
        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size, padding=padding
        )
        
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for single time step."""
        batch, _, height, width = x.shape
        
        if h is None:
            h = torch.zeros(
                batch, self.hidden_channels, height, width,
                device=x.device, dtype=x.dtype
            )
        
        # Compute gates
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r, z = gates.chunk(2, dim=1)
        
        # Compute candidate
        combined_reset = torch.cat([x, r * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        
        # Update hidden state
        h_new = (1 - z) * h + z * candidate
        
        return h_new


if __name__ == "__main__":
    # Test ConvLSTM
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config
    
    config = get_config(debug=True)
    
    # Create model
    model = ConvLSTMPredictor(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        hidden_channels=config.model.convlstm_hidden_channels,
        kernel_size=config.model.convlstm_kernel_size,
        dropout=config.model.dropout_rate,
        use_residual=config.model.use_residual
    )
    
    print(f"ConvLSTM parameters: {model.count_parameters():,}")
    
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
    
    print("\nConvLSTM tests passed!")
