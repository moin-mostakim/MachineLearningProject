"""
Neural encoder modules for feature extraction.

Implements transformer-based encoding following the architecture from:
- Vaswani et al. (2017): "Attention Is All You Need"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature extraction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(
            config.hidden_dim,
            config.max_seq_length,
            config.dropout
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.dropout,
                config.layer_norm_eps
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Encode input sequence.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) mask for padding
            
        Returns:
            sequence_output: (batch, seq_len, hidden_dim)
            pooled_output: (batch, hidden_dim)
            all_attention_weights: List of attention weights from each layer
        """
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Process attention mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None
        
        # Pass through transformer layers
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, extended_mask)
            all_attention_weights.append(attn_weights)
        
        sequence_output = self.final_norm(x)
        
        # Pool: take [CLS] token or mean pooling
        pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        return sequence_output, pooled_output, all_attention_weights


class NeuralEncoder(nn.Module):
    """
    Unified neural encoder interface.
    
    Supports multiple encoder types for feature extraction from text/symbols.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if config.encoder_type == "transformer":
            self.encoder = TransformerEncoder(config)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
        
        # Output projection for reasoning
        self.reasoning_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Encode input and produce features for reasoning.
        
        Returns:
            dict with:
                - sequence_output: Full sequence embeddings
                - pooled_output: Sentence-level embedding
                - reasoning_features: Features for symbolic reasoning
                - attention_weights: All attention patterns
        """
        sequence_output, pooled_output, attention_weights = self.encoder(
            input_ids, attention_mask
        )
        
        # Project for reasoning module
        reasoning_features = self.reasoning_projection(pooled_output)
        reasoning_features = F.gelu(reasoning_features)
        
        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "reasoning_features": reasoning_features,
            "attention_weights": attention_weights
        }
    
    def get_param_count(self) -> dict:
        """Get parameter counts for each component."""
        counts = {}
        
        if hasattr(self.encoder, 'token_embedding'):
            counts['token_embedding'] = sum(p.numel() for p in self.encoder.token_embedding.parameters())
        
        if hasattr(self.encoder, 'positional_encoding'):
            counts['positional_encoding'] = sum(p.numel() for p in self.encoder.positional_encoding.parameters())
        
        if hasattr(self.encoder, 'layers'):
            counts['transformer_layers'] = sum(
                sum(p.numel() for p in layer.parameters())
                for layer in self.encoder.layers
            )
        
        counts['reasoning_projection'] = sum(p.numel() for p in self.reasoning_projection.parameters())
        counts['total'] = sum(counts.values())
        
        return counts
