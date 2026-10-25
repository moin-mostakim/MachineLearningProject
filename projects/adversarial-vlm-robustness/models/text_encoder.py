"""Text encoders for VLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTextEncoder(nn.Module):
    """Simple embedding-based text encoder for class labels."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        output_dim: int = 256,
        max_len: int = 32
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # (batch, embed_dim)
        return self.projection(pooled)


class TransformerTextEncoder(nn.Module):
    """Transformer-based text encoder."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        output_dim: int = 256,
        max_len: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)
        
        if attention_mask is not None:
            attention_mask = attention_mask == 0
        
        encoded = self.transformer(embedded, src_key_padding_mask=attention_mask)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        return self.projection(pooled)


class LabelTextEncoder(nn.Module):
    """Text encoder specifically for CIFAR-10 class labels.
    
    Creates learnable embeddings for each class name.
    """
    
    def __init__(
        self,
        class_names: List[str],
        output_dim: int = 256
    ):
        super().__init__()
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Learnable text embeddings for each class
        self.class_embeddings = nn.Embedding(self.num_classes, output_dim)
        self.output_dim = output_dim
        
        # Initialize with reasonable variance
        nn.init.normal_(self.class_embeddings.weight, std=0.02)
    
    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for class indices."""
        return self.class_embeddings(class_indices)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get embeddings for all classes."""
        indices = torch.arange(self.num_classes, device=self.class_embeddings.weight.device)
        return self.class_embeddings(indices)


class TextEncoder(nn.Module):
    """Unified text encoder interface."""
    
    def __init__(
        self,
        encoder_type: str = "embedding",
        vocab_size: int = 10000,
        embed_dim: int = 512,
        output_dim: int = 256,
        max_len: int = 32,
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "embedding":
            self.encoder = SimpleTextEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                output_dim=output_dim,
                max_len=max_len
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerTextEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                output_dim=output_dim,
                max_len=max_len
            )
        elif encoder_type == "label" and class_names is not None:
            self.encoder = LabelTextEncoder(
                class_names=class_names,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs) if hasattr(self.encoder, '__call__') else self.encoder(x)


def get_text_encoder(config) -> TextEncoder:
    """Factory function to create text encoder from config."""
    return TextEncoder(
        encoder_type="label",
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.text_dim,
        output_dim=config.model.projection_dim,
        max_len=config.model.max_text_len,
        class_names=config.data.cifar10_labels
    )


if __name__ == "__main__":
    # Test label encoder
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    encoder = TextEncoder(encoder_type="label", class_names=class_names, output_dim=256)
    
    indices = torch.tensor([0, 1, 2])
    out = encoder(indices)
    print(f"Input indices: {indices}")
    print(f"Output shape: {out.shape}")
    
    all_emb = encoder.encoder.get_all_embeddings()
    print(f"All embeddings shape: {all_emb.shape}")
