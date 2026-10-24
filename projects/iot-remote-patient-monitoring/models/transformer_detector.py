"""
Transformer-based Anomaly Detection for Patient Monitoring.

Uses Transformer architecture for time-series anomaly detection.
Captures long-range temporal dependencies via self-attention.

References:
- Vaswani, A., et al. (2017). Attention Is All You Need.
- Tuli, S., et al. (2022). TranAD: Deep Transformer Networks for Anomaly Detection.
- Wu, N., et al. (2020). Deep Transformer Models for Time Series Forecasting.
- Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, TransformerConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time-series data.
    
    Provides temporal position information since Transformers
    don't have inherent notion of sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder for time-series anomaly detection.
    
    Architecture:
    - Input projection: Maps input features to model dimension
    - Positional encoding: Adds temporal position information
    - Transformer encoder: Self-attention for temporal dependencies
    - Decoder: Reconstructs input sequence
    
    Key advantages over LSTM:
    - Captures long-range dependencies efficiently via self-attention
    - Parallelizable training (no sequential dependence)
    - Attention weights provide interpretability
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer autoencoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Reconstructed tensor of same shape
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Project back to input dimension
        reconstruction = self.output_projection(encoded)
        
        return reconstruction
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        # This would require modifying the forward pass to return attention
        # For now, we use hooks during analysis
        pass
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Calculate number of parameters in the model."""
        input_proj_params = sum(p.numel() for p in self.input_projection.parameters())
        pos_enc_params = 0  # Positional encoding uses buffer, not parameters
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        output_proj_params = sum(p.numel() for p in self.output_projection.parameters())
        
        return {
            "input_projection": input_proj_params,
            "positional_encoding": pos_enc_params,
            "transformer_encoder": transformer_params,
            "output_projection": output_proj_params,
            "total": input_proj_params + transformer_params + output_proj_params
        }


class TransformerDetector:
    """
    Transformer-based anomaly detector for vital sign monitoring.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.tf_config = self.config.transformer
        
        self.feature_cols = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
        self.input_size = len(self.feature_cols)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model = TransformerAutoencoder(
            input_size=self.input_size,
            d_model=self.tf_config.d_model,
            n_heads=self.tf_config.n_heads,
            n_layers=self.tf_config.n_layers,
            d_ff=self.tf_config.d_ff,
            dropout=self.tf_config.dropout,
            max_len=self.tf_config.sequence_length * 2
        ).to(self.device)
        
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_trained = False
        self.training_history = []
        
    def _create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding window sequences."""
        seq_len = self.tf_config.sequence_length
        sequences = []
        seq_labels = []
        
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
            if labels is not None:
                seq_labels.append(int(np.any(labels[i:i + seq_len])))
        
        sequences = np.array(sequences)
        seq_labels = np.array(seq_labels) if labels is not None else None
        
        return sequences, seq_labels
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/inference."""
        X = df[self.feature_cols].values
        labels = df["is_anomaly"].values if "is_anomaly" in df.columns else None
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        sequences, seq_labels = self._create_sequences(X_scaled, labels)
        
        return sequences, seq_labels
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "TransformerDetector":
        """Train the Transformer autoencoder."""
        if self.config.debug:
            print("Preparing training data...")
        
        train_sequences, _ = self._prepare_data(train_df, fit_scaler=True)
        
        # Create DataLoader
        train_tensor = torch.FloatTensor(train_sequences).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.tf_config.batch_size,
            shuffle=True
        )
        
        # Validation data
        if val_df is not None:
            val_sequences, _ = self._prepare_data(val_df)
            val_tensor = torch.FloatTensor(val_sequences).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.tf_config.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Learning rate scheduler (warmup + cosine decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.tf_config.epochs,
            eta_min=1e-6
        )
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        if self.config.debug:
            print(f"Training Transformer on {self.device}...")
            print(f"  Sequences: {len(train_sequences)}")
            print(f"  d_model: {self.tf_config.d_model}")
            print(f"  n_heads: {self.tf_config.n_heads}")
            print(f"  n_layers: {self.tf_config.n_layers}")
        
        for epoch in range(self.tf_config.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                reconstruction = self.model(batch_x)
                loss = criterion(reconstruction, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            scheduler.step()
            
            # Validation
            if val_df is not None:
                self.model.eval()
                with torch.no_grad():
                    val_reconstruction = self.model(val_tensor)
                    val_loss = criterion(val_reconstruction, val_tensor).item()
                
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0]
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.tf_config.early_stopping_patience:
                    if self.config.debug:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_state)
                    break
                
                if self.config.debug and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss
                })
                if self.config.debug and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}")
        
        # Calculate threshold
        self._calculate_threshold(train_tensor)
        self.is_trained = True
        
        return self
    
    def _calculate_threshold(self, train_tensor: torch.Tensor):
        """Calculate anomaly threshold from training reconstruction errors."""
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(train_tensor)
            errors = torch.mean((train_tensor - reconstruction) ** 2, dim=(1, 2))
            errors = errors.cpu().numpy()
        
        self.threshold = np.percentile(errors, self.tf_config.threshold_percentile)
        
        if self.config.debug:
            print(f"\nThreshold (percentile {self.tf_config.threshold_percentile}): {self.threshold:.6f}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        sequences, _ = self._prepare_data(df)
        seq_tensor = torch.FloatTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(seq_tensor)
            errors = torch.mean((seq_tensor - reconstruction) ** 2, dim=(1, 2))
            errors = errors.cpu().numpy()
        
        predictions = (errors > self.threshold).astype(int)
        
        # Pad predictions to match original length
        pad_length = self.tf_config.sequence_length - 1
        predictions = np.concatenate([np.zeros(pad_length), predictions])
        errors = np.concatenate([np.zeros(pad_length), errors])
        
        return predictions, errors
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run detection and return DataFrame with results."""
        results = df.copy()
        predictions, errors = self.predict(df)
        
        results["transformer_anomaly"] = predictions[:len(df)]
        results["transformer_error"] = errors[:len(df)]
        
        return results
    
    def get_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate detection metrics."""
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_alarm_rate": false_alarm_rate,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count table."""
        return self.model.get_parameter_count()
    
    def save(self, path: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "threshold": self.threshold,
            "config": self.tf_config
        }, path)
    
    def load(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]
        self.threshold = checkpoint["threshold"]
        self.is_trained = True


def run_transformer_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    debug: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to train and evaluate Transformer detector."""
    detector = TransformerDetector()
    
    if debug:
        print("Training Transformer Autoencoder...")
        params = detector.get_parameter_count()
        print(f"\nParameter Count:")
        print(f"  Input projection: {params['input_projection']:,}")
        print(f"  Transformer encoder: {params['transformer_encoder']:,}")
        print(f"  Output projection: {params['output_projection']:,}")
        print(f"  Total: {params['total']:,}")
    
    detector.fit(train_df, val_df)
    results = detector.detect(test_df)
    
    metrics = detector.get_metrics(
        results["transformer_anomaly"].values,
        results["is_anomaly"].values
    )
    
    if debug:
        print("\nTransformer Detection Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    
    return results, metrics


if __name__ == "__main__":
    from data.sensor_simulator import generate_dataset
    
    print("Generating test dataset...")
    df, _ = generate_dataset(num_patients=5, duration_hours=2, debug=False)
    
    n = len(df)
    train_df = df.iloc[:int(n*0.6)]
    val_df = df.iloc[int(n*0.6):int(n*0.8)]
    test_df = df.iloc[int(n*0.8):]
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    results, metrics = run_transformer_detector(train_df, test_df, val_df, debug=True)
