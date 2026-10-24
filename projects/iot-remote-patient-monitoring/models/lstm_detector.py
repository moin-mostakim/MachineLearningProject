"""
LSTM-based Anomaly Detection for Patient Monitoring.

Uses LSTM autoencoder for reconstruction-based anomaly detection.
Anomalies are detected when reconstruction error exceeds threshold.

References:
- Malhotra, P., et al. (2015). Long Short Term Memory Networks for Anomaly Detection.
- Hundman, K., et al. (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.
- Park, D., et al. (2018). A Multimodal Anomaly Detector for Robot-Assisted Feeding.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, LSTMConfig


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.
    
    Architecture:
    - Encoder: LSTM layers that compress input sequence to latent representation
    - Decoder: LSTM layers that reconstruct the input from latent representation
    
    Anomaly detection via reconstruction error:
    - Normal patterns have low reconstruction error
    - Anomalous patterns have high reconstruction error
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Reconstructed tensor of same shape
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Prepare decoder input (use hidden state repeated for each timestep)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Project to input dimension
        reconstruction = self.output_layer(decoder_output)
        
        return reconstruction
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Calculate number of parameters in the model."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        output_params = sum(p.numel() for p in self.output_layer.parameters())
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "output_layer": output_params,
            "total": encoder_params + decoder_params + output_params
        }


class LSTMDetector:
    """
    LSTM-based anomaly detector for vital sign monitoring.
    
    Uses sliding window to create sequences and trains autoencoder
    to reconstruct normal patterns. Anomalies detected via high
    reconstruction error.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.lstm_config = self.config.lstm
        
        self.feature_cols = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
        self.input_size = len(self.feature_cols)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model = LSTMAutoencoder(
            input_size=self.input_size,
            hidden_size=self.lstm_config.hidden_size,
            num_layers=self.lstm_config.num_layers,
            dropout=self.lstm_config.dropout
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
        seq_len = self.lstm_config.sequence_length
        sequences = []
        seq_labels = []
        
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
            if labels is not None:
                # Label sequence as anomaly if any point is anomalous
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
    ) -> "LSTMDetector":
        """
        Train the LSTM autoencoder.
        
        Trains on primarily normal data. The model learns to reconstruct
        normal patterns, and anomalies will have high reconstruction error.
        """
        if self.config.debug:
            print("Preparing training data...")
        
        train_sequences, _ = self._prepare_data(train_df, fit_scaler=True)
        
        # Create DataLoader
        train_tensor = torch.FloatTensor(train_sequences).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.lstm_config.batch_size,
            shuffle=True
        )
        
        # Validation data
        if val_df is not None:
            val_sequences, _ = self._prepare_data(val_df)
            val_tensor = torch.FloatTensor(val_sequences).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lstm_config.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        if self.config.debug:
            print(f"Training LSTM on {self.device}...")
            print(f"  Sequences: {len(train_sequences)}")
            print(f"  Sequence length: {self.lstm_config.sequence_length}")
            print(f"  Hidden size: {self.lstm_config.hidden_size}")
        
        for epoch in range(self.lstm_config.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                reconstruction = self.model(batch_x)
                loss = criterion(reconstruction, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_df is not None:
                self.model.eval()
                with torch.no_grad():
                    val_reconstruction = self.model(val_tensor)
                    val_loss = criterion(val_reconstruction, val_tensor).item()
                
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.lstm_config.early_stopping_patience:
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
        
        # Calculate threshold on training data
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
        
        self.threshold = np.percentile(errors, self.lstm_config.threshold_percentile)
        
        if self.config.debug:
            print(f"\nThreshold (percentile {self.lstm_config.threshold_percentile}): {self.threshold:.6f}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Returns:
            - Binary predictions (1 = anomaly)
            - Reconstruction errors
        """
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
        pad_length = self.lstm_config.sequence_length - 1
        predictions = np.concatenate([np.zeros(pad_length), predictions])
        errors = np.concatenate([np.zeros(pad_length), errors])
        
        return predictions, errors
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run detection and return DataFrame with results."""
        results = df.copy()
        predictions, errors = self.predict(df)
        
        results["lstm_anomaly"] = predictions[:len(df)]
        results["lstm_error"] = errors[:len(df)]
        
        return results
    
    def get_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate detection metrics."""
        # Align lengths
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
            "config": self.lstm_config
        }, path)
    
    def load(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]
        self.threshold = checkpoint["threshold"]
        self.is_trained = True


def run_lstm_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    debug: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to train and evaluate LSTM detector.
    """
    detector = LSTMDetector()
    
    if debug:
        print("Training LSTM Autoencoder...")
        params = detector.get_parameter_count()
        print(f"\nParameter Count:")
        print(f"  Encoder: {params['encoder']:,}")
        print(f"  Decoder: {params['decoder']:,}")
        print(f"  Output layer: {params['output_layer']:,}")
        print(f"  Total: {params['total']:,}")
    
    detector.fit(train_df, val_df)
    results = detector.detect(test_df)
    
    metrics = detector.get_metrics(
        results["lstm_anomaly"].values,
        results["is_anomaly"].values
    )
    
    if debug:
        print("\nLSTM Detection Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    
    return results, metrics


if __name__ == "__main__":
    from data.sensor_simulator import generate_dataset
    
    print("Generating test dataset...")
    df, _ = generate_dataset(num_patients=5, duration_hours=2, debug=False)
    
    # Split into train/val/test
    n = len(df)
    train_df = df.iloc[:int(n*0.6)]
    val_df = df.iloc[int(n*0.6):int(n*0.8)]
    test_df = df.iloc[int(n*0.8):]
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    results, metrics = run_lstm_detector(train_df, test_df, val_df, debug=True)
