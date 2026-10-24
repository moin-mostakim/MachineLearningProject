"""
Isolation Forest Anomaly Detection for Patient Monitoring.

Statistical anomaly detection using Isolation Forest algorithm.
Effective for detecting outliers in multivariate vital sign data.

References:
- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest.
- Goldstein, M., & Uchida, S. (2016). A Comparative Evaluation of 
  Unsupervised Anomaly Detection Algorithms for Multivariate Data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, IsolationForestConfig


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detection for vital signs.
    
    The Isolation Forest algorithm isolates anomalies by randomly selecting
    features and split values. Anomalies require fewer splits to isolate,
    resulting in shorter average path lengths.
    
    Advantages:
    - No assumptions about data distribution
    - Works well with high-dimensional data
    - Fast training and inference
    - Linear time complexity O(n)
    
    Limitations:
    - Cannot capture temporal dependencies
    - May miss gradual drift anomalies
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.if_config = self.config.isolation_forest
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
        self.is_trained = False
        
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare and scale features for the model."""
        X = df[self.feature_cols].values
        return X
    
    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest model on normal data.
        
        For best results, train on data that is primarily normal
        with the expected contamination rate of anomalies.
        """
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            n_estimators=self.if_config.n_estimators,
            max_samples=min(self.if_config.max_samples, len(X)),
            contamination=self.if_config.contamination,
            max_features=self.if_config.max_features,
            random_state=self.if_config.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        if self.config.debug:
            print(f"Isolation Forest trained with {len(X)} samples")
            print(f"  n_estimators: {self.if_config.n_estimators}")
            print(f"  max_samples: {min(self.if_config.max_samples, len(X))}")
            print(f"  contamination: {self.if_config.contamination}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Binary array where 1 = anomaly, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_scaled)
        # Convert to 1 for anomaly, 0 for normal
        return (predictions == -1).astype(int)
    
    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores.
        
        Lower scores indicate more anomalous points.
        Returns scores normalized to [0, 1] where higher = more anomalous.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # decision_function returns negative scores for anomalies
        scores = self.model.decision_function(X_scaled)
        # Normalize to [0, 1] where 1 = most anomalous
        scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores_normalized
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run detection and return DataFrame with results.
        """
        results = df.copy()
        
        predictions = self.predict(df)
        scores = self.predict_scores(df)
        
        results["if_anomaly"] = predictions
        results["if_score"] = scores
        
        return results
    
    def get_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate detection metrics."""
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
        """
        Calculate the number of parameters in the model.
        
        For Isolation Forest:
        - Each tree has roughly log2(n_samples) splits
        - Each split stores: feature index (int) + threshold (float)
        """
        if not self.is_trained:
            return {"total": 0, "trees": 0, "per_tree": 0}
        
        n_trees = self.if_config.n_estimators
        n_features = len(self.feature_cols)
        max_depth = int(np.ceil(np.log2(self.if_config.max_samples)))
        max_nodes = 2 ** (max_depth + 1) - 1
        
        # Each node stores: feature index, threshold, left/right child indices
        params_per_node = 4  # feature_idx, threshold, left, right
        params_per_tree = max_nodes * params_per_node
        total_params = n_trees * params_per_tree
        
        # Add scaler parameters
        scaler_params = n_features * 2  # mean and std for each feature
        
        return {
            "total": total_params + scaler_params,
            "trees": n_trees,
            "per_tree": params_per_tree,
            "scaler": scaler_params,
            "max_depth": max_depth
        }
    
    def save(self, path: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "config": self.if_config
        }, path)
    
    def load(self, path: str):
        """Load a trained model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True


def run_isolation_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    debug: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to train and evaluate Isolation Forest.
    
    Returns:
        - DataFrame with detection results
        - Metrics dictionary
    """
    detector = IsolationForestDetector()
    
    if debug:
        print("Training Isolation Forest...")
    
    detector.fit(train_df)
    
    if debug:
        params = detector.get_parameter_count()
        print(f"\nParameter Count:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trees: {params['trees']}")
        print(f"  Per tree: {params['per_tree']:,}")
    
    results = detector.detect(test_df)
    
    # Calculate metrics
    metrics = detector.get_metrics(
        results["if_anomaly"].values,
        results["is_anomaly"].values
    )
    
    if debug:
        print("\nIsolation Forest Detection Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    
    return results, metrics


if __name__ == "__main__":
    from data.sensor_simulator import generate_dataset
    
    print("Generating test dataset...")
    df, _ = generate_dataset(num_patients=10, duration_hours=4, debug=False)
    
    # Split into train/test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    
    results, metrics = run_isolation_forest(train_df, test_df, debug=True)
