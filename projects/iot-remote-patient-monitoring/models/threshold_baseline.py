"""
Threshold-based Alert Baseline for Patient Monitoring.

This implements clinical threshold-based alerting as a baseline method.
Based on standard clinical vital sign thresholds and early warning scores.

References:
- Smith, G.B., et al. (2013). A modified Early Warning Score (MEWS).
- Royal College of Physicians (2017). National Early Warning Score (NEWS) 2.
- Jones, D., et al. (2011). Rapid Response Teams. NEJM.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, VitalSignsConfig, AlertConfig


class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class ThresholdAlert:
    """Individual threshold alert."""
    timestamp: pd.Timestamp
    patient_id: int
    vital_sign: str
    value: float
    threshold_type: str  # "low" or "high"
    level: AlertLevel
    message: str


class ThresholdBasedDetector:
    """
    Threshold-based anomaly detection using clinical vital sign thresholds.
    
    Implements a tiered alert system:
    - Normal: All vitals within normal range
    - Warning: One or more vitals in warning zone
    - Critical: One or more vitals in critical zone
    - Emergency: Multiple critical vitals or sustained critical state
    
    Also implements Modified Early Warning Score (MEWS) calculation.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vs_config = self.config.vital_signs
        self.alert_config = self.config.alert
        
        # Define thresholds for each vital sign
        self.thresholds = self._build_thresholds()
        
        # Track alert history for each patient
        self.alert_history: Dict[int, List[ThresholdAlert]] = {}
        self.consecutive_anomalies: Dict[int, int] = {}
        self.last_alert_time: Dict[int, pd.Timestamp] = {}
        
    def _build_thresholds(self) -> Dict:
        """Build threshold dictionary from config."""
        return {
            "heart_rate": {
                "critical_low": self.vs_config.heart_rate_min,
                "warning_low": self.vs_config.heart_rate_warning_low,
                "warning_high": self.vs_config.heart_rate_warning_high,
                "critical_high": self.vs_config.heart_rate_max
            },
            "spo2": {
                "critical_low": self.vs_config.spo2_min,
                "warning_low": self.vs_config.spo2_warning_low,
                "warning_high": 101,  # No upper warning for SpO2
                "critical_high": 101
            },
            "sbp": {
                "critical_low": self.vs_config.sbp_min,
                "warning_low": self.vs_config.sbp_warning_low,
                "warning_high": self.vs_config.sbp_warning_high,
                "critical_high": self.vs_config.sbp_max
            },
            "dbp": {
                "critical_low": self.vs_config.dbp_min,
                "warning_low": self.vs_config.dbp_warning_low,
                "warning_high": self.vs_config.dbp_warning_high,
                "critical_high": self.vs_config.dbp_max
            },
            "temperature": {
                "critical_low": self.vs_config.temperature_min,
                "warning_low": self.vs_config.temperature_warning_low,
                "warning_high": self.vs_config.temperature_warning_high,
                "critical_high": self.vs_config.temperature_max
            },
            "resp_rate": {
                "critical_low": self.vs_config.resp_rate_min,
                "warning_low": self.vs_config.resp_rate_warning_low,
                "warning_high": self.vs_config.resp_rate_warning_high,
                "critical_high": self.vs_config.resp_rate_max
            }
        }
    
    def calculate_mews(self, row: pd.Series) -> int:
        """
        Calculate Modified Early Warning Score (MEWS).
        
        MEWS is a clinical scoring system that triggers escalation of care.
        Score ≥5 indicates significant deterioration.
        
        Based on: Subbe, C.P., et al. (2001). Validation of a modified Early Warning Score.
        """
        score = 0
        
        # Heart Rate scoring
        hr = row["heart_rate"]
        if hr < 40:
            score += 2
        elif hr < 50:
            score += 1
        elif hr <= 100:
            score += 0
        elif hr <= 110:
            score += 1
        elif hr <= 130:
            score += 2
        else:
            score += 3
        
        # Systolic BP scoring
        sbp = row["sbp"]
        if sbp < 70:
            score += 3
        elif sbp < 80:
            score += 2
        elif sbp < 100:
            score += 1
        elif sbp <= 199:
            score += 0
        else:
            score += 2
        
        # Respiratory Rate scoring
        rr = row["resp_rate"]
        if rr < 9:
            score += 2
        elif rr <= 14:
            score += 0
        elif rr <= 20:
            score += 1
        elif rr <= 29:
            score += 2
        else:
            score += 3
        
        # Temperature scoring
        temp = row["temperature"]
        if temp < 35:
            score += 2
        elif temp < 38.5:
            score += 0
        else:
            score += 2
        
        # SpO2 contribution (custom addition to MEWS)
        spo2 = row["spo2"]
        if spo2 < 85:
            score += 3
        elif spo2 < 90:
            score += 2
        elif spo2 < 94:
            score += 1
        
        return score
    
    def check_single_vital(
        self,
        vital_name: str,
        value: float
    ) -> Tuple[AlertLevel, str]:
        """Check a single vital sign against thresholds."""
        thresholds = self.thresholds[vital_name]
        
        if value <= thresholds["critical_low"]:
            return AlertLevel.CRITICAL, f"Critical low {vital_name}: {value:.1f}"
        elif value >= thresholds["critical_high"]:
            return AlertLevel.CRITICAL, f"Critical high {vital_name}: {value:.1f}"
        elif value <= thresholds["warning_low"]:
            return AlertLevel.WARNING, f"Warning low {vital_name}: {value:.1f}"
        elif value >= thresholds["warning_high"]:
            return AlertLevel.WARNING, f"Warning high {vital_name}: {value:.1f}"
        else:
            return AlertLevel.NORMAL, ""
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run threshold-based detection on the dataset.
        
        Returns DataFrame with added columns:
        - threshold_anomaly: Binary anomaly flag
        - alert_level: AlertLevel value
        - mews_score: Modified Early Warning Score
        """
        results = df.copy()
        results["threshold_anomaly"] = 0
        results["alert_level"] = AlertLevel.NORMAL.value
        results["mews_score"] = 0
        results["alert_messages"] = ""
        
        vital_cols = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
        
        for idx, row in results.iterrows():
            alerts = []
            max_level = AlertLevel.NORMAL
            messages = []
            
            # Check each vital sign
            for vital in vital_cols:
                level, msg = self.check_single_vital(vital, row[vital])
                if level != AlertLevel.NORMAL:
                    alerts.append((vital, level))
                    messages.append(msg)
                    if level.value > max_level.value:
                        max_level = level
            
            # Calculate MEWS
            mews = self.calculate_mews(row)
            results.at[idx, "mews_score"] = mews
            
            # Escalate to emergency if multiple critical vitals or high MEWS
            critical_count = sum(1 for _, l in alerts if l == AlertLevel.CRITICAL)
            if critical_count >= 2 or mews >= 5:
                max_level = AlertLevel.EMERGENCY
            
            # Set results
            if max_level != AlertLevel.NORMAL:
                results.at[idx, "threshold_anomaly"] = 1
            results.at[idx, "alert_level"] = max_level.value
            results.at[idx, "alert_messages"] = "; ".join(messages)
        
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
    
    def get_threshold_summary(self) -> str:
        """Get a summary of all thresholds."""
        lines = ["Threshold-Based Detection Thresholds:", "=" * 50]
        for vital, thresh in self.thresholds.items():
            lines.append(f"{vital}:")
            lines.append(f"  Critical: <{thresh['critical_low']} or >{thresh['critical_high']}")
            lines.append(f"  Warning: <{thresh['warning_low']} or >{thresh['warning_high']}")
        return "\n".join(lines)


def run_threshold_baseline(df: pd.DataFrame, debug: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run threshold-based detection.
    
    Returns:
        - DataFrame with detection results
        - Metrics dictionary
    """
    detector = ThresholdBasedDetector()
    
    if debug:
        print(detector.get_threshold_summary())
        print()
    
    results = detector.detect(df)
    
    # Calculate metrics
    metrics = detector.get_metrics(
        results["threshold_anomaly"].values,
        results["is_anomaly"].values
    )
    
    if debug:
        print("\nThreshold-Based Detection Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    
    return results, metrics


if __name__ == "__main__":
    # Test with sample data
    from data.sensor_simulator import generate_dataset
    
    print("Generating test dataset...")
    df, _ = generate_dataset(num_patients=5, duration_hours=2, debug=False)
    
    print("\nRunning threshold-based detection...")
    results, metrics = run_threshold_baseline(df, debug=True)
