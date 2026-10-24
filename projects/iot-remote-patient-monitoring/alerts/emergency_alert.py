"""
Emergency Alert Mechanism for Patient Monitoring System.

Implements multi-level alert system with confirmation logic,
cooldown periods, and escalation protocols.

References:
- Cvach, M. (2012). Monitor Alarm Fatigue: An Integrative Review.
- Sendelbach, S., & Funk, M. (2013). Alarm Fatigue: A Patient Safety Concern.
- Drew, B.J., et al. (2014). Insights into the Problem of Alarm Fatigue.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, AlertConfig


class AlertPriority(Enum):
    """Alert priority levels matching clinical triage systems."""
    LOW = 1       # Advisory - routine follow-up
    MEDIUM = 2    # Warning - requires attention within 30 mins
    HIGH = 3      # Urgent - requires immediate attention
    CRITICAL = 4  # Emergency - life-threatening, immediate intervention


class AlertStatus(Enum):
    """Alert lifecycle status."""
    PENDING = "pending"         # Alert triggered, awaiting confirmation
    CONFIRMED = "confirmed"     # Alert confirmed, awaiting acknowledgment
    ACKNOWLEDGED = "acknowledged"  # Alert acknowledged by staff
    RESOLVED = "resolved"       # Condition resolved
    SUPPRESSED = "suppressed"   # Alert suppressed (cooldown or manual)


@dataclass
class Alert:
    """Individual alert record."""
    alert_id: str
    patient_id: int
    timestamp: datetime
    priority: AlertPriority
    status: AlertStatus
    vital_signs: Dict[str, float]
    anomaly_scores: Dict[str, float]
    message: str
    detection_method: str  # "threshold", "isolation_forest", "lstm", "transformer", "ensemble"
    detection_latency_ms: float = 0.0
    
    # Tracking
    confirmed_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "status": self.status.name,
            "vital_signs": self.vital_signs,
            "anomaly_scores": self.anomaly_scores,
            "message": self.message,
            "detection_method": self.detection_method,
            "detection_latency_ms": self.detection_latency_ms
        }


class AlertBuffer:
    """
    Sliding window buffer for alert confirmation logic.
    
    Prevents false alarms by requiring multiple consecutive
    anomaly detections before triggering an alert.
    """
    
    def __init__(self, window_size: int, threshold: int):
        self.window_size = window_size
        self.threshold = threshold  # Minimum anomalies to trigger alert
        self.buffer: deque = deque(maxlen=window_size)
        
    def add(self, is_anomaly: bool) -> bool:
        """
        Add observation and check if alert should trigger.
        
        Returns True if consecutive anomaly count >= threshold.
        """
        self.buffer.append(is_anomaly)
        
        if len(self.buffer) < self.threshold:
            return False
        
        # Check for consecutive anomalies
        recent = list(self.buffer)[-self.threshold:]
        return all(recent)
    
    def reset(self):
        """Reset the buffer."""
        self.buffer.clear()


class EmergencyAlertSystem:
    """
    Multi-level emergency alert system for patient monitoring.
    
    Features:
    - Multi-method anomaly fusion (threshold, ML models)
    - Alert confirmation logic to reduce false alarms
    - Cooldown periods to prevent alert fatigue
    - Priority escalation based on severity and duration
    - Alert lifecycle tracking
    - Metrics computation (latency, false alarm rate)
    
    Based on clinical decision support best practices and
    alarm fatigue mitigation strategies (Cvach, 2012).
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.alert_config = self.config.alert
        
        # Per-patient state
        self.alert_buffers: Dict[int, AlertBuffer] = {}
        self.last_alert_time: Dict[int, datetime] = {}
        self.active_alerts: Dict[int, Alert] = {}
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.alert_counter = 0
        
        # Callbacks for alert handling
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
    def _get_buffer(self, patient_id: int) -> AlertBuffer:
        """Get or create alert buffer for patient."""
        if patient_id not in self.alert_buffers:
            self.alert_buffers[patient_id] = AlertBuffer(
                window_size=10,
                threshold=self.alert_config.min_consecutive_anomalies
            )
        return self.alert_buffers[patient_id]
    
    def _is_in_cooldown(self, patient_id: int, current_time: datetime) -> bool:
        """Check if patient is in alert cooldown period."""
        if patient_id not in self.last_alert_time:
            return False
        
        time_since_last = (current_time - self.last_alert_time[patient_id]).total_seconds()
        return time_since_last < self.alert_config.alert_cooldown
    
    def _determine_priority(
        self,
        vital_signs: Dict[str, float],
        anomaly_scores: Dict[str, float],
        consecutive_count: int
    ) -> AlertPriority:
        """
        Determine alert priority based on vital signs and anomaly severity.
        
        Priority escalation logic:
        - CRITICAL: Any vital in critical range OR high anomaly score with persistence
        - HIGH: Multiple vitals in warning range OR moderate anomaly score
        - MEDIUM: Single vital in warning range
        - LOW: Trend change or minor deviation
        """
        vs = self.config.vital_signs
        
        # Check for critical vital sign values
        critical_conditions = [
            vital_signs.get("heart_rate", 75) < vs.heart_rate_min,
            vital_signs.get("heart_rate", 75) > vs.heart_rate_max,
            vital_signs.get("spo2", 97) < vs.spo2_min,
            vital_signs.get("sbp", 120) < vs.sbp_min,
            vital_signs.get("sbp", 120) > vs.sbp_max,
            vital_signs.get("temperature", 37) < vs.temperature_min,
            vital_signs.get("temperature", 37) > vs.temperature_max,
        ]
        
        if any(critical_conditions):
            return AlertPriority.CRITICAL
        
        # Check anomaly scores
        max_score = max(anomaly_scores.values()) if anomaly_scores else 0
        
        if max_score > 0.9 and consecutive_count >= 5:
            return AlertPriority.CRITICAL
        elif max_score > 0.8 or consecutive_count >= 4:
            return AlertPriority.HIGH
        elif max_score > 0.6 or consecutive_count >= 3:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW
    
    def _generate_message(
        self,
        vital_signs: Dict[str, float],
        priority: AlertPriority,
        detection_method: str
    ) -> str:
        """Generate human-readable alert message."""
        vs = self.config.vital_signs
        
        issues = []
        
        # Check each vital sign
        hr = vital_signs.get("heart_rate", 75)
        if hr < vs.heart_rate_warning_low:
            issues.append(f"Low HR: {hr:.0f} bpm")
        elif hr > vs.heart_rate_warning_high:
            issues.append(f"High HR: {hr:.0f} bpm")
        
        spo2 = vital_signs.get("spo2", 97)
        if spo2 < vs.spo2_warning_low:
            issues.append(f"Low SpO2: {spo2:.0f}%")
        
        sbp = vital_signs.get("sbp", 120)
        if sbp < vs.sbp_warning_low:
            issues.append(f"Low BP: {sbp:.0f} mmHg")
        elif sbp > vs.sbp_warning_high:
            issues.append(f"High BP: {sbp:.0f} mmHg")
        
        temp = vital_signs.get("temperature", 37)
        if temp < vs.temperature_warning_low:
            issues.append(f"Low Temp: {temp:.1f}°C")
        elif temp > vs.temperature_warning_high:
            issues.append(f"High Temp: {temp:.1f}°C")
        
        if not issues:
            issues.append(f"Anomaly detected by {detection_method}")
        
        prefix = {
            AlertPriority.CRITICAL: "EMERGENCY",
            AlertPriority.HIGH: "URGENT",
            AlertPriority.MEDIUM: "WARNING",
            AlertPriority.LOW: "ADVISORY"
        }[priority]
        
        return f"[{prefix}] {', '.join(issues)}"
    
    def process_observation(
        self,
        patient_id: int,
        timestamp: datetime,
        vital_signs: Dict[str, float],
        anomaly_predictions: Dict[str, bool],
        anomaly_scores: Dict[str, float],
        detection_start_time: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Process a single observation and potentially trigger an alert.
        
        Args:
            patient_id: Patient identifier
            timestamp: Observation timestamp
            vital_signs: Current vital sign values
            anomaly_predictions: Binary predictions from each detector
            anomaly_scores: Anomaly scores from each detector
            detection_start_time: Time when detection started (for latency)
            
        Returns:
            Alert if triggered, None otherwise
        """
        # Calculate detection latency
        if detection_start_time:
            detection_latency_ms = (time.time() - detection_start_time) * 1000
        else:
            detection_latency_ms = 0.0
        
        # Ensemble: alert if any method detects anomaly
        is_anomaly = any(anomaly_predictions.values())
        
        # Use buffer for confirmation
        buffer = self._get_buffer(patient_id)
        should_alert = buffer.add(is_anomaly)
        
        if not should_alert:
            return None
        
        # Check cooldown
        if self._is_in_cooldown(patient_id, timestamp):
            return None
        
        # Determine which method(s) triggered
        triggering_methods = [k for k, v in anomaly_predictions.items() if v]
        if len(triggering_methods) > 1:
            detection_method = "ensemble"
        else:
            detection_method = triggering_methods[0] if triggering_methods else "unknown"
        
        # Determine priority
        consecutive_count = sum(buffer.buffer)
        priority = self._determine_priority(vital_signs, anomaly_scores, consecutive_count)
        
        # Generate alert
        self.alert_counter += 1
        alert = Alert(
            alert_id=f"ALERT-{self.alert_counter:06d}",
            patient_id=patient_id,
            timestamp=timestamp,
            priority=priority,
            status=AlertStatus.CONFIRMED,
            vital_signs=vital_signs,
            anomaly_scores=anomaly_scores,
            message=self._generate_message(vital_signs, priority, detection_method),
            detection_method=detection_method,
            detection_latency_ms=detection_latency_ms,
            confirmed_at=timestamp
        )
        
        # Update state
        self.last_alert_time[patient_id] = timestamp
        self.active_alerts[patient_id] = alert
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            callback(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_at: datetime) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = acknowledged_at
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_at: datetime) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = resolved_at
                
                # Remove from active alerts
                if alert.patient_id in self.active_alerts:
                    if self.active_alerts[alert.patient_id].alert_id == alert_id:
                        del self.active_alerts[alert.patient_id]
                return True
        return False
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Register a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self, ground_truth: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate alert system metrics.
        
        Metrics:
        - Alert latency: Time from anomaly onset to alert
        - False alarm rate: Alerts without actual anomaly
        - Miss rate: Anomalies without alerts
        - Alert count by priority
        """
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "avg_latency_ms": 0,
                "false_alarm_rate": 0,
                "alerts_by_priority": {}
            }
        
        # Count alerts by priority
        priority_counts = {}
        for p in AlertPriority:
            priority_counts[p.name] = sum(
                1 for a in self.alert_history if a.priority == p
            )
        
        # Calculate average latency
        latencies = [a.detection_latency_ms for a in self.alert_history]
        avg_latency = np.mean(latencies) if latencies else 0
        
        # Calculate false alarm rate if ground truth provided
        if ground_truth is not None and "is_anomaly" in ground_truth.columns:
            alert_timestamps = set(
                (a.patient_id, a.timestamp) for a in self.alert_history
            )
            
            true_positives = 0
            false_positives = 0
            
            for alert in self.alert_history:
                # Find corresponding ground truth
                mask = (
                    (ground_truth["patient_id"] == alert.patient_id) &
                    (pd.to_datetime(ground_truth["timestamp"]) == alert.timestamp)
                )
                if mask.any():
                    if ground_truth.loc[mask, "is_anomaly"].values[0]:
                        true_positives += 1
                    else:
                        false_positives += 1
            
            total_alerts = true_positives + false_positives
            false_alarm_rate = false_positives / total_alerts if total_alerts > 0 else 0
        else:
            false_alarm_rate = 0
        
        return {
            "total_alerts": len(self.alert_history),
            "avg_latency_ms": avg_latency,
            "false_alarm_rate": false_alarm_rate,
            "alerts_by_priority": priority_counts
        }
    
    def get_alert_summary(self) -> str:
        """Generate summary of alert activity."""
        metrics = self.get_metrics(None)
        
        lines = [
            "=" * 50,
            "EMERGENCY ALERT SYSTEM SUMMARY",
            "=" * 50,
            f"Total Alerts: {metrics['total_alerts']}",
            f"Average Detection Latency: {metrics['avg_latency_ms']:.2f} ms",
            "\nAlerts by Priority:"
        ]
        
        for priority, count in metrics["alerts_by_priority"].items():
            lines.append(f"  {priority}: {count}")
        
        return "\n".join(lines)


def run_alert_system(
    results_df: pd.DataFrame,
    debug: bool = True
) -> Tuple[List[Alert], Dict]:
    """
    Run emergency alert system on detection results.
    
    Args:
        results_df: DataFrame with detection results from all methods
        debug: Print debug information
        
    Returns:
        - List of triggered alerts
        - Alert metrics
    """
    alert_system = EmergencyAlertSystem()
    
    if debug:
        print("Running Emergency Alert System...")
    
    alert_count = 0
    
    for idx, row in results_df.iterrows():
        vital_signs = {
            "heart_rate": row["heart_rate"],
            "spo2": row["spo2"],
            "sbp": row["sbp"],
            "dbp": row["dbp"],
            "temperature": row["temperature"],
            "resp_rate": row["resp_rate"]
        }
        
        # Gather predictions from different methods
        anomaly_predictions = {}
        anomaly_scores = {}
        
        if "threshold_anomaly" in row:
            anomaly_predictions["threshold"] = bool(row["threshold_anomaly"])
            anomaly_scores["threshold"] = float(row.get("mews_score", 0)) / 10
        
        if "if_anomaly" in row:
            anomaly_predictions["isolation_forest"] = bool(row["if_anomaly"])
            anomaly_scores["isolation_forest"] = float(row.get("if_score", 0))
        
        if "lstm_anomaly" in row:
            anomaly_predictions["lstm"] = bool(row["lstm_anomaly"])
            anomaly_scores["lstm"] = float(row.get("lstm_error", 0))
        
        if "transformer_anomaly" in row:
            anomaly_predictions["transformer"] = bool(row["transformer_anomaly"])
            anomaly_scores["transformer"] = float(row.get("transformer_error", 0))
        
        # Process observation
        start_time = time.time()
        alert = alert_system.process_observation(
            patient_id=int(row["patient_id"]),
            timestamp=pd.to_datetime(row["timestamp"]),
            vital_signs=vital_signs,
            anomaly_predictions=anomaly_predictions,
            anomaly_scores=anomaly_scores,
            detection_start_time=start_time
        )
        
        if alert:
            alert_count += 1
            if debug and alert_count <= 5:
                print(f"  Alert: {alert.message} (Priority: {alert.priority.name})")
    
    metrics = alert_system.get_metrics(results_df)
    
    if debug:
        print(f"\n{alert_system.get_alert_summary()}")
    
    return alert_system.alert_history, metrics


if __name__ == "__main__":
    # Test with sample data
    from data.sensor_simulator import generate_dataset
    from models.threshold_baseline import run_threshold_baseline
    
    print("Generating test dataset...")
    df, _ = generate_dataset(num_patients=3, duration_hours=1, debug=False)
    
    print("\nRunning threshold detection...")
    results, _ = run_threshold_baseline(df, debug=False)
    
    print("\nRunning alert system...")
    alerts, metrics = run_alert_system(results, debug=True)
