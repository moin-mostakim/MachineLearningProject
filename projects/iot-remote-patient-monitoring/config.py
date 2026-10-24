"""
Configuration file for Smart Remote Patient Monitoring System.
All hyperparameters with justifications based on literature and clinical guidelines.

References:
- Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database.
- Clifford, G.D., et al. (2012). PhysioNet: Open access databases for the biomedical research community.
- Goldberger, A.L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
- Rajkomar, A., et al. (2018). Scalable and accurate deep learning with electronic health records.
- Harutyunyan, H., et al. (2019). Multitask learning and benchmarking with clinical time series data.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VitalSignsConfig:
    """
    Normal ranges for vital signs based on clinical guidelines.
    
    References:
    - American Heart Association guidelines
    - WHO vital signs reference ranges
    - MIMIC-III database statistics (Johnson et al., 2016)
    """
    # Heart Rate (bpm) - Normal adult range: 60-100 bpm
    # Justification: Based on AHA guidelines, bradycardia <60, tachycardia >100
    heart_rate_mean: float = 75.0
    heart_rate_std: float = 12.0
    heart_rate_min: float = 40.0  # Critical bradycardia threshold
    heart_rate_max: float = 150.0  # Critical tachycardia threshold
    heart_rate_warning_low: float = 55.0
    heart_rate_warning_high: float = 110.0
    
    # SpO2 (%) - Normal range: 95-100%
    # Justification: WHO guidelines, hypoxemia <90% requires intervention
    spo2_mean: float = 97.0
    spo2_std: float = 1.5
    spo2_min: float = 85.0  # Critical hypoxemia
    spo2_max: float = 100.0
    spo2_warning_low: float = 92.0
    
    # Systolic Blood Pressure (mmHg) - Normal: 90-120 mmHg
    # Justification: AHA blood pressure categories
    sbp_mean: float = 120.0
    sbp_std: float = 15.0
    sbp_min: float = 70.0  # Hypotensive shock threshold
    sbp_max: float = 200.0  # Hypertensive crisis
    sbp_warning_low: float = 90.0
    sbp_warning_high: float = 140.0
    
    # Diastolic Blood Pressure (mmHg) - Normal: 60-80 mmHg
    dbp_mean: float = 75.0
    dbp_std: float = 10.0
    dbp_min: float = 40.0
    dbp_max: float = 120.0
    dbp_warning_low: float = 60.0
    dbp_warning_high: float = 90.0
    
    # Body Temperature (°C) - Normal: 36.1-37.2°C
    # Justification: WHO guidelines, fever >38°C, hypothermia <35°C
    temperature_mean: float = 36.8
    temperature_std: float = 0.4
    temperature_min: float = 34.0  # Severe hypothermia
    temperature_max: float = 41.0  # Hyperpyrexia
    temperature_warning_low: float = 35.5
    temperature_warning_high: float = 38.0
    
    # Respiratory Rate (breaths/min) - Normal adult: 12-20
    # Justification: Clinical guidelines, tachypnea >24, bradypnea <10
    resp_rate_mean: float = 16.0
    resp_rate_std: float = 3.0
    resp_rate_min: float = 8.0
    resp_rate_max: float = 35.0
    resp_rate_warning_low: float = 10.0
    resp_rate_warning_high: float = 24.0


@dataclass
class SimulationConfig:
    """
    IoT sensor simulation parameters.
    
    Justification:
    - Sampling rates based on typical wearable device capabilities
    - Noise levels based on sensor accuracy specifications
    - Anomaly rates based on MIMIC-III ICU event frequencies
    """
    # Number of simulated patients
    num_patients: int = 100
    
    # Sampling interval in seconds
    # Justification: 5s provides good resolution for vital sign monitoring
    # while being achievable by consumer IoT devices (Rajkomar et al., 2018)
    sampling_interval: int = 5
    
    # Simulation duration in hours
    simulation_hours: int = 24
    
    # Sensor noise level (percentage of signal)
    # Justification: Typical wearable sensor accuracy ±2-5%
    sensor_noise_pct: float = 0.02
    
    # Anomaly injection rate (probability per time step)
    # Justification: Based on ICU adverse event rates ~5-10% (Harutyunyan et al., 2019)
    anomaly_rate: float = 0.05
    
    # Duration of anomaly episodes (timesteps)
    anomaly_duration_min: int = 6  # 30 seconds minimum
    anomaly_duration_max: int = 60  # 5 minutes maximum
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class IsolationForestConfig:
    """
    Isolation Forest hyperparameters for anomaly detection.
    
    References:
    - Liu, F.T., et al. (2008). Isolation Forest.
    - Goldstein, M., et al. (2016). A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms.
    """
    # Number of trees in the forest
    # Justification: 100-200 trees typically sufficient (Liu et al., 2008)
    n_estimators: int = 100
    
    # Number of samples to draw for training each tree
    # Justification: 256 is often optimal for anomaly detection
    max_samples: int = 256
    
    # Contamination rate (expected proportion of anomalies)
    # Justification: Aligned with anomaly_rate in simulation
    contamination: float = 0.05
    
    # Maximum features to consider
    # Justification: Using all features for multivariate anomaly detection
    max_features: float = 1.0
    
    # Random seed
    random_state: int = 42


@dataclass
class LSTMConfig:
    """
    LSTM model hyperparameters for time-series anomaly detection.
    
    References:
    - Malhotra, P., et al. (2015). Long Short Term Memory Networks for Anomaly Detection.
    - Hundman, K., et al. (2018). Detecting Spacecraft Anomalies Using LSTMs.
    """
    # Input sequence length (timesteps)
    # Justification: 60 timesteps = 5 minutes of data at 5s intervals
    # Captures short-term temporal dependencies in vital signs
    sequence_length: int = 60
    
    # Hidden layer size
    # Justification: 64-128 units sufficient for vital sign patterns
    hidden_size: int = 64
    
    # Number of LSTM layers
    # Justification: 2 layers capture hierarchical temporal patterns
    num_layers: int = 2
    
    # Dropout for regularization
    # Justification: 0.2-0.3 typical for LSTM models (Hundman et al., 2018)
    dropout: float = 0.2
    
    # Learning rate
    # Justification: Adam optimizer works well with 1e-3 to 1e-4
    learning_rate: float = 0.001
    
    # Batch size
    # Justification: 32-64 balances gradient stability and memory
    batch_size: int = 32
    
    # Training epochs
    # Justification: Early stopping typically triggers 20-50 epochs
    # Reduced for debug mode efficiency
    epochs: int = 15
    
    # Early stopping patience
    early_stopping_patience: int = 5
    
    # Reconstruction error threshold percentile
    # Justification: 95th percentile identifies top 5% anomalies
    threshold_percentile: float = 95.0


@dataclass
class TransformerConfig:
    """
    Transformer model hyperparameters for time-series anomaly detection.
    
    References:
    - Vaswani, A., et al. (2017). Attention Is All You Need.
    - Tuli, S., et al. (2022). TranAD: Deep Transformer Networks for Anomaly Detection.
    - Wu, N., et al. (2020). Deep Transformer Models for Time Series Forecasting.
    """
    # Input sequence length
    # Justification: Same as LSTM for fair comparison
    sequence_length: int = 60
    
    # Model dimension
    # Justification: 32-64 sufficient for vital signs (smaller than NLP)
    d_model: int = 32
    
    # Number of attention heads
    # Justification: 4 heads allow diverse attention patterns
    n_heads: int = 4
    
    # Number of encoder layers
    # Justification: 2-3 layers for time-series (Tuli et al., 2022)
    n_layers: int = 2
    
    # Feed-forward dimension
    # Justification: Typically 2-4x d_model
    d_ff: int = 128
    
    # Dropout
    dropout: float = 0.1
    
    # Learning rate
    # Justification: Transformers often need lower LR
    learning_rate: float = 0.0005
    
    # Batch size
    batch_size: int = 32
    
    # Training epochs
    epochs: int = 15
    
    # Early stopping patience
    early_stopping_patience: int = 5
    
    # Threshold percentile
    threshold_percentile: float = 95.0


@dataclass
class AlertConfig:
    """
    Emergency alert mechanism configuration.
    
    References:
    - Clinical decision support systems literature
    - FDA guidelines for patient monitoring devices
    """
    # Alert levels
    ALERT_NORMAL: int = 0
    ALERT_WARNING: int = 1
    ALERT_CRITICAL: int = 2
    ALERT_EMERGENCY: int = 3
    
    # Minimum consecutive anomalies before alert
    # Justification: Reduces false alarms from transient sensor noise
    min_consecutive_anomalies: int = 3
    
    # Alert cooldown period (seconds)
    # Justification: Prevents alert fatigue
    alert_cooldown: int = 300  # 5 minutes
    
    # Confirmation window for critical alerts (seconds)
    # Justification: Allows for sensor recalibration before escalation
    confirmation_window: int = 30
    
    # Maximum false alarm rate target
    # Justification: Clinical studies suggest <5% acceptable (Clifford et al., 2012)
    target_false_alarm_rate: float = 0.05


@dataclass
class QuantizationConfig:
    """
    INT8 quantization configuration for edge deployment.
    
    References:
    - Jacob, B., et al. (2018). Quantization and Training of Neural Networks.
    - TensorFlow Lite quantization documentation
    """
    # Calibration samples for post-training quantization
    calibration_samples: int = 1000
    
    # Target precision
    target_bits: int = 8
    
    # Quantization method
    method: str = "dynamic"  # Options: "dynamic", "static", "qat"


@dataclass
class ResourceMonitorConfig:
    """Resource monitoring configuration."""
    # Monitoring interval in seconds
    interval: int = 1
    
    # Log file path
    log_path: str = "resource_usage.log"
    
    # Memory warning threshold (GB)
    memory_warning_gb: float = 4.0
    
    # GPU memory warning threshold (%)
    gpu_memory_warning_pct: float = 80.0


@dataclass
class ProjectConfig:
    """Main project configuration combining all sub-configs."""
    vital_signs: VitalSignsConfig = field(default_factory=VitalSignsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    resource_monitor: ResourceMonitorConfig = field(default_factory=ResourceMonitorConfig)
    
    # Project paths
    project_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = field(default="")
    models_dir: str = field(default="")
    results_dir: str = field(default="")
    
    # Debug mode
    debug: bool = True
    
    # Device configuration
    device: str = "cpu"  # Will be set to "cuda" if available
    
    def __post_init__(self):
        self.data_dir = os.path.join(self.project_dir, "data")
        self.models_dir = os.path.join(self.project_dir, "models")
        self.results_dir = os.path.join(self.project_dir, "results")
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)


# Global configuration instance
config = ProjectConfig()


def get_config() -> ProjectConfig:
    """Get the global configuration."""
    return config


def print_config_summary():
    """Print configuration summary with parameter counts."""
    print("=" * 60)
    print("SMART REMOTE PATIENT MONITORING SYSTEM - CONFIGURATION")
    print("=" * 60)
    
    print("\n### Vital Signs Thresholds ###")
    vs = config.vital_signs
    print(f"Heart Rate: {vs.heart_rate_mean}±{vs.heart_rate_std} bpm "
          f"(Critical: <{vs.heart_rate_min} or >{vs.heart_rate_max})")
    print(f"SpO2: {vs.spo2_mean}±{vs.spo2_std}% "
          f"(Critical: <{vs.spo2_min}%)")
    print(f"SBP: {vs.sbp_mean}±{vs.sbp_std} mmHg "
          f"(Critical: <{vs.sbp_min} or >{vs.sbp_max})")
    print(f"Temperature: {vs.temperature_mean}±{vs.temperature_std}°C "
          f"(Critical: <{vs.temperature_min} or >{vs.temperature_max}°C)")
    
    print("\n### Simulation Parameters ###")
    sim = config.simulation
    print(f"Patients: {sim.num_patients}")
    print(f"Sampling interval: {sim.sampling_interval}s")
    print(f"Duration: {sim.simulation_hours} hours")
    print(f"Anomaly rate: {sim.anomaly_rate*100}%")
    
    print("\n### Model Configurations ###")
    print(f"LSTM: {config.lstm.num_layers} layers × {config.lstm.hidden_size} units")
    print(f"Transformer: {config.transformer.n_layers} layers × "
          f"{config.transformer.d_model}d × {config.transformer.n_heads} heads")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
