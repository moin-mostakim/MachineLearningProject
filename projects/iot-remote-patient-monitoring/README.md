# Smart Remote Patient Monitoring System

Using Simulated IoT Sensors with Real-Time Health Analytics and Emergency Alert Mechanism.

## Overview

This project implements a comprehensive anomaly detection system for remote patient monitoring using simulated IoT sensor data. The system generates realistic vital sign streams based on MIMIC-III and PhysioNet distributions, then applies multiple anomaly detection methods to identify health deterioration events.

## Features

- **IoT Sensor Simulation**: Generates realistic vital sign data (heart rate, SpO2, blood pressure, temperature, respiratory rate) with:
  - Inter-patient variability based on age and condition
  - Circadian rhythm modeling
  - Realistic sensor noise
  - Configurable anomaly injection

- **Multiple Detection Methods**:
  1. **Threshold-based baseline**: Clinical vital sign thresholds with MEWS scoring
  2. **Isolation Forest**: Statistical anomaly detection for multivariate data
  3. **LSTM Autoencoder**: Deep learning reconstruction-based detection
  4. **Transformer Autoencoder**: Attention-based temporal anomaly detection

- **Emergency Alert System**: Multi-level alert mechanism with:
  - Alert confirmation logic to reduce false alarms
  - Cooldown periods to prevent alert fatigue
  - Priority escalation based on severity
  - Alert lifecycle tracking

- **INT8 Quantization**: Model optimization for edge deployment

## Project Structure

```
iot-remote-patient-monitoring/
├── config.py              # All hyperparameters with justifications
├── main.py                # Main pipeline entry point
├── data/
│   ├── sensor_simulator.py    # IoT sensor data generation
│   └── __init__.py
├── models/
│   ├── threshold_baseline.py  # Clinical threshold detection
│   ├── isolation_forest.py    # Statistical anomaly detection
│   ├── lstm_detector.py       # LSTM autoencoder
│   ├── transformer_detector.py # Transformer autoencoder
│   └── __init__.py
├── alerts/
│   ├── emergency_alert.py     # Alert system implementation
│   └── __init__.py
├── utils/
│   ├── resource_monitor.py    # CPU/Memory/GPU monitoring
│   ├── quantization.py        # INT8 quantization utilities
│   └── __init__.py
├── visualization/
│   ├── plot_anomalies.py      # Time-series visualization
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline with default settings (debug mode):

```bash
python main.py --debug
```

Customize the simulation:

```bash
python main.py --num-patients 50 --duration-hours 12 --debug
```

## Dataset

The system generates simulated IoT sensor data based on vital sign distributions from:

- **MIMIC-III** (Johnson et al., 2016): Critical care database with ~40,000 ICU stays
- **PhysioNet** (Goldberger et al., 2000): Open access physiological signal databases

Vital sign ranges and anomaly patterns are calibrated to match real clinical data characteristics.

### Simulated Vital Signs

| Vital Sign | Normal Range | Unit | Source |
|------------|--------------|------|--------|
| Heart Rate | 60-100 | bpm | AHA Guidelines |
| SpO2 | 95-100 | % | WHO Guidelines |
| Systolic BP | 90-120 | mmHg | AHA Guidelines |
| Diastolic BP | 60-80 | mmHg | AHA Guidelines |
| Temperature | 36.1-37.2 | °C | WHO Guidelines |
| Respiratory Rate | 12-20 | breaths/min | Clinical Guidelines |

### Anomaly Types

- Bradycardia / Tachycardia
- Hypoxemia
- Hypotension / Hypertension
- Hypothermia / Hyperthermia
- Tachypnea / Bradypnea
- Multi-organ (sepsis-like)

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) - Fraction of detected anomalies that are true |
| Recall | TP / (TP + FN) - Fraction of true anomalies detected |
| F1 Score | Harmonic mean of precision and recall |
| False Alarm Rate | FP / (FP + TN) - Target: <5% for clinical acceptance |
| Alert Latency | Time from anomaly onset to alert |

## Configuration

All hyperparameters are defined in `config.py` with literature-based justifications:

### LSTM Configuration
- Sequence length: 60 (5 minutes at 5s intervals)
- Hidden size: 64
- Layers: 2
- Dropout: 0.2

### Transformer Configuration
- Model dimension: 32
- Attention heads: 4
- Layers: 2
- Feed-forward dimension: 128

### Alert Configuration
- Minimum consecutive anomalies: 3
- Cooldown period: 300s (5 minutes)
- Target false alarm rate: 5%

## Literature References

1. Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data.

2. Goldberger, A.L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation.

3. Liu, F.T., et al. (2008). Isolation Forest. ICDM.

4. Malhotra, P., et al. (2015). Long Short Term Memory Networks for Anomaly Detection. ESANN.

5. Tuli, S., et al. (2022). TranAD: Deep Transformer Networks for Anomaly Detection. VLDB.

6. Cvach, M. (2012). Monitor Alarm Fatigue: An Integrative Review. Biomedical Instrumentation & Technology.

7. Smith, G.B., et al. (2013). A modified Early Warning Score (MEWS). Critical Care.

8. Harutyunyan, H., et al. (2019). Multitask learning and benchmarking with clinical time series data. Scientific Data.

## License

MIT License
