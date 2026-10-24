#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Remote Patient Monitoring System Using Simulated IoT Sensors
with Real-Time Health Analytics and Emergency Alert Mechanism.

This is the main entry point that runs the complete pipeline:
1. Generate simulated IoT sensor data
2. Run threshold-based baseline detection
3. Run Isolation Forest anomaly detection
4. Run LSTM-based anomaly detection
5. Run Transformer-based anomaly detection
6. Evaluate and compare all methods
7. Run emergency alert system
8. Generate visualizations
9. Perform INT8 quantization comparison

Literature References:
- Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database.
- Goldberger, A.L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
- Liu, F.T., et al. (2008). Isolation Forest.
- Malhotra, P., et al. (2015). Long Short Term Memory Networks for Anomaly Detection.
- Tuli, S., et al. (2022). TranAD: Deep Transformer Networks for Anomaly Detection.
- Cvach, M. (2012). Monitor Alarm Fatigue: An Integrative Review.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, print_config_summary
from data.sensor_simulator import generate_dataset, IoTSensorSimulator
from models.threshold_baseline import ThresholdBasedDetector, run_threshold_baseline
from models.isolation_forest import IsolationForestDetector, run_isolation_forest
from models.lstm_detector import LSTMDetector, run_lstm_detector
from models.transformer_detector import TransformerDetector, run_transformer_detector
from alerts.emergency_alert import EmergencyAlertSystem, run_alert_system
from utils.resource_monitor import ResourceMonitorContext
from utils.quantization import quantize_and_compare
from visualization.plot_anomalies import create_all_visualizations


def print_banner():
    """Print project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║     SMART REMOTE PATIENT MONITORING SYSTEM                                   ║
║     Using Simulated IoT Sensors with Real-Time Health Analytics              ║
║     and Emergency Alert Mechanism                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║     Methods: Threshold | Isolation Forest | LSTM | Transformer               ║
║     Dataset: Simulated from MIMIC-III/PhysioNet distributions                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_parameter_table(models_info: dict):
    """Print parameter count table for all models."""
    print("\n" + "=" * 70)
    print("MODEL PARAMETER COUNT TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'Component':<25} {'Parameters':>15}")
    print("-" * 70)
    
    for model_name, info in models_info.items():
        params = info.get("parameters", {})
        if isinstance(params, dict):
            for component, count in params.items():
                if component == "total":
                    print(f"{model_name:<25} {'TOTAL':<25} {count:>15,}")
                else:
                    print(f"{'':<25} {component:<25} {count:>15,}")
            print("-" * 70)
        else:
            print(f"{model_name:<25} {'TOTAL':<25} {params:>15,}")
    
    print("=" * 70)


def print_benchmark_results(metrics: dict):
    """Print benchmark results table."""
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method':<20} {'Precision':>12} {'Recall':>12} {'F1 Score':>12} {'FAR':>12}")
    print("-" * 80)
    
    for method, m in metrics.items():
        print(f"{method:<20} {m['precision']:>12.4f} {m['recall']:>12.4f} "
              f"{m['f1_score']:>12.4f} {m['false_alarm_rate']:>12.4f}")
    
    print("-" * 80)
    
    # Find best performers
    best_f1 = max(metrics.keys(), key=lambda k: metrics[k]["f1_score"])
    best_precision = max(metrics.keys(), key=lambda k: metrics[k]["precision"])
    best_recall = max(metrics.keys(), key=lambda k: metrics[k]["recall"])
    lowest_far = min(metrics.keys(), key=lambda k: metrics[k]["false_alarm_rate"])
    
    print(f"\nBest F1 Score: {best_f1} ({metrics[best_f1]['f1_score']:.4f})")
    print(f"Best Precision: {best_precision} ({metrics[best_precision]['precision']:.4f})")
    print(f"Best Recall: {best_recall} ({metrics[best_recall]['recall']:.4f})")
    print(f"Lowest False Alarm Rate: {lowest_far} ({metrics[lowest_far]['false_alarm_rate']:.4f})")
    print("=" * 80)


def print_literature_comparison():
    """Print comparison with literature benchmarks."""
    print("\n" + "=" * 80)
    print("LITERATURE COMPARISON")
    print("=" * 80)
    print("""
Remote Patient Monitoring and Anomaly Detection Benchmarks from Literature:

1. PhysioNet/MIMIC-III Studies:
   - Harutyunyan et al. (2019): Multitask learning on MIMIC-III
     * In-hospital mortality prediction AUC-ROC: 0.87
     * Decompensation prediction AUC-ROC: 0.90
   
2. LSTM for Anomaly Detection:
   - Malhotra et al. (2015): LSTM-based anomaly detection
     * Recall: 0.85-0.95 on ECG data
     * Precision: 0.80-0.90
   - Hundman et al. (2018): Spacecraft anomaly detection
     * F1 Score: 0.75-0.85

3. Transformer-based Methods:
   - Tuli et al. (2022): TranAD for anomaly detection
     * F1 Score: 0.82-0.91 on various time-series datasets
     * Up to 15% improvement over LSTM baselines

4. Isolation Forest:
   - Liu et al. (2008): Original Isolation Forest paper
     * AUC-ROC: 0.85-0.95 on benchmark datasets
   
5. Clinical Early Warning Scores:
   - Smith et al. (2013): Modified Early Warning Score (MEWS)
     * Sensitivity: 75-80% for deterioration prediction
     * Specificity: 85-90%
   - National Early Warning Score (NEWS):
     * AUC-ROC: 0.87-0.91 for mortality prediction

6. Alert Fatigue Studies:
   - Cvach (2012): False alarm rates in ICU: 72-99%
   - Target: <5% false alarm rate for clinical acceptance
     
Note: Our simulated dataset may show different characteristics than
real clinical data. Results should be validated on actual MIMIC-III
or PhysioNet data for clinical applications.
""")
    print("=" * 80)


def run_pipeline(args):
    """Run the complete anomaly detection pipeline."""
    config = get_config()
    config.debug = args.debug
    
    print_banner()
    
    if args.debug:
        print_config_summary()
    
    # Results storage
    all_metrics = {}
    models_info = {}
    run_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_patients": args.num_patients,
            "duration_hours": args.duration_hours,
            "debug": args.debug
        }
    }
    
    # =========================================================================
    # Step 1: Generate Simulated IoT Sensor Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: GENERATING SIMULATED IOT SENSOR DATA")
    print("=" * 60)
    
    with ResourceMonitorContext("data_generation") as ctx:
        df, data_stats = generate_dataset(
            num_patients=args.num_patients,
            duration_hours=args.duration_hours,
            debug=args.debug
        )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {data_stats['total_samples']:,}")
    print(f"  Patients: {data_stats['num_patients']}")
    print(f"  Anomaly rate: {data_stats['anomaly_rate']*100:.2f}%")
    
    run_results["dataset"] = {
        "total_samples": data_stats['total_samples'],
        "num_patients": data_stats['num_patients'],
        "anomaly_rate": data_stats['anomaly_rate'],
        "source": "Simulated from MIMIC-III/PhysioNet vital sign distributions"
    }
    
    if args.debug:
        ctx.print_summary()
    
    # Save dataset
    dataset_path = os.path.join(config.data_dir, "patient_vitals.csv")
    df.to_csv(dataset_path, index=False)
    print(f"\nDataset saved to: {dataset_path}")
    
    # Split data
    n = len(df)
    train_df = df.iloc[:int(n * 0.6)]
    val_df = df.iloc[int(n * 0.6):int(n * 0.8)]
    test_df = df.iloc[int(n * 0.8):]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # =========================================================================
    # Step 2: Threshold-Based Baseline
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: THRESHOLD-BASED BASELINE DETECTION")
    print("=" * 60)
    
    with ResourceMonitorContext("threshold_baseline") as ctx:
        detector_threshold = ThresholdBasedDetector(config)
        results_threshold = detector_threshold.detect(test_df)
        metrics_threshold = detector_threshold.get_metrics(
            results_threshold["threshold_anomaly"].values,
            results_threshold["is_anomaly"].values
        )
    
    all_metrics["Threshold"] = metrics_threshold
    models_info["Threshold"] = {
        "type": "Rule-based",
        "parameters": {"rules": 12, "thresholds": 24, "total": 36}
    }
    
    print(f"\nThreshold Detection Results:")
    print(f"  Precision: {metrics_threshold['precision']:.4f}")
    print(f"  Recall: {metrics_threshold['recall']:.4f}")
    print(f"  F1 Score: {metrics_threshold['f1_score']:.4f}")
    
    if args.debug:
        ctx.print_summary()
    
    # =========================================================================
    # Step 3: Isolation Forest
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)
    
    with ResourceMonitorContext("isolation_forest") as ctx:
        detector_if = IsolationForestDetector(config)
        detector_if.fit(train_df)
        results_if = detector_if.detect(test_df)
        metrics_if = detector_if.get_metrics(
            results_if["if_anomaly"].values,
            results_if["is_anomaly"].values
        )
    
    all_metrics["Isolation Forest"] = metrics_if
    models_info["Isolation Forest"] = {
        "type": "Statistical",
        "parameters": detector_if.get_parameter_count()
    }
    
    print(f"\nIsolation Forest Results:")
    print(f"  Precision: {metrics_if['precision']:.4f}")
    print(f"  Recall: {metrics_if['recall']:.4f}")
    print(f"  F1 Score: {metrics_if['f1_score']:.4f}")
    
    if args.debug:
        ctx.print_summary()
    
    # Save model
    if_model_path = os.path.join(config.models_dir, "isolation_forest.joblib")
    detector_if.save(if_model_path)
    print(f"Model saved to: {if_model_path}")
    
    # =========================================================================
    # Step 4: LSTM Autoencoder
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: LSTM AUTOENCODER ANOMALY DETECTION")
    print("=" * 60)
    
    with ResourceMonitorContext("lstm_training") as ctx:
        detector_lstm = LSTMDetector(config)
        detector_lstm.fit(train_df, val_df)
        results_lstm = detector_lstm.detect(test_df)
        metrics_lstm = detector_lstm.get_metrics(
            results_lstm["lstm_anomaly"].values,
            results_lstm["is_anomaly"].values
        )
    
    all_metrics["LSTM"] = metrics_lstm
    models_info["LSTM"] = {
        "type": "Deep Learning",
        "parameters": detector_lstm.get_parameter_count()
    }
    
    print(f"\nLSTM Autoencoder Results:")
    print(f"  Precision: {metrics_lstm['precision']:.4f}")
    print(f"  Recall: {metrics_lstm['recall']:.4f}")
    print(f"  F1 Score: {metrics_lstm['f1_score']:.4f}")
    
    if args.debug:
        ctx.print_summary()
    
    # Save model
    lstm_model_path = os.path.join(config.models_dir, "lstm_autoencoder.pt")
    detector_lstm.save(lstm_model_path)
    print(f"Model saved to: {lstm_model_path}")
    
    # =========================================================================
    # Step 5: Transformer Autoencoder
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: TRANSFORMER AUTOENCODER ANOMALY DETECTION")
    print("=" * 60)
    
    with ResourceMonitorContext("transformer_training") as ctx:
        detector_transformer = TransformerDetector(config)
        detector_transformer.fit(train_df, val_df)
        results_transformer = detector_transformer.detect(test_df)
        metrics_transformer = detector_transformer.get_metrics(
            results_transformer["transformer_anomaly"].values,
            results_transformer["is_anomaly"].values
        )
    
    all_metrics["Transformer"] = metrics_transformer
    models_info["Transformer"] = {
        "type": "Deep Learning",
        "parameters": detector_transformer.get_parameter_count()
    }
    
    print(f"\nTransformer Autoencoder Results:")
    print(f"  Precision: {metrics_transformer['precision']:.4f}")
    print(f"  Recall: {metrics_transformer['recall']:.4f}")
    print(f"  F1 Score: {metrics_transformer['f1_score']:.4f}")
    
    if args.debug:
        ctx.print_summary()
    
    # Save model
    transformer_model_path = os.path.join(config.models_dir, "transformer_autoencoder.pt")
    detector_transformer.save(transformer_model_path)
    print(f"Model saved to: {transformer_model_path}")
    
    # =========================================================================
    # Step 6: Combine Results and Run Alert System
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: EMERGENCY ALERT SYSTEM")
    print("=" * 60)
    
    # Merge all results
    combined_results = test_df.copy()
    combined_results["threshold_anomaly"] = results_threshold["threshold_anomaly"].values
    combined_results["mews_score"] = results_threshold["mews_score"].values
    combined_results["if_anomaly"] = results_if["if_anomaly"].values
    combined_results["if_score"] = results_if["if_score"].values
    combined_results["lstm_anomaly"] = results_lstm["lstm_anomaly"].values[:len(test_df)]
    combined_results["lstm_error"] = results_lstm["lstm_error"].values[:len(test_df)]
    combined_results["transformer_anomaly"] = results_transformer["transformer_anomaly"].values[:len(test_df)]
    combined_results["transformer_error"] = results_transformer["transformer_error"].values[:len(test_df)]
    
    # Run alert system
    alerts, alert_metrics = run_alert_system(combined_results, debug=args.debug)
    
    run_results["alerts"] = {
        "total_alerts": alert_metrics["total_alerts"],
        "avg_latency_ms": alert_metrics["avg_latency_ms"],
        "alerts_by_priority": alert_metrics["alerts_by_priority"]
    }
    
    # =========================================================================
    # Step 7: INT8 Quantization Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: INT8 QUANTIZATION COMPARISON")
    print("=" * 60)
    
    # Prepare test data for quantization
    from sklearn.preprocessing import StandardScaler
    feature_cols = ["heart_rate", "spo2", "sbp", "dbp", "temperature", "resp_rate"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(test_df[feature_cols].values[:1000])
    
    # Create sequences
    seq_len = config.lstm.sequence_length
    sequences = []
    for i in range(len(X_scaled) - seq_len + 1):
        sequences.append(X_scaled[i:i + seq_len])
    test_sequences = torch.FloatTensor(np.array(sequences[:100]))
    
    print("\n--- LSTM Model Quantization ---")
    try:
        _, lstm_quant_metrics = quantize_and_compare(
            detector_lstm.model, test_sequences, debug=args.debug
        )
        run_results["quantization"] = {"lstm": lstm_quant_metrics}
    except Exception as e:
        print(f"LSTM quantization failed: {e}")
        run_results["quantization"] = {"lstm": "failed"}
    
    # =========================================================================
    # Step 8: Print Summary Tables
    # =========================================================================
    print_parameter_table(models_info)
    print_benchmark_results(all_metrics)
    print_literature_comparison()
    
    # =========================================================================
    # Step 9: Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    viz_dir = os.path.join(config.results_dir, "visualizations")
    try:
        saved_files = create_all_visualizations(
            combined_results,
            all_metrics,
            viz_dir,
            num_patients=min(3, args.num_patients)
        )
        print(f"\nSaved {len(saved_files)} visualization files to {viz_dir}")
    except Exception as e:
        print(f"Visualization error: {e}")
        saved_files = []
    
    # =========================================================================
    # Save Final Results
    # =========================================================================
    run_results["metrics"] = all_metrics
    run_results["models_info"] = {
        k: {"type": v["type"], "total_params": v["parameters"].get("total", v["parameters"])}
        for k, v in models_info.items()
    }
    run_results["visualizations"] = saved_files
    run_results["ran_successfully"] = True
    
    results_path = os.path.join(config.results_dir, "run_results.json")
    with open(results_path, "w") as f:
        json.dump(run_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"""
Summary:
  - Dataset: {data_stats['total_samples']:,} samples from {data_stats['num_patients']} patients
  - Source: Simulated from MIMIC-III/PhysioNet vital sign distributions
  - Best F1 Score: {max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1_score'])} 
    ({max(m['f1_score'] for m in all_metrics.values()):.4f})
  - Total Alerts Generated: {alert_metrics['total_alerts']}
  - Average Alert Latency: {alert_metrics['avg_latency_ms']:.2f} ms
  
Output files:
  - Dataset: {dataset_path}
  - Models: {config.models_dir}/
  - Results: {results_path}
  - Visualizations: {viz_dir}/
""")
    print("=" * 80)
    
    return run_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Remote Patient Monitoring System"
    )
    parser.add_argument(
        "--num-patients", type=int, default=10,
        help="Number of patients to simulate (default: 10)"
    )
    parser.add_argument(
        "--duration-hours", type=float, default=4,
        help="Duration of simulation in hours (default: 4)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=True,
        help="Enable debug mode with verbose output"
    )
    parser.add_argument(
        "--no-debug", action="store_false", dest="debug",
        help="Disable debug mode"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(args)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
