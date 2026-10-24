"""
Visualization Module for Patient Monitoring Anomaly Detection.

Generates time-series plots with detected anomalies highlighted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


def plot_patient_vitals(
    df: pd.DataFrame,
    patient_id: int,
    output_path: Optional[str] = None,
    show_predictions: bool = True,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Plot vital signs time series for a single patient with anomalies.
    
    Args:
        df: DataFrame with vital signs and anomaly predictions
        patient_id: Patient to plot
        output_path: Path to save figure (optional)
        show_predictions: Show prediction overlays
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    patient_data = df[df["patient_id"] == patient_id].copy()
    
    if len(patient_data) == 0:
        raise ValueError(f"No data for patient {patient_id}")
    
    # Convert timestamp
    patient_data["timestamp"] = pd.to_datetime(patient_data["timestamp"])
    patient_data = patient_data.sort_values("timestamp")
    
    config = get_config()
    vs = config.vital_signs
    
    # Create figure with subplots for each vital sign
    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"Patient {patient_id} - Vital Signs with Anomaly Detection", fontsize=14, fontweight="bold")
    
    vitals_config = [
        ("heart_rate", "Heart Rate (bpm)", vs.heart_rate_warning_low, vs.heart_rate_warning_high, "tab:red"),
        ("spo2", "SpO2 (%)", vs.spo2_warning_low, 100, "tab:blue"),
        ("sbp", "Systolic BP (mmHg)", vs.sbp_warning_low, vs.sbp_warning_high, "tab:green"),
        ("dbp", "Diastolic BP (mmHg)", vs.dbp_warning_low, vs.dbp_warning_high, "tab:olive"),
        ("temperature", "Temperature (°C)", vs.temperature_warning_low, vs.temperature_warning_high, "tab:orange"),
        ("resp_rate", "Resp Rate (breaths/min)", vs.resp_rate_warning_low, vs.resp_rate_warning_high, "tab:purple")
    ]
    
    x = patient_data["timestamp"]
    
    for idx, (vital, ylabel, warn_low, warn_high, color) in enumerate(vitals_config):
        ax = axes[idx]
        y = patient_data[vital]
        
        # Plot vital sign
        ax.plot(x, y, color=color, linewidth=1, alpha=0.8, label=vital.replace("_", " ").title())
        
        # Add warning thresholds
        ax.axhline(y=warn_low, color="orange", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(y=warn_high, color="orange", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.fill_between(x, warn_low, warn_high, alpha=0.1, color="green", label="Normal range")
        
        # Highlight ground truth anomalies
        if "is_anomaly" in patient_data.columns:
            anomaly_mask = patient_data["is_anomaly"] == 1
            if anomaly_mask.any():
                ax.scatter(
                    x[anomaly_mask], y[anomaly_mask],
                    color="red", s=20, alpha=0.7, zorder=5,
                    label="True anomaly", marker="o"
                )
        
        # Highlight predictions from different methods
        if show_predictions:
            markers = [
                ("threshold_anomaly", "yellow", "s", "Threshold"),
                ("if_anomaly", "cyan", "^", "Isolation Forest"),
                ("lstm_anomaly", "magenta", "v", "LSTM"),
                ("transformer_anomaly", "lime", "D", "Transformer")
            ]
            
            for pred_col, pred_color, marker, label in markers:
                if pred_col in patient_data.columns:
                    pred_mask = (patient_data[pred_col] == 1)
                    # Only show false positives/negatives for clarity
                    if "is_anomaly" in patient_data.columns:
                        # Show detections that differ from ground truth
                        fp_mask = pred_mask & (patient_data["is_anomaly"] == 0)
                        if fp_mask.any():
                            ax.scatter(
                                x[fp_mask], y[fp_mask],
                                facecolors='none', edgecolors=pred_color,
                                s=60, alpha=0.7, zorder=4, marker=marker,
                                linewidths=1.5
                            )
        
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=8)
        
        # Only show legend for first subplot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    
    # Format x-axis
    axes[-1].set_xlabel("Time", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    return fig


def plot_detection_comparison(
    df: pd.DataFrame,
    metrics: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison of different detection methods.
    
    Args:
        df: DataFrame with all detection results
        metrics: Dictionary of metrics for each method
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Anomaly Detection Method Comparison", fontsize=14, fontweight="bold")
    
    methods = list(metrics.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    # Plot 1: Precision, Recall, F1
    ax1 = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.25
    
    precision = [metrics[m]["precision"] for m in methods]
    recall = [metrics[m]["recall"] for m in methods]
    f1 = [metrics[m]["f1_score"] for m in methods]
    
    bars1 = ax1.bar(x - width, precision, width, label="Precision", color="steelblue")
    bars2 = ax1.bar(x, recall, width, label="Recall", color="darkorange")
    bars3 = ax1.bar(x + width, f1, width, label="F1 Score", color="forestgreen")
    
    ax1.set_ylabel("Score")
    ax1.set_title("Detection Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    # Plot 2: False Alarm Rate
    ax2 = axes[0, 1]
    far = [metrics[m]["false_alarm_rate"] for m in methods]
    bars = ax2.bar(methods, far, color=colors)
    ax2.set_ylabel("False Alarm Rate")
    ax2.set_title("False Alarm Rate (Lower is Better)")
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, max(far) * 1.2 if max(far) > 0 else 0.1)
    ax2.axhline(y=0.05, color="red", linestyle="--", label="5% Target", alpha=0.7)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    
    # Plot 3: Confusion matrix metrics
    ax3 = axes[1, 0]
    tp = [metrics[m]["true_positives"] for m in methods]
    fp = [metrics[m]["false_positives"] for m in methods]
    fn = [metrics[m]["false_negatives"] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax3.bar(x - width, tp, width, label="True Positives", color="green")
    ax3.bar(x, fp, width, label="False Positives", color="red")
    ax3.bar(x + width, fn, width, label="False Negatives", color="orange")
    
    ax3.set_ylabel("Count")
    ax3.set_title("Detection Counts")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    # Create summary table
    best_f1 = max(methods, key=lambda m: metrics[m]["f1_score"])
    best_precision = max(methods, key=lambda m: metrics[m]["precision"])
    best_recall = max(methods, key=lambda m: metrics[m]["recall"])
    lowest_far = min(methods, key=lambda m: metrics[m]["false_alarm_rate"])
    
    summary_text = f"""
    Summary of Detection Methods
    ============================
    
    Best F1 Score: {best_f1}
      ({metrics[best_f1]['f1_score']:.4f})
    
    Best Precision: {best_precision}
      ({metrics[best_precision]['precision']:.4f})
    
    Best Recall: {best_recall}
      ({metrics[best_recall]['recall']:.4f})
    
    Lowest False Alarm Rate: {lowest_far}
      ({metrics[lowest_far]['false_alarm_rate']:.4f})
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
    
    return fig


def plot_anomaly_scores_timeline(
    df: pd.DataFrame,
    patient_id: int,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot anomaly scores from different methods over time.
    """
    patient_data = df[df["patient_id"] == patient_id].copy()
    patient_data["timestamp"] = pd.to_datetime(patient_data["timestamp"])
    patient_data = patient_data.sort_values("timestamp")
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"Patient {patient_id} - Anomaly Scores Timeline", fontsize=14, fontweight="bold")
    
    x = patient_data["timestamp"]
    
    # Ground truth
    ax0 = axes[0]
    if "is_anomaly" in patient_data.columns:
        ax0.fill_between(x, 0, patient_data["is_anomaly"], alpha=0.5, color="red", label="Ground Truth")
    ax0.set_ylabel("Ground Truth")
    ax0.set_ylim(-0.1, 1.1)
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)
    
    # MEWS Score (threshold-based)
    ax1 = axes[1]
    if "mews_score" in patient_data.columns:
        ax1.fill_between(x, 0, patient_data["mews_score"], alpha=0.5, color="blue")
        ax1.axhline(y=5, color="red", linestyle="--", alpha=0.7, label="MEWS ≥5 threshold")
    ax1.set_ylabel("MEWS Score")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Isolation Forest Score
    ax2 = axes[2]
    if "if_score" in patient_data.columns:
        ax2.fill_between(x, 0, patient_data["if_score"], alpha=0.5, color="green")
    ax2.set_ylabel("IF Score")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Deep Learning Errors
    ax3 = axes[3]
    if "lstm_error" in patient_data.columns:
        ax3.plot(x, patient_data["lstm_error"], color="purple", alpha=0.7, label="LSTM Error")
    if "transformer_error" in patient_data.columns:
        ax3.plot(x, patient_data["transformer_error"], color="orange", alpha=0.7, label="Transformer Error")
    ax3.set_ylabel("Reconstruction Error")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved scores timeline to {output_path}")
    
    return fig


def create_all_visualizations(
    df: pd.DataFrame,
    metrics: Dict[str, Dict],
    output_dir: str,
    num_patients: int = 3
) -> List[str]:
    """
    Create all visualizations and save to output directory.
    
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # Get unique patients
    patient_ids = df["patient_id"].unique()[:num_patients]
    
    # Plot vitals for each patient
    for pid in patient_ids:
        try:
            path = os.path.join(output_dir, f"patient_{pid}_vitals.png")
            plot_patient_vitals(df, pid, output_path=path)
            saved_files.append(path)
            plt.close()
        except Exception as e:
            print(f"Error plotting patient {pid}: {e}")
    
    # Plot comparison
    try:
        path = os.path.join(output_dir, "method_comparison.png")
        plot_detection_comparison(df, metrics, output_path=path)
        saved_files.append(path)
        plt.close()
    except Exception as e:
        print(f"Error plotting comparison: {e}")
    
    # Plot scores timeline
    for pid in patient_ids[:1]:  # Just first patient
        try:
            path = os.path.join(output_dir, f"patient_{pid}_scores.png")
            plot_anomaly_scores_timeline(df, pid, output_path=path)
            saved_files.append(path)
            plt.close()
        except Exception as e:
            print(f"Error plotting scores for patient {pid}: {e}")
    
    return saved_files


if __name__ == "__main__":
    # Test visualization
    from data.sensor_simulator import generate_dataset
    
    print("Generating test data...")
    df, _ = generate_dataset(num_patients=3, duration_hours=2, debug=False)
    
    # Add dummy predictions
    df["threshold_anomaly"] = (np.random.rand(len(df)) > 0.95).astype(int)
    df["if_anomaly"] = (np.random.rand(len(df)) > 0.95).astype(int)
    df["mews_score"] = np.random.randint(0, 8, len(df))
    df["if_score"] = np.random.rand(len(df))
    
    print("Creating visualizations...")
    fig = plot_patient_vitals(df, 0)
    plt.show()
