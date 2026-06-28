"""
Combined Cycle Power Plant - Electrical Power Output Prediction

Predicts net hourly electrical energy output (PE) from ambient variables:
  AT - Ambient Temperature (°C)
  V  - Exhaust Vacuum (cm Hg)
  AP - Ambient Pressure (mbar)
  RH - Relative Humidity (%)

Uses 3 Excel sheets for training and 2 sheets for testing, as specified
in the assignment. Primary model: Linear Regression (as in Tüfekci, 2014).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
TRAIN_SHEETS = ["Sheet1", "Sheet2", "Sheet3"]
TEST_SHEETS = ["Sheet4", "Sheet5"]


def load_sheets(excel_path: Path, sheet_names: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple Excel sheets."""
    frames = [pd.read_excel(excel_path, sheet_name=name) for name in sheet_names]
    return pd.concat(frames, ignore_index=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "split": label,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def print_model_equation(model: LinearRegression, feature_names: list[str]) -> None:
    """Print the fitted linear regression equation."""
    intercept = model.intercept_
    coefs = model.coef_
    terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coefs, feature_names)]
    equation = f"PE = {intercept:.4f} " + " ".join(terms)
    print(f"\nFitted equation:\n  {equation}")


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save actual vs predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual PE (MW)")
    ax.set_ylabel("Predicted PE (MW)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict power plant output using Linear Regression"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/CCPP/Folds5x2_pp.xlsx"),
        help="Path to the Excel dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for plots and summary",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combined Cycle Power Plant - Linear Regression")
    print("=" * 60)

    train_df = load_sheets(args.data, TRAIN_SHEETS)
    test_df = load_sheets(args.data, TEST_SHEETS)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    print(f"\nTraining sheets : {', '.join(TRAIN_SHEETS)}")
    print(f"Testing sheets  : {', '.join(TEST_SHEETS)}")
    print(f"Training samples  : {len(y_train):,}")
    print(f"Testing samples   : {len(y_test):,}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print_model_equation(model, FEATURES)

    print("\nCoefficients:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name:>3} (β):   {coef:+.4f}")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = evaluate(y_train, y_train_pred, "Train (Sheets 1-3)")
    test_metrics = evaluate(y_test, y_test_pred, "Test (Sheets 4-5)")

    print("\n" + "-" * 60)
    print("Performance Metrics")
    print("-" * 60)
    print(f"{'Split':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    for m in (train_metrics, test_metrics):
        print(f"{m['split']:<22} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['R2']:>10.4f}")

    plot_results(
        y_test,
        y_test_pred,
        args.output_dir / "actual_vs_predicted.png",
        "Test Set: Actual vs Predicted Power Output",
    )

    residuals = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual (Actual - Predicted) [MW]")
    ax.set_ylabel("Frequency")
    ax.set_title("Test Set Residual Distribution")
    ax.axvline(0, color="r", linestyle="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "residuals.png", dpi=150)
    plt.close(fig)

    summary_path = args.output_dir / "results_summary.txt"
    with summary_path.open("w") as f:
        f.write("Combined Cycle Power Plant - Linear Regression Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training sheets: {', '.join(TRAIN_SHEETS)}\n")
        f.write(f"Testing sheets:  {', '.join(TEST_SHEETS)}\n\n")
        f.write(f"Intercept: {model.intercept_:.6f}\n")
        for name, coef in zip(FEATURES, model.coef_):
            f.write(f"{name}: {coef:+.6f}\n")
        f.write("\nMetrics:\n")
        for m in (train_metrics, test_metrics):
            f.write(
                f"  {m['split']}: MAE={m['MAE']:.4f}, "
                f"RMSE={m['RMSE']:.4f}, R²={m['R2']:.4f}\n"
            )

    print(f"\nPlots saved to: {args.output_dir}/")
    print(f"Summary saved to: {summary_path}")
    print("\nModel evaluation complete.")


if __name__ == "__main__":
    main()
