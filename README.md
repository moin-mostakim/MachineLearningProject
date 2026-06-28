# Combined Cycle Power Plant - Power Output Prediction

Predict the **net hourly electrical energy output (PE)** of a combined cycle power plant operating at full load, using **Linear Regression**.

## Problem

Given hourly ambient measurements, predict electrical power output (MW):

| Variable | Description | Units |
|----------|-------------|-------|
| **AT** | Ambient Temperature | °C |
| **V** | Exhaust Vacuum | cm Hg |
| **AP** | Ambient Pressure | mbar |
| **RH** | Relative Humidity | % |
| **PE** | Net hourly electrical energy output *(target)* | MW |

## Dataset

[UCI Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)

The Excel file `Folds5x2_pp.xlsx` contains **5 sheets** (5 folds), each with 9,568 samples. Per the assignment:

- **Sheets 1–3** → training
- **Sheets 4–5** → testing

## Setup

```bash
pip install -r requirements.txt
```

## Run (Python)

```bash
python power_plant_regression.py
```

Optional flags:

```bash
python power_plant_regression.py --data data/CCPP/Folds5x2_pp.xlsx --output-dir results
```

## Run (MATLAB)

Open MATLAB in this directory and run:

```matlab
power_plant_regression
```

Uses `readtable` (modern equivalent of `xlsread`) to read the Excel sheets.

## Model

Linear Regression (as in Tüfekci, 2014):

```
PE = β₀ + β₁·AT + β₂·V + β₃·AP + β₄·RH
```

## Output

Results are written to `results/`:

- `results_summary.txt` — coefficients and metrics
- `actual_vs_predicted.png` — scatter plot of test predictions
- `residuals.png` — residual distribution on the test set

## Reference

Pınar Tüfekci, *Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods*, International Journal of Electrical Power & Energy Systems, 60 (2014), 126–140.
