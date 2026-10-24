# AI-Driven Climate Emulation and Uncertainty Quantification

## Overview

This project implements machine learning methods for climate emulation with uncertainty quantification, targeting high-resolution Earth system modeling. The goal is to create fast, accurate neural network surrogates for computationally expensive climate simulations.

## Key Features

- **Multiple architectures**: U-Net, ConvLSTM, and baseline models
- **Uncertainty quantification**: MC Dropout and Deep Ensembles
- **Comprehensive metrics**: RMSE, ACC, CRPS, calibration error
- **INT8 quantization**: For efficient deployment
- **Resource monitoring**: GPU/CPU usage tracking
- **Synthetic data generation**: For testing without large datasets

## Project Structure

```
climate-ai-emulation/
├── config.py              # Centralized configuration with hyperparameter justifications
├── main.py                # Main experiment script
├── requirements.txt       # Python dependencies
├── data/
│   ├── __init__.py
│   └── dataset.py         # Synthetic and WeatherBench data loaders
├── models/
│   ├── __init__.py
│   ├── baselines.py       # Persistence, climatology, linear regression
│   ├── unet.py            # U-Net with attention and residual connections
│   ├── convlstm.py        # Convolutional LSTM
│   └── ensemble.py        # Deep ensemble implementation
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # RMSE, ACC, CRPS, calibration metrics
│   ├── resource_monitor.py # GPU/CPU/memory monitoring
│   └── quantization.py    # INT8 quantization utilities
├── experiments/
│   ├── __init__.py
│   └── trainer.py         # Training loop with AMP support
├── configs/               # Additional configuration files
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Experiment (Debug Mode)

```bash
python main.py --debug
```

### Run Full Experiment

```bash
python main.py --debug=False --architecture unet
```

## Implemented Methods

### Baselines
1. **Persistence**: Predicts current state as future state
2. **Climatology**: Predicts long-term mean
3. **Linear Regression**: Ridge regression with spatial features

### Neural Networks
1. **U-Net**: Encoder-decoder with skip connections and attention gates
2. **ConvLSTM**: Convolutional LSTM for spatiotemporal learning

### Uncertainty Quantification
1. **MC Dropout**: Multiple forward passes with dropout enabled
2. **Deep Ensembles**: Train multiple independent models

## Metrics

### Deterministic
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **ACC**: Anomaly Correlation Coefficient (primary skill metric)
- **SSIM**: Structural Similarity Index

### Probabilistic
- **CRPS**: Continuous Ranked Probability Score
- **Calibration Error**: Expected coverage vs actual coverage
- **Sharpness**: Width of prediction intervals
- **Coverage**: Fraction within prediction intervals

## Configuration

All hyperparameters are in `config.py` with detailed justifications:

```python
from config import get_config

config = get_config(debug=True)
config.model.architecture = "unet"
config.training.learning_rate = 1e-4
```

## Literature References

1. **WeatherBench** (Rasp et al., 2020): Benchmark dataset and evaluation methodology
2. **Pangu-Weather** (Bi et al., 2023): 3D transformer architecture for weather prediction
3. **GraphCast** (Lam et al., 2023): Graph neural network on icosahedral mesh
4. **FourCastNet** (Pathak et al., 2022): Fourier neural operators for weather

## Results

Results are saved to `outputs/experiment_results.json` including:
- Baseline performance
- Neural network metrics
- Ensemble uncertainty metrics
- Quantization comparison

## Quantization

INT8 quantization reduces model size by ~4x with minimal accuracy loss:

```python
from utils.quantization import quantize_for_deployment

quantized_model, metrics = quantize_for_deployment(model, method="dynamic")
```

## Data

### Synthetic Data
For testing and debugging, the project generates synthetic spatiotemporal data with:
- Spatial correlations (Gaussian random fields)
- Temporal autocorrelation (AR process)
- Multi-channel atmospheric variables

### Real Data (WeatherBench)
The code supports WeatherBench data at multiple resolutions:
- 5.625° (~600km)
- 2.8125° (~300km)
- 1.40625° (~150km)

Download from: https://github.com/pangeo-data/WeatherBench

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@software{climate_ai_emulation,
  title={AI-Driven Climate Emulation and Uncertainty Quantification},
  year={2024},
  url={https://github.com/example/climate-ai-emulation}
}
```

And the relevant papers:

```bibtex
@article{rasp2020weatherbench,
  title={WeatherBench: A benchmark dataset for data-driven weather forecasting},
  author={Rasp, Stephan and others},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2020}
}

@article{bi2023pangu,
  title={Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast},
  author={Bi, Kaifeng and others},
  journal={Nature},
  year={2023}
}

@article{lam2023graphcast,
  title={GraphCast: Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and others},
  journal={Science},
  year={2023}
}
```
