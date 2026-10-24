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

## Experimental Results

**Dataset**: Synthetic Climate Data (realistic spatiotemporal patterns)
**Ran on Real Data**: No (uses physically-motivated synthetic data)
**Training**: 10 epochs on CPU

### Model Parameters

| Model | Parameters | Description |
|-------|------------|-------------|
| U-Net | 1,570,050 | Encoder-decoder with attention gates |
| ConvLSTM | ~1.2M | Convolutional LSTM for temporal modeling |

### Baseline Performance (Test Set)

| Method | RMSE | ACC | Notes |
|--------|------|-----|-------|
| Persistence | 1.216 | 0.262 | Predicts current state |
| Climatology | 1.000 | 0.000 | Predicts mean |
| Linear Regression | 0.949 | 0.265 | Ridge with spatial features |

### Neural Network Performance (Test Set)

| Model | RMSE | ACC | CRPS | Coverage@90 |
|-------|------|-----|------|-------------|
| U-Net (single) | 1.674 | 0.010 | 1.187 | 0.208 |
| U-Net (ensemble) | 1.455 | 0.115 | 0.910 | 0.687 |

### Uncertainty Quantification

| Method | Calibration Error | Sharpness | Coverage@50 | Coverage@90 |
|--------|-------------------|-----------|-------------|-------------|
| MC Dropout | 0.428 | 0.275 | 0.088 | 0.208 |
| Deep Ensemble | 0.105 | 1.308 | 0.416 | 0.687 |

*Note: Results on synthetic data demonstrate methodology. Deep Ensemble shows better calibration (lower error, coverage closer to nominal).*

## Quantization

INT8 dynamic quantization is supported for deployment:

```python
from utils.quantization import quantize_for_deployment

quantized_model, metrics = quantize_for_deployment(model, method="dynamic")
```

Expected compression: ~4x size reduction with minimal accuracy loss.

## Datasets

This project uses realistic **synthetic climate data** for experiments. The synthetic data generator creates spatiotemporal patterns with:
- Spatial correlations (Gaussian random fields with ~1000km correlation length)
- Temporal autocorrelation (AR(1) process, ρ ≈ 0.85-0.95)
- Multi-scale patterns (synoptic + mesoscale)
- 7 atmospheric channels (t2m, u10, v10, z500, t500, z850, t850)

**Note**: While HuggingFace hosts ERA5 derivatives, most require authentication or have access issues. The synthetic data provides a functional pipeline for methodology development.

| Dataset | Source | Status | License | Notes |
|---------|--------|--------|---------|-------|
| **Synthetic** | This project | **Available** | MIT | Realistic spatiotemporal patterns |
| **WeatherBench2** | Google Research | Requires setup | Apache 2.0 | Large (~100GB+) |
| **ERA5** | ECMWF CDS | Requires registration | Copernicus | Official source |

### Download Instructions

**WeatherBench2 (Recommended for full experiments):**
```bash
# Install Google Cloud SDK for data access
pip install google-cloud-storage xarray zarr

# Download specific variables at 5.625° resolution
# See: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
python -c "
import xarray as xr
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
ds_subset = ds[['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']].sel(time=slice('2020', '2021'))
ds_subset.to_netcdf('data/weatherbench/era5_subset.nc')
"
```

**Original WeatherBench (smaller, easier):**
```bash
# Download from Pangeo data catalog
pip install intake intake-xarray zarr
python -c "
import intake
cat = intake.open_catalog('https://raw.githubusercontent.com/pangeo-data/WeatherBench/master/catalog.yaml')
ds = cat.era5_5_625deg.to_dask()
ds.sel(time=slice('2017', '2018')).to_netcdf('data/weatherbench/era5_5.625.nc')
"
```

**ECMWF Climate Data Store (requires free registration):**
1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)
2. Install CDS API: `pip install cdsapi`
3. Download ERA5 data using their API

**Synthetic Data (for pipeline testing):**
```bash
python main.py --debug  # Uses synthetic by default in debug mode
```

Note: Synthetic data generates realistic spatiotemporal patterns (Gaussian random fields with AR(1) temporal correlation) suitable for testing the full pipeline, but real ERA5/WeatherBench data is required for meaningful scientific experiments.

### Data Variables

| Variable | Description | Units | Pressure Levels |
|----------|-------------|-------|-----------------|
| t2m | 2m Temperature | K | Surface |
| u10 | 10m U-wind | m/s | Surface |
| v10 | 10m V-wind | m/s | Surface |
| z | Geopotential | m²/s² | 500, 850 hPa |
| t | Temperature | K | 500, 850 hPa |

### Data Format

The dataloader expects data in NetCDF or Zarr format with dimensions:
- `time`: Timestamps
- `lat`: Latitude coordinates
- `lon`: Longitude coordinates
- Variables as data arrays

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
