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

## Datasets

The system supports multiple climate/weather datasets. Due to the large size of ERA5 data, this project provides synthetic data for quick testing while documenting how to use real data.

| Dataset | Source | URL | License | Resolution | Size |
|---------|--------|-----|---------|------------|------|
| **WeatherBench2** | Google Research | [github.com/google-research/weatherbench2](https://github.com/google-research/weatherbench2) | Apache 2.0 | 5.625°, 2.8125°, 1.40625° | ~100GB+ |
| **ERA5** | ECMWF Copernicus | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) | Copernicus License | 0.25° native | ~500TB total |
| **WeatherBench (original)** | Pangeo | [pangeo-data/WeatherBench](https://github.com/pangeo-data/WeatherBench) | CC BY 4.0 | Multiple | ~10GB |

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
