"""
Dataset classes for climate emulation.

Dataset: WeatherBench2 ERA5 Reanalysis
- Source: Google Cloud Storage (public, anonymous access)
- URL: gs://weatherbench2/datasets/era5/
- License: Copernicus Climate Data Store License (free for research)
- Resolution: 64x32 (5.625° equiangular)
- Variables: 2m_temperature, 10m_u/v_wind, geopotential@500hPa, temperature@850hPa

References:
- Rasp et al. (2020): "WeatherBench: A benchmark dataset for data-driven weather forecasting"
- Hersbach et al. (2020): "The ERA5 global reanalysis"
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DataConfig

DATASET_CONFIG = {
    "weatherbench2_era5": {
        "name": "WeatherBench2 ERA5 Reanalysis",
        "source": "Google Cloud Storage (public bucket)",
        "url": "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        "official": "https://weatherbench2.readthedocs.io/",
        "license": "Copernicus Climate Data Store License",
        "resolution": "64x32 (5.625° equiangular)",
        "time_step": "6-hourly",
        "variables": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", 
                      "geopotential@500hPa", "temperature@850hPa"],
    },
}


class WeatherBench2Dataset(Dataset):
    """
    Real ERA5 dataset from WeatherBench2 public GCS bucket.
    
    Loads actual reanalysis data for climate emulation training.
    """
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
        years: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            config: Data configuration
            split: One of "train", "val", "test"
            max_samples: Maximum number of samples to use
            years: Override year range (start, end)
            cache_dir: Directory to cache downloaded data
        """
        self.config = config
        self.split = split
        self.lat_size = config.lat_resolution
        self.lon_size = config.lon_resolution
        self.input_steps = config.input_time_steps
        self.forecast_steps = config.forecast_lead_time_hours // config.time_step_hours
        
        n_surface = len(config.surface_variables)
        n_pressure = len(config.pressure_variables) * len(config.pressure_levels)
        self.num_channels = n_surface + n_pressure
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if years is None:
            if split == "train":
                years = (2015, 2018)
            elif split == "val":
                years = (2019, 2019)
            else:
                years = (2020, 2020)
        
        self.years = years
        self._load_data(max_samples)
        
        print(f"WeatherBench2 ERA5 ({split}): {len(self)} samples from {years[0]}-{years[1]}")
    
    def _load_data(self, max_samples: Optional[int] = None):
        """Load ERA5 data from WeatherBench2 GCS bucket."""
        import gcsfs
        import xarray as xr
        
        cache_file = os.path.join(
            self.cache_dir, 
            f"era5_{self.years[0]}_{self.years[1]}_{self.split}.npz"
        )
        
        if os.path.exists(cache_file):
            print(f"  Loading cached data from {cache_file}")
            cached = np.load(cache_file)
            self.data = cached['data']
            self.mean = cached['mean']
            self.std = cached['std']
            return
        
        print(f"  Downloading ERA5 data from WeatherBench2 ({self.years[0]}-{self.years[1]})...")
        
        fs = gcsfs.GCSFileSystem(token='anon')
        store = gcsfs.GCSMap(
            'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
            gcs=fs
        )
        ds = xr.open_zarr(store, consolidated=True)
        
        time_slice = slice(f"{self.years[0]}-01-01", f"{self.years[1]}-12-31")
        ds = ds.sel(time=time_slice)
        
        variables = []
        
        t2m = ds['2m_temperature'].values
        variables.append(t2m)
        
        u10 = ds['10m_u_component_of_wind'].values
        variables.append(u10)
        
        v10 = ds['10m_v_component_of_wind'].values
        variables.append(v10)
        
        level_idx_500 = list(ds.level.values).index(500)
        level_idx_850 = list(ds.level.values).index(850)
        
        z500 = ds['geopotential'].isel(level=level_idx_500).values
        variables.append(z500)
        
        t850 = ds['temperature'].isel(level=level_idx_850).values
        variables.append(t850)
        
        self.data = np.stack(variables, axis=1).astype(np.float32)
        
        # Data comes as (time, channel, lon, lat) = (T, C, 64, 32)
        # Transpose to (time, channel, lat, lon) = (T, C, 32, 64) to match config
        self.data = np.transpose(self.data, (0, 1, 3, 2))
        
        self.mean = self.data.mean(axis=(0, 2, 3), keepdims=True)
        self.std = self.data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        self.data = (self.data - self.mean) / self.std
        
        if max_samples and len(self.data) > max_samples + self.input_steps + self.forecast_steps:
            self.data = self.data[:max_samples + self.input_steps + self.forecast_steps]
        
        print(f"  Caching to {cache_file}")
        np.savez(cache_file, data=self.data, mean=self.mean, std=self.std)
        
        print(f"  Loaded {self.data.shape[0]} timesteps, {self.data.shape[1]} channels")
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.input_steps - self.forecast_steps)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx
        end_idx = start_idx + self.input_steps
        target_idx = end_idx + self.forecast_steps - 1
        
        input_data = self.data[start_idx:end_idx]
        target_data = self.data[target_idx]
        
        return {
            'input': torch.from_numpy(input_data),
            'target': torch.from_numpy(target_data),
            'lead_time': torch.tensor(self.config.forecast_lead_time_hours, dtype=torch.float32)
        }
    
    def get_climatology(self) -> np.ndarray:
        """Return climatological mean for skill score computation."""
        return np.zeros((self.num_channels, self.lon_size, self.lat_size), dtype=np.float32)


class SyntheticClimateDataset(Dataset):
    """Synthetic climate dataset for testing only (not for real experiments)."""
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        num_samples: int = 1000,
        seed: int = 42
    ):
        self.config = config
        self.split = split
        self.num_samples = num_samples
        
        split_offset = {"train": 0, "val": 10000, "test": 20000}
        self.rng = np.random.RandomState(seed + split_offset.get(split, 0))
        
        self.lat_size = config.lat_resolution
        self.lon_size = config.lon_resolution
        
        n_surface = len(config.surface_variables)
        n_pressure = len(config.pressure_variables) * len(config.pressure_levels)
        self.num_channels = n_surface + n_pressure
        
        self.input_steps = config.input_time_steps
        self.forecast_steps = config.forecast_lead_time_hours // config.time_step_hours
        
        self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic data with AR(1) temporal correlation."""
        total_time = self.num_samples + self.input_steps + self.forecast_steps
        
        self.data = np.zeros(
            (total_time, self.num_channels, self.lon_size, self.lat_size),
            dtype=np.float32
        )
        
        ar_coef = 0.9
        for c in range(self.num_channels):
            self.data[0, c] = self.rng.randn(self.lon_size, self.lat_size)
            for t in range(1, total_time):
                innovation = self.rng.randn(self.lon_size, self.lat_size) * np.sqrt(1 - ar_coef**2)
                self.data[t, c] = ar_coef * self.data[t-1, c] + innovation
        
        self.mean = self.data.mean(axis=(0, 2, 3), keepdims=True)
        self.std = self.data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        self.data = (self.data - self.mean) / self.std
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx
        end_idx = start_idx + self.input_steps
        target_idx = end_idx + self.forecast_steps - 1
        
        input_data = self.data[start_idx:end_idx]
        target_data = self.data[target_idx]
        
        return {
            'input': torch.from_numpy(input_data),
            'target': torch.from_numpy(target_data),
            'lead_time': torch.tensor(self.config.forecast_lead_time_hours, dtype=torch.float32)
        }


def create_dataloaders(
    config: Config,
    max_samples: Optional[int] = None,
    use_real_data: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Master configuration
        max_samples: Override number of samples
        use_real_data: If True, use real WeatherBench2 ERA5 data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    num_samples = max_samples or (
        config.experiment.debug_samples if config.experiment.debug else 5000
    )
    
    if use_real_data:
        print("\n=== Using Real WeatherBench2 ERA5 Data ===")
        print(f"Source: {DATASET_CONFIG['weatherbench2_era5']['url']}")
        print(f"License: {DATASET_CONFIG['weatherbench2_era5']['license']}")
        
        train_dataset = WeatherBench2Dataset(
            config.data, "train", max_samples=num_samples,
            years=(2015, 2018)
        )
        val_dataset = WeatherBench2Dataset(
            config.data, "val", max_samples=num_samples // 5,
            years=(2019, 2019)
        )
        test_dataset = WeatherBench2Dataset(
            config.data, "test", max_samples=num_samples // 5,
            years=(2020, 2020)
        )
    else:
        print("\n=== Using Synthetic Data (testing only) ===")
        train_dataset = SyntheticClimateDataset(
            config.data, "train", num_samples=num_samples, seed=config.experiment.seed
        )
        val_dataset = SyntheticClimateDataset(
            config.data, "val", num_samples=num_samples // 5, seed=config.experiment.seed
        )
        test_dataset = SyntheticClimateDataset(
            config.data, "test", num_samples=num_samples // 5, seed=config.experiment.seed
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.resources.num_workers,
        pin_memory=config.resources.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.resources.num_workers,
        pin_memory=config.resources.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.resources.num_workers,
        pin_memory=config.resources.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import get_config
    
    config = get_config(debug=True)
    
    print("Testing WeatherBench2 ERA5 dataset loading...")
    train_loader, val_loader, test_loader = create_dataloaders(config, max_samples=100, use_real_data=True)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
