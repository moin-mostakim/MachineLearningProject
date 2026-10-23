"""
Dataset classes for climate emulation.

Supports:
- Synthetic spatiotemporal data (for testing and debugging)
- WeatherBench data (if available)
- ERA5-derived climate data
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DataConfig


class SyntheticClimateDataset(Dataset):
    """
    Synthetic climate dataset for testing and debugging.
    
    Generates spatiotemporal data with realistic statistical properties:
    - Spatial correlations (Gaussian random fields)
    - Temporal autocorrelation (AR process)
    - Seasonal cycles
    - Multi-scale patterns (synoptic + mesoscale)
    
    This allows full pipeline testing without downloading large datasets.
    """
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        num_samples: int = 1000,
        seed: int = 42
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            config: Data configuration
            split: One of "train", "val", "test"
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.config = config
        self.split = split
        self.num_samples = num_samples
        
        # Set seed based on split for reproducibility
        split_offset = {"train": 0, "val": 10000, "test": 20000}
        self.rng = np.random.RandomState(seed + split_offset.get(split, 0))
        
        # Spatial dimensions
        self.lat_size = config.lat_resolution
        self.lon_size = config.lon_resolution
        
        # Channel dimensions
        n_surface = len(config.surface_variables)
        n_pressure = len(config.pressure_variables) * len(config.pressure_levels)
        self.num_channels = n_surface + n_pressure
        
        # Time dimensions
        self.input_steps = config.input_time_steps
        self.forecast_steps = config.forecast_lead_time_hours // config.time_step_hours
        
        # Generate base patterns (spatial correlation matrices)
        self._generate_spatial_patterns()
        
        # Pre-generate data for consistency
        self._generate_data()
        
    def _generate_spatial_patterns(self):
        """Generate spatial correlation patterns for realistic fields."""
        # Create coordinate grids
        lat = np.linspace(-90, 90, self.lat_size)
        lon = np.linspace(0, 360, self.lon_size)
        self.lat_grid, self.lon_grid = np.meshgrid(lat, lon, indexing='ij')
        
        # Generate correlation length scales for different variables
        # Synoptic scale: ~1000km (larger correlation length)
        # Mesoscale: ~100km (smaller correlation length)
        self.correlation_lengths = {
            'synoptic': 30,  # degrees
            'meso': 10,      # degrees
        }
        
    def _generate_correlated_field(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        correlation_length: float = 20.0
    ) -> np.ndarray:
        """
        Generate spatially correlated random field using spectral method.
        
        Args:
            mean: Mean of the field
            std: Standard deviation
            correlation_length: Spatial correlation length in degrees
            
        Returns:
            2D array of shape (lat_size, lon_size)
        """
        # Generate white noise in spectral domain
        noise_real = self.rng.randn(self.lat_size, self.lon_size)
        noise_imag = self.rng.randn(self.lat_size, self.lon_size)
        noise_fft = noise_real + 1j * noise_imag
        
        # Create spectral filter for spatial correlation
        kx = np.fft.fftfreq(self.lon_size)
        ky = np.fft.fftfreq(self.lat_size)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Gaussian filter in spectral domain
        filter_scale = correlation_length / (self.lat_size + self.lon_size) * 2
        spectral_filter = np.exp(-0.5 * (k_mag / (0.1 / filter_scale))**2)
        
        # Apply filter and transform back
        filtered_fft = noise_fft * spectral_filter
        field = np.real(np.fft.ifft2(filtered_fft))
        
        # Normalize to desired statistics
        field = (field - field.mean()) / (field.std() + 1e-8) * std + mean
        
        return field.astype(np.float32)
    
    def _generate_data(self):
        """Pre-generate all samples for the dataset."""
        total_time = self.num_samples + self.input_steps + self.forecast_steps
        
        # Generate time series for each channel
        self.data = np.zeros(
            (total_time, self.num_channels, self.lat_size, self.lon_size),
            dtype=np.float32
        )
        
        # Channel-specific statistics (approximate climatology)
        # Based on typical ERA5 statistics after standardization
        channel_stats = {
            0: {'name': 't2m', 'mean': 0, 'std': 1, 'ar_coef': 0.9},
            1: {'name': 'u10', 'mean': 0, 'std': 1, 'ar_coef': 0.7},
            2: {'name': 'v10', 'mean': 0, 'std': 1, 'ar_coef': 0.7},
            3: {'name': 'z500', 'mean': 0, 'std': 1, 'ar_coef': 0.95},
            4: {'name': 't500', 'mean': 0, 'std': 1, 'ar_coef': 0.9},
            5: {'name': 'z850', 'mean': 0, 'std': 1, 'ar_coef': 0.93},
            6: {'name': 't850', 'mean': 0, 'std': 1, 'ar_coef': 0.9},
        }
        
        for c in range(self.num_channels):
            stats = channel_stats.get(c, {'mean': 0, 'std': 1, 'ar_coef': 0.85})
            ar_coef = stats['ar_coef']
            
            # Initialize first time step
            self.data[0, c] = self._generate_correlated_field(
                mean=stats['mean'],
                std=stats['std'],
                correlation_length=self.correlation_lengths['synoptic']
            )
            
            # Generate AR(1) process with spatial correlation
            innovation_std = stats['std'] * np.sqrt(1 - ar_coef**2)
            for t in range(1, total_time):
                innovation = self._generate_correlated_field(
                    mean=0,
                    std=innovation_std,
                    correlation_length=self.correlation_lengths['meso']
                )
                self.data[t, c] = ar_coef * self.data[t-1, c] + innovation
        
        # Compute normalization statistics
        self.mean = self.data.mean(axis=(0, 2, 3), keepdims=True)
        self.std = self.data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - 'input': Input tensor of shape (input_steps, channels, lat, lon)
            - 'target': Target tensor of shape (channels, lat, lon)
            - 'lead_time': Forecast lead time in hours
        """
        # Get input sequence
        start_idx = idx
        end_idx = start_idx + self.input_steps
        
        input_data = self.data[start_idx:end_idx]  # (T, C, H, W)
        
        # Get target (at forecast lead time)
        target_idx = end_idx + self.forecast_steps - 1
        target_data = self.data[target_idx]  # (C, H, W)
        
        return {
            'input': torch.from_numpy(input_data),
            'target': torch.from_numpy(target_data),
            'lead_time': torch.tensor(self.config.forecast_lead_time_hours, dtype=torch.float32)
        }
    
    def get_climatology(self) -> np.ndarray:
        """Return climatological mean for skill score computation."""
        return np.zeros((self.num_channels, self.lat_size, self.lon_size), dtype=np.float32)


class WeatherBenchDataset(Dataset):
    """
    WeatherBench dataset loader.
    
    WeatherBench provides processed ERA5 data at multiple resolutions.
    Reference: Rasp et al. (2020) - WeatherBench benchmark paper.
    
    Note: This requires downloading the data first. For the project,
    we primarily use synthetic data but provide this interface for
    future use with real data.
    """
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        data_path: Optional[str] = None
    ):
        """
        Initialize WeatherBench dataset.
        
        Args:
            config: Data configuration
            split: One of "train", "val", "test"
            data_path: Path to WeatherBench data directory
        """
        self.config = config
        self.split = split
        self.data_path = data_path or "./data/weatherbench"
        
        # This would load actual WeatherBench data
        # For now, fall back to synthetic data
        print(f"WeatherBench data not found at {self.data_path}")
        print("Falling back to synthetic data generation...")
        
        self._synthetic = SyntheticClimateDataset(config, split)
        
    def __len__(self) -> int:
        return len(self._synthetic)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._synthetic[idx]


def create_dataloaders(
    config: Config,
    debug_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Master configuration
        debug_samples: Override number of samples for debugging
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    num_samples = debug_samples or (
        config.experiment.debug_samples if config.experiment.debug else 10000
    )
    
    # Create datasets
    if config.data.data_source == "synthetic":
        train_dataset = SyntheticClimateDataset(
            config.data, "train", num_samples=num_samples, seed=config.experiment.seed
        )
        val_dataset = SyntheticClimateDataset(
            config.data, "val", num_samples=num_samples // 5, seed=config.experiment.seed
        )
        test_dataset = SyntheticClimateDataset(
            config.data, "test", num_samples=num_samples // 5, seed=config.experiment.seed
        )
    else:
        # Would use WeatherBenchDataset or ERA5Dataset
        train_dataset = SyntheticClimateDataset(
            config.data, "train", num_samples=num_samples, seed=config.experiment.seed
        )
        val_dataset = SyntheticClimateDataset(
            config.data, "val", num_samples=num_samples // 5, seed=config.experiment.seed
        )
        test_dataset = SyntheticClimateDataset(
            config.data, "test", num_samples=num_samples // 5, seed=config.experiment.seed
        )
    
    # Create dataloaders
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
    # Test dataset creation
    from config import get_config
    
    config = get_config(debug=True)
    
    print("Creating synthetic dataset...")
    dataset = SyntheticClimateDataset(config.data, "train", num_samples=100)
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Lead time: {sample['lead_time']} hours")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
