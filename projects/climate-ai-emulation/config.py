"""
Configuration file for Climate AI Emulation project.

All hyperparameters are centralized here with detailed justifications based on:
- WeatherBench benchmark (Rasp et al., 2020)
- Pangu-Weather (Bi et al., 2023)
- GraphCast (Lam et al., 2023)
- FourCastNet (Pathak et al., 2022)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Data source: WeatherBench or synthetic
    data_source: str = "synthetic"  # Options: "weatherbench", "era5", "synthetic"
    
    # Spatial resolution
    # WeatherBench provides 5.625° (~600km), 2.8125° (~300km), 1.40625° (~150km)
    # We use coarse resolution for computational efficiency in debug mode
    lat_resolution: int = 32  # Number of latitude points
    lon_resolution: int = 64  # Number of longitude points
    
    # Temporal settings
    # ERA5 provides hourly data; WeatherBench uses 6-hourly
    # 6-hour intervals are standard for NWP comparison (Rasp et al., 2020)
    time_step_hours: int = 6
    input_time_steps: int = 2  # Past time steps for input (following WeatherBench)
    forecast_lead_time_hours: int = 72  # 3-day forecast (common benchmark horizon)
    
    # Variables (following WeatherBench core variables)
    # Z500: geopotential at 500hPa - key for synoptic-scale patterns
    # T850: temperature at 850hPa - important for surface weather
    # T2m: 2m temperature - direct surface impact
    # U10, V10: 10m wind components
    surface_variables: List[str] = field(default_factory=lambda: ["t2m", "u10", "v10"])
    pressure_levels: List[int] = field(default_factory=lambda: [500, 850])  # hPa
    pressure_variables: List[str] = field(default_factory=lambda: ["z", "t"])
    
    # Number of input channels = surface_vars + pressure_vars * pressure_levels
    # = 3 + 2*2 = 7 channels
    
    # Data splits (following ML best practices)
    train_years: Tuple[int, int] = (1979, 2015)  # 37 years training
    val_years: Tuple[int, int] = (2016, 2017)    # 2 years validation
    test_years: Tuple[int, int] = (2018, 2020)   # 3 years test
    
    # Normalization
    # Per-variable standardization is standard in weather/climate ML
    normalize: bool = True
    normalization_type: str = "standard"  # Options: "standard", "minmax"
    
    # Data augmentation (spatial only - temporal augmentation can break physics)
    augment_flip: bool = False  # Horizontal flip - careful with Coriolis
    augment_rotate: bool = False  # Random rotation - breaks lat/lon grid


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Base architecture
    # ConvLSTM: Good for spatiotemporal, used in early weather ML
    # U-Net: Encoder-decoder with skip connections, used in Pangu-Weather
    # Vision Transformer: Used in FourCastNet
    architecture: str = "unet"  # Options: "convlstm", "unet", "simple_cnn"
    
    # U-Net specific parameters
    # Following medical imaging U-Net with adaptations for weather data
    # Note: Reduced for debug mode; production would use 64 base channels, depth 5
    unet_base_channels: int = 16  # Start with 16 in debug, 32-64 in production
    unet_depth: int = 3  # Number of encoder/decoder levels (3 for debug, 4-5 for production)
    unet_use_attention: bool = True  # Attention gates (Oktay et al., 2018)
    
    # ConvLSTM specific parameters
    # Based on Shi et al., 2015 precipitation nowcasting
    convlstm_hidden_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    convlstm_kernel_size: int = 3
    convlstm_num_layers: int = 3
    
    # Common parameters
    input_channels: int = 7  # Computed from DataConfig
    output_channels: int = 7  # Same variables for autoregressive prediction
    
    # Activation function
    # GELU preferred in transformers and modern architectures
    activation: str = "gelu"  # Options: "relu", "gelu", "silu"
    
    # Normalization layers
    # Instance norm often better for spatial data than batch norm
    normalization: str = "instance"  # Options: "batch", "instance", "group", "layer"
    
    # Dropout for regularization
    # Also enables MC Dropout for uncertainty quantification
    dropout_rate: float = 0.1
    
    # Residual connections (critical for weather prediction)
    # Weather changes slowly; residual learning captures anomalies
    use_residual: bool = True


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""
    
    # Method selection
    # MC Dropout: Simple, no architecture change (Gal & Ghahramani, 2016)
    # Deep Ensemble: Better calibration but 5x compute (Lakshminarayanan et al., 2017)
    # Heteroscedastic: Predict mean and variance (Kendall & Gal, 2017)
    method: str = "mc_dropout"  # Options: "mc_dropout", "ensemble", "heteroscedastic"
    
    # MC Dropout parameters
    mc_samples: int = 20  # Number of forward passes with dropout
    # 20 samples provides good balance of quality vs compute
    # Increasing to 50-100 only marginally improves calibration
    
    # Deep Ensemble parameters
    ensemble_size: int = 5  # Standard in literature
    # Increasing beyond 5 has diminishing returns (Fort et al., 2019)
    
    # Heteroscedastic output
    # Model outputs both mean and log-variance
    predict_variance: bool = False
    variance_activation: str = "softplus"  # Ensures positive variance
    min_variance: float = 1e-6  # Numerical stability
    
    # Calibration
    # Temperature scaling for post-hoc calibration
    use_temperature_scaling: bool = True
    temperature_init: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Batch size
    # WeatherBench uses 32; limited by GPU memory for high-res data
    # GraphCast uses 1 due to massive model size
    batch_size: int = 8  # Reduced for debug mode
    
    # Learning rate
    # Cosine schedule common in weather ML (FourCastNet, Pangu-Weather)
    # AdamW with 1e-4 is a safe starting point
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6  # For cosine annealing
    
    # Learning rate schedule
    scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    warmup_epochs: int = 5  # Gradual warmup prevents early divergence
    
    # Optimizer
    optimizer: str = "adamw"  # AdamW standard for transformers and modern CNNs
    weight_decay: float = 1e-5  # Mild regularization
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Training duration
    epochs: int = 100
    early_stopping_patience: int = 15  # Stop if no improvement
    
    # Loss function
    # MSE standard for regression; can weight by variable importance
    loss: str = "mse"  # Options: "mse", "mae", "huber", "weighted_mse"
    
    # Gradient clipping (important for RNNs and transformers)
    gradient_clip_val: float = 1.0
    
    # Mixed precision training
    # Reduces memory and speeds up training on modern GPUs
    use_amp: bool = True
    
    # Checkpointing
    save_top_k: int = 3  # Keep best 3 checkpoints
    checkpoint_metric: str = "val_rmse"


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration."""
    
    # Core metrics (following WeatherBench)
    # RMSE: Root Mean Squared Error - standard regression metric
    # ACC: Anomaly Correlation Coefficient - skill vs climatology
    # SSIM: Structural similarity - spatial pattern quality
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "acc", "ssim"])
    
    # Lead times to evaluate (hours)
    # WeatherBench evaluates at 72h, 120h (3-day, 5-day)
    eval_lead_times: List[int] = field(default_factory=lambda: [6, 24, 72])
    
    # Uncertainty metrics
    # CRPS: Continuous Ranked Probability Score - proper scoring rule
    # Calibration: Reliability diagram metrics
    # Sharpness: Width of prediction intervals
    uncertainty_metrics: List[str] = field(default_factory=lambda: [
        "crps", "calibration_error", "sharpness", "coverage_90"
    ])
    
    # Climatology baseline
    # Used to compute skill scores
    use_climatology_baseline: bool = True
    climatology_years: Tuple[int, int] = (1979, 2010)


@dataclass
class ResourceConfig:
    """Resource monitoring and constraints."""
    
    # GPU settings
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    num_gpus: int = 1
    
    # Memory limits
    max_gpu_memory_gb: float = 8.0  # Limit GPU memory usage
    
    # CPU settings
    num_workers: int = 0  # DataLoader workers (0 for debug to avoid overhead)
    pin_memory: bool = False  # Disable for CPU
    
    # Monitoring
    log_memory_usage: bool = True
    log_compute_time: bool = True
    profile_first_epoch: bool = True
    
    # Quantization (for deployment)
    quantize_int8: bool = False  # Enable INT8 quantization
    quantization_backend: str = "fbgemm"  # Options: "fbgemm", "qnnpack"


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    
    # Experiment name
    name: str = "climate_emulation_v1"
    
    # Debug mode (reduced data and epochs)
    debug: bool = True
    debug_samples: int = 50  # Number of samples in debug mode
    debug_epochs: int = 3  # Reduced for faster testing on CPU
    
    # Logging
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "climate-ai-emulation"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Output
    output_dir: str = "outputs"
    save_predictions: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Compute input channels from data config
        n_surface = len(self.data.surface_variables)
        n_pressure = len(self.data.pressure_variables) * len(self.data.pressure_levels)
        self.model.input_channels = n_surface + n_pressure
        self.model.output_channels = self.model.input_channels
        
        # Adjust for debug mode
        if self.experiment.debug:
            self.training.epochs = self.experiment.debug_epochs
            self.training.batch_size = min(self.training.batch_size, 4)
            self.uncertainty.mc_samples = min(self.uncertainty.mc_samples, 5)
            self.uncertainty.ensemble_size = min(self.uncertainty.ensemble_size, 2)
        
        # Create directories
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.output_dir, exist_ok=True)


def get_config(debug: bool = True) -> Config:
    """Get configuration with optional debug mode override."""
    config = Config()
    config.experiment.debug = debug
    config.__post_init__()
    return config


# Literature References for Hyperparameter Choices:
"""
[1] Rasp, S., et al. (2020). WeatherBench: A benchmark dataset for data-driven 
    weather forecasting. Journal of Advances in Modeling Earth Systems.
    - Establishes standard variables, lead times, and metrics for weather ML
    - Uses Z500, T850 as primary variables; 72h and 120h lead times
    
[2] Bi, K., et al. (2023). Pangu-Weather: A 3D High-Resolution Model for Fast 
    and Accurate Global Weather Forecast. Nature.
    - Uses 3D Earth-specific transformer with hierarchical temporal aggregation
    - 24 vertical levels, 0.25° resolution, 13 surface + 5 upper-air variables
    
[3] Lam, R., et al. (2023). GraphCast: Learning skillful medium-range global 
    weather forecasting. Science.
    - Graph neural network on icosahedral mesh
    - 37 vertical levels, 0.25° resolution, trained on ERA5
    - Uses autoregressive multi-step training
    
[4] Pathak, J., et al. (2022). FourCastNet: A Global Data-driven High-resolution 
    Weather Model using Adaptive Fourier Neural Operators. arXiv.
    - Adaptive Fourier Neural Operator (AFNO) architecture
    - 0.25° resolution, 20 variables
    
[5] Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: 
    Representing Model Uncertainty in Deep Learning. ICML.
    - MC Dropout for uncertainty quantification
    
[6] Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive 
    Uncertainty Estimation using Deep Ensembles. NeurIPS.
    - Deep ensembles for uncertainty; 5 members is sufficient
"""
