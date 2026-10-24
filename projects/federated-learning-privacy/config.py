"""
Configuration for Privacy-Preserving Federated Learning with Heterogeneous Clients.

Hyperparameters are organized by category with justifications based on literature.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """Data configuration for federated learning experiments."""
    
    dataset: str = "cifar10"
    # CIFAR-10: 60,000 images (50k train, 10k test), 10 classes, 32x32 RGB
    # FEMNIST: ~800,000 images, 62 classes, 28x28 grayscale, naturally federated
    
    num_clients: int = 10
    # Literature uses 100-1000 clients, but 10-20 common for debugging
    # McMahan et al. 2017 used 100-10,000 clients
    
    participation_rate: float = 1.0
    # Fraction of clients participating per round
    # Typical: 0.1-1.0; higher for debugging, lower for realistic simulation
    
    non_iid_alpha: float = 0.5
    # Dirichlet distribution parameter for non-IID data partitioning
    # alpha=0.5: highly non-IID (label skew)
    # alpha=1.0: moderately non-IID
    # alpha=100: approximately IID
    # Li et al. 2020 (FedProx) used alpha=0.5 for heterogeneous experiments
    
    batch_size: int = 32
    # Local batch size for client training
    # McMahan et al. 2017 used B=10-50
    
    test_batch_size: int = 128
    # Larger batch for faster evaluation
    
    data_dir: str = "./data"
    # Directory to store downloaded datasets


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    model_name: str = "cnn"
    # Options: "cnn" (small CNN), "resnet18", "mlp"
    # CNN: ~100k params, suitable for CIFAR-10/FEMNIST
    # ResNet18: ~11M params, for larger experiments
    
    num_classes: int = 10
    # CIFAR-10: 10, FEMNIST: 62
    
    hidden_dim: int = 256
    # Hidden dimension for MLP/CNN layers
    
    dropout: float = 0.2
    # Dropout rate for regularization


@dataclass
class TrainingConfig:
    """Training configuration for federated learning."""
    
    num_rounds: int = 50
    # Number of communication rounds
    # McMahan et al. 2017: 100-2000 rounds
    # Reduced for debugging (10-50)
    
    local_epochs: int = 5
    # Number of local training epochs per round
    # McMahan et al. 2017: E=1-20, commonly E=5
    # FedProx: E=20 for heterogeneous settings
    
    learning_rate: float = 0.01
    # Client learning rate
    # McMahan et al. 2017: 0.01-0.1
    # Lower rates (0.001-0.01) with DP for stability
    
    momentum: float = 0.9
    # SGD momentum, standard value
    
    weight_decay: float = 1e-4
    # L2 regularization, prevents overfitting
    
    lr_decay: float = 0.995
    # Learning rate decay per round
    # Gradual decay helps convergence in later rounds
    
    optimizer: str = "sgd"
    # Options: "sgd", "adam"
    # SGD preferred in FL for theoretical guarantees
    
    seed: int = 42
    # Random seed for reproducibility


@dataclass
class FedProxConfig:
    """FedProx-specific configuration."""
    
    mu: float = 0.01
    # Proximal term coefficient
    # Li et al. 2020: mu=0.001-1.0
    # Higher mu: more regularization toward global model
    # mu=0.01 is a good default for moderate heterogeneity


@dataclass
class DifferentialPrivacyConfig:
    """Differential privacy configuration."""
    
    enabled: bool = True
    # Whether to apply DP during training
    
    epsilon: float = 8.0
    # Privacy budget (total)
    # epsilon=1-10: strong privacy
    # epsilon=8: reasonable privacy-utility tradeoff (Abadi et al. 2016)
    # Lower epsilon = more privacy = more noise = lower accuracy
    
    delta: float = 1e-5
    # Probability of privacy breach
    # Standard: 1/n where n is dataset size
    # delta < 1/|D| where |D| is total dataset size
    
    max_grad_norm: float = 1.0
    # Gradient clipping bound (C)
    # Abadi et al. 2016: C=1.0-4.0
    # Lower C: tighter privacy but may clip useful gradients
    
    noise_multiplier: float = 1.0
    # Noise multiplier (sigma = noise_multiplier * max_grad_norm)
    # Higher = more privacy, lower accuracy
    # Typical: 0.5-2.0 depending on epsilon target
    # Calculated based on epsilon using privacy accounting
    
    secure_mode: bool = False
    # Use secure aggregation (simulated)
    # In practice, uses cryptographic protocols
    
    accountant: str = "rdp"
    # Privacy accountant type
    # Options: "rdp" (Rényi DP), "gdp" (Gaussian DP), "prv"
    # RDP provides tighter bounds for Gaussian mechanism


@dataclass
class PersonalizationConfig:
    """Personalization configuration for heterogeneous clients."""
    
    enabled: bool = True
    # Enable personalized aggregation
    
    method: str = "weighted"
    # Personalization method:
    # "weighted": Weighted average based on local performance
    # "local_adapt": Local adaptation layers (keep some layers local)
    # "mixture": Mixture of global and local models
    
    local_layers: List[str] = field(default_factory=lambda: ["fc", "classifier"])
    # Layers to keep personalized (not aggregated)
    # Typically last layers for task-specific adaptation
    
    mixing_weight: float = 0.5
    # Weight for local model in mixture (1-alpha for global)
    # Higher = more personalization
    
    adaptation_steps: int = 5
    # Local adaptation steps after receiving global model


@dataclass
class ExperimentConfig:
    """Experiment and logging configuration."""
    
    experiment_name: str = "federated_dp_experiment"
    
    results_dir: str = "./results"
    
    log_interval: int = 1
    # Log every N rounds
    
    eval_interval: int = 5
    # Evaluate global model every N rounds
    
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    
    wandb_enabled: bool = False
    wandb_project: str = "federated-learning-privacy"
    
    debug_mode: bool = True
    # Debug mode: fewer clients, fewer rounds
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_workers: int = 2
    # DataLoader workers


@dataclass
class QuantizationConfig:
    """Quantization configuration for model compression."""
    
    enabled: bool = True
    # Enable INT8 quantization comparison
    
    method: str = "dynamic"
    # Options: "dynamic", "static", "qat" (quantization-aware training)
    # Dynamic: no calibration needed, runtime quantization
    # Static: requires calibration data
    
    backend: str = "fbgemm"
    # Quantization backend: "fbgemm" (x86), "qnnpack" (ARM)


@dataclass
class Config:
    """Main configuration aggregating all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fedprox: FedProxConfig = field(default_factory=FedProxConfig)
    dp: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    def to_dict(self):
        """Convert config to dictionary for logging."""
        import dataclasses
        return {
            "data": dataclasses.asdict(self.data),
            "model": dataclasses.asdict(self.model),
            "training": dataclasses.asdict(self.training),
            "fedprox": dataclasses.asdict(self.fedprox),
            "dp": dataclasses.asdict(self.dp),
            "personalization": dataclasses.asdict(self.personalization),
            "experiment": dataclasses.asdict(self.experiment),
            "quantization": dataclasses.asdict(self.quantization),
        }


def get_debug_config() -> Config:
    """Get configuration for debug/testing mode."""
    config = Config()
    config.data.num_clients = 3
    config.data.batch_size = 64
    config.training.num_rounds = 3
    config.training.local_epochs = 1
    config.experiment.debug_mode = True
    config.experiment.eval_interval = 1
    config.experiment.checkpoint_interval = 3
    config.experiment.num_workers = 0  # Faster for small runs
    return config


def get_full_config() -> Config:
    """Get configuration for full experiments."""
    config = Config()
    config.data.num_clients = 100
    config.training.num_rounds = 200
    config.training.local_epochs = 5
    config.experiment.debug_mode = False
    return config
