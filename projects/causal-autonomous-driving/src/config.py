"""Configuration for causal representation learning experiments."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import json
from pathlib import Path


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    feature_dim: int = 512
    num_classes: int = 5  # Discretized steering angles (5 bins from -45° to +45°)
    dropout: float = 0.3
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "backbone": "ResNet-18: lightweight CNN suitable for driving images, CPU-friendly",
        "feature_dim": "512 matches ResNet-18 output dimension",
        "num_classes": "4 discrete steering actions (left, slight-left, slight-right, right)",
        "dropout": "0.3 for regularization against distribution shift"
    }, repr=False)


@dataclass
class IRMConfig:
    """Invariant Risk Minimization config (Arjovsky et al., 2019)."""
    penalty_weight: float = 1e4
    penalty_anneal_iters: int = 100
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "penalty_weight": "1e4 from Arjovsky et al. (2019) for strong invariance",
        "penalty_anneal_iters": "100 iterations warmup before applying IRM penalty"
    }, repr=False)


@dataclass
class TrainingConfig:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_environments: int = 3  # Different weather/lighting conditions
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "num_epochs": "20 epochs for convergence on small driving datasets",
        "batch_size": "32 balances gradient stability and memory",
        "learning_rate": "1e-3 standard for CNN training",
        "weight_decay": "1e-4 L2 regularization",
        "num_environments": "3 environments (clear, rain, night) for IRM training"
    }, repr=False)


@dataclass
class DataConfig:
    image_size: int = 64
    num_samples_per_env: int = 500
    train_split: float = 0.7
    val_split: float = 0.15
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "image_size": "64x64 for CPU efficiency; production uses 224x224",
        "num_samples_per_env": "500 samples per environment for quick training",
        "train_split": "70% train standard split"
    }, repr=False)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    irm: IRMConfig = field(default_factory=IRMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    
    @classmethod
    def default(cls): return cls()
    
    @classmethod
    def small_debug(cls):
        config = cls()
        config.data.num_samples_per_env = 50
        config.data.image_size = 16
        config.training.num_epochs = 2
        config.training.batch_size = 8
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        def filter_dict(d):
            if isinstance(d, dict):
                return {k: filter_dict(v) for k, v in d.items() if not k.startswith('_')}
            elif hasattr(d, '__dataclass_fields__'):
                return filter_dict(asdict(d))
            return d
        return filter_dict(asdict(self))
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
