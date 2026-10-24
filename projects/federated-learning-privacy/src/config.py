"""Configuration for federated learning experiments."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import json
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    num_classes: int = 10
    dropout: float = 0.2
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "hidden_dims": "[128, 64] lightweight MLP for CIFAR-10/MNIST",
        "num_classes": "10 classes for CIFAR-10",
        "dropout": "0.2 standard regularization"
    }, repr=False)


@dataclass
class FederatedConfig:
    num_clients: int = 10
    clients_per_round: int = 5
    num_rounds: int = 20
    local_epochs: int = 2
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "num_clients": "10 clients standard for FL experiments (McMahan et al., 2017)",
        "clients_per_round": "50% client participation per round",
        "num_rounds": "20 communication rounds sufficient for convergence",
        "local_epochs": "2 local epochs balances communication and computation"
    }, repr=False)


@dataclass
class DPConfig:
    """Differential Privacy configuration."""
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "epsilon": "ε=8.0 provides moderate privacy (Abadi et al., 2016)",
        "delta": "δ=1e-5 standard for DP-SGD",
        "max_grad_norm": "1.0 gradient clipping for bounded sensitivity",
        "noise_multiplier": "σ=1.1 noise scale for privacy budget"
    }, repr=False)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "batch_size": "32 standard for local training",
        "learning_rate": "0.01 for SGD on CIFAR-10",
        "weight_decay": "1e-4 L2 regularization"
    }, repr=False)


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    iid: bool = False
    alpha: float = 0.5  # Dirichlet concentration for non-IID
    samples_per_client: int = 500
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "dataset": "CIFAR-10 standard FL benchmark",
        "iid": "False for realistic non-IID setting",
        "alpha": "0.5 Dirichlet for moderate heterogeneity (Hsu et al., 2019)",
        "samples_per_client": "500 samples per client for fast iteration"
    }, repr=False)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    
    @classmethod
    def default(cls): return cls()
    
    @classmethod
    def small_debug(cls):
        config = cls()
        config.federated.num_clients = 5
        config.federated.clients_per_round = 3
        config.federated.num_rounds = 3
        config.federated.local_epochs = 1
        config.data.samples_per_client = 50
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
