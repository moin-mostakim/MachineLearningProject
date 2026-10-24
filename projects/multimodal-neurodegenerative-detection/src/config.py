"""Configuration module for multimodal neurodegenerative disease detection."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class ImageEncoderConfig:
    """Image encoder configuration."""
    model_name: str = "microsoft/resnet-18"
    hidden_dim: int = 512
    pretrained: bool = True
    freeze_backbone: bool = True
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "model_name": "ResNet-18: lightweight CNN suitable for medical imaging, good CPU performance",
        "hidden_dim": "512 matches ResNet-18 output, sufficient for fusion",
        "pretrained": "ImageNet pretraining provides useful low-level features",
        "freeze_backbone": "Freeze to reduce trainable params and prevent overfitting on small datasets"
    }, repr=False)


@dataclass
class ClinicalEncoderConfig:
    """Clinical/tabular data encoder configuration."""
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 32
    dropout: float = 0.3
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "input_dim": "10 clinical features typical for Alzheimer's datasets",
        "hidden_dims": "[64, 32] - two-layer MLP for tabular encoding",
        "output_dim": "32 matches fusion requirements",
        "dropout": "0.3 higher dropout for tabular to prevent memorization"
    }, repr=False)


@dataclass
class FusionConfig:
    """Multimodal fusion configuration."""
    fusion_type: str = "attention"
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.2
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "fusion_type": "Attention-based fusion learns modality importance dynamically",
        "hidden_dim": "128 balances capacity and efficiency for small datasets",
        "num_heads": "4 heads for multi-view attention without excessive params",
        "dropout": "0.2 standard for attention layers"
    }, repr=False)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "name": "AdamW standard for deep learning with proper weight decay",
        "learning_rate": "1e-4 conservative for medical data, prevents overfitting",
        "weight_decay": "0.01 regularization for generalization"
    }, repr=False)


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 20
    batch_size: int = 16
    num_workers: int = 0
    early_stopping_patience: int = 5
    seed: int = 42
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "num_epochs": "20 epochs sufficient for small medical datasets with early stopping",
        "batch_size": "16 fits CPU memory while providing stable gradients",
        "num_workers": "0 for CPU compatibility and debugging",
        "early_stopping_patience": "5 epochs patience prevents overfitting"
    }, repr=False)


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "alzheimers_synthetic"
    image_size: int = 64
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    max_samples: int = 500
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "dataset_name": "Synthetic Alzheimer's-style data; real ADNI requires DUA",
        "image_size": "64x64 for CPU efficiency; real runs use 224x224",
        "max_samples": "500 samples for quick iteration; scale up for full training"
    }, repr=False)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    clinical_encoder: ClinicalEncoderConfig = field(default_factory=ClinicalEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    num_classes: int = 4  # Non/Very Mild/Mild/Moderate Demented (Falah/Alzheimer_MRI)
    
    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls()
    
    @classmethod
    def small_debug(cls) -> "ExperimentConfig":
        config = cls()
        config.data.max_samples = 100
        config.data.image_size = 32
        config.training.num_epochs = 3
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
    
    def get_hyperparameter_table(self) -> str:
        lines = [
            "| Category | Hyperparameter | Value | Justification |",
            "|----------|---------------|-------|---------------|"
        ]
        
        categories = [
            ("Image Encoder", self.image_encoder, self.image_encoder._justifications),
            ("Clinical Encoder", self.clinical_encoder, self.clinical_encoder._justifications),
            ("Fusion", self.fusion, self.fusion._justifications),
            ("Optimizer", self.training.optimizer, self.training.optimizer._justifications),
            ("Training", self.training, self.training._justifications),
            ("Data", self.data, self.data._justifications),
        ]
        
        for cat_name, config, justifications in categories:
            for field_name, value in asdict(config).items():
                if field_name.startswith('_') or field_name == 'optimizer':
                    continue
                if isinstance(value, (dict, list)):
                    value = str(value)
                justification = justifications.get(field_name, "Standard default")
                lines.append(f"| {cat_name} | `{field_name}` | `{value}` | {justification} |")
        
        return "\n".join(lines)
    
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
