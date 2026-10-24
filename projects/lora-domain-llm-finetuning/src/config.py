"""Configuration module for LoRA fine-tuning experiments.

All hyperparameters are centralized here with justifications for each choice.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "distilgpt2"
    max_length: int = 128
    pad_token: str = "<|pad|>"
    
    # Justifications
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "model_name": "DistilGPT-2 (82M params) chosen for fast iteration; scales to larger models",
        "max_length": "128 tokens balances context length with memory/speed for CPU training",
        "pad_token": "Standard padding token for batch processing"
    }, repr=False)


@dataclass
class LoRAConfig:
    """LoRA-specific configuration.
    
    References:
    - Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
    - Zhang et al. (2023) "AdaLoRA: Adaptive Budget Allocation", ICLR 2023
    """
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1  # Dropout for regularization
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "r": "Rank 8 from Hu et al. (2022); good balance of capacity vs efficiency",
        "lora_alpha": "Alpha=2*r following Hu et al. (2022) recommendation for stable training",
        "lora_dropout": "0.1 dropout prevents overfitting on small datasets (standard practice)",
        "target_modules": "Attention projections (c_attn, c_proj) most impactful per Hu et al.",
        "bias": "'none' - training bias adds minimal benefit per ablations in LoRA paper"
    }, repr=False)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "AdamW"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "name": "AdamW standard for transformer fine-tuning (Loshchilov & Hutter, 2019)",
        "learning_rate": "2e-4 from Hu et al. (2022) LoRA experiments; higher than full FT",
        "weight_decay": "0.01 standard regularization for AdamW",
        "beta1": "0.9 default, works well for language models",
        "beta2": "0.999 default for Adam-family optimizers",
        "eps": "1e-8 for numerical stability"
    }, repr=False)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine_with_warmup"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "name": "Cosine annealing with warmup standard for LLM fine-tuning",
        "warmup_steps": "100 steps (~10% of training) for stable early training",
        "warmup_ratio": "Alternative: 10% of total steps for warmup",
        "min_lr_ratio": "Decay to 10% of peak LR by end of training"
    }, repr=False)


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 32  # batch_size * gradient_accumulation_steps
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42
    fp16: bool = False  # Disabled for CPU compatibility
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "num_epochs": "3 epochs sufficient for LoRA convergence per Hu et al. (2022)",
        "batch_size": "8 fits in CPU/single GPU memory",
        "gradient_accumulation_steps": "4 steps gives effective batch size of 32",
        "effective_batch_size": "32 balances gradient noise and training speed",
        "max_grad_norm": "1.0 gradient clipping prevents instability",
        "seed": "42 for reproducibility"
    }, repr=False)
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_train_samples: Optional[int] = 1000
    max_val_samples: Optional[int] = 200
    max_test_samples: Optional[int] = 200
    
    _justifications: Dict[str, str] = field(default_factory=lambda: {
        "dataset_name": "WikiText-2 standard LM benchmark (Merity et al., 2017)",
        "max_train_samples": "1000 samples for fast iteration; scale up for full runs",
        "max_val_samples": "200 samples sufficient for validation signal",
        "max_test_samples": "200 samples for benchmark evaluation"
    }, repr=False)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with all hyperparameters."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls()
    
    @classmethod
    def small_debug(cls) -> "ExperimentConfig":
        """Minimal config for quick testing."""
        config = cls()
        config.data.max_train_samples = 100
        config.data.max_val_samples = 50
        config.data.max_test_samples = 50
        config.training.num_epochs = 1
        config.training.batch_size = 4
        config.model.max_length = 64
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding justifications."""
        def filter_dict(d):
            if isinstance(d, dict):
                return {k: filter_dict(v) for k, v in d.items() 
                       if not k.startswith('_')}
            elif hasattr(d, '__dataclass_fields__'):
                return filter_dict(asdict(d))
            else:
                return d
        return filter_dict(asdict(self))
    
    def get_hyperparameter_table(self) -> str:
        """Generate markdown table of all hyperparameters with justifications."""
        lines = [
            "| Category | Hyperparameter | Value | Justification |",
            "|----------|---------------|-------|---------------|"
        ]
        
        categories = [
            ("Model", self.model, self.model._justifications),
            ("LoRA", self.lora, self.lora._justifications),
            ("Optimizer", self.training.optimizer, self.training.optimizer._justifications),
            ("Scheduler", self.training.scheduler, self.training.scheduler._justifications),
            ("Training", self.training, self.training._justifications),
            ("Data", self.data, self.data._justifications),
        ]
        
        for cat_name, config, justifications in categories:
            for field_name, value in asdict(config).items():
                if field_name.startswith('_') or field_name in ['optimizer', 'scheduler']:
                    continue
                if isinstance(value, (dict, list)):
                    value = str(value)
                justification = justifications.get(field_name, "Standard default")
                lines.append(f"| {cat_name} | `{field_name}` | `{value}` | {justification} |")
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        if 'model' in data:
            for k, v in data['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if 'lora' in data:
            for k, v in data['lora'].items():
                if hasattr(config.lora, k):
                    setattr(config.lora, k, v)
        return config
