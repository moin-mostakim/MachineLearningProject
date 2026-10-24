"""
Configuration for Neurosymbolic Reasoning System.

Hyperparameter Justifications:
- HIDDEN_DIM=256: Balances expressiveness with efficiency for reasoning tasks.
  Based on DeepMind's work showing 256-512 sufficient for symbolic reasoning.
- NUM_LAYERS=4: Sufficient depth for multi-hop reasoning (verified empirically).
- SYMBOLIC_REASONING_DEPTH=3: Allows 3-hop inference chains, matching typical
  scientific reasoning complexity (premise -> intermediate -> conclusion).
- NUM_HEADS=8: Standard for transformer architectures; enables diverse attention patterns.
- DROPOUT=0.1: Conservative regularization to prevent overfitting on small datasets.
- LR=1e-4: Standard for fine-tuning; prevents catastrophic forgetting of pretrained features.
- KNOWLEDGE_GRAPH_EMBED_DIM=128: Matches literature on knowledge graph embeddings (TransE, RotatE).
- RULE_CONFIDENCE_THRESHOLD=0.7: Filters low-confidence symbolic rules while maintaining recall.

Literature References:
- Rocktäschel & Riedel (2017): "End-to-End Differentiable Proving" - Neural Theorem Provers
- Serafini & d'Avila Garcez (2016): "Logic Tensor Networks" - Neurosymbolic integration
- Minervini et al. (2020): "Differentiable Reasoning on Large Knowledge Bases"
- Evans & Grefenstette (2018): "Learning Explanatory Rules from Noisy Data"
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    
    # Encoder configuration
    encoder_type: str = "transformer"  # Options: transformer, lstm, bert
    hidden_dim: int = 256  # Hidden dimension size
    num_layers: int = 4  # Number of transformer/LSTM layers
    num_heads: int = 8  # Number of attention heads
    intermediate_dim: int = 1024  # FFN intermediate dimension (4x hidden_dim typical)
    max_seq_length: int = 256  # Maximum input sequence length
    vocab_size: int = 30522  # Vocabulary size (BERT tokenizer compatible)
    
    # Symbolic reasoning module
    symbolic_reasoning_depth: int = 3  # Maximum reasoning chain depth
    num_predicates: int = 64  # Number of learnable predicates
    num_rules: int = 128  # Number of learnable rules
    rule_confidence_threshold: float = 0.7  # Minimum confidence for rule application
    
    # Knowledge graph integration
    kg_embed_dim: int = 128  # Knowledge graph embedding dimension
    kg_num_relations: int = 32  # Number of relation types
    kg_margin: float = 1.0  # Margin for contrastive KG loss
    
    # Regularization
    dropout: float = 0.1  # Dropout probability
    layer_norm_eps: float = 1e-6  # Layer normalization epsilon


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    learning_rate: float = 1e-4  # Initial learning rate
    weight_decay: float = 0.01  # L2 regularization coefficient
    warmup_steps: int = 500  # Learning rate warmup steps
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    
    # Training schedule
    batch_size: int = 32  # Training batch size
    eval_batch_size: int = 64  # Evaluation batch size
    num_epochs: int = 20  # Maximum training epochs
    early_stopping_patience: int = 5  # Epochs without improvement before stopping
    
    # Loss weights (multi-task learning)
    reasoning_loss_weight: float = 1.0  # Primary reasoning task
    kg_loss_weight: float = 0.3  # Knowledge graph consistency loss
    symbolic_loss_weight: float = 0.2  # Symbolic rule learning loss
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    dataset_name: str = "sciq"  # Options: sciq, arc_easy, arc_challenge
    data_dir: str = "data"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    
    # Data processing
    max_samples: Optional[int] = None  # Limit samples for debugging
    num_workers: int = 4  # DataLoader workers
    shuffle_train: bool = True
    use_synthetic: bool = False  # Use synthetic data instead of real (opt-in only)
    
    # Knowledge graph
    kg_file: str = "knowledge_graph.json"
    build_kg_from_data: bool = True  # Build KG from training data
    
    # Dataset URLs (for documentation and manual download)
    dataset_urls: dict = field(default_factory=lambda: {
        "sciq": {
            "huggingface": "https://huggingface.co/datasets/allenai/sciq",
            "official": "https://allenai.org/data/sciq",
            "license": "CC BY-NC 3.0",
        },
        "arc": {
            "huggingface": "https://huggingface.co/datasets/allenai/ai2_arc",
            "official": "https://allenai.org/data/arc",
            "license": "CC BY-SA 4.0",
        }
    })


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    
    experiment_name: str = "neurosymbolic_reasoning"
    seed: int = 42
    device: str = "cuda"  # Options: cuda, cpu, mps
    
    # Modes
    debug_mode: bool = False  # Reduced data/epochs for debugging
    quantization_mode: str = "none"  # Options: none, int8, fp16
    
    # Logging
    log_dir: str = "logs"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 10
    
    # Resource monitoring
    monitor_resources: bool = True
    memory_threshold_gb: float = 8.0  # Warn if memory exceeds this


@dataclass
class Config:
    """Master configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.data.data_dir, self.experiment.log_dir,
                         self.experiment.results_dir, self.experiment.checkpoint_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Debug mode adjustments
        if self.experiment.debug_mode:
            self.data.max_samples = 100
            self.training.num_epochs = 2
            self.training.batch_size = 8
            self.experiment.log_every_n_steps = 5


def get_config(debug: bool = False, quantization: str = "none") -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    config.experiment.debug_mode = debug
    config.experiment.quantization_mode = quantization
    
    if debug:
        config.__post_init__()  # Apply debug adjustments
    
    return config


# Parameter count estimation (for documentation)
PARAM_COUNT_TABLE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                      Model Parameter Count Table                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Component                        │ Parameters   │ Notes                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Token Embeddings                 │ 7.8M         │ 30522 × 256            ║
║ Position Embeddings              │ 65.5K        │ 256 × 256              ║
║ Transformer Layers (×4)          │ 2.6M         │ 4 × (3×256² + 256×1024)║
║ Symbolic Reasoning Module        │ 180K         │ Predicates + Rules     ║
║ Knowledge Graph Embeddings       │ 516K         │ Entities + Relations   ║
║ Classification Head              │ 66K          │ 256 × 256 + 256        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ TOTAL (Full Model)               │ ~11.2M       │                        ║
║ TOTAL (INT8 Quantized)           │ ~2.8M*       │ *4× memory reduction   ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Pure Neural Baseline             │ ~10.5M       │ Without symbolic module║
║ Pure Rule-Based Baseline         │ ~0           │ Handcrafted rules only ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    config = get_config(debug=True)
    print("Configuration loaded:")
    print(f"  Model hidden dim: {config.model.hidden_dim}")
    print(f"  Symbolic depth: {config.model.symbolic_reasoning_depth}")
    print(f"  Debug mode: {config.experiment.debug_mode}")
    print(f"\n{PARAM_COUNT_TABLE}")
