"""
Configuration file for Generative AI for Inverse Design of Functional Materials.

Hyperparameter Justifications:
-----------------------------
1. LATENT_DIM = 64: Standard for molecular VAEs (Junction Tree VAE uses 56).
   Balances expressiveness vs. overfitting for ~134k QM9 molecules.

2. HIDDEN_DIM = 256: Following GraphVAE architecture recommendations.
   Sufficient capacity for learning molecular representations.

3. NUM_LAYERS = 3: Common choice for GNN-based molecular encoders.
   Allows 3-hop message passing, covering most molecular substructures.

4. BATCH_SIZE = 128: Standard for molecular datasets. QM9 molecules are
   small (~9 heavy atoms avg), allowing larger batches.

5. LEARNING_RATE = 1e-3: Adam optimizer default, works well for VAEs.
   With warmup schedule to stabilize KL divergence training.

6. KL_WEIGHT = 0.1: Beta-VAE style weighting. Prevents posterior collapse
   while maintaining reconstruction quality. (Higgins et al., 2017)

7. NUM_EPOCHS = 100: Sufficient for convergence on QM9.
   Early stopping prevents overfitting.

8. NUM_OBJECTIVES = 3: Multi-objective optimization targets.
   QM9 properties: HOMO, LUMO, Gap (electronically relevant).

9. PARETO_POPULATION = 100: NSGA-II population size for multi-objective
   optimization. Balance between diversity and computational cost.

Literature References:
---------------------
- Junction Tree VAE (Jin et al., 2018): Uses hierarchical latent space
  for molecular generation with 100% validity.
- GraphVAE (Simonovsky & Komodakis, 2018): Direct graph generation
  approach with node/edge prediction.
- SELFIES (Krenn et al., 2020): String representation ensuring
  100% valid molecule generation.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "QM9"
    data_root: str = "./data/qm9"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    max_atoms: int = 29  # QM9 max atoms
    num_atom_types: int = 5  # H, C, N, O, F in QM9
    num_bond_types: int = 4  # single, double, triple, aromatic


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Encoder
    latent_dim: int = 64  # Latent space dimensionality
    hidden_dim: int = 256  # Hidden layer size
    num_encoder_layers: int = 3  # GNN layers for encoding
    
    # Decoder
    num_decoder_layers: int = 3  # GNN layers for decoding
    
    # Property predictor
    num_property_layers: int = 2
    num_properties: int = 12  # QM9 has 12 computed properties
    
    # Regularization
    dropout: float = 0.1
    
    # Architecture choices
    encoder_type: str = "gcn"  # 'gcn', 'gat', 'gin'
    decoder_type: str = "mlp"  # 'mlp', 'autoregressive'
    
    # Activation
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # VAE loss weights
    kl_weight: float = 0.1  # Beta for KL divergence
    recon_weight: float = 1.0  # Reconstruction loss weight
    property_weight: float = 0.5  # Property prediction weight
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    
    # Debug mode
    debug_mode: bool = True
    debug_samples: int = 1000  # Use subset for debugging


@dataclass
class OptimizationConfig:
    """Multi-objective optimization configuration."""
    # Target properties (indices in QM9)
    # 0: mu (dipole moment), 1: alpha (polarizability)
    # 4: homo, 5: lumo, 6: gap, 7: r2, 11: U0
    target_properties: List[int] = field(default_factory=lambda: [4, 5, 6])
    property_names: List[str] = field(default_factory=lambda: ["HOMO", "LUMO", "Gap"])
    
    # Optimization directions (True = maximize, False = minimize)
    maximize: List[bool] = field(default_factory=lambda: [True, False, True])
    
    # Target values (optional, for inverse design)
    target_values: Optional[List[float]] = None
    
    # NSGA-II parameters
    population_size: int = 100
    num_generations: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    
    # Latent space optimization
    latent_optimization_steps: int = 100
    latent_lr: float = 0.01
    
    # Baseline comparison
    random_search_samples: int = 1000


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration."""
    # Generation metrics
    num_generation_samples: int = 1000
    
    # Molecular validity checks
    check_validity: bool = True
    check_uniqueness: bool = True
    check_novelty: bool = True
    
    # Property evaluation
    evaluate_properties: bool = True
    
    # Quantization
    quantize_int8: bool = True
    compare_quantized: bool = True


@dataclass
class ResourceConfig:
    """Resource monitoring configuration."""
    log_memory: bool = True
    log_gpu: bool = True
    log_interval: int = 10  # Log every N batches
    profile_training: bool = False


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


def get_config(debug: bool = True) -> Config:
    """Get configuration with optional debug mode."""
    config = Config()
    config.training.debug_mode = debug
    return config


# Parameter count estimation
def estimate_parameters(config: Config) -> dict:
    """Estimate model parameter counts."""
    mc = config.model
    dc = config.data
    
    # Encoder parameters (GCN-based)
    encoder_params = (
        # Input projection
        dc.num_atom_types * mc.hidden_dim +
        # GNN layers
        mc.num_encoder_layers * (mc.hidden_dim * mc.hidden_dim + mc.hidden_dim) +
        # Mean/var projection
        2 * mc.hidden_dim * mc.latent_dim
    )
    
    # Decoder parameters
    decoder_params = (
        # Latent projection
        mc.latent_dim * mc.hidden_dim +
        # MLP layers
        mc.num_decoder_layers * (mc.hidden_dim * mc.hidden_dim + mc.hidden_dim) +
        # Output projection (atom types + adjacency)
        mc.hidden_dim * (dc.max_atoms * dc.num_atom_types) +
        mc.hidden_dim * (dc.max_atoms * dc.max_atoms * dc.num_bond_types)
    )
    
    # Property predictor parameters
    predictor_params = (
        mc.latent_dim * mc.hidden_dim +
        mc.num_property_layers * (mc.hidden_dim * mc.hidden_dim + mc.hidden_dim) +
        mc.hidden_dim * mc.num_properties
    )
    
    total = encoder_params + decoder_params + predictor_params
    
    return {
        "encoder": encoder_params,
        "decoder": decoder_params,
        "property_predictor": predictor_params,
        "total": total,
        "total_millions": total / 1e6
    }


if __name__ == "__main__":
    config = get_config(debug=True)
    params = estimate_parameters(config)
    
    print("=" * 60)
    print("PARAMETER COUNT TABLE")
    print("=" * 60)
    print(f"{'Component':<25} {'Parameters':>15} {'MB (FP32)':>12}")
    print("-" * 60)
    for name, count in params.items():
        if name != "total_millions":
            mb = count * 4 / 1e6  # 4 bytes per FP32 param
            print(f"{name:<25} {count:>15,} {mb:>12.2f}")
    print("-" * 60)
    print(f"Total: {params['total_millions']:.2f}M parameters")
    print("=" * 60)
