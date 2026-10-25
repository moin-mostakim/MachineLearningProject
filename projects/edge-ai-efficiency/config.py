"""
Configuration for Edge AI Efficiency Project
==============================================

Hyperparameters with detailed justifications based on literature:
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- Once-for-All NAS: Cai et al., "Once-for-All: Train One Network and Specialize it for Efficient Deployment", ICLR 2020
- Quantization: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    # Dataset choice: CIFAR-10 (smaller, faster) or ImageNette (ImageNet subset)
    # CIFAR-10: 60K 32x32 images, 10 classes - good for rapid prototyping
    # ImageNette: 9K 160x160+ images, 10 classes - better for realistic edge deployment
    dataset: str = "cifar10"
    
    # Data augmentation parameters
    # Justification: Standard augmentation from MobileNet/EfficientNet papers
    # Random crop and horizontal flip are essential for preventing overfitting
    random_crop_padding: int = 4  # CIFAR-10 standard padding
    horizontal_flip_prob: float = 0.5  # Standard probability
    
    # Normalization values (CIFAR-10 mean/std)
    # Computed from training set for proper preprocessing
    cifar10_mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    cifar10_std: Tuple[float, ...] = (0.2470, 0.2435, 0.2616)
    
    # ImageNette uses ImageNet statistics
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Number of classes
    num_classes: int = 10
    
    # Image size for input
    # CIFAR-10: 32x32, ImageNette: 224x224 (standard for MobileNet/EfficientNet)
    image_size: int = 32  # Will be overridden based on dataset


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Width multiplier for MobileNet
    # Justification: Controls model capacity vs efficiency trade-off
    # α = 1.0 is full model, α = 0.5 reduces channels by 50%
    # Smaller values for edge deployment (0.25, 0.5, 0.75)
    # Reference: MobileNetV2 paper Table 6
    width_multiplier: float = 1.0
    
    # Depth multiplier (for scaling number of layers)
    # EfficientNet uses compound scaling: depth = α^φ where φ is scaling coefficient
    depth_multiplier: float = 1.0
    
    # Resolution multiplier (for input size)
    resolution_multiplier: float = 1.0
    
    # Dropout rate
    # Justification: Prevents overfitting, standard 0.2 for MobileNet
    # EfficientNet uses 0.2-0.4 depending on model size
    dropout_rate: float = 0.2
    
    # Inverted residual expansion ratio for MobileNetV2
    # Justification: Expansion ratio t=6 is standard from MobileNetV2
    # Expands channels before depthwise conv, then projects back
    expansion_ratio: int = 6
    
    # Squeeze-and-Excitation ratio for EfficientNet
    # Justification: SE ratio of 0.25 is standard, balances accuracy vs compute
    se_ratio: float = 0.25


@dataclass
class NASConfig:
    """Neural Architecture Search configuration."""
    
    # Search space definition
    # Justification: Based on Once-for-All (OFA) search space
    # Covers key architectural decisions that impact efficiency
    
    # Kernel sizes to search (depthwise convolutions)
    # 3x3 is most efficient, 5x5/7x7 increase receptive field but add compute
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])
    
    # Expansion ratios for inverted residuals
    # Lower expansion = fewer parameters, higher expansion = more capacity
    expansion_ratios: List[int] = field(default_factory=lambda: [3, 4, 6])
    
    # Depth options (number of blocks per stage)
    # More depth = more capacity but higher latency
    depth_options: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Width multiplier options
    width_options: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    
    # NAS algorithm settings
    # Population size for evolutionary search
    # Justification: 20-50 is common in NAS literature for manageable compute
    population_size: int = 20
    
    # Number of generations for evolutionary search
    # Justification: 20-50 generations typically sufficient for convergence
    num_generations: int = 10  # Reduced for debug mode
    
    # Mutation probability
    # Justification: 0.1-0.3 is standard for architecture search
    mutation_prob: float = 0.2
    
    # Crossover probability
    crossover_prob: float = 0.5
    
    # Number of random samples (for random search baseline)
    num_random_samples: int = 20
    
    # Latency constraint (ms) for constrained search
    # Target edge device latency budget
    latency_constraint_ms: float = 50.0
    
    # Model size constraint (MB)
    size_constraint_mb: float = 10.0


@dataclass
class QuantizationConfig:
    """Quantization-Aware Training configuration."""
    
    # Quantization bit widths
    # INT8: Standard for edge deployment, good accuracy-efficiency trade-off
    # INT4: More aggressive, significant speedup but accuracy drop
    # Reference: Jacob et al., CVPR 2018
    
    # Weight quantization bits
    weight_bits: int = 8
    
    # Activation quantization bits
    activation_bits: int = 8
    
    # Per-channel vs per-tensor quantization
    # Per-channel generally better for weights (reduces quantization error)
    per_channel_weights: bool = True
    
    # Symmetric vs asymmetric quantization
    # Symmetric simpler but asymmetric better for activations (ReLU)
    symmetric_weights: bool = True
    symmetric_activations: bool = False
    
    # QAT warmup epochs before enabling quantization
    # Justification: Allow model to converge first, then fine-tune with quantization
    qat_warmup_epochs: int = 2
    
    # Learning rate for QAT fine-tuning
    # Justification: Lower LR for fine-tuning, typically 10x smaller than training
    qat_learning_rate: float = 1e-4
    
    # Number of QAT epochs
    qat_epochs: int = 5
    
    # Observer settings for calibration
    # Number of batches for observer calibration
    observer_batches: int = 100
    
    # Moving average coefficient for observers
    averaging_constant: float = 0.01


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Batch size
    # Justification: 128 is standard for CIFAR-10, balances memory and convergence
    # Larger batches (256, 512) possible with learning rate scaling
    batch_size: int = 128
    
    # Number of training epochs
    # Justification: CIFAR-10 typically converges in 100-200 epochs
    # ImageNette may need 50-100 epochs
    num_epochs: int = 50
    
    # Learning rate
    # Justification: 0.1 is standard for SGD with momentum
    # Will use cosine annealing schedule
    learning_rate: float = 0.1
    
    # Momentum for SGD
    # Justification: 0.9 is standard, helps escape local minima
    momentum: float = 0.9
    
    # Weight decay (L2 regularization)
    # Justification: 4e-5 is standard for MobileNet, prevents overfitting
    # EfficientNet uses 1e-5 to 5e-5
    weight_decay: float = 4e-5
    
    # Learning rate schedule
    # Cosine annealing: smooth decay, often better than step decay
    lr_schedule: str = "cosine"
    
    # Warmup epochs
    # Justification: Gradual warmup helps training stability
    warmup_epochs: int = 5
    
    # Label smoothing
    # Justification: Improves generalization, 0.1 is standard
    label_smoothing: float = 0.1
    
    # Gradient clipping
    # Justification: Prevents gradient explosion, especially useful with QAT
    gradient_clip_norm: float = 1.0
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Pin memory for faster GPU transfer
    pin_memory: bool = True
    
    # Mixed precision training
    use_amp: bool = True


@dataclass 
class ExperimentConfig:
    """Experiment and logging configuration."""
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Debug mode: reduced epochs, smaller search space
    debug_mode: bool = True
    
    # Debug settings
    debug_epochs: int = 3
    debug_nas_generations: int = 3
    debug_nas_population: int = 5
    debug_batch_size: int = 64
    
    # Logging frequency
    log_every_n_steps: int = 50
    
    # Checkpoint settings
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Results directory
    results_dir: str = "results"
    
    # Device settings
    device: str = "cuda"  # or "cpu" for edge simulation
    
    # Benchmark settings
    # Number of inference runs for latency measurement
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 100
    
    # CPU inference for edge simulation
    benchmark_on_cpu: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def apply_debug_mode(self):
        """Apply debug mode settings for faster iteration."""
        if self.experiment.debug_mode:
            self.training.num_epochs = self.experiment.debug_epochs
            self.training.batch_size = self.experiment.debug_batch_size
            self.nas.num_generations = self.experiment.debug_nas_generations
            self.nas.population_size = self.experiment.debug_nas_population
            self.nas.num_random_samples = self.experiment.debug_nas_population
            self.quantization.qat_epochs = 2
            self.quantization.observer_batches = 20
            print("[DEBUG MODE] Reduced training epochs, NAS iterations, and batch size")
    
    def get_image_size(self) -> int:
        """Get image size based on dataset."""
        if self.data.dataset == "cifar10":
            return 32
        elif self.data.dataset == "imagenette":
            return 224
        else:
            return 32
    
    def get_normalization_stats(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Get normalization mean and std based on dataset."""
        if self.data.dataset == "cifar10":
            return self.data.cifar10_mean, self.data.cifar10_std
        else:
            return self.data.imagenet_mean, self.data.imagenet_std


def get_config(debug_mode: bool = True) -> Config:
    """Get configuration with optional debug mode."""
    config = Config()
    config.experiment.debug_mode = debug_mode
    config.apply_debug_mode()
    return config


# Literature references for hyperparameter choices
REFERENCES = """
Key References:
===============

1. MobileNetV2 (Sandler et al., 2018)
   - Inverted residuals with linear bottlenecks
   - Width multiplier α ∈ {0.25, 0.5, 0.75, 1.0}
   - Expansion ratio t = 6
   - Input resolution: 224×224
   - Training: SGD, momentum=0.9, weight_decay=4e-5

2. EfficientNet (Tan & Le, 2019)
   - Compound scaling: depth α^φ, width β^φ, resolution γ^φ
   - α=1.2, β=1.1, γ=1.15 with α·β²·γ² ≈ 2
   - Squeeze-and-Excitation with ratio 0.25
   - Swish activation, dropout 0.2-0.4

3. Once-for-All NAS (Cai et al., 2020)
   - Progressive shrinking training
   - Elastic kernel: {3, 5, 7}
   - Elastic depth: {2, 3, 4} blocks per stage
   - Elastic width: {0.5, 0.75, 1.0} expansion
   - Accuracy predictors for efficient search

4. Quantization (Jacob et al., 2018)
   - INT8 inference with minimal accuracy loss
   - Per-channel quantization for weights
   - Fake quantization during training (QAT)
   - Batch normalization folding

5. Additional References:
   - ProxylessNAS (Cai et al., 2019): Hardware-aware search
   - FBNet (Wu et al., 2019): Differentiable NAS with latency
   - HAQ (Wang et al., 2019): Hardware-aware mixed precision
"""

if __name__ == "__main__":
    # Print configuration for verification
    config = get_config(debug_mode=True)
    print("Edge AI Efficiency Configuration")
    print("=" * 50)
    print(f"Dataset: {config.data.dataset}")
    print(f"Image Size: {config.get_image_size()}")
    print(f"Debug Mode: {config.experiment.debug_mode}")
    print(f"Training Epochs: {config.training.num_epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"NAS Generations: {config.nas.num_generations}")
    print(f"QAT Epochs: {config.quantization.qat_epochs}")
    print("\n" + REFERENCES)
