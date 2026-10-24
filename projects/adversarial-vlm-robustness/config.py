"""
Configuration file for Adversarial Robustness of Vision-Language Models.

Hyperparameter justifications are based on literature:
- Madry et al. (2018): "Towards Deep Learning Models Resistant to Adversarial Attacks"
- Cohen et al. (2019): "Certified Adversarial Robustness via Randomized Smoothing"
- Sinha et al. (2018): "Certifiable Distributional Robustness with Principled Adversarial Training"
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """Dataset configuration.
    
    Datasets:
    ---------
    CIFAR-10: Standard benchmark for image classification
    - Source: torchvision (auto-download)
    - URL: https://www.cs.toronto.edu/~kriz/cifar.html
    - HuggingFace: https://huggingface.co/datasets/cifar10
    - License: MIT-like (research use)
    - Size: ~170MB (60,000 32x32 color images in 10 classes)
    """
    dataset: str = "cifar10"  # Options: cifar10, flickr8k
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    use_synthetic: bool = False  # Set True only for quick testing without download
    
    # Dataset URLs for reference
    dataset_url: str = "https://www.cs.toronto.edu/~kriz/cifar.html"
    dataset_hf: str = "https://huggingface.co/datasets/cifar10"
    
    # CIFAR-10 text labels for VLM setup
    cifar10_labels: List[str] = field(default_factory=lambda: [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ])
    
    # Justification: Standard batch size for CIFAR-10 training
    # Larger batches help stabilize adversarial training gradients


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Vision encoder
    vision_encoder: str = "resnet18"  # Options: resnet18, resnet50, vit_small
    vision_dim: int = 512
    
    # Text encoder (simple for CIFAR-10 labels)
    text_encoder: str = "embedding"  # Options: embedding, transformer
    text_dim: int = 512
    vocab_size: int = 10000
    max_text_len: int = 32
    
    # Projection head for contrastive learning
    projection_dim: int = 256
    
    # Justification: ResNet-18 provides good balance of capacity and efficiency
    # 512-dim features standard for CLIP-style models


@dataclass
class AttackConfig:
    """Adversarial attack configuration.
    
    Based on Madry et al. (2018) recommendations for L-infinity attacks.
    """
    # PGD attack parameters
    epsilon: float = 8/255  # L-infinity perturbation bound
    alpha: float = 2/255    # Step size per iteration
    num_steps: int = 7      # Number of PGD steps
    
    # Justification for epsilon=8/255:
    # - Standard benchmark value for CIFAR-10 (Madry et al., 2018)
    # - Represents ~3% perturbation in pixel intensity
    # - Imperceptible to human observers
    
    # Justification for alpha=2/255:
    # - Recommended as epsilon/4 to epsilon/3 for stable convergence
    # - Ensures smooth gradient ascent without overshooting
    
    # Justification for num_steps=7:
    # - Madry et al. found 7-20 steps sufficient for strong attacks
    # - Trade-off between attack strength and training time
    
    random_start: bool = True  # Random initialization within epsilon ball
    norm: str = "linf"  # Options: linf, l2


@dataclass  
class CertifiedConfig:
    """Certified defense configuration via randomized smoothing.
    
    Based on Cohen et al. (2019) "Certified Adversarial Robustness via Randomized Smoothing"
    """
    # Noise scale for Gaussian smoothing
    sigma: float = 0.25
    
    # Justification for sigma=0.25:
    # - Provides certified radius ~0.5 for high-confidence predictions
    # - Trade-off: higher sigma = larger certified radius but lower clean accuracy
    # - Cohen et al. found 0.12-0.50 effective range for CIFAR-10
    
    # Certification parameters
    n_samples_certify: int = 1000   # Samples for certification (Monte Carlo)
    n_samples_predict: int = 100    # Samples for prediction
    alpha: float = 0.001            # Confidence level for certification
    
    # Justification for n_samples:
    # - 1000 samples gives tight confidence bounds
    # - More samples = tighter bounds but slower certification
    
    # Training with noise augmentation
    noise_augmentation: bool = True


@dataclass
class DROConfig:
    """Distributionally Robust Optimization configuration.
    
    Based on Sinha et al. (2018) and recent advances in Wasserstein DRO.
    """
    # Wasserstein ball radius
    rho: float = 0.1
    
    # Justification for rho=0.1:
    # - Controls size of distributional uncertainty set
    # - Larger rho = more conservative but potentially lower clean accuracy
    # - Typical range: 0.01-1.0 depending on domain
    
    # Dual variable for Lagrangian relaxation
    gamma_init: float = 1.0
    gamma_lr: float = 0.01
    
    # Number of inner maximization steps
    inner_steps: int = 5
    inner_lr: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training mode
    mode: str = "standard"  # Options: standard, adversarial, certified, dro
    
    # Optimization
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Justification for lr=0.1:
    # - Standard for SGD with momentum on CIFAR-10
    # - Combined with cosine annealing for smooth convergence
    
    # Schedule
    epochs: int = 100
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # Options: cosine, step, linear
    
    # For adversarial training
    adv_train_epsilon: float = 8/255
    adv_train_steps: int = 7
    
    # Early stopping
    patience: int = 20
    
    # Mixed precision
    use_amp: bool = True
    
    # Debug mode
    debug: bool = False
    debug_batches: int = 10
    debug_epochs: int = 3


@dataclass
class QuantizationConfig:
    """INT8 quantization configuration for efficiency analysis."""
    enable_quantization: bool = True
    quantization_backend: str = "fbgemm"  # Options: fbgemm, qnnpack
    calibration_batches: int = 100
    
    # Dynamic vs static quantization
    quantization_type: str = "dynamic"  # Options: dynamic, static


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment identification
    name: str = "adversarial_vlm_robustness"
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_interval: int = 50
    
    # Resource monitoring
    monitor_resources: bool = True
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    certified: CertifiedConfig = field(default_factory=CertifiedConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


def get_config(debug: bool = False) -> ExperimentConfig:
    """Get experiment configuration."""
    config = ExperimentConfig()
    if debug:
        config.training.debug = True
        config.training.epochs = config.training.debug_epochs
        config.data.batch_size = 32
        config.certified.n_samples_certify = 100
        config.certified.n_samples_predict = 10
    return config


# Literature references for hyperparameter choices
REFERENCES = """
Key Literature References:

1. Madry et al. (2018) - "Towards Deep Learning Models Resistant to Adversarial Attacks"
   - Introduced PGD adversarial training
   - Epsilon=8/255 for CIFAR-10, 7-20 PGD steps
   - URL: https://arxiv.org/abs/1706.06083

2. Cohen et al. (2019) - "Certified Adversarial Robustness via Randomized Smoothing"
   - Randomized smoothing for certified defenses
   - Gaussian noise with sigma=0.12-0.50
   - Monte Carlo certification with 100-10000 samples
   - URL: https://arxiv.org/abs/1902.02918

3. Sinha et al. (2018) - "Certifiable Distributional Robustness with Principled Adversarial Training"
   - Wasserstein DRO for robust optimization
   - Lagrangian relaxation approach
   - URL: https://arxiv.org/abs/1710.10571

4. Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision"
   - CLIP architecture for vision-language models
   - Contrastive learning objective
   - URL: https://arxiv.org/abs/2103.00020

5. Carlini & Wagner (2017) - "Towards Evaluating the Robustness of Neural Networks"
   - Strong C&W attack for evaluation
   - URL: https://arxiv.org/abs/1608.04644
"""


if __name__ == "__main__":
    config = get_config(debug=True)
    print(f"Configuration: {config.name}")
    print(f"Device: {config.device}")
    print(f"Debug mode: {config.training.debug}")
    print(f"\nEpsilon: {config.attack.epsilon:.4f} ({config.attack.epsilon*255:.1f}/255)")
    print(f"Certified sigma: {config.certified.sigma}")
    print(f"DRO rho: {config.dro.rho}")
    print(f"\n{REFERENCES}")
