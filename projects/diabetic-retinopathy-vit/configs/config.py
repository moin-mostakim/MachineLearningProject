"""
Configuration file for Adaptive Multi-Scale Explainable Vision Transformer
with Lesion-Aware Attention for Early Diabetic Retinopathy Grading.

All hyperparameters are documented with justifications from literature.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    # Image size: 224x224 is standard for ViT pretrained models
    # Higher resolutions (512, 768) used in DR competitions but increase compute
    # Reference: APTOS 2019 winning solutions used 224-512
    image_size: int = 224
    
    # Number of DR grades (0-4): No DR, Mild, Moderate, Severe, Proliferative
    # Standard 5-class classification per International Clinical DR Severity Scale
    num_classes: int = 5
    
    # Batch size: 16-32 common for ViT training on single GPU
    # Smaller batches help with class imbalance in DR datasets
    batch_size: int = 16
    
    # Number of workers: 4 is typically optimal for data loading
    num_workers: int = 4
    
    # Train/val/test split ratios
    # 80/10/10 split standard in medical imaging
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data augmentation parameters
    # Heavy augmentation needed for small medical datasets
    # Reference: EyePACS studies show augmentation crucial for DR
    augmentation: dict = field(default_factory=lambda: {
        'horizontal_flip': 0.5,  # Fundus images symmetric
        'vertical_flip': 0.5,    # Also applicable
        'rotation': 15,          # Small rotation to preserve lesion structure
        'color_jitter': 0.3,     # Account for imaging device variation
        'gaussian_blur': 0.1,    # Simulate focus issues
        'random_crop_scale': (0.8, 1.0),  # Standard crop
    })
    
    # Class weights for imbalanced data
    # DR datasets heavily imbalanced: most images are No DR (0) or Mild (1)
    # Reference: APTOS 2019 had ~50% class 0, class 4 < 5%
    use_class_weights: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # ViT base configuration
    # Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    
    # Patch size: 16x16 standard, smaller (8x8) for fine-grained features
    # 16x16 balances detail vs computational cost
    patch_size: int = 16
    
    # Hidden dimension: 768 for ViT-Base, 384 for ViT-Small
    # 384 for efficiency while maintaining performance
    hidden_dim: int = 384
    
    # Number of transformer layers: 12 for ViT-Base
    # 8 layers sufficient for medical image tasks
    # Reference: Medical imaging often uses shallower networks
    num_layers: int = 8
    
    # Number of attention heads: 12 for ViT-Base
    # 6 heads with dim 384 gives head_dim=64 (standard)
    num_heads: int = 6
    
    # MLP ratio: 4x hidden dim is standard
    mlp_ratio: float = 4.0
    
    # Dropout rates
    # Higher dropout for medical imaging (limited data)
    # Reference: Common in APTOS solutions
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path: float = 0.1  # Stochastic depth
    
    # Multi-scale configuration
    # Use 3 scales for capturing lesions at different sizes
    # Microaneurysms: small, Hemorrhages: medium-large
    multi_scale_sizes: List[int] = field(default_factory=lambda: [224, 384, 512])
    multi_scale_weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    
    # Lesion-aware attention configuration
    # Lesion types in DR: microaneurysms, hemorrhages, exudates, cotton wool spots
    num_lesion_types: int = 4
    lesion_attention_heads: int = 4
    
    # ResNet baseline configuration
    resnet_variant: str = 'resnet50'  # ResNet-50 common baseline for DR
    resnet_pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Learning rate: 1e-4 to 3e-4 typical for ViT fine-tuning
    # Reference: "Training data-efficient image transformers" (Touvron et al.)
    learning_rate: float = 1e-4
    
    # Weight decay: 0.01-0.05 for ViT regularization
    # Higher than CNNs due to transformer overfitting tendency
    weight_decay: float = 0.05
    
    # Learning rate scheduler
    # Cosine annealing works well for ViT
    scheduler: str = 'cosine'
    warmup_epochs: int = 1  # Reduced warmup for CPU-friendly 5-epoch training
    
    # Training epochs
    # 50-100 epochs common for medical imaging
    # Reference: APTOS solutions used 30-100 epochs
    # Using 5 epochs for CPU-friendly meaningful evaluation
    epochs: int = 5
    
    # Early stopping patience
    early_stopping_patience: int = 10
    
    # Gradient clipping: prevents exploding gradients in transformers
    max_grad_norm: float = 1.0
    
    # Mixed precision training: speeds up ViT training significantly
    use_amp: bool = True
    
    # Label smoothing: helps with noisy DR labels
    # Reference: Inter-rater agreement in DR grading ~80%
    label_smoothing: float = 0.1
    
    # Loss function: Cross-entropy with ordinal awareness
    # Quadratic penalty makes sense for ordinal DR grades
    use_ordinal_loss: bool = True
    ordinal_weight: float = 0.3
    
    # Optimizer
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Debug mode settings
    debug_mode: bool = False
    debug_batches: int = 2
    debug_epochs: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    
    # Primary metric: Quadratic Weighted Kappa
    # Standard for DR grading (APTOS, EyePACS competitions)
    # Penalizes disagreements more when predictions are far from truth
    primary_metric: str = 'quadratic_weighted_kappa'
    
    # Secondary metrics
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'accuracy',
        'auc_macro',
        'auc_weighted',
        'f1_macro',
        'sensitivity',
        'specificity',
    ])
    
    # Confusion matrix threshold optimization
    # Reference: APTOS winners used threshold optimization
    optimize_thresholds: bool = True
    
    # Test time augmentation
    use_tta: bool = True
    tta_transforms: int = 4  # Number of TTA variants


@dataclass
class QuantizationConfig:
    """INT8 quantization configuration."""
    
    # Post-training quantization
    quantization_type: str = 'dynamic'  # 'dynamic' or 'static'
    
    # Calibration settings for static quantization
    calibration_batches: int = 100
    
    # Backend for quantization
    backend: str = 'fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM


@dataclass 
class VisualizationConfig:
    """Explainability and visualization configuration."""
    
    # Attention visualization
    save_attention_maps: bool = True
    attention_head_fusion: str = 'mean'  # 'mean', 'max', 'min'
    
    # Grad-CAM settings
    use_gradcam: bool = True
    gradcam_layer: str = 'last_block'
    
    # Saliency maps
    use_saliency: bool = True
    saliency_method: str = 'integrated_gradients'
    
    # Output settings
    num_samples_to_visualize: int = 10
    figure_dpi: int = 300


@dataclass
class ResourceConfig:
    """Resource monitoring configuration."""
    
    # GPU memory tracking
    track_gpu_memory: bool = True
    
    # Training time tracking
    track_time: bool = True
    
    # Model profiling
    profile_model: bool = True
    
    # Logging interval
    log_interval: int = 10  # Log every N batches


@dataclass
class Config:
    """Main configuration aggregating all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Experiment settings
    experiment_name: str = 'dr_vit_lesion_aware'
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output paths
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'outputs/checkpoints'
    log_dir: str = 'outputs/logs'
    figure_dir: str = 'outputs/figures'


# Literature references for hyperparameter choices
LITERATURE_REFERENCES = """
Hyperparameter Justifications from Literature:

1. Image Size (224-512):
   - Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words" - ViT pretrained at 224
   - APTOS 2019 winners: Used 224-512 with multi-scale approaches
   - Gulshan et al. (2016): "Development of a Deep Learning Algorithm for DR" - 299x299

2. Patch Size (16):
   - Standard in ViT-Base; 8x8 increases sequence length 4x
   - Trade-off between fine-grained detail and computation

3. Learning Rate (1e-4):
   - Touvron et al. (2021): "Training data-efficient image transformers" - 1e-4 to 3e-4
   - Lower than CNN fine-tuning (1e-3) due to transformer sensitivity

4. Weight Decay (0.05):
   - Higher than CNNs; transformers prone to overfitting
   - Reference: DeiT paper recommends 0.05

5. Label Smoothing (0.1):
   - Krause et al. (2018): DR grading has ~80% inter-rater agreement
   - Smoothing accounts for inherent label noise

6. Quadratic Weighted Kappa:
   - Standard metric for APTOS, EyePACS, Kaggle DR competitions
   - Penalizes distant misclassifications (e.g., 0 vs 4) more heavily

7. Multi-scale approach:
   - Microaneurysms: 10-100 micrometers (small patches)
   - Hemorrhages: 100-1000 micrometers (medium patches)
   - Large lesions visible at lower resolution

8. Ordinal Loss:
   - DR grades are ordinal (0 < 1 < 2 < 3 < 4)
   - Reference: "Ordinal Regression with Multiple Output CNN for Age Estimation"
"""


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def get_debug_config() -> Config:
    """Get configuration for debug/quick testing."""
    config = Config()
    config.training.debug_mode = True
    config.training.epochs = config.training.debug_epochs
    config.data.batch_size = 8
    return config
