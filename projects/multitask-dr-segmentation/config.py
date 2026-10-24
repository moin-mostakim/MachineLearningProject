"""
Configuration for Multi-Task Learning Framework for Joint Lesion Segmentation and DR Grading.

Hyperparameter Justifications:
- IMAGE_SIZE (512): Standard for retinal image analysis; balances detail preservation with memory
  constraints. IDRiD images are 4288x2848, downsampled to preserve lesion visibility while 
  maintaining computational feasibility (Porwal et al., 2018).
  
- BATCH_SIZE (4): Small batch due to large image size and multi-task architecture memory 
  requirements. Similar to settings in attention-based DR methods (Foo et al., 2020).
  
- LEARNING_RATE (1e-4): Conservative rate for fine-tuning pretrained encoders. Adam optimizer
  with this rate is standard for medical imaging (He et al., 2019).
  
- ENCODER_CHANNELS ([64, 128, 256, 512]): Progressive feature extraction following U-Net
  architecture principles (Ronneberger et al., 2015). Doubled at each level for hierarchical
  feature learning.
  
- SEGMENTATION_WEIGHT (0.4) / CLASSIFICATION_WEIGHT (0.6): Task weighting based on gradient
  magnitude balancing. Classification as primary task gets higher weight following 
  Chen et al. (2018) multi-task learning recommendations.
  
- DROPOUT_RATE (0.3): Moderate dropout for regularization without excessive information loss.
  Standard for medical imaging to prevent overfitting on small datasets (Quellec et al., 2017).
  
- NUM_CLASSES (5): IDRiD grading scale: 0-No DR, 1-Mild, 2-Moderate, 3-Severe, 4-Proliferative.
  
- NUM_LESION_TYPES (4): Microaneurysms (MA), Hemorrhages (HE), Soft Exudates (SE), Hard 
  Exudates (EX) following IDRiD challenge protocol.
  
- UNCERTAINTY_WEIGHTING (True): Homoscedastic uncertainty for automatic task balancing
  (Kendall et al., 2018). Learns task-specific noise parameters during training.

References:
- Porwal et al. (2018) "IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge"
- Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
- Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
"""

import os

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_NAME = "IDRiD_Synthetic"  # Using synthetic data mimicking IDRiD structure
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_TRAIN_SAMPLES = 200  # Synthetic training samples
NUM_VAL_SAMPLES = 50     # Synthetic validation samples
NUM_TEST_SAMPLES = 50    # Synthetic test samples

# =============================================================================
# Image Configuration
# =============================================================================
IMAGE_SIZE = 256         # Input image size (height, width) - reduced for debug
IMAGE_CHANNELS = 3       # RGB fundus images
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization (pretrained encoder)
STD = [0.229, 0.224, 0.225]

# =============================================================================
# Task Configuration
# =============================================================================
NUM_CLASSES = 5          # DR grades: 0-4 (No DR to Proliferative)
NUM_LESION_TYPES = 4     # MA, HE, SE, EX lesion types
LESION_NAMES = ["Microaneurysms", "Hemorrhages", "Soft_Exudates", "Hard_Exudates"]
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]

# =============================================================================
# Model Architecture
# =============================================================================
ENCODER_NAME = "resnet34"  # Pretrained encoder backbone
ENCODER_CHANNELS = [64, 128, 256, 512]  # Feature channels at each level
DECODER_CHANNELS = [256, 128, 64, 32]   # U-Net decoder channels
BOTTLENECK_DIM = 512      # Shared bottleneck dimension
CLASSIFIER_HIDDEN = 256   # Classification head hidden dimension
USE_ATTENTION = True      # Attention gates in decoder
DROPOUT_RATE = 0.3        # Dropout for regularization

# =============================================================================
# Training Configuration
# =============================================================================
BATCH_SIZE = 4           # Batch size (memory constrained)
NUM_EPOCHS = 50          # Training epochs
LEARNING_RATE = 1e-4     # Initial learning rate
WEIGHT_DECAY = 1e-5      # L2 regularization
SCHEDULER_PATIENCE = 5   # LR scheduler patience
SCHEDULER_FACTOR = 0.5   # LR reduction factor
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

# =============================================================================
# Multi-Task Learning Configuration
# =============================================================================
SEGMENTATION_WEIGHT = 0.4    # Segmentation loss weight
CLASSIFICATION_WEIGHT = 0.6  # Classification loss weight
UNCERTAINTY_WEIGHTING = True # Use learned uncertainty weights (Kendall et al.)
GRADNORM_ALPHA = 1.5         # GradNorm restoring force

# Task Weighting Strategies
TASK_WEIGHTING_METHODS = [
    "fixed",          # Fixed weights (SEGMENTATION_WEIGHT, CLASSIFICATION_WEIGHT)
    "uncertainty",    # Homoscedastic uncertainty (Kendall et al., 2018)
    "gradnorm",       # GradNorm (Chen et al., 2018)
    "dynamic",        # Dynamic Weight Average (Liu et al., 2019)
]
DEFAULT_WEIGHTING = "uncertainty"

# =============================================================================
# Loss Configuration
# =============================================================================
SEGMENTATION_LOSS = "dice_bce"  # Combined Dice + BCE loss
CLASSIFICATION_LOSS = "focal"   # Focal loss for class imbalance
FOCAL_GAMMA = 2.0               # Focal loss focusing parameter
DICE_SMOOTH = 1e-6              # Dice loss smoothing

# =============================================================================
# Evaluation Metrics
# =============================================================================
CLASSIFICATION_METRICS = ["accuracy", "kappa", "f1_macro", "auc"]
SEGMENTATION_METRICS = ["dice", "iou", "precision", "recall"]

# =============================================================================
# Debug Mode Configuration
# =============================================================================
DEBUG_MODE = True        # Enable debug mode
DEBUG_SAMPLES = 10       # Number of samples in debug mode
DEBUG_EPOCHS = 2         # Number of epochs in debug mode

# =============================================================================
# Resource Monitoring
# =============================================================================
MONITOR_RESOURCES = True
LOG_INTERVAL = 10        # Log every N batches

# =============================================================================
# Quantization Configuration
# =============================================================================
QUANTIZATION_ENABLED = True
QUANTIZATION_BACKEND = "fbgemm"  # x86 backend
CALIBRATION_SAMPLES = 50

# =============================================================================
# Paths Configuration
# =============================================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")

# =============================================================================
# Random Seeds for Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# Effective configuration based on debug mode
# =============================================================================
def get_effective_config():
    """Return effective configuration based on debug mode."""
    config = {
        'num_epochs': DEBUG_EPOCHS if DEBUG_MODE else NUM_EPOCHS,
        'num_train_samples': DEBUG_SAMPLES if DEBUG_MODE else NUM_TRAIN_SAMPLES,
        'num_val_samples': DEBUG_SAMPLES // 2 if DEBUG_MODE else NUM_VAL_SAMPLES,
        'num_test_samples': DEBUG_SAMPLES // 2 if DEBUG_MODE else NUM_TEST_SAMPLES,
        'batch_size': min(BATCH_SIZE, 2) if DEBUG_MODE else BATCH_SIZE,
    }
    return config
