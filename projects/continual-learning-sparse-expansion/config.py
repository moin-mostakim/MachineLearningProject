"""
Configuration for Continual Learning with Dynamic Sparse Expansion.

Hyperparameter Justifications:
- LEARNING_RATE = 0.001: Standard Adam learning rate; lower than SGD default (0.01)
  to ensure stable convergence with adaptive optimizer.
- EWC_LAMBDA = 5000: Controls regularization strength in EWC. Higher values
  preserve old task performance but may limit plasticity. 5000 is from the
  original EWC paper (Kirkpatrick et al., 2017).
- FISHER_SAMPLE_SIZE = 200: Number of samples for Fisher Information estimation.
  Larger improves accuracy but increases computation.
- SPARSITY_RATIO = 0.1: Only 10% of new parameters activated per task.
  Balances capacity with efficiency (Progressive Neural Networks use 100%).
- EXPANSION_THRESHOLD = 0.85: Expand network when task accuracy drops below this.
  Prevents unnecessary expansion while ensuring adequate capacity.
- BATCH_SIZE = 64: Standard batch size balancing memory and gradient noise.
- EPOCHS_PER_TASK = 5 (debug) / 20 (full): Sufficient for convergence on CIFAR subsets.
- NUM_TASKS = 10: CIFAR-100 split into 10 tasks of 10 classes each.

Literature References:
- EWC: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
- PackNet: Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network
  by Iterative Pruning" (2018)
- Progressive Neural Networks: Rusu et al., "Progressive Neural Networks" (2016)
- Sparse Networks: Frankle & Carlin, "The Lottery Ticket Hypothesis" (2019)
"""

import torch

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
DEBUG_MODE = True  # Set to False for full training runs
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
DATASET = "split_cifar100"  # Options: "split_cifar100", "permuted_mnist"
NUM_TASKS = 10  # Number of sequential tasks
CLASSES_PER_TASK = 10  # For Split CIFAR-100: 100 classes / 10 tasks

# For Permuted MNIST
PERMUTATION_SEED_START = 0  # Starting seed for permutations

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Base model configuration
BASE_HIDDEN_SIZES = [256, 256]  # Hidden layer sizes for MLP
CNN_CHANNELS = [32, 64, 128]  # Channel sizes for CNN backbone
DROPOUT_RATE = 0.2

# Dynamic Sparse Expansion settings
SPARSITY_RATIO = 0.1  # Fraction of new weights activated per expansion
EXPANSION_THRESHOLD = 0.85  # Accuracy threshold triggering expansion
MAX_EXPANSION_FACTOR = 3.0  # Maximum network size relative to original
SPARSE_INIT_STD = 0.01  # Std for initializing sparse connections

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
LEARNING_RATE = 0.001  # Adam default; stable for continual learning
WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting
BATCH_SIZE = 64 if not DEBUG_MODE else 32
EPOCHS_PER_TASK = 5 if DEBUG_MODE else 20  # Reduced for debug mode

# Learning rate scheduling
LR_SCHEDULER = "cosine"  # Options: "cosine", "step", "none"
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = [10, 15]  # For step scheduler

# =============================================================================
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# =============================================================================
# Lambda controls the importance of preserving old task performance
# Higher values = more regularization = less forgetting but less plasticity
EWC_LAMBDA = 5000  # From original paper; tune between 100-10000
FISHER_SAMPLE_SIZE = 200 if not DEBUG_MODE else 50  # Samples for Fisher estimation
ONLINE_EWC = True  # Use online EWC variant for efficiency
EWC_GAMMA = 0.9  # Decay factor for online EWC

# =============================================================================
# DYNAMIC SPARSE EXPANSION SETTINGS
# =============================================================================
# Our proposed method: expand network sparsely for new tasks
EXPANSION_HIDDEN_UNITS = 64  # Units to add per expansion
SPARSE_MASK_UPDATE_FREQ = 5  # Update sparse masks every N epochs
GRADIENT_BASED_GROWTH = True  # Use gradient magnitude for sparse growth
MAGNITUDE_PRUNING = True  # Prune by weight magnitude

# Task-specific adapter settings
USE_TASK_ADAPTERS = True
ADAPTER_BOTTLENECK_DIM = 32  # Dimension of adapter bottleneck

# =============================================================================
# QUANTIZATION SETTINGS
# =============================================================================
QUANTIZE_MODEL = True
QUANTIZATION_BITS = 8  # INT8 quantization
DYNAMIC_QUANTIZATION = True  # Use dynamic quantization for inference
CALIBRATION_SAMPLES = 100  # Samples for static quantization calibration

# =============================================================================
# EVALUATION METRICS
# =============================================================================
# Backward Transfer: BWT = (1/T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
# Forgetting Measure: F_j = max_{l in {1,...,T-1}} (a_{l,j} - a_{T,j}) for task j
EVAL_FREQUENCY = 1  # Evaluate all tasks every N epochs
COMPUTE_BACKWARD_TRANSFER = True
COMPUTE_FORGETTING = True

# =============================================================================
# RESOURCE MONITORING
# =============================================================================
MONITOR_RESOURCES = True
LOG_GPU_MEMORY = True
LOG_CPU_MEMORY = True
PROFILE_INFERENCE_TIME = True
RESOURCE_LOG_INTERVAL = 10  # Log every N batches

# =============================================================================
# LOGGING AND CHECKPOINTING
# =============================================================================
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 5  # Save every N epochs
TENSORBOARD_LOGGING = False  # Set True if tensorboard available
VERBOSE = True

# =============================================================================
# METHODS TO COMPARE
# =============================================================================
METHODS = [
    "finetune",           # Baseline: Sequential fine-tuning (catastrophic forgetting)
    "ewc",                # Elastic Weight Consolidation
    "sparse_expansion",   # Our method: Dynamic Sparse Expansion
]

# =============================================================================
# DEBUG MODE OVERRIDES
# =============================================================================
if DEBUG_MODE:
    EPOCHS_PER_TASK = 2
    NUM_TASKS = 3  # Only 3 tasks for quick testing
    FISHER_SAMPLE_SIZE = 20
    BATCH_SIZE = 32
    CALIBRATION_SAMPLES = 20
    RESOURCE_LOG_INTERVAL = 5
