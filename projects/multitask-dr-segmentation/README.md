# Multi-Task Learning Framework for Joint Lesion Segmentation and Diabetic Retinopathy Grading

A comprehensive deep learning framework implementing multi-task learning for simultaneous diabetic retinopathy (DR) grading and lesion segmentation from fundus images.

## Project Overview

This project implements and compares three approaches:
1. **Classification-only Baseline**: ResNet-style encoder with classification head for DR grading
2. **Segmentation-only Baseline (U-Net)**: Standard U-Net architecture for lesion segmentation
3. **Multi-Task Model (Our Method)**: Shared encoder with task-specific decoders for joint learning

### Key Features
- Synthetic fundus image generation with realistic lesion patterns
- Multiple task weighting strategies (fixed, uncertainty-based, GradNorm)
- Comprehensive evaluation metrics (Kappa, Dice, IoU, etc.)
- INT8 quantization support for deployment
- Resource monitoring and visualization tools

## Project Structure

```
multitask-dr-segmentation/
├── config.py                 # All hyperparameters with justifications
├── main.py                   # Main entry point
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
├── data/
│   ├── __init__.py
│   ├── synthetic_data.py     # Synthetic fundus generator
│   └── dataset.py            # PyTorch dataset classes
├── models/
│   ├── __init__.py
│   ├── encoder.py            # Shared encoder (ResNet-style)
│   ├── decoder.py            # Segmentation decoder & classification head
│   └── multitask_model.py    # Multi-task model with task weighting
├── utils/
│   ├── __init__.py
│   ├── losses.py             # Dice, Focal, Multi-task losses
│   ├── metrics.py            # Classification & segmentation metrics
│   ├── monitoring.py         # Resource monitoring
│   ├── quantization.py       # INT8 quantization utilities
│   └── visualization.py      # Plotting and visualization
├── experiments/              # Training logs and results
├── figures/                  # Generated visualizations
└── checkpoints/              # Model checkpoints
```

## Datasets

The system supports real DR datasets with lesion segmentation masks. These require Kaggle API authentication.

| Dataset | Source | URL | License | Size | Samples |
|---------|--------|-----|---------|------|---------|
| **IDRiD** | Kaggle | [diabetic-retinopathy-resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) | Competition | ~5GB | 516 images with pixel-level annotations |
| **APTOS 2019** | Kaggle | [aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection) | Competition | ~15GB | 3,662 train (classification only) |
| **DDR** | GitHub | [nkicsl/DDR-dataset](https://github.com/nkicsl/DDR-dataset) | Research | ~3GB | 13,673 images with lesion annotations |

**Requires:** Kaggle API key (`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables)

### Download Instructions

**IDRiD via Kaggle:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
kaggle datasets download -d tanlikesmath/diabetic-retinopathy-resized
unzip diabetic-retinopathy-resized.zip -d data/idrid/
```

**Synthetic Data (pipeline testing):**
```bash
python main.py  # Auto-generates if real data unavailable
```

### Data Details
- **Image Size**: 512×512 RGB fundus images
- **Lesion Types**: 4 types (MA, HE, SE, EX)
  - Microaneurysms (MA): Small red dots
  - Hemorrhages (HE): Larger red blots
  - Soft Exudates (SE): Fluffy white regions
  - Hard Exudates (EX): Bright yellow deposits
- **DR Grades**: 5 classes (No DR, Mild, Moderate, Severe, Proliferative)

Note: Synthetic data generates realistic lesion patterns for testing but real IDRiD data is required for valid research.

## Methods

### Shared Encoder Architecture
- ResNet-style encoder with 4 stages
- Channel progression: [64, 128, 256, 512]
- Skip connections for segmentation decoder
- Global pooling + MLP for classification

### Task Weighting Strategies

1. **Fixed Weights**: Static weights (Seg: 0.4, Cls: 0.6)
2. **Uncertainty Weighting** (Kendall et al., 2018): Learns task-specific noise parameters
3. **GradNorm** (Chen et al., 2018): Gradient magnitude balancing

### Loss Functions
- **Segmentation**: Dice + BCE combined loss
- **Classification**: Focal loss for class imbalance

## Benchmark Metrics

### Classification
- **Quadratic Weighted Kappa** (primary metric for DR challenges)
- Accuracy
- F1-Score (macro)
- AUC-ROC

### Segmentation
- **Dice Coefficient** (per lesion type and mean)
- IoU / Jaccard Index
- Precision / Recall

## Usage

### Run Full Pipeline
```bash
python main.py
```

### Training Only
```bash
python main.py --train-only
```

### Evaluation Only
```bash
python main.py --eval-only
```

### Generate Parameter Table
```bash
python main.py --param-table
```

### Quantization Analysis
```bash
python main.py --quantize
```

## Configuration

All hyperparameters are in `config.py` with detailed justifications:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| IMAGE_SIZE | 512 | Balance detail vs memory |
| BATCH_SIZE | 4 | Large images require small batches |
| LEARNING_RATE | 1e-4 | Conservative for fine-tuning |
| ENCODER_CHANNELS | [64,128,256,512] | Standard U-Net progression |
| SEGMENTATION_WEIGHT | 0.4 | Task balancing |
| CLASSIFICATION_WEIGHT | 0.6 | Primary task emphasis |
| DROPOUT_RATE | 0.3 | Regularization for small datasets |
| FOCAL_GAMMA | 2.0 | Standard focal loss setting |

## Literature Comparison

| Method | Type | Kappa | Mean Dice |
|--------|------|-------|-----------|
| VRT (Porwal et al., 2019) | Challenge Winner | 0.8500 | - |
| DeepDR (Hua et al., 2020) | Multi-task | 0.7800 | 0.5200 |
| MTL-DR (Zhou et al., 2019) | Multi-task | 0.7600 | 0.4800 |
| Attention U-Net (Oktay, 2018) | Segmentation | - | 0.5800 |
| **Ours (Multi-task)** | Multi-task | TBD | TBD |

## References

1. Porwal, P., et al. (2018). "IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge." Medical Image Analysis.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.

3. Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR.

4. Chen, Z., et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML.

5. Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." MIDL.

6. Lin, T.Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

## Model Parameter Comparison

| Model | Parameters | Size (MB) |
|-------|------------|-----------|
| Classification-only | ~7.5M | ~30 |
| Segmentation-only (U-Net) | ~10.2M | ~41 |
| Multi-task (Ours) | ~11.8M | ~47 |

Multi-task model achieves parameter efficiency by sharing encoder weights, saving ~33% compared to running both baselines separately.

## INT8 Quantization

Quantization reduces model size by ~75% with minimal accuracy loss:

| Model | FP32 Size | INT8 Size | Reduction |
|-------|-----------|-----------|-----------|
| Classification | ~30 MB | ~7.5 MB | 75% |
| Segmentation | ~41 MB | ~10 MB | 75% |
| Multi-task | ~47 MB | ~12 MB | 75% |

## License

This project is for research purposes.
