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

This project uses the **IDRiD dataset from HuggingFace** (no Kaggle authentication required).

| Dataset | Source | URL | License | Samples |
|---------|--------|-----|---------|---------|
| **IDRiD** | HuggingFace | [MahsaTorki/IDRiD_Dataset](https://huggingface.co/datasets/MahsaTorki/IDRiD_Dataset) | CC BY 4.0 | 54 train + 27 test with pixel-level annotations |

The IDRiD (Indian Diabetic Retinopathy Image Dataset) contains:
- **Training**: 54 fundus images with lesion segmentation masks (split 41/13 for train/val)
- **Testing**: 27 fundus images with lesion segmentation masks
- **Lesion Types**: Microaneurysms (MA), Hemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE)
- **Classification**: 5-class DR grading (0-4)
- **Original Source**: IEEE DataPort IDRiD Challenge 2018

### Data Loading

The dataset is automatically downloaded from HuggingFace on first run:

```python
from data.dataset import get_dataloaders

# Load real IDRiD data
train_loader, val_loader, test_loader = get_dataloaders(use_huggingface=True)
```

**Synthetic Data (pipeline testing):**
```bash
python main.py --generate-data  # Generates synthetic fundus images
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

## Experimental Results

**Dataset**: IDRiD from HuggingFace (41 train / 13 val / 27 test images)
**Ran on Real Data**: Yes

### Model Parameters

| Model | Parameters | Size (MB) |
|-------|------------|-----------|
| Classification-only | 16,062,021 | 61.27 |
| Segmentation-only (U-Net) | 20,103,812 | 76.69 |
| **Multi-task (Ours)** | **20,268,683** | **77.32** |

Parameter efficiency: 44.0% reduction vs. running both baselines separately

### Classification Performance (Test Set)

| Model | Accuracy | Kappa | F1 (macro) | AUC |
|-------|----------|-------|------------|-----|
| Classification-only | 0.1111 | 0.0150 | 0.0462 | 0.568 |
| **Multi-task (Ours)** | **0.1111** | **0.0000** | **0.0400** | **0.568** |

### Segmentation Performance (Test Set - Mean Dice)

| Model | MA | HE | SE | EX | Mean |
|-------|-----|-----|-----|-----|------|
| Segmentation-only | 0.0002 | 0.0029 | 0.0006 | 0.0103 | 0.0035 |
| **Multi-task (Ours)** | **0.0004** | **0.0157** | **0.0023** | **0.0121** | **0.0076** |

*Note: Performance limited by small dataset (41 training images) and training from scratch without pretrained weights.*

### INT8 Quantization

| Model | FP32 Size | INT8 Size | Reduction | FP32 Latency | INT8 Latency |
|-------|-----------|-----------|-----------|--------------|--------------|
| Classification | 61.32 MB | 60.69 MB | 1.0% | 12.5 ms | 12.4 ms |
| Segmentation | 76.74 MB | 76.74 MB | 0.0% | 22.9 ms | 23.3 ms |
| Multi-task | 77.37 MB | 76.74 MB | 0.8% | 23.0 ms | 23.2 ms |

## Literature Comparison

| Method | Type | Kappa | Mean Dice |
|--------|------|-------|-----------|
| VRT (Porwal et al., 2019) | Challenge Winner | 0.8500 | - |
| DeepDR (Hua et al., 2020) | Multi-task | 0.7800 | 0.5200 |
| MTL-DR (Zhou et al., 2019) | Multi-task | 0.7600 | 0.4800 |
| Attention U-Net (Oktay, 2018) | Segmentation | - | 0.5800 |
| **Ours (Multi-task)** | Multi-task | 0.0000 | 0.0076 |

*Note: Our results use a minimal training setup (41 images, no pretrained encoder). Published methods use full datasets (5000+ images) with pretrained ImageNet encoders. Direct comparison is not meaningful.*

## References

1. Porwal, P., et al. (2018). "IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge." Medical Image Analysis.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.

3. Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR.

4. Chen, Z., et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML.

5. Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." MIDL.

6. Lin, T.Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

## Model Architecture

### Component Breakdown (Multi-task)
- **Encoder**: 15,897,152 params (60.64 MB) - Shared ResNet-style backbone
- **Segmentation Decoder**: 4,206,660 params (16.05 MB) - U-Net style decoder
- **Classification Head**: 164,869 params (0.63 MB) - MLP classifier

## License

This project is for research purposes.
