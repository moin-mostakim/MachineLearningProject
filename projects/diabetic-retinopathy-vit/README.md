# Adaptive Multi-Scale Explainable Vision Transformer with Lesion-Aware Attention for Early Diabetic Retinopathy Grading

## Overview

This project implements a novel Vision Transformer architecture for Diabetic Retinopathy (DR) grading that combines:

1. **Multi-scale patch embedding** - Captures lesions at different sizes (microaneurysms to large hemorrhages)
2. **Lesion-aware attention** - Dedicated attention heads for different DR lesion types
3. **Built-in explainability** - Attention visualization for clinical interpretability

## DR Grading Scale

| Grade | Description | Clinical Features |
|-------|-------------|-------------------|
| 0 | No DR | No visible retinopathy |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than just microaneurysms |
| 3 | Severe NPDR | Extensive hemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularization, vitreous hemorrhage |

## Project Structure

```
diabetic-retinopathy-vit/
├── configs/
│   └── config.py          # Hyperparameters with justifications
├── src/
│   ├── models/
│   │   ├── resnet_baseline.py      # ResNet-50 baseline
│   │   ├── vit_baseline.py         # Standard ViT baseline
│   │   └── multiscale_lesion_vit.py # Our proposed method
│   ├── data/
│   │   └── dataset.py     # Dataset handling with synthetic data fallback
│   ├── utils/
│   │   ├── metrics.py     # QWK and evaluation metrics
│   │   ├── training.py    # Training utilities
│   │   └── quantization.py # INT8 quantization
│   └── visualization/
│       └── attention_viz.py # Attention and explainability visualization
├── experiments/
│   └── train.py           # Main training script
├── outputs/
│   ├── figures/           # Generated visualizations
│   ├── checkpoints/       # Model checkpoints
│   └── logs/              # Training logs
├── LITERATURE.md          # Literature review and references
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Models

### 1. ResNet-50 Baseline
Standard ResNet-50 pretrained on ImageNet with custom classifier head for 5-class DR grading.

### 2. ViT Baseline
Standard Vision Transformer adapted for DR grading:
- Patch size: 16x16
- Hidden dim: 384
- Depth: 8 layers
- Heads: 6

### 3. Multi-Scale Lesion-Aware ViT (Our Method)
Novel architecture with:

**Multi-scale Processing:**
- Patch sizes: 8x8, 16x16, 32x32
- Captures fine (microaneurysms) to coarse (large hemorrhages) features

**Lesion-Aware Attention:**
- 4 lesion-type specific attention heads
- Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates
- Learnable lesion embeddings
- Gating mechanism for lesion importance

## Metrics

**Primary Metric: Quadratic Weighted Kappa (QWK)**
- Standard for DR grading competitions (APTOS, EyePACS)
- Accounts for ordinal nature of grades
- Penalizes large disagreements more heavily

**Secondary Metrics:**
- Accuracy
- AUC (macro and weighted)
- F1 Score
- Sensitivity/Specificity for referable DR (grade ≥ 2)

## Usage

### Training

```bash
cd experiments
python train.py
```

### Debug Mode (Quick Test)
By default, training runs in debug mode with:
- 2 epochs
- 5 batches per epoch
- 500 synthetic samples

### Full Training
Edit `configs/config.py` to set `debug_mode = False` for full training.

## Hyperparameters

All hyperparameters are documented with literature justifications in `configs/config.py`:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 1e-4 | DeiT paper recommendation for ViT fine-tuning |
| Weight Decay | 0.05 | Higher than CNNs due to transformer overfitting tendency |
| Batch Size | 16 | Common for ViT on single GPU |
| Label Smoothing | 0.1 | Accounts for ~80% inter-rater agreement in DR |
| Warmup Epochs | 5 | Critical for ViT training stability |

## Data

### Supported Datasets
1. **APTOS 2019** - Place in `outputs/data/aptos2019/`
2. **Messidor** - Place in `outputs/data/messidor/`
3. **Synthetic** - Auto-generated if real data unavailable

### Synthetic Data
When real datasets are unavailable, the system generates synthetic fundus-like images with:
- Realistic fundus structure (optic disc, macula, vessels)
- Grade-appropriate lesion simulation
- Class imbalance matching real distribution

## INT8 Quantization

The project includes INT8 quantization comparison:
- Dynamic quantization for deployment
- Size reduction: ~2-4x
- Inference speedup: ~1.5-2x on CPU

## Explainability

### Attention Maps
- Standard attention visualization
- Lesion-specific attention heatmaps
- Multi-scale attention weights

### Grad-CAM
- Class activation maps for any target class
- Works with both CNN and ViT architectures

### Integrated Gradients
- Attribution maps showing input feature importance
- Satisfies axiomatic properties

## References

See `LITERATURE.md` for detailed literature review including:
- APTOS 2019 winning solutions
- Google/Verily DR detection paper
- Vision Transformer papers
- Medical imaging explainability

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite:

```bibtex
@misc{multiscale_lesion_vit_dr,
  title={Adaptive Multi-Scale Explainable Vision Transformer with Lesion-Aware Attention for Early Diabetic Retinopathy Grading},
  year={2024},
  note={Implementation repository}
}
```
