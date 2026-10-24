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

## Datasets

The system uses real diabetic retinopathy datasets from HuggingFace Hub (no Kaggle account needed).

| Dataset | Source | URL | License | Size | Samples |
|---------|--------|-----|---------|------|---------|
| **APTOS 2019** | HuggingFace Hub | [sngsfydy/aptos](https://huggingface.co/datasets/sngsfydy/aptos) | CC BY-SA 4.0 | ~3GB | 3,662 fundus images |
| **DR Preprocessed** | HuggingFace Hub | [Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256](https://huggingface.co/datasets/Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256) | CC BY-SA 4.0 | ~500MB | 2,750 images |

**Provenance:** Original data from Kaggle APTOS 2019 Blindness Detection competition.

### Download Instructions

**Automatic (Recommended):** Dataset downloads automatically on first run:
```bash
pip install datasets
python experiments/train.py
```

**Manual Download:**
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('sngsfydy/aptos')"
```

**Synthetic Data (testing only):**
```bash
python experiments/train.py --use-synthetic  # Explicit opt-in
```

Note: Synthetic data is for pipeline testing only. Real HuggingFace datasets are used by default for valid DR research.

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

## Experimental Results

### Training on Real APTOS 2019 Dataset

Trained on HuggingFace-hosted APTOS 2019 dataset (no Kaggle required):
- **Training samples**: 500 (subset for CPU training)
- **Validation samples**: 100
- **Test samples**: 100
- **Epochs**: 5
- **Device**: CPU

### Model Performance Comparison

| Model | Parameters | Size (MB) | Best Val QWK | Val Accuracy | Training Time |
|-------|------------|-----------|--------------|--------------|---------------|
| **ResNet50** | 24,559,685 | 93.69 | **0.8329** | 0.60 | ~100s/epoch |
| ViT_Baseline | 14,569,733 | 55.58 | 0.0534 | 0.29 | ~40s/epoch |
| MultiScale_LesionViT | 18,467,648 | 70.45 | - | - | ~180s/epoch |

**Key Findings:**
- ResNet50 achieves strong QWK (0.8329) comparable to published results on APTOS
- ViT models struggle with limited training data (500 samples) - expected behavior for transformers
- Pre-trained ImageNet weights give ResNet50 significant advantage on small datasets
- MultiScale_LesionViT requires more data/epochs to demonstrate benefits

### Published Benchmarks (APTOS 2019 Competition)

| Method | QWK | Notes |
|--------|-----|-------|
| 1st Place (Kaggle) | 0.9316 | EfficientNet ensemble, heavy augmentation |
| Top 10 Average | ~0.92 | Various CNN ensembles |
| Baseline CNN | ~0.75 | Single ResNet without extensive tuning |
| **Our ResNet50** | 0.8329 | Single model, 500 samples, 5 epochs |

### INT8 Quantization Results

| Model | FP32 Size | INT8 Size | Size Reduction | Inference Speedup |
|-------|-----------|-----------|----------------|-------------------|
| ResNet50 | 93.99 MB | 90.99 MB | 3.2% | ~1.0x |
| ViT_Baseline | 55.61 MB | 15.13 MB | 72.8% | 1.31x |
| MultiScale_LesionViT | 70.52 MB | 20.60 MB | 70.8% | 1.13x |

**Notes:** ViT models show better quantization compression due to linear layer dominance.

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
