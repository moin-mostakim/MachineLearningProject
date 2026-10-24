# Hybrid Foundation Model with Kolmogorov-Arnold Network for Diabetic Retinopathy Diagnosis

This project implements and benchmarks a novel hybrid architecture combining CNN feature extraction with Kolmogorov-Arnold Network (KAN) classification for automated diabetic retinopathy (DR) grading.

## Overview

Diabetic retinopathy is a leading cause of preventable blindness. This project explores using KAN layers - which employ learnable activation functions via B-splines - as a parameter-efficient alternative to traditional MLP classifiers in medical imaging pipelines.

## Methods

### Models Implemented

1. **MLP Baseline**: Standard Multi-Layer Perceptron classifier
2. **CNN Baseline**: ResNet-18 with ImageNet pretraining
3. **KAN Classifier**: Kolmogorov-Arnold Network with learnable B-spline activations
4. **Hybrid CNN+KAN**: ResNet-18 feature extractor + KAN classification head

### Key Innovation: Kolmogorov-Arnold Networks

Based on Liu et al. (2024), KAN replaces fixed activation functions with learnable univariate functions (B-splines) on network edges:

- **Mathematical Foundation**: Kolmogorov-Arnold representation theorem guarantees any multivariate continuous function can be decomposed into univariate functions
- **Architecture**: `output = scale_base * base_fn(x) + scale_spline * spline(x)`
- **Efficiency**: 2-4x fewer parameters than equivalent MLPs while maintaining accuracy

## Datasets

The system supports real diabetic retinopathy datasets. Due to patient privacy, these require Kaggle API authentication.

| Dataset | Source | URL | License | Size | Samples |
|---------|--------|-----|---------|------|---------|
| **APTOS 2019** | Kaggle | [aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection) | Competition | ~15GB | 3,662 train, 1,928 test |
| **EyePACS** | Kaggle | [diabetic-retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) | Competition | ~89GB | 35,126 train |

**Classes**: 5-level DR grading (No DR, Mild, Moderate, Severe, Proliferative)

**Requires:** Kaggle API key (`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables)

### Download Instructions

**APTOS 2019 via Kaggle:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/aptos2019/
```

**Synthetic Data (pipeline testing only):**
```bash
python main.py --debug  # Auto-generates synthetic if real data unavailable
```

Note: Synthetic data is for pipeline testing only. Real APTOS 2019 data is required for valid DR research.

## Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)**: Primary metric for ordinal classification
- **Accuracy**: Overall classification accuracy
- **AUC (ROC)**: Multi-class area under ROC curve
- **F1-Score**: Per-class and macro-averaged F1

## Project Structure

```
kan-diabetic-retinopathy/
├── config.py              # Hyperparameters with literature justifications
├── main.py                # Main experiment runner
├── requirements.txt       # Dependencies
├── models/
│   ├── kan_layer.py       # KAN layer implementation (B-spline basis)
│   ├── kan_model.py       # KAN classifier
│   ├── mlp_model.py       # MLP baseline
│   ├── cnn_model.py       # CNN baseline (ResNet)
│   └── hybrid_model.py    # Hybrid CNN+KAN model
├── data/
│   └── dataset.py         # Dataset loading and synthetic generation
├── utils/
│   ├── metrics.py         # Evaluation metrics (QWK, AUC, etc.)
│   ├── training.py        # Training utilities
│   ├── quantization.py    # INT8 quantization comparison
│   ├── resource_monitor.py # Resource tracking
│   └── visualization.py   # Result plotting
├── results/               # Experiment outputs
└── checkpoints/           # Model weights
```

## Usage

### Run in Debug Mode
```bash
cd /workspace/projects/kan-diabetic-retinopathy
python main.py --debug
```

### Full Training
```bash
python main.py --models mlp kan cnn hybrid_kan
```

### Specific Models
```bash
python main.py --models hybrid_kan --output-dir ./results/hybrid_only
```

## Hyperparameter Configuration

All hyperparameters are documented in `config.py` with literature justifications:

### KAN Parameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| grid_size | 5 | Liu et al. recommend 3-10; 5 is default |
| spline_order | 3 | Cubic splines for smoothness |
| scale_base | 1.0 | Equal weighting of base and spline paths |
| base_activation | SiLU | Recommended by Liu et al. |

### Training Parameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| learning_rate | 1e-4 | Lower for fine-tuning pretrained models |
| optimizer | AdamW | Better weight decay handling |
| scheduler | Cosine | Smooth decay, works well with pretrained |
| epochs | 50 | Standard for medical imaging transfer learning |

## Expected Results

### Parameter Efficiency
| Model | Total Params | Classifier Params |
|-------|-------------|-------------------|
| MLP Classifier | ~400K | ~400K |
| KAN Classifier | ~100K | ~100K |
| CNN (ResNet-18) | 11.7M | 2.5K |
| Hybrid CNN+KAN | 11.6M | ~100K |
| Hybrid CNN+MLP | 11.9M | ~200K |

### INT8 Quantization
- Expected size reduction: 2-4x
- Expected accuracy drop: <0.5%
- Speed improvement: 1.5-3x on CPU

## Literature References

1. **Liu et al. (2024)** - "KAN: Kolmogorov-Arnold Networks" - arXiv:2404.19756
   - Introduces learnable B-spline activations
   - Shows 2-3x parameter efficiency vs MLP

2. **Gulshan et al. (2016)** - JAMA
   - Google's DR detection with deep learning
   - 97.5% sensitivity, 93.4% specificity

3. **APTOS 2019** - Kaggle Competition
   - Standard DR detection benchmark
   - 5-class severity grading

4. **He et al. (2016)** - "Deep Residual Learning" - CVPR
   - ResNet architecture for image classification

## License

Research use only. See LICENSE file for details.
