# Literature Review: Diabetic Retinopathy Detection and Grading

## Overview

This document reviews the key literature informing our approach to Diabetic Retinopathy (DR) grading using Vision Transformers with lesion-aware attention.

## Key Datasets

### 1. APTOS 2019 Blindness Detection
- **Source**: Kaggle competition by Asia Pacific Tele-Ophthalmology Society
- **Size**: ~3,600 training images, ~1,800 test images
- **Classes**: 5-class severity scale (0-4)
- **Distribution**: Highly imbalanced (~50% No DR)
- **Reference**: [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)

### 2. EyePACS / Kaggle Diabetic Retinopathy Detection (2015)
- **Size**: 88,702 images
- **Significance**: Largest public DR dataset
- **Reference**: Cuadros & Bresnick (2009), "EyePACS: An Adaptable Telemedicine System"

### 3. Messidor / Messidor-2
- **Size**: 1,200 / 1,748 images
- **Source**: French research program
- **Reference**: Decencière et al. (2014)

### 4. IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Size**: 516 images with pixel-level lesion annotations
- **Significance**: Ground truth for lesion detection
- **Reference**: Porwal et al. (2018)

## APTOS 2019 Winning Solutions

### 1st Place Solution
- **Approach**: Ensemble of EfficientNets (B3-B5) with heavy augmentation
- **Key Techniques**:
  - Multi-scale training (224, 256, 288, 380, 456, 528)
  - Test-time augmentation (8 variations)
  - Threshold optimization for ordinal regression
  - Class-balanced sampling
- **QWK Score**: 0.936 (private leaderboard)

### 2nd Place Solution
- **Approach**: SE-ResNeXt ensemble with regression head
- **Key Techniques**:
  - Treating DR grading as regression problem
  - Focal loss for class imbalance
  - Ben Graham's preprocessing (circle crop + normalization)
- **Reference**: [Kaggle Discussion](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion)

### 3rd Place Solution
- **Approach**: EfficientNet-B4 with pseudo-labeling
- **Key Techniques**:
  - 2015 competition data for pretraining
  - Label smoothing
  - Mixup augmentation

## Deep Learning for DR Detection

### Seminal Work: Google/Verily (2016)
- **Paper**: Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs"
- **Journal**: JAMA
- **Key Findings**:
  - Inception-v3 achieved AUC 0.991 for referable DR
  - Performance comparable to ophthalmologists
  - Validated on EyePACS and Messidor-2

### Reproducibility Study
- **Paper**: Voets et al. (2019), "Reproduction study using public dataset of the development and validation of a deep learning algorithm for detection of diabetic retinopathy"
- **Key Points**:
  - Reproduced Google's results with lower performance
  - Highlighted importance of preprocessing
  - Discussed data quality issues in public datasets

## Vision Transformers for Medical Imaging

### Original ViT
- **Paper**: Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Key Insight**: Transformers can match or exceed CNNs on image classification when pretrained on large datasets

### DeiT (Data-efficient Image Transformers)
- **Paper**: Touvron et al. (2021), "Training data-efficient image transformers"
- **Relevance**: Better training recipes for ViT on smaller datasets (important for medical imaging)
- **Recommendations**:
  - Learning rate: 1e-4 to 3e-4
  - Weight decay: 0.05
  - Strong augmentation (RandAugment, MixUp)

### Swin Transformer
- **Paper**: Liu et al. (2021), "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Relevance**: Multi-scale processing through hierarchical structure

### Medical Imaging Applications
- **TransMed** (2021): Transformer for medical image segmentation
- **COVID-VIT** (2021): ViT for COVID-19 detection from CT
- **TransFuse** (2021): Combining CNNs and Transformers for medical segmentation

## Attention-based Methods for DR

### Attention U-Net
- **Paper**: Schlemper et al. (2019), "Attention gated networks: Learning to leverage salient regions in medical images"
- **Contribution**: Attention gates for focusing on relevant regions
- **Application**: Segmentation with interpretability

### CABNet
- **Paper**: He et al. (2020), "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading"
- **Approach**: Category-specific attention for handling imbalance
- **Relevance**: Lesion-type specific attention (similar to our approach)

### Lesion-based Approaches
- **Paper**: Foo et al. (2020), "Multi-task Learning for Diabetic Retinopathy Grading and Lesion Segmentation"
- **Approach**: Joint learning of grading and lesion detection
- **Key Insight**: Lesion features improve grading accuracy

## Explainability in Medical AI

### Grad-CAM
- **Paper**: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks"
- **Method**: Gradient-weighted class activation maps
- **Use**: Standard for CNN visualization in medical imaging

### Integrated Gradients
- **Paper**: Sundararajan et al. (2017), "Axiomatic Attribution for Deep Networks"
- **Method**: Attribution along gradient path from baseline
- **Advantage**: Satisfies sensitivity and implementation invariance axioms

### Attention Rollout
- **Paper**: Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
- **Method**: Propagating attention weights through layers
- **Use**: ViT-specific explainability

### Clinical Validation
- **Paper**: Sayres et al. (2019), "Using a Deep Learning Algorithm and Integrated Gradients Explanation to Assist Grading for Diabetic Retinopathy"
- **Finding**: AI explanations can help clinicians, but need careful design

## Ordinal Regression for DR

### Problem Definition
- DR grades are ordinal: 0 < 1 < 2 < 3 < 4
- Standard classification ignores ordering
- Misclassifying 0 as 4 worse than 0 as 1

### Approaches
1. **Threshold-based**: Predict continuous value, bin with thresholds
2. **Cumulative Link Models**: Model P(Y > k) for each threshold
3. **Soft Labels**: Use ordinal-aware label smoothing

### Reference
- **Paper**: Niu et al. (2016), "Ordinal Regression with Multiple Output CNN for Age Estimation"
- **Application to DR**: Several APTOS solutions used ordinal regression

## Quadratic Weighted Kappa (QWK)

### Definition
- Cohen's Kappa with quadratic weights
- Standard metric for DR grading competitions
- Formula: κ = 1 - (Σw_ij·O_ij) / (Σw_ij·E_ij)
- Where w_ij = (i-j)² / (N-1)²

### Interpretation
- 0.0-0.20: Slight agreement
- 0.21-0.40: Fair agreement  
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

### Reference
- Cohen (1968), "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit"

## State-of-the-Art Performance

| Dataset | Method | QWK | AUC |
|---------|--------|-----|-----|
| APTOS 2019 | EfficientNet Ensemble | 0.936 | - |
| EyePACS | Inception-v3 (Google) | - | 0.991 |
| Messidor-2 | Inception-v3 (Google) | - | 0.990 |
| IDRiD | EfficientNet-B5 | 0.911 | - |

## Our Contribution

Based on this literature, our Multi-Scale Lesion-Aware ViT addresses several gaps:

1. **Multi-scale Processing**: Captures lesions at different sizes (microaneurysms vs hemorrhages)
2. **Lesion-aware Attention**: Dedicated attention heads for different lesion types
3. **Built-in Explainability**: Attention maps show what the model focuses on
4. **Ordinal-aware Training**: Loss function respects DR grade ordering
5. **Efficient Architecture**: Competitive with ensembles using single model

## References

1. Gulshan V, et al. (2016). Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy. JAMA.
2. Dosovitskiy A, et al. (2020). An Image is Worth 16x16 Words. ICLR.
3. Touvron H, et al. (2021). Training data-efficient image transformers. ICML.
4. Liu Z, et al. (2021). Swin Transformer. ICCV.
5. Schlemper J, et al. (2019). Attention gated networks. MedIA.
6. Selvaraju RR, et al. (2017). Grad-CAM. ICCV.
7. Voets M, et al. (2019). Reproduction study using public dataset. PLOS ONE.
8. Cohen J. (1968). Weighted kappa. Psychological Bulletin.
9. Porwal P, et al. (2018). IDRiD: Diabetic Retinopathy Dataset. Data.
