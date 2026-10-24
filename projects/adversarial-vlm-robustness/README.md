# Adversarial Robustness of Vision-Language Models through Certified Training and Distributionally Robust Optimization

This project implements and compares multiple approaches for improving the adversarial robustness of Vision-Language Models (VLMs), including standard training, adversarial training, certified defenses, and distributionally robust optimization.

## Overview

Modern vision-language models achieve impressive performance but remain vulnerable to adversarial perturbations. This project provides a comprehensive framework for:

1. **Standard Training** - Baseline training without adversarial examples
2. **Adversarial Training (PGD)** - Training on worst-case adversarial examples (Madry et al., 2018)
3. **Certified Training** - Randomized smoothing for provable robustness (Cohen et al., 2019)
4. **DRO** - Distributionally robust optimization (Sinha et al., 2018)

## Project Structure

```
adversarial-vlm-robustness/
├── config.py           # Hyperparameters with justifications
├── main.py             # Main experiment runner
├── models/
│   ├── vision_encoder.py   # ResNet-based vision encoder
│   ├── text_encoder.py     # Text/label encoder
│   └── vlm.py              # Vision-Language Model (CLIP-style)
├── trainers/
│   ├── base.py             # Base trainer class
│   ├── standard.py         # Standard training
│   ├── adversarial.py      # PGD adversarial training
│   ├── certified.py        # Randomized smoothing training
│   └── dro.py              # Distributionally robust optimization
├── attacks/
│   ├── pgd.py              # PGD attack implementation
│   └── certified.py        # Certification via randomized smoothing
├── data/
│   └── datasets.py         # CIFAR-10 data loading
└── utils/
    ├── metrics.py          # Evaluation metrics
    ├── monitoring.py       # Resource monitoring
    └── quantization.py     # INT8 quantization
```

## Datasets

### CIFAR-10 (Primary Dataset)

| Attribute | Value |
|-----------|-------|
| **Name** | CIFAR-10 |
| **Source** | torchvision (auto-download) |
| **Official URL** | https://www.cs.toronto.edu/~kriz/cifar.html |
| **HuggingFace** | https://huggingface.co/datasets/cifar10 |
| **License** | MIT-like (research use) |
| **Size** | ~170MB |
| **Train Samples** | 50,000 |
| **Test Samples** | 10,000 |
| **Image Size** | 32×32 RGB |
| **Classes** | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |

### Download Instructions

The dataset downloads automatically when running the experiment:

```bash
# Real CIFAR-10 (default - downloads automatically)
python main.py --debug

# Synthetic data for quick testing (no download)
python main.py --debug --use-synthetic
```

Manual download:
```python
import torchvision
torchvision.datasets.CIFAR10(root='./data', download=True)
```

Or via HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("cifar10")
```

## Methods

### 1. Standard Training (Baseline)
Simple cross-entropy training without adversarial examples.

### 2. Adversarial Training (Madry et al., 2018)
```
min_θ E[(x,y)~D] [max_{δ: ||δ||∞ ≤ ε} L(f_θ(x+δ), y)]
```
Uses PGD to generate worst-case perturbations during training.

**Key hyperparameters:**
- ε = 8/255 (standard for CIFAR-10)
- α = 2/255 (step size)
- K = 7 (PGD steps)

### 3. Certified Training (Cohen et al., 2019)
Trains with Gaussian noise augmentation to enable provable robustness via randomized smoothing.

**Certified radius:** `r = σ * Φ⁻¹(p_A)` where p_A is the probability of the most likely class.

**Key hyperparameters:**
- σ = 0.25 (noise standard deviation)
- n = 1000 (Monte Carlo samples for certification)

### 4. Distributionally Robust Optimization (Sinha et al., 2018)
```
min_θ sup_{Q: W(Q, P_n) ≤ ρ} E_Q[L(θ; ξ)]
```
Optimizes for worst-case distributions within a Wasserstein ball.

**Key hyperparameters:**
- ρ = 0.1 (Wasserstein ball radius)
- γ = 1.0 (Lagrangian dual variable)

## Hyperparameter Justifications

All hyperparameters in `config.py` include literature references:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| ε (epsilon) | 8/255 | Standard CIFAR-10 benchmark (Madry et al., 2018) |
| α (step size) | 2/255 | ε/4 for stable gradient ascent |
| K (PGD steps) | 7 | Sufficient for strong attacks (7-20 typical) |
| σ (smoothing) | 0.25 | Balances certified radius and clean accuracy |
| ρ (DRO radius) | 0.1 | Moderate distributional robustness |

## Usage

### Debug Mode (Quick Test)
```bash
python main.py --debug
```
Runs with reduced epochs (3) and batches (10) for testing.

### Full Training
```bash
python main.py
```

### Select Specific Methods
```bash
python main.py --methods standard adversarial
```

## Benchmarks

### Metrics
- **Clean Accuracy**: Accuracy on unperturbed test images
- **Adversarial Accuracy**: Accuracy under PGD attack (ε=8/255)
- **Certified Accuracy**: Accuracy with provable guarantees at radius r

### Literature Comparison (CIFAR-10, ε=8/255)

| Method | Clean Acc | PGD Acc | Certified |
|--------|-----------|---------|-----------|
| Madry et al. (2018) | 87.3% | 45.8% | - |
| TRADES (Zhang 2019) | 84.9% | 56.6% | - |
| Cohen et al. (2019) | 75.0% | - | 60%@r=0.5 |
| Wong et al. (2020) | 83.3% | 43.2% | - |

## Model Architecture

**SimpleCLIP**: CLIP-style contrastive model for classification
- Vision Encoder: ResNet-18 (modified for CIFAR-10)
- Text Encoder: Learnable class embeddings
- Projection dimension: 256

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Vision Encoder | ~11M |
| Text Encoder | ~2.5K |
| Total | ~11M |

## INT8 Quantization

The project includes INT8 quantization comparison:
- Dynamic quantization reduces model size ~4x
- Minimal accuracy degradation (<1%)
- Significant inference speedup on CPU

## Key References

1. **Madry et al. (2018)** - "Towards Deep Learning Models Resistant to Adversarial Attacks"
   - Introduced PGD adversarial training
   - https://arxiv.org/abs/1706.06083

2. **Cohen et al. (2019)** - "Certified Adversarial Robustness via Randomized Smoothing"
   - Provable robustness via randomized smoothing
   - https://arxiv.org/abs/1902.02918

3. **Sinha et al. (2018)** - "Certifiable Distributional Robustness with Principled Adversarial Training"
   - Wasserstein DRO for robust optimization
   - https://arxiv.org/abs/1710.10571

4. **Zhang et al. (2019)** - "Theoretically Principled Trade-off between Robustness and Accuracy"
   - TRADES defense
   - https://arxiv.org/abs/1901.08573

5. **Radford et al. (2021)** - "Learning Transferable Visual Models From Natural Language Supervision"
   - CLIP architecture
   - https://arxiv.org/abs/2103.00020

## Resource Monitoring

Built-in resource monitoring tracks:
- CPU utilization
- Memory usage
- GPU memory (when available)
- Training duration

## License

MIT License
