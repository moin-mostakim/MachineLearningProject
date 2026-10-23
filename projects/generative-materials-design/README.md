# Generative AI for Inverse Design of Functional Materials with Multi-Objective Optimization

This project implements a Variational Autoencoder (VAE) for molecular generation with multi-objective optimization capabilities, targeting inverse design of functional materials.

## Overview

### Problem Statement
Design molecules with desired properties by learning a latent representation and optimizing in that space for multiple objectives simultaneously.

### Method
- **Model**: Molecular VAE with GNN encoder and MLP decoder
- **Dataset**: QM9 (~134k molecules with quantum mechanical properties)
- **Optimization**: Multi-objective Pareto optimization (NSGA-II style)
- **Properties**: HOMO, LUMO, Gap (electronic properties)

## Project Structure

```
generative-materials-design/
├── config.py              # Hyperparameters with justifications
├── main.py                # Training entry point
├── requirements.txt       # Dependencies
├── data/
│   ├── __init__.py
│   └── dataset.py         # QM9 dataset loader
├── models/
│   ├── __init__.py
│   └── vae.py             # VAE architecture
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Training loop
│   └── losses.py          # Loss functions
├── optimization/
│   ├── __init__.py
│   └── multi_objective.py # Pareto optimization
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         # Generation metrics
│   └── quantization.py    # INT8 comparison
└── utils/
    ├── __init__.py
    ├── visualization.py   # Plotting utilities
    └── resource_monitor.py # Resource tracking
```

## Literature References

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Junction Tree VAE (Jin et al.) | ICML 2018 | 100% validity via tree decomposition |
| GraphVAE (Simonovsky & Komodakis) | 2018 | Direct graph generation |
| MolGAN (De Cao & Kipf) | 2018 | Adversarial + RL for properties |
| CGVAE (Liu et al.) | NeurIPS 2018 | Constrained graph VAE |
| SELFIES (Krenn et al.) | 2020 | String representation with validity |

## Hyperparameter Justifications

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `latent_dim` | 64 | Standard for molecular VAEs (JT-VAE uses 56) |
| `hidden_dim` | 256 | Following GraphVAE recommendations |
| `num_layers` | 3 | 3-hop message passing covers most substructures |
| `batch_size` | 128 | QM9 molecules are small (~9 heavy atoms avg) |
| `learning_rate` | 1e-3 | Adam optimizer default, works well for VAEs |
| `kl_weight` | 0.1 | β-VAE weighting to prevent posterior collapse |
| `num_epochs` | 100 | Sufficient for QM9 convergence |

## Benchmarks

### Generation Metrics (Target)
- **Validity**: >90% (chemically valid molecules)
- **Uniqueness**: >80% (distinct molecules)
- **Novelty**: >90% (not in training set)

### Literature Comparison

| Method | Validity | Uniqueness | Novelty |
|--------|----------|------------|---------|
| Junction Tree VAE | 100.0% | N/A | 100.0% |
| GraphVAE | 55.7% | 76.0% | 61.6% |
| MolGAN | 98.1% | 10.4% | 94.2% |
| Our VAE | >90% | >80% | >90% |

## Usage

### Debug Mode (Quick Test)
```bash
python main.py --debug
```

### Full Training
```bash
python main.py
```

### Evaluation Only
```bash
python main.py --eval-only --checkpoint best_model.pt
```

## Multi-Objective Optimization

The project implements three approaches for comparison:
1. **Random Search**: Sample random latent vectors
2. **Single-Objective**: Optimize for one property (HOMO)
3. **Multi-Objective (Pareto)**: NSGA-II style optimization for HOMO, LUMO, Gap

## INT8 Quantization

Compares FP32 vs INT8 quantized models:
- Model size reduction
- Inference speedup
- Accuracy degradation

## Output

Results are saved to `outputs/run_<timestamp>/`:
- `checkpoints/` - Model checkpoints
- `training_curves.png` - Loss plots
- `evaluation_results.json` - Generation metrics
- `optimization_results.json` - Pareto optimization results
- `quantization_results.json` - INT8 comparison
- `config.json` - Configuration used

## Resource Requirements

- **Memory**: ~2GB RAM (debug), ~8GB RAM (full)
- **GPU**: Optional, works on CPU
- **Storage**: ~500MB for QM9 dataset
- **Time**: ~5 min (debug), ~2 hours (full, CPU)
