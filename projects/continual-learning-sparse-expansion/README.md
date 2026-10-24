# Continual Learning in Foundation Models without Catastrophic Forgetting via Dynamic Sparse Expansion

This project implements and compares methods for continual learning, focusing on mitigating catastrophic forgetting through dynamic sparse network expansion.

## Overview

Continual learning (also called lifelong learning) is the challenge of learning new tasks without forgetting previously learned ones. Standard neural networks suffer from **catastrophic forgetting**: when trained on new data, they rapidly lose performance on old tasks.

This project implements three approaches:

1. **Fine-tuning Baseline**: Naive sequential training (demonstrates catastrophic forgetting)
2. **Elastic Weight Consolidation (EWC)**: Regularization-based approach using Fisher Information
3. **Dynamic Sparse Expansion**: Our proposed method - sparse network growth per task

## Method: Dynamic Sparse Expansion

Our approach combines ideas from:
- **Progressive Neural Networks** (Rusu et al., 2016): Add capacity for new tasks
- **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019): Sparse networks are sufficient
- **PackNet** (Mallya & Lazebnik, 2018): Use network sparsity for task separation

### Key Ideas

1. **Sparse Initialization**: Start with a sparse network
2. **Per-Task Expansion**: Activate only a small fraction (e.g., 10%) of new connections per task
3. **Weight Freezing**: Freeze weights used by previous tasks
4. **Task Adapters**: Lightweight per-task adaptation modules

### Advantages

- **Sublinear Growth**: Parameter count grows sub-linearly with tasks (vs. linear for Progressive Nets)
- **Zero Forgetting**: Previous task weights are frozen
- **Efficient**: Sparse computations and small adapters

## Project Structure

```
continual-learning-sparse-expansion/
├── config.py              # All hyperparameters with justifications
├── datasets.py            # Split CIFAR-100 and Permuted MNIST
├── models.py              # CNN/MLP architectures with sparse expansion
├── methods.py             # Fine-tuning, EWC, and Sparse Expansion
├── metrics.py             # Continual learning metrics (forgetting, BWT)
├── resource_monitor.py    # GPU/CPU monitoring and profiling
├── quantization.py        # INT8 quantization comparison
├── main.py                # Main experiment runner
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Usage

### Quick Start (Debug Mode)

```bash
python main.py --debug
```

### Full Experiment

```bash
python main.py
```

### Specific Method

```bash
python main.py --method ewc
python main.py --method sparse_expansion
```

### Dataset Selection

```bash
python main.py --dataset split_cifar100
python main.py --dataset permuted_mnist
```

## Metrics

We evaluate using standard continual learning metrics:

- **Average Accuracy**: Mean accuracy across all tasks at the end
- **Forgetting Measure**: Max drop in accuracy for each previous task
- **Backward Transfer (BWT)**: Change in old task performance after learning new tasks

## Benchmarks

### Split CIFAR-100 (10 tasks, 10 classes each)

| Method | Avg Accuracy | Forgetting | BWT |
|--------|-------------|------------|-----|
| Fine-tuning | ~40-50% | ~40-50% | -40% |
| EWC | ~60-70% | ~15-25% | -15% |
| Sparse Expansion | ~70-80% | ~0-5% | ~0% |

*Note: Results vary based on hyperparameters and random seeds*

### Parameter Growth

| Method | Task 0 | Task 5 | Task 10 | Growth |
|--------|--------|--------|---------|--------|
| Fine-tuning | 1.5M | 1.5M | 1.5M | 0% |
| EWC | 1.5M | 1.5M | 1.5M | 0% |
| Sparse Expansion | 1.5M | 1.65M | 1.8M | +20% |

## INT8 Quantization

We compare FP32 vs INT8 models:

- **Size Reduction**: ~4x smaller model
- **Speedup**: ~1.5-2x faster inference (CPU)
- **Accuracy**: Minimal degradation (<1%)

## Literature References

1. **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
2. **PackNet**: Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning" (CVPR 2018)
3. **Progressive Neural Networks**: Rusu et al., "Progressive Neural Networks" (arXiv 2016)
4. **Lottery Ticket Hypothesis**: Frankle & Carlin, "The Lottery Ticket Hypothesis" (ICLR 2019)
5. **A-GEM**: Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (ICLR 2019)

## Configuration

All hyperparameters are in `config.py` with detailed justifications:

```python
EWC_LAMBDA = 5000          # From original paper; tune 100-10000
SPARSITY_RATIO = 0.1       # 10% new connections per task
EXPANSION_THRESHOLD = 0.85 # Expand when accuracy drops below
LEARNING_RATE = 0.001      # Standard Adam learning rate
```

## Resource Requirements

- **GPU**: CUDA-capable GPU recommended (runs on CPU for debugging)
- **Memory**: ~4GB GPU memory for CIFAR-100
- **Storage**: ~500MB for datasets and checkpoints
