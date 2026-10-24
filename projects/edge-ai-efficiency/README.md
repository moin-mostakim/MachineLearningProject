# Energy-Efficient Edge AI via Neural Architecture Search and Quantization-Aware Training

This project implements a comprehensive pipeline for creating efficient neural networks suitable for edge deployment, combining Neural Architecture Search (NAS) with Quantization-Aware Training (QAT).

## Overview

Edge AI deployment requires models that are:
- **Compact**: Small memory footprint for limited device storage
- **Fast**: Low latency for real-time inference
- **Accurate**: Maintain high accuracy despite compression
- **Energy-efficient**: Minimize power consumption

This project addresses these requirements through:
1. **Efficient baseline architectures** (MobileNetV2, EfficientNet-Lite)
2. **Neural Architecture Search** (Random Search, Evolutionary Search)
3. **Quantization-Aware Training** (INT8/INT4 quantization)
4. **Combined NAS + QAT pipeline**

## Project Structure

```
edge-ai-efficiency/
├── config.py              # Configuration with hyperparameter justifications
├── main.py               # Main experiment runner
├── requirements.txt      # Dependencies
├── README.md            # This file
├── data/
│   ├── __init__.py
│   └── datasets.py      # Data loading utilities
├── models/
│   ├── __init__.py
│   ├── mobilenet.py     # MobileNetV2 implementation
│   ├── efficientnet.py  # EfficientNet-Lite implementation
│   └── search_space.py  # NAS search space definition
├── nas/
│   ├── __init__.py
│   ├── random_search.py # Random search algorithm
│   ├── evolutionary.py  # Evolutionary search algorithm
│   └── searcher.py      # Unified NAS interface
├── quantization/
│   ├── __init__.py
│   ├── qat.py          # Quantization-Aware Training
│   └── ptq.py          # Post-Training Quantization
├── experiments/
│   ├── __init__.py
│   ├── trainer.py      # Training utilities
│   └── benchmark.py    # Benchmarking utilities
├── utils/
│   ├── __init__.py
│   ├── metrics.py      # FLOPs, params, accuracy metrics
│   └── profiler.py     # Latency measurement
└── results/            # Experiment outputs
```

## Key Features

### 1. Efficient Baseline Models

- **MobileNetV2**: Inverted residuals with linear bottlenecks
  - Width multiplier support (0.25, 0.5, 0.75, 1.0)
  - Expansion ratio t=6 for depth-wise separable convolutions
  
- **EfficientNet-Lite**: Edge-optimized EfficientNet
  - ReLU6 instead of Swish for mobile efficiency
  - Optional Squeeze-and-Excitation removal
  - Compound scaling support

### 2. Neural Architecture Search

**Search Space (OFA-style)**:
- Kernel sizes: {3, 5}
- Expansion ratios: {3, 4, 6}
- Depth options: {2, 3, 4} blocks per stage
- Width multipliers: {0.5, 0.75, 1.0}

**Algorithms**:
- Random Search: Simple baseline
- Evolutionary Search: Regularized evolution with tournament selection

### 3. Quantization

**Quantization-Aware Training (QAT)**:
- Fake quantization during training
- Per-channel weight quantization
- Batch normalization folding

**Post-Training Quantization (PTQ)**:
- INT8 static/dynamic quantization
- INT4 simulation for aggressive compression

## Usage

### Basic Run (Debug Mode)

```bash
cd /workspace/projects/edge-ai-efficiency
python main.py --debug
```

### Full Training

```bash
python main.py --debug=false --device cuda
```

### Configuration

Edit `config.py` to modify:
- Dataset (CIFAR-10 or ImageNette)
- Training hyperparameters
- NAS search space
- Quantization settings

## Benchmarks

The project measures:
- **Accuracy**: Top-1 accuracy on validation set
- **Latency**: CPU/GPU inference time
- **Model Size**: Memory footprint in MB
- **FLOPs**: Computational cost

### Example Results (CIFAR-10, Debug Mode)

| Model | Params | Size | FLOPs | Accuracy | Latency |
|-------|--------|------|-------|----------|---------|
| MobileNetV2-1.0 | 2.3M | 9.0 MB | 91M | ~85% | 5.2ms |
| MobileNetV2-0.5 | 0.7M | 2.8 MB | 27M | ~80% | 2.1ms |
| EfficientNet-Lite | 1.5M | 5.8 MB | 65M | ~83% | 4.0ms |
| NAS-Found | ~1.0M | ~4.0 MB | ~40M | ~82% | ~3.0ms |

### Quantization Impact

| Method | Accuracy | Size Reduction |
|--------|----------|----------------|
| Float32 | 100% (baseline) | 1x |
| INT8 | 98-99% | 4x |
| INT4 | 92-96% | 8x |

## Key References

1. **MobileNetV2** - Sandler et al., CVPR 2018
   - Inverted residuals and linear bottlenecks
   - https://arxiv.org/abs/1801.04381

2. **EfficientNet** - Tan & Le, ICML 2019
   - Compound scaling for CNNs
   - https://arxiv.org/abs/1905.11946

3. **Once-for-All NAS** - Cai et al., ICLR 2020
   - Train once, deploy anywhere
   - https://arxiv.org/abs/1908.09791

4. **Quantization** - Jacob et al., CVPR 2018
   - Quantization and training for efficient inference
   - https://arxiv.org/abs/1712.05877

5. **Regularized Evolution** - Real et al., AAAI 2019
   - Evolution for architecture search
   - https://arxiv.org/abs/1802.01548

## Hyperparameter Justifications

See `config.py` for detailed justifications of all hyperparameters based on the literature.

Key choices:
- **Batch size 128**: Standard for CIFAR-10, balances memory and convergence
- **Learning rate 0.1**: Standard for SGD with momentum
- **Weight decay 4e-5**: MobileNet standard, prevents overfitting
- **Expansion ratio 6**: MobileNetV2 default for inverted residuals
- **QAT learning rate 1e-4**: 10x smaller for fine-tuning

## License

MIT License
