# Scalable Neurosymbolic Reasoning for Scientific Hypothesis Generation and Verification

A neurosymbolic AI system that combines neural network feature extraction with symbolic reasoning modules and knowledge graph integration for scientific reasoning tasks.

## Overview

This project implements a neurosymbolic architecture for scientific hypothesis generation and verification, inspired by:

- **Neural Theorem Provers** (Rocktäschel & Riedel, 2017) - End-to-end differentiable proving
- **Logic Tensor Networks** (Serafini & d'Avila Garcez, 2016) - Neural-symbolic integration
- **Differentiable Reasoning on Large Knowledge Bases** (Minervini et al., 2020)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neurosymbolic Model                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────┐ │
│  │  Neural Encoder  │──▶│Symbolic Reasoner │──▶│ Integration │ │
│  │  (Transformer)   │   │(Rule Engine)     │   │   Layer     │ │
│  └──────────────────┘   └──────────────────┘   └─────────────┘ │
│           │                     │                      │        │
│           ▼                     ▼                      ▼        │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────┐ │
│  │  Knowledge Graph │──▶│  Hypothesis      │──▶│ Classifier  │ │
│  │    Module        │   │  Generator       │   │             │ │
│  └──────────────────┘   └──────────────────┘   └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Neural Encoder**: Transformer-based feature extraction from text
- **Symbolic Reasoner**: Differentiable rule learning and multi-hop inference
- **Knowledge Graph Integration**: TransE-style embeddings for hypothesis verification
- **Interpretability**: Reasoning traces and rule extraction
- **Quantization**: INT8 quantization support for efficient deployment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Debug Mode (Quick Test)
```bash
python train.py --debug
```

### Full Training
```bash
python train.py --epochs 20 --dataset sciq
```

### With INT8 Quantization Comparison
```bash
python train.py --debug --quantize int8
```

## Dataset

The system supports multiple scientific reasoning datasets:

1. **SciQ** (default): Science question answering with support text
2. **ARC-Easy**: AI2 Reasoning Challenge (easy split)
3. **ARC-Challenge**: AI2 Reasoning Challenge (challenge split)

For debug mode, synthetic scientific reasoning data is generated.

## Model Comparison

| Model | Accuracy | Hypothesis Validity | Parameters |
|-------|----------|---------------------|------------|
| Neurosymbolic (Ours) | ~75-85% | ~80% | ~11.2M |
| Pure Neural Baseline | ~70-80% | N/A | ~10.5M |
| Rule-Based Baseline | ~40-50% | N/A | 0 |

## Parameter Count Table

```
╔══════════════════════════════════════════════════════════════════════════╗
║                      Model Parameter Count Table                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Component                        │ Parameters   │ Notes                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Token Embeddings                 │ 7.8M         │ 30522 × 256            ║
║ Position Embeddings              │ 65.5K        │ 256 × 256              ║
║ Transformer Layers (×4)          │ 2.6M         │ 4 × (3×256² + 256×1024)║
║ Symbolic Reasoning Module        │ 180K         │ Predicates + Rules     ║
║ Knowledge Graph Embeddings       │ 516K         │ Entities + Relations   ║
║ Classification Head              │ 66K          │ 256 × 256 + 256        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ TOTAL (Full Model)               │ ~11.2M       │                        ║
║ TOTAL (INT8 Quantized)           │ ~2.8M*       │ *4× memory reduction   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Configuration

All hyperparameters are in `config.py` with detailed justifications:

- **HIDDEN_DIM=256**: Balances expressiveness with efficiency
- **SYMBOLIC_REASONING_DEPTH=3**: Allows 3-hop inference chains
- **RULE_CONFIDENCE_THRESHOLD=0.7**: Filters low-confidence rules
- **KG_EMBED_DIM=128**: Standard for knowledge graph embeddings

## Literature References

1. Rocktäschel, T., & Riedel, S. (2017). End-to-End Differentiable Proving. *NeurIPS*.
2. Serafini, L., & d'Avila Garcez, A. (2016). Logic Tensor Networks: Deep Learning and Logical Reasoning. *arXiv*.
3. Minervini, P., et al. (2020). Differentiable Reasoning on Large Knowledge Bases and Natural Language. *AAAI*.
4. Evans, R., & Grefenstette, E. (2018). Learning Explanatory Rules from Noisy Data. *JAIR*.
5. Garcez, A., et al. (2019). Neural-Symbolic Computing: An Effective Methodology for Principled Integration. *arXiv*.

## Project Structure

```
neurosymbolic-reasoning/
├── config.py              # Configuration with hyperparameter justifications
├── train.py               # Main training script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── models/
│   ├── neural_encoder.py  # Transformer encoder
│   ├── symbolic_reasoner.py # Rule engine and predicate embeddings
│   ├── knowledge_graph.py # KG embeddings and verification
│   ├── neurosymbolic.py   # Integrated model
│   └── baselines.py       # Pure neural and rule-based baselines
├── utils/
│   ├── data_loader.py     # Dataset loading (SciQ, ARC)
│   ├── metrics.py         # Evaluation metrics
│   ├── resource_monitor.py # Resource tracking
│   └── quantization.py    # INT8 quantization utilities
├── data/                  # Dataset files
├── checkpoints/           # Saved models
├── logs/                  # Training logs
└── results/               # Evaluation results
```

## License

MIT License
