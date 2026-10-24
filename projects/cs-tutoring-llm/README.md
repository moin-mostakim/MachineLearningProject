# CS Tutoring LLM: Intelligent Tutoring System for Computer Science Education

An intelligent tutoring system powered by Large Language Models for Computer Science education, implementing retrieval-augmented generation (RAG) for code explanation and educational assistance.

## Project Overview

This project implements and compares multiple approaches for automated CS tutoring:

1. **BM25 Baseline** - Classic probabilistic retrieval
2. **Fine-tuned LLM** - CodeT5 for code explanation generation
3. **RAG System** - Retrieval-Augmented Generation combining dense retrieval with generation
4. **INT8 Quantization** - Efficient inference with quantized models

## Datasets

The system uses CodeSearchNet from HuggingFace, the standard benchmark for code search and understanding.

| Dataset | Source | URL | License | Size | Samples |
|---------|--------|-----|---------|------|---------|
| **CodeSearchNet** | HuggingFace Hub | [code_search_net](https://huggingface.co/datasets/code_search_net) | MIT | ~2GB | 6M functions (Python: 500K+, JS: 300K+, etc.) |

**Official Source:**
- GitHub: [github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet)
- Reference: Husain et al. (2019) "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search"

### Dataset Details

CodeSearchNet contains code-docstring pairs from 6 programming languages:
- Python (~500K functions)
- JavaScript (~300K functions)
- Java (~500K functions)
- Go (~300K functions)
- Ruby (~50K functions)
- PHP (~500K functions)

For this project, we focus on **Python** code snippets with their documentation strings.

### Download Instructions

**Automatic (Recommended):** Dataset downloads automatically on first run:
```bash
pip install datasets transformers
python main.py --mode debug
```

**Manual Download via HuggingFace:**
```bash
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('code_search_net', 'python')"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CS Tutoring System                            │
├─────────────────────────────────────────────────────────────────┤
│  Input: Student Question                                         │
│         ↓                                                        │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  BM25 Retriever │    │ Dense Retriever │                     │
│  │  (Sparse)       │    │ (MiniLM-L6)     │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           └──────────┬───────────┘                               │
│                      ↓                                           │
│           ┌─────────────────────┐                               │
│           │  Cross-Encoder      │                               │
│           │  Reranker           │                               │
│           └──────────┬──────────┘                               │
│                      ↓                                           │
│           ┌─────────────────────┐                               │
│           │  CodeT5 Generator   │                               │
│           │  (60M params)       │                               │
│           └──────────┬──────────┘                               │
│                      ↓                                           │
│  Output: Educational Explanation                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Model Parameter Counts

| Model | Parameters | Notes |
|-------|------------|-------|
| BM25 | 0 | Non-parametric statistical method |
| CodeBERT-base | 125M | Pre-trained code understanding |
| CodeT5-small | 60M | Code generation/summarization |
| MiniLM-L6-v2 | 22.7M | Sentence embedding retrieval |
| Cross-Encoder | 22M | Reranking |

## Installation

```bash
cd cs-tutoring-llm
pip install -r requirements.txt
```

## Usage

### Debug Mode (Quick Test)
```bash
python main.py --mode debug
```

### Full Training and Evaluation
```bash
python main.py --mode full
```

### Evaluation Only
```bash
python main.py --mode eval_only
```

## Project Structure

```
cs-tutoring-llm/
├── config.py              # Configuration with hyperparameter justifications
├── main.py                # Main execution script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading (CodeSearchNet)
│   ├── bm25_retriever.py  # BM25 baseline retrieval
│   ├── model.py           # Neural model architectures
│   ├── rag_system.py      # RAG implementation
│   ├── trainer.py         # Training utilities
│   ├── evaluator.py       # Evaluation metrics
│   ├── quantization.py    # INT8 quantization
│   └── resource_monitor.py # Resource monitoring
├── data/                  # Data and indices
├── results/               # Output results
├── models/                # Saved models
└── notebooks/             # Jupyter notebooks
```

## Evaluation Metrics

### Answer Relevance
- ROUGE-1, ROUGE-2, ROUGE-L scores
- BERTScore for semantic similarity
- Classification accuracy for relevance prediction

### Code Correctness
- Syntax validation using AST parsing
- Execution testing where applicable

### Educational Quality
- Flesch Reading Ease score
- Flesch-Kincaid Grade Level
- Target: Grade 12-16 (undergraduate level)

## Literature References

1. **CodeBERT** (Feng et al., 2020)
   - "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
   - Pre-training on bimodal data with MLM + Replaced Token Detection

2. **CodeT5** (Wang et al., 2021)
   - "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models"
   - Identifier-aware denoising pre-training

3. **Codex** (Chen et al., 2021)
   - "Evaluating Large Language Models Trained on Code"
   - GPT-based code generation

4. **RAG** (Lewis et al., 2020)
   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - Combining retrieval with generation

5. **CodeSearchNet** (Husain et al., 2019)
   - "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search"
   - Benchmark dataset

## Hyperparameter Justifications

All hyperparameters in `config.py` include detailed justifications based on:
- Original paper recommendations
- Empirical findings from related work
- Resource constraints and efficiency considerations

Key hyperparameters:
- Learning rate: 2e-5 (standard for BERT fine-tuning)
- Batch size: 8 with 4x gradient accumulation (effective 32)
- BM25 k1=1.5, b=0.75 (standard defaults)
- RAG retrieval: 3 documents with reranking

## INT8 Quantization

Dynamic quantization is applied for efficient inference:
- Quantizes Linear layers to INT8
- Typical 2-4x speedup on CPU
- ~4x memory reduction
- Minimal accuracy degradation (<1%)

## Resource Requirements

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM
- **Training**: ~2GB GPU memory for CodeT5-small with batch size 8

## License

This project is for educational and research purposes.

## Citation

If you use this work, please cite the relevant papers listed in the Literature References section.
