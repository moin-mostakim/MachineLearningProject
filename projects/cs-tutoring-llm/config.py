"""
Configuration file for CS Tutoring LLM System.

All hyperparameters are centralized here with detailed justifications
based on literature and empirical findings.

References:
- CodeBERT (Feng et al., 2020): Pre-training for Programming Language Understanding
- Codex (Chen et al., 2021): Evaluating Large Language Models on Code
- CodeT5 (Wang et al., 2021): Identifier-aware Unified Pre-trained Encoder-Decoder
- BERT-based Educational QA (Kumar et al., 2020): Automatic Question Answering
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Dataset selection
    dataset_name: str = "code_search_net"
    # Justification: CodeSearchNet is a large-scale benchmark with 6M functions
    # across 6 programming languages, suitable for code understanding tasks.
    # Reference: Husain et al., 2019 "CodeSearchNet Challenge"
    
    dataset_subset: str = "python"
    # Justification: Python is most common in CS education and has clear syntax.
    # Starting with one language allows focused model training.
    
    max_samples_train: int = 10000
    # Justification: For debug/demo mode, limit samples to ensure reasonable
    # training time. Production would use full dataset (~500K Python samples).
    
    max_samples_eval: int = 1000
    # Justification: 1000 samples provides statistically meaningful evaluation
    # while keeping evaluation time reasonable.
    
    max_seq_length: int = 512
    # Justification: Matches BERT/CodeBERT default. Most code snippets and
    # explanations fit within this limit. Longer sequences increase memory
    # quadratically due to attention mechanism.
    
    max_code_length: int = 256
    # Justification: Average function length in CodeSearchNet is ~100 tokens.
    # 256 covers 95th percentile while limiting padding overhead.
    
    max_explanation_length: int = 256
    # Justification: Docstrings/explanations are typically shorter than code.
    # 256 tokens sufficient for detailed explanations.
    
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    # Justification: Standard 80/10/10 split provides sufficient training data
    # while maintaining robust validation and test sets.
    
    seed: int = 42
    # Justification: Fixed seed for reproducibility.


@dataclass
class BM25Config:
    """BM25 retrieval baseline configuration."""
    
    k1: float = 1.5
    # Justification: k1 controls term frequency saturation.
    # Value of 1.5 is standard default from original BM25 paper (Robertson et al., 1994).
    # Higher values give more weight to term frequency.
    
    b: float = 0.75
    # Justification: b controls document length normalization.
    # 0.75 is empirically optimal for most text retrieval tasks.
    # 0 = no length normalization, 1 = full normalization.
    
    top_k: int = 5
    # Justification: Retrieve top 5 relevant documents for RAG context.
    # Balances context richness vs. noise and token limits.
    
    index_path: str = "data/bm25_index.pkl"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Base model selection
    base_model: str = "microsoft/codebert-base"
    # Justification: CodeBERT is pre-trained on bimodal data (code + NL) with
    # MLM and replaced token detection objectives. Shows SOTA on code search,
    # documentation generation. (Feng et al., 2020)
    # Alternative: "Salesforce/codet5-small" for encoder-decoder tasks
    
    generation_model: str = "Salesforce/codet5-small"
    # Justification: CodeT5-small (60M params) provides good balance between
    # performance and computational efficiency for code explanation generation.
    # Pre-trained with identifier-aware denoising objective.
    
    # Model dimensions (for reference, determined by base model)
    hidden_size: int = 768
    # Justification: Standard BERT-base hidden dimension.
    
    num_attention_heads: int = 12
    # Justification: 768/12 = 64 dims per head, standard for BERT-base.
    
    num_hidden_layers: int = 12
    # Justification: 12 layers provides good depth for code understanding.
    
    intermediate_size: int = 3072
    # Justification: 4x hidden_size is standard feed-forward expansion.
    
    # Custom head configuration
    num_labels: int = 2
    # Justification: Binary classification for answer relevance.
    
    dropout_rate: float = 0.1
    # Justification: 0.1 dropout is standard for transformer fine-tuning.
    # Helps prevent overfitting without excessive regularization.


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    learning_rate: float = 2e-5
    # Justification: 2e-5 is recommended for BERT fine-tuning (Devlin et al., 2019).
    # Higher rates cause catastrophic forgetting of pre-trained knowledge.
    # Lower rates require more epochs.
    
    weight_decay: float = 0.01
    # Justification: L2 regularization coefficient. 0.01 is standard for
    # AdamW optimizer with transformers.
    
    adam_epsilon: float = 1e-8
    # Justification: Numerical stability term for Adam optimizer.
    
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    # Justification: Standard Adam momentum parameters.
    
    max_grad_norm: float = 1.0
    # Justification: Gradient clipping prevents exploding gradients,
    # especially important for code with variable-length sequences.
    
    # Training schedule
    num_epochs: int = 3
    # Justification: 3 epochs typical for transformer fine-tuning.
    # More epochs risk overfitting to training data patterns.
    
    batch_size: int = 8
    # Justification: Small batch size for memory efficiency on consumer GPUs.
    # Effective batch size can be increased with gradient accumulation.
    
    gradient_accumulation_steps: int = 4
    # Justification: Effective batch size = 8 * 4 = 32, which provides
    # stable gradients while fitting in GPU memory.
    
    warmup_ratio: float = 0.1
    # Justification: 10% of training steps for linear warmup.
    # Helps stabilize early training and improve final performance.
    
    # Regularization
    label_smoothing: float = 0.1
    # Justification: Prevents overconfident predictions, improves calibration.
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 3
    # Justification: Stop if validation loss doesn't improve for 3 evaluations.
    
    # Mixed precision
    fp16: bool = True
    # Justification: FP16 training halves memory usage and speeds up training
    # on GPUs with Tensor Cores (V100, A100, RTX series).


@dataclass
class RAGConfig:
    """Retrieval-Augmented Generation configuration."""
    
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Justification: Lightweight (22M params) but effective sentence encoder.
    # Provides good balance between retrieval quality and speed.
    
    num_retrieved_docs: int = 3
    # Justification: 3 documents provide sufficient context without
    # exceeding token limits or introducing too much noise.
    
    max_combined_length: int = 1024
    # Justification: Total context length for RAG (query + retrieved + generation).
    
    retrieval_score_threshold: float = 0.5
    # Justification: Minimum similarity score to include retrieved document.
    # Filters out low-relevance results.
    
    use_reranking: bool = True
    # Justification: Reranking improves precision of retrieved documents.
    
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Justification: Lightweight cross-encoder for reranking.


@dataclass
class QuantizationConfig:
    """INT8 quantization configuration."""
    
    quantization_type: str = "dynamic"
    # Justification: Dynamic quantization doesn't require calibration data
    # and provides good speedup for inference-bound workloads.
    # Options: "dynamic", "static", "qat" (quantization-aware training)
    
    quantize_embeddings: bool = False
    # Justification: Embedding quantization often hurts quality significantly.
    # Keep embeddings in FP32 for better representation.
    
    per_channel: bool = True
    # Justification: Per-channel quantization provides better accuracy than
    # per-tensor by allowing different scales per output channel.


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration."""
    
    # Answer relevance metrics
    relevance_threshold: float = 0.7
    # Justification: Minimum relevance score to consider answer helpful.
    
    # Code correctness evaluation
    syntax_check: bool = True
    # Justification: Verify generated code is syntactically valid.
    
    # Educational quality metrics
    explanation_clarity_model: str = "textstat"
    # Justification: Use readability metrics (Flesch-Kincaid, etc.)
    
    target_grade_level: float = 12.0
    # Justification: Target undergraduate level (grade 12-16).
    # Explanations should be accessible but not oversimplified.
    
    # Evaluation batch size
    eval_batch_size: int = 16


@dataclass
class ResourceConfig:
    """Resource monitoring configuration."""
    
    monitor_gpu: bool = True
    monitor_cpu: bool = True
    monitor_memory: bool = True
    
    log_interval_seconds: int = 10
    # Justification: Log resource usage every 10 seconds.
    
    memory_warning_threshold: float = 0.9
    # Justification: Warn when memory usage exceeds 90%.
    
    gpu_memory_limit_gb: Optional[float] = None
    # Justification: Optional GPU memory limit for multi-tenant environments.


@dataclass
class Config:
    """Master configuration."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    
    # General settings
    project_name: str = "cs-tutoring-llm"
    output_dir: str = "results"
    cache_dir: str = "data/cache"
    debug_mode: bool = True
    # Justification: Debug mode limits data samples and enables verbose logging.
    
    random_seed: int = 42
    device: str = "auto"
    # Justification: "auto" selects CUDA if available, else CPU.
    
    num_workers: int = 4
    # Justification: 4 workers for data loading provides good parallelism
    # without excessive overhead.
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.bm25.index_path), exist_ok=True)


# Singleton configuration instance
config = Config()


# Parameter count reference table
PARAMETER_COUNTS = {
    "CodeBERT-base": {
        "total": 125_000_000,
        "embeddings": 23_440_896,
        "encoder": 85_054_464,
        "pooler": 590_592,
        "breakdown": {
            "word_embeddings": 23_440_896,
            "position_embeddings": 393_216,
            "token_type_embeddings": 1_536,
            "attention_per_layer": 2_359_296,
            "ffn_per_layer": 4_722_432,
        }
    },
    "CodeT5-small": {
        "total": 60_000_000,
        "encoder": 30_000_000,
        "decoder": 30_000_000,
        "breakdown": {
            "embedding_dim": 512,
            "num_layers": 6,
            "attention_heads": 8,
        }
    },
    "all-MiniLM-L6-v2": {
        "total": 22_700_000,
        "layers": 6,
        "hidden_size": 384,
    },
    "BM25": {
        "total": 0,
        "note": "Non-parametric statistical method",
    }
}


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def print_config_summary():
    """Print a summary of the configuration."""
    print("=" * 60)
    print("CS Tutoring LLM Configuration Summary")
    print("=" * 60)
    print(f"\nDataset: {config.data.dataset_name}/{config.data.dataset_subset}")
    print(f"Train samples: {config.data.max_samples_train}")
    print(f"Eval samples: {config.data.max_samples_eval}")
    print(f"\nBase Model: {config.model.base_model}")
    print(f"Generation Model: {config.model.generation_model}")
    print(f"\nTraining:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"\nRAG Configuration:")
    print(f"  Retriever: {config.rag.retriever_model}")
    print(f"  Retrieved docs: {config.rag.num_retrieved_docs}")
    print(f"\nDebug mode: {config.debug_mode}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
