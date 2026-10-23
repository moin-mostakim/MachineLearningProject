"""
CS Tutoring LLM System - Intelligent Tutoring for Computer Science Education.

This package implements a retrieval-augmented generation (RAG) system for
providing educational assistance in computer science topics.

Modules:
    - data_loader: Dataset loading and preprocessing
    - bm25_retriever: BM25 baseline retrieval
    - model: Neural model architectures
    - rag_system: RAG implementation
    - trainer: Training utilities
    - evaluator: Evaluation metrics
    - quantization: INT8 quantization utilities
    - resource_monitor: Resource monitoring
"""

__version__ = "0.1.0"
__author__ = "CS Tutoring Team"

from .data_loader import DataLoader, CodeSearchNetDataset
from .bm25_retriever import BM25Retriever
from .model import CodeExplanationModel, TutoringModel
from .rag_system import RAGSystem
from .trainer import Trainer
from .evaluator import Evaluator
from .quantization import QuantizationManager
from .resource_monitor import ResourceMonitor

__all__ = [
    "DataLoader",
    "CodeSearchNetDataset",
    "BM25Retriever",
    "CodeExplanationModel",
    "TutoringModel",
    "RAGSystem",
    "Trainer",
    "Evaluator",
    "QuantizationManager",
    "ResourceMonitor",
]
