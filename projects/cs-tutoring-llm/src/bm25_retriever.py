"""
BM25 Retrieval Baseline for CS Tutoring System.

Implements BM25 (Best Matching 25) algorithm for document retrieval,
serving as a non-neural baseline for comparison with learned retrievers.

Reference:
- Robertson & Zaragoza, 2009. "The Probabilistic Relevance Framework: BM25 and Beyond"
- Robertson et al., 1994. "Some Simple Effective Approximations to the 2-Poisson Model"
"""

import os
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any
import re

from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25-based retrieval system for code and documentation.
    
    BM25 is a probabilistic ranking function that considers:
    - Term frequency (TF) with saturation
    - Inverse document frequency (IDF)
    - Document length normalization
    
    Formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
    
    where:
    - k1: Term frequency saturation parameter (default: 1.5)
    - b: Document length normalization parameter (default: 0.75)
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.bm25_code = None
        self.bm25_docs = None
        self.code_corpus = []
        self.doc_corpus = []
        self.tokenized_code = []
        self.tokenized_docs = []
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def _preprocess_code(self, code: str) -> List[str]:
        """
        Preprocess code for BM25 indexing.
        
        - Split camelCase and snake_case identifiers
        - Remove comments and string literals
        - Tokenize into words
        """
        # Remove string literals
        code = re.sub(r'"[^"]*"', ' ', code)
        code = re.sub(r"'[^']*'", ' ', code)
        
        # Remove comments
        code = re.sub(r'#.*$', ' ', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', ' ', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", ' ', code, flags=re.DOTALL)
        
        # Split camelCase
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        
        # Split snake_case
        code = code.replace('_', ' ')
        
        # Tokenize
        tokens = word_tokenize(code.lower())
        
        # Filter tokens
        tokens = [t for t in tokens if len(t) > 1 and t.isalnum()]
        
        return tokens
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess natural language text for BM25."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 1 and t.isalnum()]
        return tokens
    
    def build_index(
        self,
        code_corpus: List[str],
        doc_corpus: List[str],
        save: bool = True
    ) -> None:
        """
        Build BM25 indices for code and documentation.
        
        Args:
            code_corpus: List of code snippets
            doc_corpus: List of documentation strings
            save: Whether to save the index to disk
        """
        logger.info(f"Building BM25 index for {len(code_corpus)} documents...")
        
        self.code_corpus = code_corpus
        self.doc_corpus = doc_corpus
        
        # Tokenize code corpus
        logger.info("Tokenizing code corpus...")
        self.tokenized_code = [self._preprocess_code(c) for c in code_corpus]
        
        # Tokenize doc corpus
        logger.info("Tokenizing documentation corpus...")
        self.tokenized_docs = [self._preprocess_text(d) for d in doc_corpus]
        
        # Build BM25 indices
        logger.info("Building BM25 index for code...")
        self.bm25_code = BM25Okapi(
            self.tokenized_code,
            k1=self.config.bm25.k1,
            b=self.config.bm25.b
        )
        
        logger.info("Building BM25 index for documentation...")
        self.bm25_docs = BM25Okapi(
            self.tokenized_docs,
            k1=self.config.bm25.k1,
            b=self.config.bm25.b
        )
        
        if save:
            self.save_index()
        
        logger.info("BM25 index built successfully")
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save BM25 index to disk."""
        path = path or self.config.bm25.index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        index_data = {
            "bm25_code": self.bm25_code,
            "bm25_docs": self.bm25_docs,
            "code_corpus": self.code_corpus,
            "doc_corpus": self.doc_corpus,
            "tokenized_code": self.tokenized_code,
            "tokenized_docs": self.tokenized_docs,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """Load BM25 index from disk."""
        path = path or self.config.bm25.index_path
        
        if not os.path.exists(path):
            logger.warning(f"Index not found at {path}")
            return False
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25_code = index_data["bm25_code"]
        self.bm25_docs = index_data["bm25_docs"]
        self.code_corpus = index_data["code_corpus"]
        self.doc_corpus = index_data["doc_corpus"]
        self.tokenized_code = index_data["tokenized_code"]
        self.tokenized_docs = index_data["tokenized_docs"]
        
        logger.info(f"Index loaded from {path}")
        return True
    
    def search_by_code(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str, str]]:
        """
        Search for relevant code snippets given a natural language query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of (index, score, code, docstring) tuples
        """
        if self.bm25_code is None:
            raise ValueError("Index not built. Call build_index first.")
        
        top_k = top_k or self.config.bm25.top_k
        
        # Tokenize query
        query_tokens = self._preprocess_text(query)
        
        # Search code corpus
        scores = self.bm25_code.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(scores[idx]),
                self.code_corpus[idx],
                self.doc_corpus[idx]
            ))
        
        return results
    
    def search_by_description(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str, str]]:
        """
        Search for relevant documentation given a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (index, score, code, docstring) tuples
        """
        if self.bm25_docs is None:
            raise ValueError("Index not built. Call build_index first.")
        
        top_k = top_k or self.config.bm25.top_k
        
        # Tokenize query
        query_tokens = self._preprocess_text(query)
        
        # Search documentation corpus
        scores = self.bm25_docs.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(scores[idx]),
                self.code_corpus[idx],
                self.doc_corpus[idx]
            ))
        
        return results
    
    def search_combined(
        self,
        query: str,
        top_k: Optional[int] = None,
        code_weight: float = 0.5
    ) -> List[Tuple[int, float, str, str]]:
        """
        Combined search over code and documentation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            code_weight: Weight for code search (1 - code_weight for doc)
            
        Returns:
            List of (index, score, code, docstring) tuples
        """
        if self.bm25_code is None or self.bm25_docs is None:
            raise ValueError("Index not built. Call build_index first.")
        
        top_k = top_k or self.config.bm25.top_k
        
        # Tokenize query
        query_tokens = self._preprocess_text(query)
        
        # Get scores from both indices
        code_scores = self.bm25_code.get_scores(query_tokens)
        doc_scores = self.bm25_docs.get_scores(query_tokens)
        
        # Normalize scores
        code_scores = code_scores / (np.max(code_scores) + 1e-8)
        doc_scores = doc_scores / (np.max(doc_scores) + 1e-8)
        
        # Combine scores
        combined_scores = code_weight * code_scores + (1 - code_weight) * doc_scores
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(combined_scores[idx]),
                self.code_corpus[idx],
                self.doc_corpus[idx]
            ))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index."""
        if self.bm25_code is None:
            return {"error": "Index not built"}
        
        code_lengths = [len(t) for t in self.tokenized_code]
        doc_lengths = [len(t) for t in self.tokenized_docs]
        
        return {
            "num_documents": len(self.code_corpus),
            "code_stats": {
                "avg_length": np.mean(code_lengths),
                "min_length": np.min(code_lengths),
                "max_length": np.max(code_lengths),
                "std_length": np.std(code_lengths),
            },
            "doc_stats": {
                "avg_length": np.mean(doc_lengths),
                "min_length": np.min(doc_lengths),
                "max_length": np.max(doc_lengths),
                "std_length": np.std(doc_lengths),
            },
            "bm25_params": {
                "k1": self.config.bm25.k1,
                "b": self.config.bm25.b,
            }
        }


if __name__ == "__main__":
    # Test BM25 retriever
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.load_dataset()
    codes, docs = loader.get_raw_texts()
    
    # Build index
    retriever = BM25Retriever()
    retriever.build_index(codes, docs)
    
    # Test search
    query = "binary search algorithm"
    results = retriever.search_combined(query, top_k=3)
    
    print(f"\nSearch results for: '{query}'")
    for idx, score, code, doc in results:
        print(f"\n[{idx}] Score: {score:.4f}")
        print(f"Doc: {doc[:100]}...")
