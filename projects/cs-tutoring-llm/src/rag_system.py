"""
Retrieval-Augmented Generation (RAG) System for CS Tutoring.

Implements RAG combining:
1. Dense retrieval (sentence-transformers)
2. BM25 sparse retrieval (hybrid)
3. Cross-encoder reranking
4. CodeT5 generation

References:
- Lewis et al., 2020. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Karpukhin et al., 2020. "Dense Passage Retrieval for Open-Domain QA"
- Izacard & Grave, 2021. "Leveraging Passage Retrieval with Generative Models"
"""

import os
import logging
from typing import Optional, List, Tuple, Dict, Any

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config
from src.bm25_retriever import BM25Retriever
from src.model import CodeExplanationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system for code tutoring.
    
    Architecture:
    1. Query encoding using sentence-transformer
    2. Dense retrieval from FAISS index
    3. Optional BM25 fusion (hybrid retrieval)
    4. Cross-encoder reranking
    5. Context-augmented generation with CodeT5
    
    This approach combines the precision of neural retrieval with
    the generation capabilities of seq2seq models.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Initialize components
        self.retriever_model = None
        self.reranker = None
        self.generator = None
        self.bm25 = None
        
        # Document store
        self.documents = []  # (code, docstring) pairs
        self.index = None
        self.embeddings = None
        
        self._load_models()
    
    def _load_models(self):
        """Load retriever, reranker, and generator models."""
        # Dense retriever
        logger.info(f"Loading retriever: {self.config.rag.retriever_model}")
        self.retriever_model = SentenceTransformer(
            self.config.rag.retriever_model,
            cache_folder=self.config.cache_dir
        )
        
        # Cross-encoder reranker
        if self.config.rag.use_reranking:
            logger.info(f"Loading reranker: {self.config.rag.reranker_model}")
            self.reranker = CrossEncoder(
                self.config.rag.reranker_model,
                max_length=self.config.data.max_seq_length
            )
        
        # Generator
        logger.info("Loading generator model...")
        self.generator = CodeExplanationModel(self.config)
        
        # BM25 for hybrid retrieval
        self.bm25 = BM25Retriever(self.config)
    
    def build_index(
        self,
        codes: List[str],
        docstrings: List[str],
        save: bool = True
    ) -> None:
        """
        Build retrieval indices for the document collection.
        
        Args:
            codes: List of code snippets
            docstrings: List of corresponding docstrings
            save: Whether to save indices to disk
        """
        logger.info(f"Building indices for {len(codes)} documents...")
        
        # Store documents
        self.documents = list(zip(codes, docstrings))
        
        # Create combined text for embedding
        combined_texts = [
            f"Code: {code}\nDocumentation: {doc}"
            for code, doc in self.documents
        ]
        
        # Compute dense embeddings
        logger.info("Computing dense embeddings...")
        self.embeddings = self.retriever_model.encode(
            combined_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        self.index.add(self.embeddings.astype(np.float32))
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25.build_index(codes, docstrings, save=save)
        
        if save:
            self._save_index()
        
        logger.info("Indices built successfully")
    
    def _save_index(self, path: Optional[str] = None):
        """Save FAISS index and embeddings."""
        path = path or os.path.join(self.config.cache_dir, "rag_index")
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save embeddings and documents
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        import pickle
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """Load FAISS index and embeddings."""
        path = path or os.path.join(self.config.cache_dir, "rag_index")
        
        if not os.path.exists(path):
            return False
        
        try:
            self.index = faiss.read_index(os.path.join(path, "faiss.index"))
            self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
            
            import pickle
            with open(os.path.join(path, "documents.pkl"), 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load BM25
            self.bm25.load_index()
            
            logger.info(f"Index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: bool = True,
        hybrid_weight: float = 0.7
    ) -> List[Tuple[int, float, str, str]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Natural language query
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid (dense + BM25) retrieval
            hybrid_weight: Weight for dense retrieval (1 - weight for BM25)
            
        Returns:
            List of (index, score, code, docstring) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        top_k = top_k or self.config.rag.num_retrieved_docs
        
        # Dense retrieval
        query_embedding = self.retriever_model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        
        dense_scores, dense_indices = self.index.search(query_embedding, top_k * 2)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        
        if use_hybrid and self.bm25.bm25_code is not None:
            # BM25 retrieval
            bm25_results = self.bm25.search_combined(query, top_k=top_k * 2)
            
            # Normalize BM25 scores
            bm25_scores = np.array([r[1] for r in bm25_results])
            bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-8)
            bm25_indices = [r[0] for r in bm25_results]
            
            # Create score dictionaries
            dense_score_dict = {
                int(idx): float(score) 
                for idx, score in zip(dense_indices, dense_scores)
            }
            bm25_score_dict = {
                int(idx): float(score)
                for idx, score in zip(bm25_indices, bm25_scores)
            }
            
            # Combine scores
            all_indices = set(dense_score_dict.keys()) | set(bm25_score_dict.keys())
            combined_scores = {}
            
            for idx in all_indices:
                dense_s = dense_score_dict.get(idx, 0.0)
                bm25_s = bm25_score_dict.get(idx, 0.0)
                combined_scores[idx] = hybrid_weight * dense_s + (1 - hybrid_weight) * bm25_s
            
            # Sort by combined score
            sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
            
            results = [
                (idx, combined_scores[idx], self.documents[idx][0], self.documents[idx][1])
                for idx in sorted_indices[:top_k * 2]
            ]
        else:
            results = [
                (int(idx), float(score), self.documents[idx][0], self.documents[idx][1])
                for idx, score in zip(dense_indices, dense_scores)
            ]
        
        # Rerank with cross-encoder
        if self.config.rag.use_reranking and self.reranker is not None:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Filter by score threshold
        results = [
            r for r in results
            if r[1] >= self.config.rag.retrieval_score_threshold
        ]
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, str, str]],
        top_k: int
    ) -> List[Tuple[int, float, str, str]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return candidates
        
        # Prepare pairs for reranking
        pairs = [
            (query, f"Code: {code}\nDocumentation: {doc}")
            for _, _, code, doc in candidates
        ]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original scores (weighted average)
        reranked = []
        for (idx, orig_score, code, doc), rerank_score in zip(candidates, rerank_scores):
            combined_score = 0.3 * orig_score + 0.7 * float(rerank_score)
            reranked.append((idx, combined_score, code, doc))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Tuple[int, float, str, str]],
        max_length: int = 256
    ) -> str:
        """
        Generate tutoring response given query and retrieved context.
        
        Args:
            query: Student's question
            retrieved_docs: Retrieved (idx, score, code, docstring) tuples
            max_length: Maximum response length
            
        Returns:
            Generated tutoring response
        """
        # Build context from retrieved documents
        context_parts = []
        for i, (_, score, code, doc) in enumerate(retrieved_docs):
            context_parts.append(f"Example {i+1}:\nCode: {code[:200]}...\nExplanation: {doc}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following examples, provide a helpful explanation for the student's question.

{context}

Student's Question: {query}

Helpful Explanation:"""
        
        # Truncate if too long
        if len(prompt) > self.config.rag.max_combined_length:
            prompt = prompt[:self.config.rag.max_combined_length]
        
        # Generate response
        response = self.generator.generate_explanation(
            prompt,
            max_length=max_length,
            num_beams=4,
            temperature=0.7
        )
        
        return response
    
    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        End-to-end question answering pipeline.
        
        Args:
            question: Student's question
            top_k: Number of documents to retrieve
            verbose: Whether to include retrieval details
            
        Returns:
            Dictionary with answer and metadata
        """
        top_k = top_k or self.config.rag.num_retrieved_docs
        
        # Retrieve relevant documents
        retrieved = self.retrieve(question, top_k=top_k)
        
        # Generate response
        response = self.generate_response(question, retrieved)
        
        result = {
            "question": question,
            "answer": response,
            "num_retrieved": len(retrieved),
        }
        
        if verbose:
            result["retrieved_docs"] = [
                {
                    "index": idx,
                    "score": score,
                    "code": code[:100] + "...",
                    "docstring": doc
                }
                for idx, score, code, doc in retrieved
            ]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            "num_documents": len(self.documents) if self.documents else 0,
            "index_built": self.index is not None,
            "retriever_model": self.config.rag.retriever_model,
            "generator_model": self.config.model.generation_model,
            "use_reranking": self.config.rag.use_reranking,
        }
        
        if self.embeddings is not None:
            stats["embedding_dim"] = self.embeddings.shape[1]
        
        return stats


class RAGEvaluator:
    """Evaluation utilities for RAG system."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[int]],  # List of relevant doc indices for each query
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Metrics:
        - Recall@K
        - Precision@K
        - MRR (Mean Reciprocal Rank)
        """
        results = {f"recall@{k}": [] for k in k_values}
        results.update({f"precision@{k}": [] for k in k_values})
        results["mrr"] = []
        
        for query, relevant_indices in zip(queries, ground_truth):
            retrieved = self.rag.retrieve(query, top_k=max(k_values))
            retrieved_indices = [r[0] for r in retrieved]
            
            # Calculate metrics
            for k in k_values:
                retrieved_at_k = set(retrieved_indices[:k])
                relevant_set = set(relevant_indices)
                
                # Recall@K
                if relevant_set:
                    recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
                else:
                    recall = 0.0
                results[f"recall@{k}"].append(recall)
                
                # Precision@K
                precision = len(retrieved_at_k & relevant_set) / k
                results[f"precision@{k}"].append(precision)
            
            # MRR
            rr = 0.0
            for i, idx in enumerate(retrieved_indices):
                if idx in relevant_indices:
                    rr = 1.0 / (i + 1)
                    break
            results["mrr"].append(rr)
        
        # Average results
        return {key: np.mean(values) for key, values in results.items()}


if __name__ == "__main__":
    # Test RAG system
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.load_dataset()
    codes, docs = loader.get_raw_texts()
    
    # Build RAG system
    rag = RAGSystem()
    
    # Use subset for demo
    rag.build_index(codes[:100], docs[:100])
    
    # Test query
    question = "How does binary search work?"
    result = rag.answer_question(question, verbose=True)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRetrieved {result['num_retrieved']} documents")
    
    if 'retrieved_docs' in result:
        print("\nTop retrieved documents:")
        for doc in result['retrieved_docs'][:2]:
            print(f"  Score: {doc['score']:.4f}")
            print(f"  Doc: {doc['docstring'][:100]}...")
