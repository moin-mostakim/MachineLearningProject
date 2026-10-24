"""
Evaluation metrics for CS Tutoring LLM System.

Implements comprehensive evaluation across multiple dimensions:
1. Answer Relevance - How relevant is the explanation to the code?
2. Code Correctness - Is the explained code syntactically valid?
3. Educational Quality - Is the explanation pedagogically sound?
4. Generation Quality - BLEU, ROUGE, BERTScore

References:
- Papineni et al., 2002. "BLEU: A Method for Automatic Evaluation of Machine Translation"
- Lin, 2004. "ROUGE: A Package for Automatic Evaluation of Summaries"
- Zhang et al., 2020. "BERTScore: Evaluating Text Generation with BERT"
"""

import os
import ast
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
import textstat

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Retrieval metrics
    retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_mrr: float = 0.0
    
    # Generation metrics
    bleu_score: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    
    # Educational metrics
    avg_readability: float = 0.0
    avg_clarity: float = 0.0
    syntax_correct_rate: float = 0.0
    
    # Classification metrics
    relevance_accuracy: float = 0.0
    relevance_f1: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval_recall": self.retrieval_recall,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_mrr": self.retrieval_mrr,
            "bleu_score": self.bleu_score,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "bert_score": self.bert_score,
            "avg_readability": self.avg_readability,
            "avg_clarity": self.avg_clarity,
            "syntax_correct_rate": self.syntax_correct_rate,
            "relevance_accuracy": self.relevance_accuracy,
            "relevance_f1": self.relevance_f1,
        }


class Evaluator:
    """
    Comprehensive evaluator for CS tutoring system.
    
    Metrics categories:
    1. Information Retrieval: Recall, Precision, MRR
    2. Text Generation: BLEU, ROUGE, BERTScore
    3. Educational Quality: Readability, Clarity
    4. Code Analysis: Syntax correctness
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # BERTScore (lazy load)
        self._bertscore = None
    
    @property
    def bertscore(self):
        """Lazy load BERTScore."""
        if self._bertscore is None:
            try:
                from bert_score import BERTScorer
                self._bertscore = BERTScorer(
                    model_type="microsoft/deberta-xlarge-mnli",
                    num_layers=40,
                    batch_size=8,
                    rescale_with_baseline=True
                )
            except Exception as e:
                logger.warning(f"BERTScore unavailable: {e}")
                self._bertscore = "unavailable"
        return self._bertscore
    
    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate text generation quality.
        
        Args:
            predictions: Generated explanations
            references: Ground truth explanations
            
        Returns:
            Dictionary of generation metrics
        """
        results = {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bertscore": 0.0,
        }
        
        if not predictions or not references:
            return results
        
        # Calculate ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougel_scores.append(scores['rougeL'].fmeasure)
        
        results["rouge1"] = np.mean(rouge1_scores)
        results["rouge2"] = np.mean(rouge2_scores)
        results["rougeL"] = np.mean(rougel_scores)
        
        # Calculate simple BLEU (word overlap based)
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if pred_tokens:
                precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
                bleu_scores.append(precision)
        results["bleu"] = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # BERTScore (if available)
        if self.bertscore != "unavailable":
            try:
                P, R, F1 = self.bertscore.score(predictions, references)
                results["bertscore"] = F1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")
        
        return results
    
    def evaluate_retrieval(
        self,
        retrieved_indices: List[List[int]],
        relevant_indices: List[List[int]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_indices: List of retrieved doc indices per query
            relevant_indices: List of relevant doc indices per query
            k_values: K values for Recall@K and Precision@K
            
        Returns:
            Dictionary of retrieval metrics
        """
        results = {}
        
        for k in k_values:
            recalls = []
            precisions = []
            
            for retrieved, relevant in zip(retrieved_indices, relevant_indices):
                retrieved_at_k = set(retrieved[:k])
                relevant_set = set(relevant)
                
                if relevant_set:
                    recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
                else:
                    recall = 0.0
                recalls.append(recall)
                
                if retrieved_at_k:
                    precision = len(retrieved_at_k & relevant_set) / len(retrieved_at_k)
                else:
                    precision = 0.0
                precisions.append(precision)
            
            results[f"recall@{k}"] = np.mean(recalls)
            results[f"precision@{k}"] = np.mean(precisions)
        
        # MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for retrieved, relevant in zip(retrieved_indices, relevant_indices):
            rr = 0.0
            relevant_set = set(relevant)
            for i, idx in enumerate(retrieved):
                if idx in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            mrr_scores.append(rr)
        
        results["mrr"] = np.mean(mrr_scores)
        
        return results
    
    def evaluate_educational_quality(
        self,
        explanations: List[str],
        codes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate educational quality of explanations.
        
        Metrics:
        - Readability (Flesch-Kincaid Grade Level)
        - Clarity (sentence structure analysis)
        - Completeness (presence of key elements)
        
        Args:
            explanations: List of explanation texts
            codes: Optional list of corresponding code snippets
            
        Returns:
            Dictionary of educational metrics
        """
        results = {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_sentence_length": 0.0,
            "vocabulary_complexity": 0.0,
            "syntax_correct_rate": 0.0,
        }
        
        if not explanations:
            return results
        
        reading_ease_scores = []
        grade_level_scores = []
        sentence_lengths = []
        
        for explanation in explanations:
            if len(explanation.split()) < 10:
                continue
            
            try:
                # Flesch Reading Ease (higher = easier)
                reading_ease = textstat.flesch_reading_ease(explanation)
                reading_ease_scores.append(max(0, min(100, reading_ease)))
                
                # Flesch-Kincaid Grade Level
                grade = textstat.flesch_kincaid_grade(explanation)
                grade_level_scores.append(max(0, grade))
                
                # Average sentence length
                sentences = explanation.split('.')
                avg_len = np.mean([len(s.split()) for s in sentences if s.strip()])
                sentence_lengths.append(avg_len)
            except Exception:
                continue
        
        if reading_ease_scores:
            results["flesch_reading_ease"] = np.mean(reading_ease_scores)
        if grade_level_scores:
            results["flesch_kincaid_grade"] = np.mean(grade_level_scores)
        if sentence_lengths:
            results["avg_sentence_length"] = np.mean(sentence_lengths)
        
        # Check code syntax if provided
        if codes:
            syntax_correct = 0
            for code in codes:
                if self._check_python_syntax(code):
                    syntax_correct += 1
            results["syntax_correct_rate"] = syntax_correct / len(codes)
        
        return results
    
    def _check_python_syntax(self, code: str) -> bool:
        """Check if Python code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def evaluate_classification(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate classification performance.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            
        Returns:
            Dictionary of classification metrics
        """
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='weighted', zero_division=0),
            "precision": precision_score(labels, predictions, average='weighted', zero_division=0),
            "recall": recall_score(labels, predictions, average='weighted', zero_division=0),
        }
    
    def full_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        codes: Optional[List[str]] = None,
        retrieved_indices: Optional[List[List[int]]] = None,
        relevant_indices: Optional[List[List[int]]] = None,
        relevance_preds: Optional[List[int]] = None,
        relevance_labels: Optional[List[int]] = None
    ) -> EvaluationResults:
        """
        Run full evaluation across all metrics.
        
        Returns:
            EvaluationResults dataclass with all metrics
        """
        results = EvaluationResults()
        
        # Generation metrics
        gen_metrics = self.evaluate_generation(predictions, references)
        results.bleu_score = gen_metrics["bleu"]
        results.rouge_1 = gen_metrics["rouge1"]
        results.rouge_2 = gen_metrics["rouge2"]
        results.rouge_l = gen_metrics["rougeL"]
        results.bert_score = gen_metrics["bertscore"]
        
        # Educational quality
        edu_metrics = self.evaluate_educational_quality(predictions, codes)
        results.avg_readability = edu_metrics["flesch_reading_ease"]
        results.avg_clarity = 100 - edu_metrics.get("flesch_kincaid_grade", 12) * 5  # Normalize
        results.syntax_correct_rate = edu_metrics["syntax_correct_rate"]
        
        # Retrieval metrics
        if retrieved_indices and relevant_indices:
            ret_metrics = self.evaluate_retrieval(retrieved_indices, relevant_indices)
            results.retrieval_recall = ret_metrics.get("recall@5", 0.0)
            results.retrieval_precision = ret_metrics.get("precision@5", 0.0)
            results.retrieval_mrr = ret_metrics.get("mrr", 0.0)
        
        # Classification metrics
        if relevance_preds and relevance_labels:
            cls_metrics = self.evaluate_classification(relevance_preds, relevance_labels)
            results.relevance_accuracy = cls_metrics["accuracy"]
            results.relevance_f1 = cls_metrics["f1"]
        
        return results
    
    def compare_methods(
        self,
        results_dict: Dict[str, EvaluationResults]
    ) -> str:
        """
        Generate comparison table for multiple methods.
        
        Args:
            results_dict: Dictionary mapping method names to results
            
        Returns:
            Formatted comparison table string
        """
        table = []
        table.append("=" * 90)
        table.append("METHOD COMPARISON")
        table.append("=" * 90)
        
        # Header
        header = f"{'Metric':<25}"
        for method in results_dict.keys():
            header += f"{method:<15}"
        table.append(header)
        table.append("-" * 90)
        
        # Metrics to compare
        metrics = [
            ("ROUGE-1", "rouge_1"),
            ("ROUGE-2", "rouge_2"),
            ("ROUGE-L", "rouge_l"),
            ("BLEU", "bleu_score"),
            ("BERTScore", "bert_score"),
            ("Readability", "avg_readability"),
            ("Clarity", "avg_clarity"),
            ("Syntax Correct", "syntax_correct_rate"),
            ("Retrieval Recall", "retrieval_recall"),
            ("Retrieval MRR", "retrieval_mrr"),
            ("Relevance Acc", "relevance_accuracy"),
        ]
        
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<25}"
            for method, results in results_dict.items():
                value = getattr(results, metric_key, 0.0)
                row += f"{value:>13.4f}  "
            table.append(row)
        
        table.append("=" * 90)
        
        return "\n".join(table)


def generate_benchmark_report(results: Dict[str, EvaluationResults]) -> str:
    """
    Generate comprehensive benchmark report.
    
    References literature baselines for comparison.
    """
    report = []
    report.append("\n" + "=" * 80)
    report.append("CS TUTORING LLM BENCHMARK REPORT")
    report.append("=" * 80)
    
    report.append("\n## Literature Comparison")
    report.append("-" * 60)
    report.append("""
Reference baselines from literature:
- CodeBERT (Feng et al., 2020): Code search MRR ~0.58 on CodeSearchNet
- CodeT5 (Wang et al., 2021): Code summarization BLEU ~20 on CodeSearchNet
- Codex (Chen et al., 2021): Code generation pass@1 ~28.8% on HumanEval
- GPT-3.5 for education (Kumar et al., 2023): Explanation quality ~0.72

Note: Direct comparison limited due to different evaluation setups.
    """)
    
    evaluator = Evaluator()
    report.append("\n## Our Results")
    report.append(evaluator.compare_methods(results))
    
    report.append("\n## Educational Quality Analysis")
    report.append("-" * 60)
    
    for method, res in results.items():
        report.append(f"\n{method}:")
        report.append(f"  Readability (Flesch): {res.avg_readability:.1f}")
        if res.avg_readability >= 60:
            report.append("    -> Good for undergraduate level")
        elif res.avg_readability >= 30:
            report.append("    -> Suitable for advanced students")
        else:
            report.append("    -> May be too complex")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator()
    
    # Sample data
    predictions = [
        "This function implements binary search to find an element in a sorted array.",
        "The quicksort algorithm divides the array and sorts recursively.",
    ]
    references = [
        "Binary search algorithm finds target in sorted array with O(log n) time.",
        "Quicksort uses divide-and-conquer to sort arrays efficiently.",
    ]
    codes = [
        "def binary_search(arr, target): pass",
        "def quicksort(arr): return sorted(arr)",
    ]
    
    # Run evaluation
    results = evaluator.full_evaluation(predictions, references, codes)
    
    print("Evaluation Results:")
    for key, value in results.to_dict().items():
        print(f"  {key}: {value:.4f}")
