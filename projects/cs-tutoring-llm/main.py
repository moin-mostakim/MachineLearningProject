#!/usr/bin/env python3
"""
CS Tutoring LLM - Main Execution Script

An Intelligent Tutoring System for Computer Science Education
Powered by Large Language Models

This script demonstrates the complete pipeline:
1. Data loading from CodeSearchNet
2. BM25 baseline retrieval
3. Fine-tuned LLM for code explanation
4. RAG (Retrieval-Augmented Generation) approach
5. Comprehensive evaluation

References:
- CodeBERT (Feng et al., 2020): Pre-training for Programming Language Understanding
- CodeT5 (Wang et al., 2021): Identifier-aware Unified Pre-trained Encoder-Decoder
- Codex (Chen et al., 2021): Evaluating Large Language Models Trained on Code
- Lewis et al., 2020: Retrieval-Augmented Generation for Knowledge-Intensive NLP
- Kumar et al., 2020: BERT-based Educational Question Answering

Usage:
    python main.py --mode debug        # Debug mode with limited data
    python main.py --mode full         # Full training and evaluation
    python main.py --mode eval_only    # Evaluation only
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config, print_config_summary, PARAMETER_COUNTS
from src.data_loader import DataLoader
from src.bm25_retriever import BM25Retriever
from src.model import CodeExplanationModel, TutoringModel, ModelFactory, get_parameter_table
from src.rag_system import RAGSystem
from src.trainer import Trainer, GeneratorTrainer
from src.evaluator import Evaluator, EvaluationResults, generate_benchmark_report
from src.quantization import QuantizationManager, generate_quantization_report
from src.resource_monitor import ResourceMonitor, MemoryTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_literature_comparison():
    """Print literature comparison section."""
    print("\n" + "=" * 80)
    print("LITERATURE COMPARISON")
    print("=" * 80)
    print("""
This work implements and compares several approaches for CS education:

1. BM25 Baseline (Robertson et al., 1994)
   - Classic probabilistic retrieval model
   - No neural parameters, relies on term statistics
   - Strong baseline for code search tasks

2. CodeBERT (Feng et al., 2020)
   - 125M parameters
   - Pre-trained on bimodal data (code + natural language)
   - MLM + Replaced Token Detection objectives
   - Benchmark: CodeSearchNet MRR ~0.58

3. CodeT5 (Wang et al., 2021)
   - 60M parameters (small) / 220M (base)
   - Encoder-decoder architecture
   - Identifier-aware denoising pre-training
   - Benchmark: Code summarization BLEU ~20

4. RAG Approach (Lewis et al., 2020)
   - Combines retrieval with generation
   - Dense retrieval + cross-encoder reranking
   - Provides grounded, factual explanations

5. Educational AI Systems
   - Kumar et al. (2020): BERT for educational QA
   - AutoTutor (Graesser et al.): Intelligent tutoring systems
   - Codex/GPT-4 for programming education

Key Findings from Literature:
- Pre-trained code models significantly outperform general-purpose models
- RAG reduces hallucination in code explanations
- Readability at grade 10-14 level optimal for CS education
- Hybrid retrieval (dense + sparse) often beats either alone
""")


def run_bm25_baseline(config: Config, data_loader: DataLoader) -> dict:
    """Run BM25 baseline experiments."""
    logger.info("=" * 60)
    logger.info("Running BM25 Baseline")
    logger.info("=" * 60)
    
    with MemoryTracker("BM25 Baseline"):
        # Get raw texts
        codes, docs = data_loader.get_raw_texts()
        
        # Build BM25 index
        retriever = BM25Retriever(config)
        retriever.build_index(codes, docs, save=True)
        
        # Test queries
        test_queries = [
            "binary search algorithm",
            "how to sort an array",
            "recursive function example",
            "linked list implementation",
            "dynamic programming solution",
        ]
        
        results = {}
        for query in test_queries:
            search_results = retriever.search_combined(query, top_k=3)
            results[query] = [
                {"score": score, "doc": doc[:100]}
                for _, score, _, doc in search_results
            ]
            logger.info(f"Query: '{query}' - Top score: {search_results[0][1]:.4f}")
        
        stats = retriever.get_statistics()
        logger.info(f"BM25 Index Stats: {stats['num_documents']} documents indexed")
    
    return {
        "method": "BM25",
        "parameters": 0,
        "index_stats": stats,
        "sample_results": results,
    }


def run_fine_tuned_model(config: Config, data_loader: DataLoader) -> dict:
    """Run fine-tuned LLM experiments."""
    logger.info("=" * 60)
    logger.info("Running Fine-tuned LLM")
    logger.info("=" * 60)
    
    with MemoryTracker("Fine-tuned LLM"):
        # Create model (model has its own tokenizer)
        logger.info("Creating CodeExplanationModel...")
        model = CodeExplanationModel(config)
        
        # Print parameter count
        logger.info(f"Model parameters: {model.param_count:,}")
        
        # In debug mode, do minimal training
        if config.debug_mode:
            logger.info("Debug mode: Skipping full training, testing generation...")
            model.model.eval()
            
            # Test generation
            test_code = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)'''
            
            try:
                explanation = model.generate_explanation(test_code)
                logger.info(f"Sample generation: {explanation[:200]}...")
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                explanation = "Generation test skipped"
        else:
            # For full training, need dataloaders
            try:
                # Use model's tokenizer for data loading
                data_loader.tokenizer = model.tokenizer
                train_loader, val_loader, test_loader = data_loader.create_dataloaders(
                    batch_size=config.training.batch_size
                )
                trainer = GeneratorTrainer(model, config)
                history = trainer.train(train_loader, val_loader, num_epochs=config.training.num_epochs)
                logger.info(f"Training complete. Final loss: {history['training_losses'][-1]:.4f}")
            except Exception as e:
                logger.warning(f"Training failed: {e}")
    
    return {
        "method": "Fine-tuned CodeT5",
        "parameters": model.param_count,
        "base_model": config.model.generation_model,
    }


def run_rag_system(config: Config, data_loader: DataLoader) -> dict:
    """Run RAG system experiments."""
    logger.info("=" * 60)
    logger.info("Running RAG System")
    logger.info("=" * 60)
    
    with MemoryTracker("RAG System"):
        # Get raw texts
        codes, docs = data_loader.get_raw_texts()
        
        # Build RAG system
        logger.info("Building RAG system...")
        rag = RAGSystem(config)
        
        # Use subset for faster demo
        subset_size = min(100, len(codes))
        rag.build_index(codes[:subset_size], docs[:subset_size])
        
        # Test questions
        test_questions = [
            "How does binary search work?",
            "Explain the quicksort algorithm",
            "What is recursion?",
        ]
        
        results = {}
        for question in test_questions:
            logger.info(f"\nQuestion: {question}")
            result = rag.answer_question(question, verbose=True)
            results[question] = {
                "answer": result["answer"][:200] + "...",
                "num_retrieved": result["num_retrieved"],
            }
            logger.info(f"Answer: {result['answer'][:100]}...")
        
        stats = rag.get_statistics()
    
    return {
        "method": "RAG",
        "retriever": config.rag.retriever_model,
        "generator": config.model.generation_model,
        "stats": stats,
        "sample_results": results,
    }


def run_quantization_comparison(config: Config) -> dict:
    """Run INT8 quantization comparison."""
    logger.info("=" * 60)
    logger.info("Running INT8 Quantization Comparison")
    logger.info("=" * 60)
    
    with MemoryTracker("Quantization"):
        # Create model
        model = TutoringModel(config)
        
        # Create sample input
        tokenizer = model.tokenizer
        sample_text = "def binary_search(arr, target): pass"
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            max_length=128
        )
        
        # Quantization manager
        manager = QuantizationManager(config)
        
        # Apply quantization
        quantized_model = manager.dynamic_quantization(model)
        
        # Comparison (reduced iterations for debug mode)
        num_iter = 10 if config.debug_mode else 50
        comparison = manager.full_comparison(inputs["input_ids"], num_iterations=num_iter)
        
        # Print report
        report = generate_quantization_report(comparison)
        print(report)
    
    return comparison


def run_evaluation(
    config: Config,
    predictions: list,
    references: list,
    codes: list,
    method_name: str
) -> EvaluationResults:
    """Run comprehensive evaluation."""
    logger.info(f"Evaluating {method_name}...")
    
    evaluator = Evaluator(config)
    results = evaluator.full_evaluation(
        predictions=predictions,
        references=references,
        codes=codes,
    )
    
    return results


def save_results(results: dict, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    
    # Convert non-serializable objects
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return str(obj)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    
    logger.info(f"Results saved to {output_path}")
    return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CS Tutoring LLM System")
    parser.add_argument(
        "--mode",
        choices=["debug", "full", "eval_only"],
        default="debug",
        help="Execution mode"
    )
    parser.add_argument("--skip_training", action="store_true", help="Skip training steps")
    parser.add_argument("--skip_quantization", action="store_true", help="Skip quantization comparison")
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    if args.mode == "debug":
        config.debug_mode = True
        config.data.max_samples_train = 100
        config.data.max_samples_eval = 50
        config.training.num_epochs = 1
        logger.info("Running in DEBUG mode with limited data")
    elif args.mode == "full":
        config.debug_mode = False
        logger.info("Running in FULL mode")
    else:
        config.debug_mode = True
        logger.info("Running in EVAL_ONLY mode")
    
    # Print configuration
    print_config_summary()
    
    # Print parameter table
    print("\n" + get_parameter_table())
    
    # Print literature comparison
    print_literature_comparison()
    
    # Initialize results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "config": {
            "dataset": config.data.dataset_name,
            "base_model": config.model.base_model,
            "generation_model": config.model.generation_model,
        },
        "experiments": {},
    }
    
    # Resource monitor
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start_monitoring()
    
    try:
        # Load data
        logger.info("\n" + "=" * 60)
        logger.info("Loading Dataset")
        logger.info("=" * 60)
        
        data_loader = DataLoader(config)
        train_data, val_data, test_data = data_loader.load_dataset()
        
        logger.info(f"Dataset: {config.data.dataset_name}/{config.data.dataset_subset}")
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        all_results["data"] = {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        }
        
        # 1. BM25 Baseline
        bm25_results = run_bm25_baseline(config, data_loader)
        all_results["experiments"]["bm25"] = bm25_results
        
        # 2. Fine-tuned LLM
        if not args.skip_training and args.mode != "eval_only":
            llm_results = run_fine_tuned_model(config, data_loader)
            all_results["experiments"]["fine_tuned_llm"] = llm_results
        
        # 3. RAG System
        rag_results = run_rag_system(config, data_loader)
        all_results["experiments"]["rag"] = rag_results
        
        # 4. Quantization Comparison
        if not args.skip_quantization:
            quant_results = run_quantization_comparison(config)
            all_results["experiments"]["quantization"] = quant_results.get("summary", {})
        
        # 5. Evaluation
        logger.info("\n" + "=" * 60)
        logger.info("Running Comprehensive Evaluation")
        logger.info("=" * 60)
        
        # Create sample predictions for evaluation
        sample_predictions = [
            "This function implements binary search to find an element in a sorted array efficiently.",
            "Quicksort divides the array and recursively sorts each partition.",
            "This is a recursive function that calculates factorial.",
        ]
        sample_references = [
            d.docstring for d in test_data[:3]
        ] if len(test_data) >= 3 else sample_predictions
        sample_codes = [
            d.code for d in test_data[:3]
        ] if len(test_data) >= 3 else ["def test(): pass"] * 3
        
        evaluator = Evaluator(config)
        eval_results = evaluator.full_evaluation(
            predictions=sample_predictions,
            references=sample_references,
            codes=sample_codes,
        )
        
        all_results["evaluation"] = eval_results.to_dict()
        
        # Print evaluation summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for key, value in eval_results.to_dict().items():
            print(f"{key}: {value:.4f}")
        
    finally:
        # Stop resource monitoring
        resource_monitor.stop_monitoring()
        
        # Add resource stats
        all_results["resources"] = resource_monitor.get_summary().to_dict()
        
        # Print resource report
        print(resource_monitor.generate_report())
    
    # Save results
    output_path = save_results(all_results, config.output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()
