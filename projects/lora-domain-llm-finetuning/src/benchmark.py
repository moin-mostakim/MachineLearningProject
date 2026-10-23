"""Benchmark evaluation module."""

import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, Any, List, Optional
from sklearn.metrics import matthews_corrcoef, accuracy_score
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluator for standard benchmarks."""
    
    SUPPORTED_BENCHMARKS = [
        "wikitext2",      # Language modeling perplexity
        "lambada",        # Zero-shot language understanding
        "glue_cola",      # Linguistic acceptability
        "glue_sst2",      # Sentiment analysis
    ]
    
    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_wikitext2(self, max_samples: int = 500) -> Dict[str, Any]:
        """Evaluate on WikiText-2 benchmark (standard LM evaluation)."""
        logger.info("Evaluating on WikiText-2...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        except Exception:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        
        texts = [t for t in dataset["text"] if t.strip()][:max_samples]
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    continue
                    
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                if encodings.input_ids.shape[1] < 2:
                    continue
                
                outputs = self.model(**encodings, labels=encodings.input_ids)
                
                num_tokens = encodings.attention_mask.sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        
        return {
            "benchmark": "wikitext2",
            "metric": "perplexity",
            "value": perplexity,
            "loss": avg_loss,
            "total_tokens": total_tokens,
            "samples_evaluated": len(texts)
        }
    
    def evaluate_lambada(self, max_samples: int = 200) -> Dict[str, Any]:
        """Evaluate on LAMBADA benchmark (word prediction accuracy)."""
        logger.info("Evaluating on LAMBADA...")
        
        try:
            dataset = load_dataset("lambada", split="test")
        except Exception:
            dataset = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        
        samples = list(dataset)[:max_samples]
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sample in samples:
                text = sample["text"]
                words = text.split()
                if len(words) < 2:
                    continue
                
                context = " ".join(words[:-1])
                target_word = words[-1]
                
                inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip().split()[0] if outputs[0].shape[0] > inputs.input_ids.shape[1] else ""
                
                if generated.lower().startswith(target_word.lower()[:3]):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "benchmark": "lambada",
            "metric": "accuracy",
            "value": accuracy * 100,
            "correct": correct,
            "total": total
        }
    
    def run_all_benchmarks(self, max_samples: int = 200) -> Dict[str, Dict[str, Any]]:
        """Run all supported benchmarks."""
        results = {}
        
        results["wikitext2"] = self.evaluate_wikitext2(max_samples)
        
        try:
            results["lambada"] = self.evaluate_lambada(max_samples)
        except Exception as e:
            logger.warning(f"LAMBADA evaluation failed: {e}")
            results["lambada"] = {"error": str(e)}
        
        return results


def compare_methods_on_benchmark(
    methods: Dict[str, torch.nn.Module],
    tokenizer,
    benchmark: str = "wikitext2",
    max_samples: int = 200,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Compare multiple methods on a benchmark."""
    results = {}
    
    for method_name, model in methods.items():
        logger.info(f"Evaluating {method_name}...")
        model.to(device)
        model.eval()
        
        evaluator = BenchmarkEvaluator(model, tokenizer, device)
        
        if benchmark == "wikitext2":
            result = evaluator.evaluate_wikitext2(max_samples)
        elif benchmark == "lambada":
            result = evaluator.evaluate_lambada(max_samples)
        else:
            result = {"error": f"Unknown benchmark: {benchmark}"}
        
        results[method_name] = result
        
        model.to("cpu")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results
