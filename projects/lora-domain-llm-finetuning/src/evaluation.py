"""Evaluation module for model assessment."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for language models."""
    
    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_perplexity(self, dataloader) -> Dict[str, float]:
        """Compute perplexity on a dataset."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                num_tokens = attention_mask.sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens
        }
    
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """Evaluate generation quality with sample outputs."""
        self.model.eval()
        generations = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append({
                "prompt": prompt,
                "generated": generated_text,
                "generated_only": generated_text[len(prompt):].strip()
            })
        
        avg_length = np.mean([
            len(g["generated_only"].split()) for g in generations
        ])
        
        return {
            "generations": generations,
            "avg_generation_length": avg_length,
            "num_samples": len(generations)
        }
    
    def compare_base_vs_finetuned(
        self,
        base_model,
        finetuned_model,
        test_dataloader
    ) -> Dict[str, Any]:
        """Compare base model vs fine-tuned model."""
        self.model = base_model
        base_metrics = self.compute_perplexity(test_dataloader)
        
        self.model = finetuned_model
        finetuned_metrics = self.compute_perplexity(test_dataloader)
        
        improvement = {
            "perplexity_reduction": base_metrics["perplexity"] - finetuned_metrics["perplexity"],
            "perplexity_reduction_pct": 100 * (base_metrics["perplexity"] - finetuned_metrics["perplexity"]) / base_metrics["perplexity"],
            "loss_reduction": base_metrics["loss"] - finetuned_metrics["loss"]
        }
        
        return {
            "base_model": base_metrics,
            "finetuned_model": finetuned_metrics,
            "improvement": improvement
        }


def evaluate_domain_adaptation(
    model,
    tokenizer,
    domain_test_data,
    general_test_data,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluate how well model adapted to domain vs general text."""
    evaluator = ModelEvaluator(model, tokenizer, device)
    
    domain_metrics = evaluator.compute_perplexity(domain_test_data)
    general_metrics = evaluator.compute_perplexity(general_test_data)
    
    domain_specificity = general_metrics["perplexity"] / domain_metrics["perplexity"]
    
    return {
        "domain_perplexity": domain_metrics["perplexity"],
        "general_perplexity": general_metrics["perplexity"],
        "domain_specificity_ratio": domain_specificity,
        "is_domain_specialized": domain_specificity > 1.0
    }
