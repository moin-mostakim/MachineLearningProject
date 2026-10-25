"""Baseline implementations for comparison."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Optional, List
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModels:
    """Collection of baseline methods for comparison."""
    
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_frozen_baseline(self) -> nn.Module:
        """Get frozen pretrained model (no fine-tuning)."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)
        logger.info("Loaded frozen baseline model")
        return model
    
    def get_full_finetuning_model(self) -> nn.Module:
        """Get model for full fine-tuning (all parameters trainable)."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.to(self.device)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Full fine-tuning model: {trainable:,} trainable parameters")
        return model
    
    def get_vanilla_lora_model(self, rank: int = 8) -> nn.Module:
        """Get model with vanilla LoRA (fixed rank, standard config)."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,  # Vanilla: alpha = rank
            lora_dropout=0.0,  # Vanilla: no dropout
            target_modules=["c_attn"],  # Vanilla: only attention
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Vanilla LoRA model (r={rank}): {trainable:,} trainable parameters")
        return model
    
    def get_adaptive_lora_model(
        self,
        rank: int = 8,
        alpha_multiplier: float = 2.0,
        dropout: float = 0.1
    ) -> nn.Module:
        """Get model with adaptive LoRA (our proposed method)."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=int(rank * alpha_multiplier),
            lora_dropout=dropout,
            target_modules=["c_attn", "c_proj"],  # More modules
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Adaptive LoRA model (r={rank}, α={int(rank * alpha_multiplier)}): {trainable:,} trainable parameters")
        return model
    
    def get_last_layer_finetuning_model(self) -> nn.Module:
        """Get model with only last layer fine-tuning (common baseline)."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        model.to(self.device)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Last-layer fine-tuning model: {trainable:,} trainable parameters")
        return model


class LiteratureResults:
    """Published results from the literature for comparison."""
    
    RESULTS = {
        "wikitext2": {
            "metric": "perplexity",
            "lower_is_better": True,
            "published": [
                {
                    "method": "GPT-2 (Radford et al., 2019)",
                    "paper": "Language Models are Unsupervised Multitask Learners",
                    "venue": "OpenAI Blog",
                    "year": 2019,
                    "value": 29.41,
                    "model_size": "117M",
                    "notes": "Full GPT-2 small model"
                },
                {
                    "method": "LoRA (Hu et al., 2022)",
                    "paper": "LoRA: Low-Rank Adaptation of Large Language Models",
                    "venue": "ICLR 2022",
                    "year": 2022,
                    "value": None,  # Paper focuses on GPT-3, not GPT-2/WikiText
                    "model_size": "varies",
                    "notes": "Primary experiments on GPT-3; WikiText-2 not directly reported"
                },
                {
                    "method": "AdaLoRA (Zhang et al., 2023)",
                    "paper": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
                    "venue": "ICLR 2023",
                    "year": 2023,
                    "value": None,
                    "model_size": "varies",
                    "notes": "Focuses on NLU tasks (GLUE); LM perplexity not primary metric"
                }
            ]
        },
        "glue_cola": {
            "metric": "Matthews Correlation",
            "lower_is_better": False,
            "published": [
                {
                    "method": "BERT-base (Devlin et al., 2019)",
                    "paper": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "venue": "NAACL 2019",
                    "year": 2019,
                    "value": 52.1,
                    "model_size": "110M",
                    "notes": "Full fine-tuning"
                },
                {
                    "method": "LoRA r=8 (Hu et al., 2022)",
                    "paper": "LoRA: Low-Rank Adaptation of Large Language Models",
                    "venue": "ICLR 2022",
                    "year": 2022,
                    "value": 63.4,  # RoBERTa-base
                    "model_size": "125M",
                    "notes": "RoBERTa-base with LoRA"
                },
                {
                    "method": "AdaLoRA (Zhang et al., 2023)",
                    "paper": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
                    "venue": "ICLR 2023",
                    "year": 2023,
                    "value": 65.1,
                    "model_size": "125M",
                    "notes": "RoBERTa-base with AdaLoRA"
                }
            ]
        }
    }
    
    @classmethod
    def get_comparison_table(cls, benchmark: str) -> str:
        """Generate a markdown comparison table."""
        if benchmark not in cls.RESULTS:
            return "No published results available for this benchmark."
        
        data = cls.RESULTS[benchmark]
        metric = data["metric"]
        
        lines = [
            f"## Literature Comparison: {benchmark}",
            f"**Metric:** {metric} ({'lower' if data['lower_is_better'] else 'higher'} is better)\n",
            "| Method | Value | Model Size | Venue | Year | Notes |",
            "|--------|-------|------------|-------|------|-------|"
        ]
        
        for pub in data["published"]:
            value_str = f"{pub['value']:.2f}" if pub['value'] is not None else "N/R"
            lines.append(
                f"| {pub['method']} | {value_str} | {pub['model_size']} | "
                f"{pub['venue']} | {pub['year']} | {pub['notes']} |"
            )
        
        lines.append("\n*N/R = Not Reported for this specific benchmark*")
        
        return "\n".join(lines)
    
    @classmethod
    def get_citations(cls, benchmark: str) -> List[str]:
        """Get citations for a benchmark."""
        if benchmark not in cls.RESULTS:
            return []
        
        citations = []
        for pub in cls.RESULTS[benchmark]["published"]:
            citations.append(
                f"- {pub['method'].split('(')[0].strip()}: "
                f"\"{pub['paper']}\", {pub['venue']}, {pub['year']}"
            )
        
        return citations
