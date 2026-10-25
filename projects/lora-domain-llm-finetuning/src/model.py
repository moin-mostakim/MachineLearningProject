"""Model module with LoRA adaptation."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from typing import Optional, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveLoRAModel:
    """Wrapper for LLM with Adaptive LoRA fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["c_attn", "c_proj"]
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.base_model = None
        self.peft_model = None
        self.tokenizer = None
        
    def load_base_model(self):
        """Load the base pretrained model."""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info(f"Base model loaded with {self._count_parameters(self.base_model):,} parameters")
        return self
    
    def apply_lora(self) -> "AdaptiveLoRAModel":
        """Apply LoRA adaptation to the model."""
        if self.base_model is None:
            self.load_base_model()
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.peft_model = get_peft_model(self.base_model, lora_config)
        self.peft_model.to(self.device)
        
        trainable_params = self._count_trainable_parameters(self.peft_model)
        total_params = self._count_parameters(self.peft_model)
        
        logger.info(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
                   f"({100 * trainable_params / total_params:.2f}%)")
        
        return self
    
    def get_model(self):
        """Get the PEFT model for training."""
        if self.peft_model is None:
            self.apply_lora()
        return self.peft_model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        if self.tokenizer is None:
            self.load_base_model()
        return self.tokenizer
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        return self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save(self, path: str):
        """Save the LoRA weights."""
        self.peft_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, base_model_name: str = "distilgpt2"):
        """Load a saved LoRA model."""
        instance = cls(model_name=base_model_name)
        instance.load_base_model()
        instance.peft_model = PeftModel.from_pretrained(
            instance.base_model,
            path
        )
        instance.peft_model.to(instance.device)
        return instance
    
    @staticmethod
    def _count_parameters(model) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def _count_trainable_parameters(model) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_lora_stats(self) -> Dict[str, Any]:
        """Get statistics about LoRA configuration."""
        if self.peft_model is None:
            return {}
        
        return {
            "rank": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "trainable_params": self._count_trainable_parameters(self.peft_model),
            "total_params": self._count_parameters(self.peft_model),
            "trainable_percentage": 100 * self._count_trainable_parameters(self.peft_model) / 
                                   self._count_parameters(self.peft_model)
        }


def create_model_with_adaptive_lora(
    model_name: str = "distilgpt2",
    rank_schedule: Optional[Dict[str, int]] = None
) -> AdaptiveLoRAModel:
    """Factory function to create model with adaptive LoRA ranks."""
    
    if rank_schedule is None:
        rank_schedule = {
            "attention": 8,
            "mlp": 4
        }
    
    avg_rank = sum(rank_schedule.values()) // len(rank_schedule)
    
    model = AdaptiveLoRAModel(
        model_name=model_name,
        lora_r=avg_rank,
        lora_alpha=avg_rank * 2
    )
    
    return model.apply_lora()
