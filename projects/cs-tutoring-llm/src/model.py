"""
Neural Model Architectures for CS Tutoring System.

Implements code explanation models based on pre-trained language models:
- CodeBERT for code understanding
- CodeT5 for code explanation generation
- Custom tutoring head for educational quality

References:
- Feng et al., 2020. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
- Wang et al., 2021. "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models"
- Chen et al., 2021. "Evaluating Large Language Models Trained on Code" (Codex)
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_config, PARAMETER_COUNTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExplanationModel(nn.Module):
    """
    Code Explanation Model based on CodeT5.
    
    Architecture:
    - CodeT5 encoder-decoder backbone
    - Fine-tuned for code-to-explanation generation
    
    This model takes code as input and generates natural language
    explanations suitable for educational purposes.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or get_config()
        
        # Load pre-trained CodeT5
        logger.info(f"Loading generation model: {self.config.model.generation_model}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.generation_model,
            cache_dir=self.config.cache_dir
        )
        
        # Try to load tokenizer - use T5 tokenizer directly if AutoTokenizer fails
        try:
            from transformers import RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(
                self.config.model.generation_model,
                cache_dir=self.config.cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with RobertaTokenizer: {e}")
            try:
                from transformers import T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(
                    "t5-small",
                    cache_dir=self.config.cache_dir
                )
            except Exception as e2:
                logger.warning(f"Failed to load T5Tokenizer: {e2}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "gpt2",
                    cache_dir=self.config.cache_dir
                )
        
        # Track parameter counts
        self.param_count = self._count_parameters()
        logger.info(f"Model parameters: {self.param_count:,}")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Tokenized code input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target explanation tokens [batch, seq_len]
            
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def generate_explanation(
        self,
        code: str,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate explanation for given code.
        
        Args:
            code: Source code to explain
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated explanation string
        """
        # Prepare input
        inputs = self.tokenizer(
            code,
            max_length=self.config.data.max_code_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
            )
        
        # Decode
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return explanation


class TutoringModel(nn.Module):
    """
    Educational Tutoring Model with answer relevance scoring.
    
    Architecture:
    - CodeBERT encoder for code understanding
    - Pooler layer for sequence representation
    - Classification head for relevance scoring
    - Optional explanation decoder
    
    This model assesses the educational quality and relevance
    of code explanations for tutoring purposes.
    
    References:
    - Kumar et al., 2020. "Educational Question Answering using BERT"
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or get_config()
        
        # Load CodeBERT encoder
        logger.info(f"Loading base model: {self.config.model.base_model}")
        self.encoder = AutoModel.from_pretrained(
            self.config.model.base_model,
            cache_dir=self.config.cache_dir
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            cache_dir=self.config.cache_dir
        )
        
        hidden_size = self.encoder.config.hidden_size
        
        # Relevance classification head
        self.relevance_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.model.dropout_rate),
            nn.Linear(hidden_size // 2, self.config.model.num_labels),
        )
        
        # Educational quality regression head
        self.quality_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.model.dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Track parameter counts
        self.param_count = self._count_parameters()
        logger.info(f"Model parameters: {self.param_count:,}")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        relevance_labels: Optional[torch.Tensor] = None,
        quality_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training and inference.
        
        Args:
            input_ids: Tokenized input (code + explanation) [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            relevance_labels: Binary relevance labels [batch]
            quality_labels: Quality scores in [0, 1] [batch]
            
        Returns:
            Dictionary with losses, logits, and scores
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Pool [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        relevance_logits = self.relevance_classifier(pooled)
        quality_scores = self.quality_regressor(pooled).squeeze(-1)
        
        result = {
            "relevance_logits": relevance_logits,
            "quality_scores": quality_scores,
            "pooled_output": pooled,
        }
        
        # Compute losses if labels provided
        total_loss = 0.0
        
        if relevance_labels is not None:
            relevance_loss = F.cross_entropy(relevance_logits, relevance_labels)
            result["relevance_loss"] = relevance_loss
            total_loss += relevance_loss
        
        if quality_labels is not None:
            quality_loss = F.mse_loss(quality_scores, quality_labels)
            result["quality_loss"] = quality_loss
            total_loss += quality_loss
        
        if relevance_labels is not None or quality_labels is not None:
            result["loss"] = total_loss
        
        return result
    
    def predict_relevance(
        self,
        code: str,
        explanation: str,
    ) -> Tuple[int, float]:
        """
        Predict relevance of explanation for code.
        
        Args:
            code: Source code
            explanation: Explanation text
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        # Combine code and explanation
        combined = f"Code: {code}\nExplanation: {explanation}"
        
        # Tokenize
        inputs = self.tokenizer(
            combined,
            max_length=self.config.data.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.forward(**inputs)
        
        logits = outputs["relevance_logits"]
        probs = F.softmax(logits, dim=-1)
        
        predicted = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted].item()
        
        return predicted, confidence
    
    def predict_quality(
        self,
        code: str,
        explanation: str,
    ) -> float:
        """
        Predict educational quality of explanation.
        
        Args:
            code: Source code
            explanation: Explanation text
            
        Returns:
            Quality score in [0, 1]
        """
        combined = f"Code: {code}\nExplanation: {explanation}"
        
        inputs = self.tokenizer(
            combined,
            max_length=self.config.data.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
        
        return outputs["quality_scores"].item()


def get_parameter_table() -> str:
    """Generate parameter count table for all models."""
    table = []
    table.append("=" * 70)
    table.append("MODEL PARAMETER COUNTS")
    table.append("=" * 70)
    table.append(f"{'Model':<30} {'Parameters':<15} {'Notes':<25}")
    table.append("-" * 70)
    
    for model_name, info in PARAMETER_COUNTS.items():
        total = info.get("total", 0)
        if total > 0:
            param_str = f"{total:,}"
        else:
            param_str = "N/A"
        
        note = info.get("note", "")
        if not note and "breakdown" in info:
            note = f"{len(info['breakdown'])} components"
        
        table.append(f"{model_name:<30} {param_str:<15} {note:<25}")
    
    table.append("=" * 70)
    
    # Add breakdown for CodeBERT
    table.append("\nCodeBERT-base Breakdown:")
    table.append("-" * 50)
    for component, count in PARAMETER_COUNTS["CodeBERT-base"]["breakdown"].items():
        table.append(f"  {component:<30} {count:>15,}")
    
    return "\n".join(table)


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_explanation_model(
        config: Optional[Config] = None,
        pretrained_path: Optional[str] = None,
    ) -> CodeExplanationModel:
        """Create code explanation model."""
        model = CodeExplanationModel(config)
        
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {pretrained_path}")
        
        return model
    
    @staticmethod
    def create_tutoring_model(
        config: Optional[Config] = None,
        pretrained_path: Optional[str] = None,
    ) -> TutoringModel:
        """Create tutoring model."""
        model = TutoringModel(config)
        
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {pretrained_path}")
        
        return model


if __name__ == "__main__":
    # Test models
    print(get_parameter_table())
    
    config = get_config()
    
    # Test explanation model
    print("\nTesting CodeExplanationModel...")
    exp_model = CodeExplanationModel(config)
    
    test_code = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)'''
    
    explanation = exp_model.generate_explanation(test_code)
    print(f"Code:\n{test_code}")
    print(f"\nGenerated explanation:\n{explanation}")
    
    # Test tutoring model
    print("\n" + "=" * 50)
    print("Testing TutoringModel...")
    tut_model = TutoringModel(config)
    
    relevance, confidence = tut_model.predict_relevance(
        test_code,
        "This function calculates factorial recursively."
    )
    quality = tut_model.predict_quality(
        test_code,
        "This function calculates factorial recursively."
    )
    
    print(f"Relevance: {relevance} (confidence: {confidence:.4f})")
    print(f"Quality: {quality:.4f}")
