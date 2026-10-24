"""
Baseline models for comparison with neurosymbolic approach.

1. PureNeuralBaseline: Transformer-only model (no symbolic reasoning)
2. RuleBasedBaseline: Pure rule-based system (no neural learning)

These baselines enable ablation studies to understand the contribution
of each component in the neurosymbolic architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import re
from collections import defaultdict

from config import ModelConfig
from .neural_encoder import NeuralEncoder


class PureNeuralBaseline(nn.Module):
    """
    Pure neural baseline using transformer architecture.
    
    This is essentially the neurosymbolic model with symbolic
    reasoning and KG components disabled, serving as an ablation.
    
    Architecture:
    - Transformer encoder for feature extraction
    - Classification head for answer prediction
    - No symbolic rules or knowledge graph integration
    """
    
    def __init__(self, config: ModelConfig, num_classes: int = 4):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Neural encoder
        self.encoder = NeuralEncoder(config)
        
        # Classification head (more capacity since no symbolic reasoning)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass through pure neural model.
        
        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len) attention mask
            labels: Optional (batch,) ground truth labels
            
        Returns:
            Dict with logits, loss, features
        """
        # Encode
        encoder_output = self.encoder(input_ids, attention_mask)
        features = encoder_output["reasoning_features"]
        
        # Classify
        logits = self.classifier(features)
        
        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "features": features,
            "attention_weights": encoder_output["attention_weights"]
        }
    
    def get_param_count(self) -> Dict:
        """Get parameter counts."""
        return {
            "encoder": self.encoder.get_param_count(),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }


class RuleBasedBaseline:
    """
    Pure rule-based baseline using handcrafted rules.
    
    This baseline uses no neural networks - only pattern matching
    and logical rules for scientific reasoning.
    
    Rules are inspired by scientific reasoning patterns:
    1. Causal rules: "X causes Y" implies Y given X
    2. Property rules: "X has property P" enables inference about X
    3. Classification rules: Type hierarchies for entities
    4. Elimination rules: Contradict wrong answers
    """
    
    def __init__(self, config: ModelConfig = None):
        self.rules = []
        self.entity_properties = defaultdict(set)
        self.entity_types = defaultdict(set)
        self.causal_relations = defaultdict(set)
        
        # Add default scientific reasoning rules
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add handcrafted scientific reasoning rules."""
        # Causal patterns
        self.causal_patterns = [
            (r"(\w+)\s+causes?\s+(\w+)", "causes"),
            (r"(\w+)\s+leads?\s+to\s+(\w+)", "leads_to"),
            (r"(\w+)\s+results?\s+in\s+(\w+)", "results_in"),
            (r"(\w+)\s+produces?\s+(\w+)", "produces"),
        ]
        
        # Property patterns
        self.property_patterns = [
            (r"(\w+)\s+is\s+(\w+)", "is_property"),
            (r"(\w+)\s+are\s+(\w+)", "is_property"),
            (r"(\w+)\s+has\s+(\w+)", "has_property"),
            (r"(\w+)\s+contains?\s+(\w+)", "contains"),
        ]
        
        # Type patterns
        self.type_patterns = [
            (r"(\w+)\s+is\s+a\s+type\s+of\s+(\w+)", "type_of"),
            (r"(\w+)\s+is\s+an?\s+(\w+)", "instance_of"),
        ]
        
        # Scientific domain rules
        self.domain_rules = {
            # Physics
            "force": ["motion", "acceleration", "velocity"],
            "energy": ["work", "power", "heat"],
            "mass": ["weight", "gravity", "inertia"],
            
            # Chemistry
            "reaction": ["product", "reactant", "catalyst"],
            "element": ["atom", "electron", "proton"],
            "compound": ["molecule", "bond", "formula"],
            
            # Biology
            "cell": ["nucleus", "membrane", "organelle"],
            "organism": ["species", "population", "ecosystem"],
            "gene": ["dna", "protein", "trait"],
            
            # General science
            "hypothesis": ["experiment", "observation", "conclusion"],
            "variable": ["independent", "dependent", "control"],
        }
    
    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract facts from text using pattern matching."""
        text_lower = text.lower()
        facts = []
        
        # Extract causal relations
        for pattern, relation in self.causal_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    facts.append((match[0], relation, match[1]))
        
        # Extract properties
        for pattern, relation in self.property_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    facts.append((match[0], relation, match[1]))
        
        # Extract types
        for pattern, relation in self.type_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    facts.append((match[0], relation, match[1]))
        
        return facts
    
    def score_answer(
        self,
        question: str,
        answer: str,
        support_text: str = ""
    ) -> float:
        """
        Score an answer using rule-based reasoning.
        
        Returns a confidence score in [0, 1].
        """
        score = 0.0
        
        # Combine question and support text
        context = question.lower() + " " + support_text.lower()
        answer_lower = answer.lower()
        
        # Extract facts from context
        facts = self.extract_facts(context)
        
        # Rule 1: Direct mention score
        answer_words = set(answer_lower.split())
        context_words = set(context.split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        score += 0.3 * overlap
        
        # Rule 2: Causal chain score
        for head, rel, tail in facts:
            if rel in ["causes", "leads_to", "results_in", "produces"]:
                if head in answer_lower or tail in answer_lower:
                    score += 0.2
        
        # Rule 3: Domain relevance score
        for domain, keywords in self.domain_rules.items():
            if domain in context:
                keyword_matches = sum(1 for kw in keywords if kw in answer_lower)
                score += 0.1 * min(keyword_matches, 2)
        
        # Rule 4: Consistency check
        # Answers that contradict known facts get penalized
        for head, rel, tail in facts:
            if rel == "is_property":
                # Check for contradictions
                if head in answer_lower and tail not in context:
                    score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def predict(
        self,
        question: str,
        choices: List[str],
        support_text: str = ""
    ) -> Tuple[int, List[float]]:
        """
        Predict the best answer from choices.
        
        Args:
            question: The question text
            choices: List of answer choices
            support_text: Optional supporting context
            
        Returns:
            predicted_idx: Index of predicted answer
            scores: Scores for all choices
        """
        scores = [
            self.score_answer(question, choice, support_text)
            for choice in choices
        ]
        
        # Normalize scores
        total = sum(scores) + 1e-8
        scores = [s / total for s in scores]
        
        predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        return predicted_idx, scores
    
    def forward(
        self,
        questions: List[str],
        choices_list: List[List[str]],
        support_texts: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Batch prediction interface matching neural models.
        
        Args:
            questions: List of question strings
            choices_list: List of choice lists for each question
            support_texts: Optional support text for each question
            labels: Optional ground truth labels
            
        Returns:
            Dict with predictions and scores
        """
        if support_texts is None:
            support_texts = [""] * len(questions)
        
        predictions = []
        all_scores = []
        
        for q, choices, support in zip(questions, choices_list, support_texts):
            pred_idx, scores = self.predict(q, choices, support)
            predictions.append(pred_idx)
            all_scores.append(scores)
        
        # Convert to tensors for compatibility
        predictions_tensor = torch.tensor(predictions)
        scores_tensor = torch.tensor(all_scores)
        
        # Compute accuracy if labels provided
        loss = None
        if labels is not None:
            # Pseudo-loss for logging (not used for training)
            correct = (predictions_tensor == labels).float()
            loss = 1.0 - correct.mean()
        
        return {
            "logits": scores_tensor,
            "predictions": predictions_tensor,
            "loss": loss,
            "rule_activations": None  # No learned rules
        }
    
    def get_param_count(self) -> Dict:
        """Rule-based system has no learnable parameters."""
        return {
            "num_rules": len(self.causal_patterns) + len(self.property_patterns) + len(self.type_patterns),
            "domain_rules": sum(len(v) for v in self.domain_rules.values()),
            "total": 0  # No learnable parameters
        }


def compare_baselines():
    """Print comparison of baseline architectures."""
    from config import ModelConfig
    
    config = ModelConfig()
    
    neural_baseline = PureNeuralBaseline(config)
    rule_baseline = RuleBasedBaseline()
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    print("\n1. Pure Neural Baseline (Transformer):")
    neural_counts = neural_baseline.get_param_count()
    print(f"   Total Parameters: {neural_counts['total']:,}")
    
    print("\n2. Rule-Based Baseline:")
    rule_counts = rule_baseline.get_param_count()
    print(f"   Number of Rules: {rule_counts['num_rules']}")
    print(f"   Domain Rules: {rule_counts['domain_rules']}")
    print(f"   Learnable Parameters: {rule_counts['total']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    compare_baselines()
