"""
Symbolic reasoning module for neurosymbolic integration.

Implements differentiable rule learning and inference, drawing from:
- Rocktäschel & Riedel (2017): "End-to-End Differentiable Proving"
- Evans & Grefenstette (2018): "Learning Explanatory Rules from Noisy Data" (δILP)
- Serafini & d'Avila Garcez (2016): "Logic Tensor Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import ModelConfig


@dataclass
class Rule:
    """Represents a learnable logical rule."""
    head_predicate: int  # Index of head predicate
    body_predicates: List[int]  # Indices of body predicates
    confidence: float  # Learned confidence score
    
    def __repr__(self):
        return f"Rule(head={self.head_predicate}, body={self.body_predicates}, conf={self.confidence:.3f})"


class PredicateEmbeddings(nn.Module):
    """Learnable predicate representations for symbolic reasoning."""
    
    def __init__(self, num_predicates: int, hidden_dim: int):
        super().__init__()
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        
        # Predicate embeddings
        self.predicate_embeddings = nn.Embedding(num_predicates, hidden_dim)
        
        # Predicate-to-neural projection
        self.neural_to_predicate = nn.Linear(hidden_dim, num_predicates)
        
        # Initialize with small values for stable learning
        nn.init.normal_(self.predicate_embeddings.weight, std=0.02)
    
    def forward(self, neural_features: torch.Tensor) -> torch.Tensor:
        """
        Map neural features to predicate space.
        
        Args:
            neural_features: (batch, hidden_dim) neural representations
            
        Returns:
            predicate_scores: (batch, num_predicates) scores for each predicate
        """
        # Project to predicate space
        predicate_scores = self.neural_to_predicate(neural_features)
        
        # Apply sigmoid to get truth values in [0, 1]
        predicate_probs = torch.sigmoid(predicate_scores)
        
        return predicate_probs
    
    def get_predicate_embedding(self, predicate_idx: torch.Tensor) -> torch.Tensor:
        """Get embedding for specific predicates."""
        return self.predicate_embeddings(predicate_idx)


class RuleEngine(nn.Module):
    """
    Differentiable rule learning and inference engine.
    
    Implements soft unification and rule chaining following Neural Theorem Provers.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_rules = config.num_rules
        self.num_predicates = config.num_predicates
        
        # Learnable rule templates
        # Each rule: head <- body1, body2 (at most 2 body predicates for tractability)
        self.rule_heads = nn.Parameter(torch.randn(config.num_rules, config.num_predicates))
        self.rule_body1 = nn.Parameter(torch.randn(config.num_rules, config.num_predicates))
        self.rule_body2 = nn.Parameter(torch.randn(config.num_rules, config.num_predicates))
        
        # Rule confidence weights
        self.rule_confidences = nn.Parameter(torch.zeros(config.num_rules))
        
        # Attention for rule selection
        self.rule_attention = nn.Linear(config.num_predicates, config.num_rules)
    
    def forward(
        self,
        predicate_probs: torch.Tensor,
        num_steps: int = 3
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply differentiable forward chaining.
        
        Args:
            predicate_probs: (batch, num_predicates) initial truth values
            num_steps: Number of reasoning steps
            
        Returns:
            final_probs: (batch, num_predicates) updated truth values
            reasoning_trace: Dict with intermediate reasoning states
        """
        batch_size = predicate_probs.size(0)
        current_probs = predicate_probs
        
        reasoning_trace = {
            "steps": [],
            "rule_activations": [],
            "confidence_scores": []
        }
        
        # Soft rule application
        rule_head_probs = F.softmax(self.rule_heads, dim=-1)  # (num_rules, num_predicates)
        rule_body1_probs = F.softmax(self.rule_body1, dim=-1)
        rule_body2_probs = F.softmax(self.rule_body2, dim=-1)
        rule_confs = torch.sigmoid(self.rule_confidences)  # (num_rules,)
        
        for step in range(num_steps):
            # Compute body satisfaction for each rule
            # body1_sat = sum(body1_probs[r, p] * current_probs[:, p])
            body1_sat = torch.matmul(current_probs, rule_body1_probs.T)  # (batch, num_rules)
            body2_sat = torch.matmul(current_probs, rule_body2_probs.T)  # (batch, num_rules)
            
            # AND combination (product t-norm)
            body_sat = body1_sat * body2_sat  # (batch, num_rules)
            
            # Apply rule confidence
            rule_activation = body_sat * rule_confs.unsqueeze(0)  # (batch, num_rules)
            
            # Compute head updates
            # For each predicate, aggregate contributions from all rules
            head_update = torch.matmul(rule_activation, rule_head_probs)  # (batch, num_predicates)
            
            # Update predicate probabilities (OR combination - max)
            current_probs = torch.max(current_probs, head_update)
            
            # Clamp to valid probability range
            current_probs = torch.clamp(current_probs, 0.0, 1.0)
            
            # Record trace
            reasoning_trace["steps"].append(current_probs.detach())
            reasoning_trace["rule_activations"].append(rule_activation.detach())
            reasoning_trace["confidence_scores"].append(rule_confs.detach())
        
        return current_probs, reasoning_trace
    
    def get_active_rules(self, threshold: float = 0.5) -> List[Rule]:
        """Extract high-confidence rules for interpretability."""
        rule_confs = torch.sigmoid(self.rule_confidences)
        
        active_rules = []
        for i in range(self.num_rules):
            if rule_confs[i].item() >= threshold:
                head_idx = torch.argmax(self.rule_heads[i]).item()
                body1_idx = torch.argmax(self.rule_body1[i]).item()
                body2_idx = torch.argmax(self.rule_body2[i]).item()
                
                active_rules.append(Rule(
                    head_predicate=head_idx,
                    body_predicates=[body1_idx, body2_idx],
                    confidence=rule_confs[i].item()
                ))
        
        return sorted(active_rules, key=lambda r: r.confidence, reverse=True)


class SymbolicReasoner(nn.Module):
    """
    Complete symbolic reasoning module.
    
    Combines predicate grounding and differentiable rule inference.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Predicate embeddings
        self.predicate_embeddings = PredicateEmbeddings(
            config.num_predicates,
            config.hidden_dim
        )
        
        # Rule engine
        self.rule_engine = RuleEngine(config)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.num_predicates, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(
        self,
        neural_features: torch.Tensor,
        reasoning_depth: Optional[int] = None
    ) -> Dict:
        """
        Perform symbolic reasoning on neural features.
        
        Args:
            neural_features: (batch, hidden_dim) features from neural encoder
            reasoning_depth: Override for number of reasoning steps
            
        Returns:
            Dict with:
                - initial_predicates: Initial predicate groundings
                - final_predicates: After reasoning chain
                - symbolic_features: Output features for integration
                - reasoning_trace: Interpretable reasoning steps
        """
        depth = reasoning_depth or self.config.symbolic_reasoning_depth
        
        # Ground neural features to predicates
        initial_predicates = self.predicate_embeddings(neural_features)
        
        # Apply rule-based reasoning
        final_predicates, reasoning_trace = self.rule_engine(
            initial_predicates,
            num_steps=depth
        )
        
        # Project back to neural space
        symbolic_features = self.output_projection(final_predicates)
        
        return {
            "initial_predicates": initial_predicates,
            "final_predicates": final_predicates,
            "symbolic_features": symbolic_features,
            "reasoning_trace": reasoning_trace
        }
    
    def compute_symbolic_loss(
        self,
        final_predicates: torch.Tensor,
        target_predicates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for symbolic reasoning supervision.
        
        Includes:
        - Consistency loss (predicates should be coherent)
        - Sparsity loss (encourage focused predicate activation)
        """
        losses = {}
        
        # Consistency: high-confidence predicates shouldn't contradict
        # Using entropy as a proxy for coherent beliefs
        entropy = -torch.sum(
            final_predicates * torch.log(final_predicates + 1e-8) +
            (1 - final_predicates) * torch.log(1 - final_predicates + 1e-8),
            dim=-1
        )
        losses["entropy_loss"] = entropy.mean()
        
        # Sparsity: encourage focused activation
        l1_sparsity = torch.mean(torch.abs(final_predicates))
        losses["sparsity_loss"] = l1_sparsity
        
        # Supervised loss if targets provided
        if target_predicates is not None:
            losses["supervision_loss"] = F.binary_cross_entropy(
                final_predicates,
                target_predicates
            )
        
        total_loss = losses["entropy_loss"] * 0.1 + losses["sparsity_loss"] * 0.05
        if "supervision_loss" in losses:
            total_loss = total_loss + losses["supervision_loss"]
        
        return total_loss
    
    def get_param_count(self) -> dict:
        """Get parameter counts."""
        return {
            "predicate_embeddings": sum(p.numel() for p in self.predicate_embeddings.parameters()),
            "rule_engine": sum(p.numel() for p in self.rule_engine.parameters()),
            "output_projection": sum(p.numel() for p in self.output_projection.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }
