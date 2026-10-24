"""
Neurosymbolic reasoning model integrating neural and symbolic components.

Architecture combines:
- Neural encoder for feature extraction (transformers)
- Symbolic reasoner for rule-based inference
- Knowledge graph for hypothesis verification

References:
- Garcez et al. (2019): "Neural-Symbolic Computing: An Effective Methodology"
- Mao et al. (2019): "The Neuro-Symbolic Concept Learner" (NS-CL)
- Minervini et al. (2020): "Learning Reasoning Strategies in End-to-End Differentiable Proving"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from config import ModelConfig
from .neural_encoder import NeuralEncoder
from .symbolic_reasoner import SymbolicReasoner
from .knowledge_graph import KnowledgeGraphModule


class NeurosymbolicModel(nn.Module):
    """
    Complete neurosymbolic model for scientific reasoning.
    
    Combines:
    1. Neural encoder: Extracts features from text/questions
    2. Symbolic reasoner: Applies learned rules for multi-hop reasoning
    3. Knowledge graph: Verifies hypothesis consistency
    
    The model can operate in multiple modes:
    - Full: All components active
    - Neural-only: Skip symbolic reasoning (for ablation)
    - Symbolic-only: Minimal neural processing (for ablation)
    """
    
    def __init__(self, config: ModelConfig, num_classes: int = 4):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Core components
        self.neural_encoder = NeuralEncoder(config)
        self.symbolic_reasoner = SymbolicReasoner(config)
        self.knowledge_graph_module = KnowledgeGraphModule(config)
        
        # Integration layers
        self.neural_symbolic_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid()
        )
        
        self.feature_integration = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )
        
        # Hypothesis generation head
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.kg_embed_dim * 3)  # head, relation, tail
        )
        
        # Mode flags
        self.use_symbolic = True
        self.use_kg = True
    
    def set_mode(self, use_symbolic: bool = True, use_kg: bool = True):
        """Set which components to use."""
        self.use_symbolic = use_symbolic
        self.use_kg = use_kg
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass through the neurosymbolic model.
        
        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len) attention mask
            labels: Optional (batch,) ground truth labels
            
        Returns:
            Dict with:
                - logits: (batch, num_classes) classification logits
                - loss: Scalar loss (if labels provided)
                - neural_features: Features from neural encoder
                - symbolic_output: Output from symbolic reasoner
                - kg_output: Output from knowledge graph module
                - hypothesis_validity: Validity scores for generated hypotheses
        """
        batch_size = input_ids.size(0)
        
        # 1. Neural encoding
        encoder_output = self.neural_encoder(input_ids, attention_mask)
        neural_features = encoder_output["reasoning_features"]
        
        # 2. Symbolic reasoning (if enabled)
        if self.use_symbolic:
            symbolic_output = self.symbolic_reasoner(neural_features)
            symbolic_features = symbolic_output["symbolic_features"]
        else:
            symbolic_output = None
            symbolic_features = torch.zeros_like(neural_features)
        
        # 3. Knowledge graph integration (if enabled)
        if self.use_kg:
            kg_output = self.knowledge_graph_module(neural_features)
            kg_features = kg_output["kg_features"]
        else:
            kg_output = None
            kg_features = torch.zeros_like(neural_features)
        
        # 4. Integrate features with gating mechanism
        # Gate determines how much to rely on neural vs symbolic
        gate_input = torch.cat([neural_features, symbolic_features], dim=-1)
        gate = self.neural_symbolic_gate(gate_input)
        
        # Weighted combination
        gated_neural = gate * neural_features
        gated_symbolic = (1 - gate) * symbolic_features
        
        # Combine all features
        combined = torch.cat([gated_neural, gated_symbolic, kg_features], dim=-1)
        integrated_features = self.feature_integration(combined)
        
        # 5. Classification
        logits = self.classifier(integrated_features)
        
        # 6. Hypothesis generation and verification
        hypothesis_components = self.hypothesis_generator(integrated_features)
        hypothesis_components = hypothesis_components.view(batch_size, 3, self.config.kg_embed_dim)
        
        head_features = hypothesis_components[:, 0, :]
        relation_features = hypothesis_components[:, 1, :]
        tail_features = hypothesis_components[:, 2, :]
        
        hypothesis_validity = self.knowledge_graph_module.verify_hypothesis(
            head_features, relation_features, tail_features
        )
        
        # 7. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Classification loss
            ce_loss = F.cross_entropy(logits, labels)
            
            # Symbolic consistency loss
            if self.use_symbolic and symbolic_output is not None:
                symbolic_loss = self.symbolic_reasoner.compute_symbolic_loss(
                    symbolic_output["final_predicates"]
                )
            else:
                symbolic_loss = torch.tensor(0.0, device=input_ids.device)
            
            # Hypothesis validity should correlate with correct predictions
            # (correct answers should produce valid hypotheses)
            correct_mask = (torch.argmax(logits, dim=-1) == labels).float()
            validity_loss = F.binary_cross_entropy(
                hypothesis_validity,
                correct_mask.detach()  # Don't backprop through correctness
            )
            
            # Combined loss
            loss = (
                ce_loss +
                self.config.dropout * symbolic_loss +  # Placeholder weight
                0.1 * validity_loss
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "neural_features": neural_features,
            "symbolic_output": symbolic_output,
            "kg_output": kg_output,
            "hypothesis_validity": hypothesis_validity,
            "integrated_features": integrated_features
        }
    
    def generate_hypothesis(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Generate and verify hypotheses for the input.
        
        Returns interpretable hypothesis components.
        """
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
        
        hypothesis_features = self.hypothesis_generator(output["integrated_features"])
        batch_size = input_ids.size(0)
        hypothesis_features = hypothesis_features.view(batch_size, 3, self.config.kg_embed_dim)
        
        return {
            "head_embedding": hypothesis_features[:, 0, :],
            "relation_embedding": hypothesis_features[:, 1, :],
            "tail_embedding": hypothesis_features[:, 2, :],
            "validity_score": output["hypothesis_validity"],
            "prediction": torch.argmax(output["logits"], dim=-1)
        }
    
    def get_reasoning_trace(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Get interpretable reasoning trace for debugging/analysis.
        """
        output = self.forward(input_ids, attention_mask)
        
        trace = {
            "prediction": torch.argmax(output["logits"], dim=-1),
            "confidence": F.softmax(output["logits"], dim=-1).max(dim=-1).values,
            "hypothesis_validity": output["hypothesis_validity"]
        }
        
        if output["symbolic_output"] is not None:
            trace["symbolic_reasoning"] = {
                "initial_predicates": output["symbolic_output"]["initial_predicates"],
                "final_predicates": output["symbolic_output"]["final_predicates"],
                "reasoning_steps": output["symbolic_output"]["reasoning_trace"]["steps"]
            }
            
            # Get active rules
            trace["active_rules"] = self.symbolic_reasoner.rule_engine.get_active_rules(
                threshold=self.config.rule_confidence_threshold
            )
        
        return trace
    
    def get_param_count(self) -> Dict:
        """Get detailed parameter counts for all components."""
        counts = {
            "neural_encoder": self.neural_encoder.get_param_count(),
            "symbolic_reasoner": self.symbolic_reasoner.get_param_count(),
            "knowledge_graph": self.knowledge_graph_module.get_param_count(),
            "integration_layers": sum(p.numel() for p in self.neural_symbolic_gate.parameters()) +
                                 sum(p.numel() for p in self.feature_integration.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "hypothesis_generator": sum(p.numel() for p in self.hypothesis_generator.parameters())
        }
        
        counts["total"] = sum(
            v if isinstance(v, int) else v.get("total", 0)
            for v in counts.values()
        )
        
        return counts


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: NeurosymbolicModel):
    """Print detailed model summary."""
    print("\n" + "=" * 70)
    print("NEUROSYMBOLIC MODEL SUMMARY")
    print("=" * 70)
    
    counts = model.get_param_count()
    
    print("\nComponent Parameter Counts:")
    print("-" * 50)
    
    for name, value in counts.items():
        if isinstance(value, dict):
            print(f"\n{name}:")
            for k, v in value.items():
                print(f"  {k}: {v:,}")
        else:
            print(f"{name}: {value:,}")
    
    print("\n" + "-" * 50)
    print(f"TOTAL PARAMETERS: {counts['total']:,}")
    print(f"Trainable: {count_parameters(model):,}")
    print("=" * 70 + "\n")
