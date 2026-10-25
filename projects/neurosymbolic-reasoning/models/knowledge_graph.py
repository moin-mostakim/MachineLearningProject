"""
Knowledge Graph module for hypothesis verification.

Implements knowledge graph embeddings and reasoning, based on:
- Bordes et al. (2013): "Translating Embeddings for Modeling Multi-relational Data" (TransE)
- Sun et al. (2019): "RotatE: Knowledge Graph Embedding by Relational Rotation"
- Minervini et al. (2020): "Differentiable Reasoning on Large Knowledge Bases"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict

from config import ModelConfig


class KnowledgeGraphEmbedding(nn.Module):
    """
    Knowledge graph embedding layer using TransE-style scoring.
    
    Implements: h + r ≈ t for valid triples (h, r, t).
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        margin: float = 1.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.margin = margin
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        
        # Initialize embeddings
        nn.init.uniform_(self.entity_embeddings.weight, -0.5, 0.5)
        nn.init.uniform_(self.relation_embeddings.weight, -0.5, 0.5)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TransE scores for triples.
        
        Args:
            heads: (batch,) head entity indices
            relations: (batch,) relation indices
            tails: (batch,) tail entity indices
            
        Returns:
            scores: (batch,) negative distance scores (higher = more valid)
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # TransE score: -||h + r - t||
        score = -torch.norm(h + r - t, p=2, dim=1)
        
        return score
    
    def compute_loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_relations: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.
        
        L = max(0, margin + d(h+r, t) - d(h'+r, t'))
        """
        pos_scores = self.forward(pos_heads, pos_relations, pos_tails)
        neg_scores = self.forward(neg_heads, neg_relations, neg_tails)
        
        # Margin ranking loss
        loss = F.relu(self.margin - pos_scores + neg_scores)
        
        return loss.mean()
    
    def get_entity_embedding(self, entity_idx: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific entities."""
        return self.entity_embeddings(entity_idx)
    
    def get_relation_embedding(self, relation_idx: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific relations."""
        return self.relation_embeddings(relation_idx)


class KnowledgeGraph:
    """
    In-memory knowledge graph data structure.
    
    Stores triples and provides lookup operations.
    """
    
    def __init__(self):
        self.triples: List[Tuple[str, str, str]] = []
        self.entity_to_idx: Dict[str, int] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.idx_to_relation: Dict[int, str] = {}
        
        # Adjacency structures for efficient lookup
        self.head_to_triples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.tail_to_triples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.relation_to_triples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    
    def add_triple(self, head: str, relation: str, tail: str):
        """Add a triple to the knowledge graph."""
        triple = (head, relation, tail)
        self.triples.append(triple)
        
        # Update entity index
        for entity in [head, tail]:
            if entity not in self.entity_to_idx:
                idx = len(self.entity_to_idx)
                self.entity_to_idx[entity] = idx
                self.idx_to_entity[idx] = entity
        
        # Update relation index
        if relation not in self.relation_to_idx:
            idx = len(self.relation_to_idx)
            self.relation_to_idx[relation] = idx
            self.idx_to_relation[idx] = relation
        
        # Update adjacency
        self.head_to_triples[head].append(triple)
        self.tail_to_triples[tail].append(triple)
        self.relation_to_triples[relation].append(triple)
    
    def get_neighbors(self, entity: str) -> List[Tuple[str, str, str]]:
        """Get all triples involving an entity."""
        return self.head_to_triples[entity] + self.tail_to_triples[entity]
    
    def get_triples_by_relation(self, relation: str) -> List[Tuple[str, str, str]]:
        """Get all triples with a specific relation."""
        return self.relation_to_triples[relation]
    
    @property
    def num_entities(self) -> int:
        return len(self.entity_to_idx)
    
    @property
    def num_relations(self) -> int:
        return len(self.relation_to_idx)
    
    def save(self, path: str):
        """Save knowledge graph to JSON."""
        data = {
            "triples": self.triples,
            "entity_to_idx": self.entity_to_idx,
            "relation_to_idx": self.relation_to_idx
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load knowledge graph from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        for h, r, t in data["triples"]:
            kg.add_triple(h, r, t)
        
        return kg


class KnowledgeGraphModule(nn.Module):
    """
    Neural module for knowledge graph integration in reasoning.
    
    Provides:
    - Entity/relation embeddings
    - Triple verification
    - Hypothesis consistency checking
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Placeholder dimensions (updated when KG is loaded)
        self.num_entities = 1000  # Default
        self.num_relations = config.kg_num_relations
        
        # KG embeddings
        self.kg_embeddings = KnowledgeGraphEmbedding(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            embed_dim=config.kg_embed_dim,
            margin=config.kg_margin
        )
        
        # Neural-to-KG projection
        self.neural_to_entity = nn.Linear(config.hidden_dim, config.kg_embed_dim)
        self.entity_to_neural = nn.Linear(config.kg_embed_dim, config.hidden_dim)
        
        # Verification network
        self.verification_head = nn.Sequential(
            nn.Linear(config.kg_embed_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Knowledge graph data
        self.knowledge_graph: Optional[KnowledgeGraph] = None
    
    def set_knowledge_graph(self, kg: KnowledgeGraph):
        """Set the knowledge graph and resize embeddings if needed."""
        self.knowledge_graph = kg
        
        if kg.num_entities > self.num_entities:
            # Expand entity embeddings
            old_embeddings = self.kg_embeddings.entity_embeddings
            self.num_entities = kg.num_entities
            
            self.kg_embeddings.entity_embeddings = nn.Embedding(
                kg.num_entities,
                self.config.kg_embed_dim
            )
            
            # Copy old embeddings
            with torch.no_grad():
                self.kg_embeddings.entity_embeddings.weight[:old_embeddings.num_embeddings] = \
                    old_embeddings.weight
    
    def forward(
        self,
        neural_features: torch.Tensor,
        entity_indices: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Integrate knowledge graph information.
        
        Args:
            neural_features: (batch, hidden_dim) features from neural encoder
            entity_indices: Optional (batch,) specific entity indices to look up
            
        Returns:
            Dict with:
                - kg_features: Knowledge-enhanced features
                - entity_scores: Similarity to all entities
        """
        # Project neural features to KG space
        entity_query = self.neural_to_entity(neural_features)
        entity_query = F.normalize(entity_query, p=2, dim=-1)
        
        # If specific entities requested, retrieve their embeddings
        if entity_indices is not None:
            entity_embeds = self.kg_embeddings.get_entity_embedding(entity_indices)
        else:
            # Compute similarity to all entity embeddings
            all_entity_embeds = self.kg_embeddings.entity_embeddings.weight
            entity_scores = torch.matmul(entity_query, all_entity_embeds.T)
            
            # Soft retrieval: weighted combination of entities
            entity_weights = F.softmax(entity_scores, dim=-1)
            entity_embeds = torch.matmul(entity_weights, all_entity_embeds)
        
        # Project back to neural space
        kg_features = self.entity_to_neural(entity_embeds)
        
        return {
            "kg_features": kg_features,
            "entity_embeds": entity_embeds,
            "entity_query": entity_query
        }
    
    def verify_hypothesis(
        self,
        head_features: torch.Tensor,
        relation_features: torch.Tensor,
        tail_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Verify if a hypothesis (head, relation, tail) is consistent with KG.
        
        Args:
            head_features: (batch, kg_embed_dim) head entity features
            relation_features: (batch, kg_embed_dim) relation features
            tail_features: (batch, kg_embed_dim) tail entity features
            
        Returns:
            validity_scores: (batch,) probability that hypothesis is valid
        """
        # Concatenate features
        combined = torch.cat([head_features, relation_features, tail_features], dim=-1)
        
        # Predict validity
        validity_scores = self.verification_head(combined).squeeze(-1)
        
        return validity_scores
    
    def compute_kg_loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_relations: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Compute KG embedding loss for training."""
        return self.kg_embeddings.compute_loss(
            pos_heads, pos_relations, pos_tails,
            neg_heads, neg_relations, neg_tails
        )
    
    def get_param_count(self) -> dict:
        """Get parameter counts."""
        return {
            "entity_embeddings": self.kg_embeddings.entity_embeddings.weight.numel(),
            "relation_embeddings": self.kg_embeddings.relation_embeddings.weight.numel(),
            "projections": sum(p.numel() for p in self.neural_to_entity.parameters()) + \
                          sum(p.numel() for p in self.entity_to_neural.parameters()),
            "verification_head": sum(p.numel() for p in self.verification_head.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }


def build_kg_from_data(
    questions: List[Dict],
    min_entity_freq: int = 2
) -> KnowledgeGraph:
    """
    Build a knowledge graph from scientific reasoning data.
    
    Extracts entities and relations from questions and correct answers.
    """
    kg = KnowledgeGraph()
    entity_counts = defaultdict(int)
    
    # First pass: count entities
    for q in questions:
        # Extract key terms (simplified - real implementation would use NER)
        text = q.get("question", "") + " " + q.get("correct_answer", "")
        words = text.lower().split()
        
        # Count potential entities (nouns, typically longer words)
        for word in words:
            if len(word) > 3 and word.isalpha():
                entity_counts[word] += 1
    
    # Filter to frequent entities
    frequent_entities = {e for e, c in entity_counts.items() if c >= min_entity_freq}
    
    # Second pass: extract relations
    relation_patterns = {
        "is_a": ["is a", "is an", "are"],
        "causes": ["causes", "cause", "leads to"],
        "has_property": ["has", "have", "contains"],
        "part_of": ["part of", "component of"],
        "related_to": ["related to", "associated with"]
    }
    
    for q in questions:
        text = (q.get("question", "") + " " + q.get("correct_answer", "")).lower()
        
        # Simple pattern matching for relations
        for relation, patterns in relation_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    # Find entities around the pattern
                    parts = text.split(pattern)
                    if len(parts) >= 2:
                        # Extract nearby entities
                        before_words = parts[0].split()
                        after_words = parts[1].split()
                        
                        head_candidates = [w for w in before_words[-3:] if w in frequent_entities]
                        tail_candidates = [w for w in after_words[:3] if w in frequent_entities]
                        
                        for head in head_candidates:
                            for tail in tail_candidates:
                                if head != tail:
                                    kg.add_triple(head, relation, tail)
    
    return kg
