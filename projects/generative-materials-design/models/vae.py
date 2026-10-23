"""
Variational Autoencoder (VAE) for Molecular Generation.

Architecture inspired by:
- Junction Tree VAE (Jin et al., 2018): Hierarchical generation
- GraphVAE (Simonovsky & Komodakis, 2018): Graph generation
- MolGAN (De Cao & Kipf, 2018): Adversarial molecular generation

This implementation uses a GNN-based encoder and MLP-based decoder
for simplicity and efficiency, with property prediction head for
multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class GraphConvLayer(nn.Module):
    """Simple Graph Convolutional Layer."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
        
        Returns:
            Updated node features [batch, num_nodes, out_dim]
        """
        # Normalize adjacency (add self-loops and degree normalization)
        adj = adj + torch.eye(adj.size(-1), device=adj.device)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        
        # Message passing
        h = torch.bmm(adj_norm, x)
        h = self.linear(h)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h


class MolecularEncoder(nn.Module):
    """GNN-based molecular encoder."""
    
    def __init__(
        self,
        num_atom_types: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode molecular graph to latent distribution.
        
        Args:
            x: Atom types [batch, num_atoms]
            adj: Adjacency matrix [batch, num_atoms, num_atoms]
            mask: Atom mask [batch, num_atoms]
        
        Returns:
            mu, logvar: Latent distribution parameters [batch, latent_dim]
        """
        # Embed atoms
        h = self.atom_embedding(x)  # [batch, num_atoms, hidden_dim]
        
        # GNN message passing
        for gnn in self.gnn_layers:
            h = gnn(h, adj)
        
        # Global pooling (masked mean)
        mask_expanded = mask.unsqueeze(-1).float()
        h_sum = (h * mask_expanded).sum(dim=1)
        h_count = mask_expanded.sum(dim=1).clamp(min=1)
        h_global = h_sum / h_count
        
        # Project to latent space
        mu = self.fc_mu(h_global)
        logvar = self.fc_logvar(h_global)
        
        return mu, logvar


class MolecularDecoder(nn.Module):
    """MLP-based molecular decoder."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        max_atoms: int,
        num_atom_types: int,
        num_bond_types: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        
        # Build MLP
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        self.atom_head = nn.Linear(hidden_dim, max_atoms * num_atom_types)
        self.bond_head = nn.Linear(hidden_dim, max_atoms * max_atoms * num_bond_types)
        self.size_head = nn.Linear(hidden_dim, max_atoms)  # Predict molecule size
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent vector to molecular graph.
        
        Args:
            z: Latent vector [batch, latent_dim]
        
        Returns:
            Dictionary with:
                - atom_logits: [batch, max_atoms, num_atom_types]
                - bond_logits: [batch, max_atoms, max_atoms, num_bond_types]
                - size_logits: [batch, max_atoms] (for predicting molecule size)
        """
        h = self.mlp(z)
        
        # Atom type predictions
        atom_logits = self.atom_head(h)
        atom_logits = atom_logits.view(-1, self.max_atoms, self.num_atom_types)
        
        # Bond predictions
        bond_logits = self.bond_head(h)
        bond_logits = bond_logits.view(-1, self.max_atoms, self.max_atoms, self.num_bond_types)
        
        # Make bond matrix symmetric
        bond_logits = (bond_logits + bond_logits.transpose(1, 2)) / 2
        
        # Size prediction
        size_logits = self.size_head(h)
        
        return {
            'atom_logits': atom_logits,
            'bond_logits': bond_logits,
            'size_logits': size_logits
        }


class PropertyPredictor(nn.Module):
    """Property prediction head."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_properties: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, num_properties))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict properties from latent vector."""
        return self.mlp(z)


class MolecularVAE(nn.Module):
    """
    Variational Autoencoder for Molecular Generation.
    
    Combines:
    - GNN encoder for molecular graph encoding
    - MLP decoder for molecular graph generation
    - Property predictor for multi-task learning
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        mc = config.model
        dc = config.data
        
        self.encoder = MolecularEncoder(
            num_atom_types=dc.num_atom_types,
            hidden_dim=mc.hidden_dim,
            latent_dim=mc.latent_dim,
            num_layers=mc.num_encoder_layers,
            dropout=mc.dropout
        )
        
        self.decoder = MolecularDecoder(
            latent_dim=mc.latent_dim,
            hidden_dim=mc.hidden_dim,
            max_atoms=dc.max_atoms,
            num_atom_types=dc.num_atom_types,
            num_bond_types=dc.num_bond_types,
            num_layers=mc.num_decoder_layers,
            dropout=mc.dropout
        )
        
        self.property_predictor = PropertyPredictor(
            latent_dim=mc.latent_dim,
            hidden_dim=mc.hidden_dim,
            num_properties=mc.num_properties,
            num_layers=mc.num_property_layers,
            dropout=mc.dropout
        )
        
        self.latent_dim = mc.latent_dim
    
    def encode(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode molecule to latent distribution."""
        return self.encoder(x, adj, mask)
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent vector to molecule."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            batch: Dictionary with x, adj, mask, y
        
        Returns:
            Dictionary with all outputs needed for loss computation
        """
        x = batch['x']
        adj = batch['adj']
        mask = batch['mask']
        
        # Encode
        mu, logvar = self.encode(x, adj, mask)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z)
        
        # Predict properties
        pred_props = self.property_predictor(z)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'atom_logits': recon['atom_logits'],
            'bond_logits': recon['bond_logits'],
            'size_logits': recon['size_logits'],
            'pred_properties': pred_props
        }
    
    def sample(self, num_samples: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Sample molecules from prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        recon = self.decode(z)
        props = self.property_predictor(z)
        
        # Convert logits to predictions
        atom_types = recon['atom_logits'].argmax(dim=-1)
        bond_types = recon['bond_logits'].argmax(dim=-1)
        mol_sizes = F.softmax(recon['size_logits'], dim=-1).argmax(dim=-1) + 1
        
        return {
            'z': z,
            'atom_types': atom_types,
            'bond_types': bond_types,
            'mol_sizes': mol_sizes,
            'properties': props
        }
    
    def generate_with_properties(
        self, 
        target_props: torch.Tensor,
        num_samples: int = 100,
        optimization_steps: int = 100,
        lr: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Generate molecules with target properties (inverse design).
        
        Args:
            target_props: Target property values [num_objectives]
            num_samples: Number of molecules to generate
            optimization_steps: Latent space optimization steps
            lr: Learning rate for optimization
        
        Returns:
            Generated molecules optimized for target properties
        """
        device = next(self.parameters()).device
        
        # Initialize latent vectors
        z = torch.randn(num_samples, self.latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        target_props = target_props.to(device)
        
        for _ in range(optimization_steps):
            optimizer.zero_grad()
            pred_props = self.property_predictor(z)
            
            # MSE loss to target properties
            loss = F.mse_loss(pred_props[:, :len(target_props)], 
                            target_props.unsqueeze(0).expand(num_samples, -1))
            
            loss.backward()
            optimizer.step()
        
        # Generate final molecules
        with torch.no_grad():
            recon = self.decode(z)
            final_props = self.property_predictor(z)
        
        return {
            'z': z.detach(),
            'atom_logits': recon['atom_logits'],
            'bond_logits': recon['bond_logits'],
            'properties': final_props
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        predictor_params = sum(p.numel() for p in self.property_predictor.parameters())
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'property_predictor': predictor_params,
            'total': encoder_params + decoder_params + predictor_params
        }


class SingleObjectiveVAE(MolecularVAE):
    """
    Single-objective VAE baseline for comparison.
    
    Optimizes for a single property instead of multiple objectives.
    """
    
    def __init__(self, config, target_property_idx: int = 0):
        super().__init__(config)
        self.target_property_idx = target_property_idx
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward with single property focus."""
        outputs = super().forward(batch)
        
        # Keep only target property prediction
        outputs['pred_single_property'] = outputs['pred_properties'][:, self.target_property_idx]
        
        return outputs


def build_model(config) -> MolecularVAE:
    """Build VAE model from config."""
    return MolecularVAE(config)


def build_single_objective_model(config, target_idx: int = 0) -> SingleObjectiveVAE:
    """Build single-objective VAE baseline."""
    return SingleObjectiveVAE(config, target_idx)


if __name__ == "__main__":
    # Test model
    import sys
    sys.path.append("..")
    from config import get_config
    
    config = get_config(debug=True)
    model = build_model(config)
    
    # Print parameter counts
    params = model.count_parameters()
    print("\nParameter Counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch = {
        'x': torch.randint(0, 5, (4, 29)),
        'adj': torch.rand(4, 29, 29),
        'mask': torch.ones(4, 29, dtype=torch.bool),
        'y': torch.randn(4, 12)
    }
    
    outputs = model(batch)
    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tensor.shape}")
    
    # Test sampling
    samples = model.sample(10)
    print("\nSample shapes:")
    for name, tensor in samples.items():
        print(f"  {name}: {tensor.shape}")
