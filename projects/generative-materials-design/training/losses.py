"""
Loss functions for Molecular VAE training.

References:
-----------
- Beta-VAE (Higgins et al., 2017): KL weighting for disentanglement
- Evidence Lower Bound (ELBO): Standard VAE objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VAELoss(nn.Module):
    """Combined VAE loss with reconstruction, KL divergence, and property prediction."""
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        property_weight: float = 0.5,
        num_atom_types: int = 5,
        num_bond_types: int = 4
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.property_weight = property_weight
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
    
    def reconstruction_loss(
        self,
        atom_logits: torch.Tensor,
        bond_logits: torch.Tensor,
        target_atoms: torch.Tensor,
        target_adj: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss for atoms and bonds.
        
        Args:
            atom_logits: Predicted atom type logits [batch, max_atoms, num_atom_types]
            bond_logits: Predicted bond type logits [batch, max_atoms, max_atoms, num_bond_types]
            target_atoms: True atom types [batch, max_atoms]
            target_adj: True adjacency matrix [batch, max_atoms, max_atoms]
            mask: Valid atom mask [batch, max_atoms]
        
        Returns:
            atom_loss, bond_loss
        """
        batch_size = atom_logits.size(0)
        
        # Atom reconstruction loss (cross-entropy)
        atom_logits_flat = atom_logits.view(-1, self.num_atom_types)
        target_atoms_flat = target_atoms.view(-1)
        mask_flat = mask.view(-1).float()
        
        atom_loss = F.cross_entropy(atom_logits_flat, target_atoms_flat, reduction='none')
        atom_loss = (atom_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        
        # Bond reconstruction loss
        # Convert continuous adjacency to discrete bond types
        target_bonds = (target_adj > 0).long()  # Binary for now
        
        # Create bond mask (only between valid atoms)
        bond_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        bond_mask = bond_mask.float()
        
        # Simple binary cross-entropy for bond existence
        bond_probs = bond_logits[:, :, :, 0]  # Use first channel as "no bond" vs "bond"
        bond_loss = F.binary_cross_entropy_with_logits(
            bond_probs, 
            target_bonds.float(),
            reduction='none'
        )
        bond_loss = (bond_loss * bond_mask).sum() / bond_mask.sum().clamp(min=1)
        
        return atom_loss, bond_loss
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from standard normal prior.
        
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    def property_loss(
        self,
        pred_props: torch.Tensor,
        target_props: torch.Tensor
    ) -> torch.Tensor:
        """Compute property prediction MSE loss."""
        return F.mse_loss(pred_props, target_props)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined VAE loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Reconstruction losses
        atom_loss, bond_loss = self.reconstruction_loss(
            outputs['atom_logits'],
            outputs['bond_logits'],
            batch['x'],
            batch['adj'],
            batch['mask']
        )
        recon_loss = atom_loss + bond_loss
        
        # KL divergence
        kl_loss = self.kl_divergence(outputs['mu'], outputs['logvar'])
        
        # Property prediction loss
        prop_loss = self.property_loss(outputs['pred_properties'], batch['y'])
        
        # Combined loss (ELBO + property prediction)
        total_loss = (
            self.recon_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.property_weight * prop_loss
        )
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'atom_loss': atom_loss,
            'bond_loss': bond_loss,
            'kl_loss': kl_loss,
            'property_loss': prop_loss
        }


class SingleObjectiveLoss(VAELoss):
    """Loss for single-objective VAE baseline."""
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        property_weight: float = 0.5,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        target_property_idx: int = 0
    ):
        super().__init__(recon_weight, kl_weight, property_weight, 
                        num_atom_types, num_bond_types)
        self.target_property_idx = target_property_idx
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with single property focus."""
        losses = super().forward(outputs, batch)
        
        # Replace property loss with single-objective loss
        single_prop_loss = F.mse_loss(
            outputs['pred_properties'][:, self.target_property_idx],
            batch['y'][:, self.target_property_idx]
        )
        
        losses['single_property_loss'] = single_prop_loss
        
        # Recompute total with single property
        losses['loss'] = (
            self.recon_weight * losses['recon_loss'] +
            self.kl_weight * losses['kl_loss'] +
            self.property_weight * single_prop_loss
        )
        
        return losses


class MultiObjectiveLoss(VAELoss):
    """
    Multi-objective loss with Pareto-aware weighting.
    
    Implements scalarization methods for multi-objective optimization:
    - Weighted sum
    - Chebyshev scalarization
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        property_weight: float = 0.5,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        target_property_indices: list = [4, 5, 6],  # HOMO, LUMO, Gap
        property_weights: list = None,
        scalarization: str = "weighted_sum"
    ):
        super().__init__(recon_weight, kl_weight, property_weight,
                        num_atom_types, num_bond_types)
        self.target_property_indices = target_property_indices
        self.property_weights = property_weights or [1.0] * len(target_property_indices)
        self.scalarization = scalarization
    
    def multi_objective_property_loss(
        self,
        pred_props: torch.Tensor,
        target_props: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-objective property loss."""
        losses = {}
        
        # Individual property losses
        for i, prop_idx in enumerate(self.target_property_indices):
            prop_loss = F.mse_loss(pred_props[:, prop_idx], target_props[:, prop_idx])
            losses[f'property_{prop_idx}_loss'] = prop_loss
        
        # Scalarization
        if self.scalarization == "weighted_sum":
            total = sum(
                self.property_weights[i] * losses[f'property_{prop_idx}_loss']
                for i, prop_idx in enumerate(self.target_property_indices)
            )
        elif self.scalarization == "chebyshev":
            # Min-max scalarization
            scaled_losses = [
                self.property_weights[i] * losses[f'property_{prop_idx}_loss']
                for i, prop_idx in enumerate(self.target_property_indices)
            ]
            total = max(scaled_losses)
        else:
            total = sum(losses.values()) / len(losses)
        
        return total, losses
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss."""
        # Base losses
        atom_loss, bond_loss = self.reconstruction_loss(
            outputs['atom_logits'],
            outputs['bond_logits'],
            batch['x'],
            batch['adj'],
            batch['mask']
        )
        recon_loss = atom_loss + bond_loss
        kl_loss = self.kl_divergence(outputs['mu'], outputs['logvar'])
        
        # Multi-objective property loss
        mo_prop_loss, prop_losses = self.multi_objective_property_loss(
            outputs['pred_properties'],
            batch['y']
        )
        
        # Combined loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.property_weight * mo_prop_loss
        )
        
        result = {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'atom_loss': atom_loss,
            'bond_loss': bond_loss,
            'kl_loss': kl_loss,
            'mo_property_loss': mo_prop_loss,
            **prop_losses
        }
        
        return result


def build_loss(config, loss_type: str = "multi_objective") -> nn.Module:
    """Build loss function from config."""
    if loss_type == "single_objective":
        return SingleObjectiveLoss(
            recon_weight=config.training.recon_weight,
            kl_weight=config.training.kl_weight,
            property_weight=config.training.property_weight,
            num_atom_types=config.data.num_atom_types,
            num_bond_types=config.data.num_bond_types,
            target_property_idx=config.optimization.target_properties[0]
        )
    elif loss_type == "multi_objective":
        return MultiObjectiveLoss(
            recon_weight=config.training.recon_weight,
            kl_weight=config.training.kl_weight,
            property_weight=config.training.property_weight,
            num_atom_types=config.data.num_atom_types,
            num_bond_types=config.data.num_bond_types,
            target_property_indices=config.optimization.target_properties
        )
    else:
        return VAELoss(
            recon_weight=config.training.recon_weight,
            kl_weight=config.training.kl_weight,
            property_weight=config.training.property_weight,
            num_atom_types=config.data.num_atom_types,
            num_bond_types=config.data.num_bond_types
        )
