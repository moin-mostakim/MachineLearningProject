"""
Evaluation metrics for molecular generation.

Metrics:
--------
1. Validity: Fraction of chemically valid molecules
2. Uniqueness: Fraction of unique valid molecules
3. Novelty: Fraction of valid molecules not in training set
4. Property MAE: Mean absolute error of property predictions

References:
----------
- GuacaMol (Brown et al., 2019): Molecular generation benchmarks
- MOSES (Polykovskiy et al., 2020): Molecular generation metrics
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularMetrics:
    """Compute molecular generation metrics."""
    
    def __init__(
        self,
        training_smiles: Optional[Set[str]] = None,
        num_atom_types: int = 5,
        max_atoms: int = 29
    ):
        """
        Initialize metrics calculator.
        
        Args:
            training_smiles: Set of SMILES from training data (for novelty)
            num_atom_types: Number of atom types
            max_atoms: Maximum atoms per molecule
        """
        self.training_smiles = training_smiles or set()
        self.num_atom_types = num_atom_types
        self.max_atoms = max_atoms
        
        # Try to import RDKit for validity checking
        try:
            from rdkit import Chem
            self.rdkit_available = True
        except ImportError:
            logger.warning("RDKit not available. Using heuristic validity checks.")
            self.rdkit_available = False
    
    def check_validity_heuristic(
        self,
        atom_types: torch.Tensor,
        adjacency: torch.Tensor,
        mol_size: int
    ) -> bool:
        """
        Heuristic validity check without RDKit.
        
        Checks:
        - Molecule has at least 1 atom
        - Adjacency is symmetric
        - Valence constraints (approximate)
        """
        if mol_size < 1:
            return False
        
        # Check symmetry
        adj = adjacency[:mol_size, :mol_size]
        if not torch.allclose(adj, adj.t(), atol=1e-6):
            return False
        
        # Check connectivity (at least weakly connected)
        degree = adj.sum(dim=1)
        if (degree < 0.5).any():  # Isolated atoms
            return False
        
        # Check valence (approximate)
        # H: 1, C: 4, N: 3, O: 2, F: 1
        max_valence = torch.tensor([1, 4, 3, 2, 1], dtype=torch.float)
        atoms = atom_types[:mol_size]
        valences = max_valence[atoms]
        
        actual_valence = degree
        if (actual_valence > valences + 0.5).any():  # Allow some tolerance
            return False
        
        return True
    
    def check_validity_rdkit(
        self,
        atom_types: torch.Tensor,
        adjacency: torch.Tensor,
        mol_size: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check validity using RDKit.
        
        Returns:
            (is_valid, smiles or None)
        """
        from rdkit import Chem
        
        # Convert to molecule
        atom_map = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}  # H, C, N, O, F
        
        try:
            mol = Chem.RWMol()
            
            # Add atoms
            atoms = atom_types[:mol_size].tolist()
            for atom_type in atoms:
                atomic_num = atom_map.get(atom_type, 6)
                mol.AddAtom(Chem.Atom(atomic_num))
            
            # Add bonds
            adj = adjacency[:mol_size, :mol_size]
            for i in range(mol_size):
                for j in range(i + 1, mol_size):
                    if adj[i, j] > 0.5:
                        bond_order = int(round(adj[i, j].item()))
                        bond_order = min(max(bond_order, 1), 3)
                        
                        if bond_order == 1:
                            bond_type = Chem.BondType.SINGLE
                        elif bond_order == 2:
                            bond_type = Chem.BondType.DOUBLE
                        else:
                            bond_type = Chem.BondType.TRIPLE
                        
                        mol.AddBond(i, j, bond_type)
            
            # Sanitize
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            
            return True, smiles
            
        except Exception:
            return False, None
    
    def compute_validity(
        self,
        atom_types: torch.Tensor,
        adjacency: torch.Tensor,
        mol_sizes: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute validity metrics for a batch of molecules.
        
        Args:
            atom_types: [batch, max_atoms]
            adjacency: [batch, max_atoms, max_atoms]
            mol_sizes: [batch]
        
        Returns:
            Dictionary with validity metrics
        """
        batch_size = atom_types.size(0)
        valid_count = 0
        valid_smiles = []
        
        for i in range(batch_size):
            size = mol_sizes[i].item()
            
            if self.rdkit_available:
                is_valid, smiles = self.check_validity_rdkit(
                    atom_types[i], adjacency[i], size
                )
                if is_valid and smiles:
                    valid_smiles.append(smiles)
            else:
                is_valid = self.check_validity_heuristic(
                    atom_types[i], adjacency[i], size
                )
                if is_valid:
                    valid_smiles.append(f"mol_{i}")  # Placeholder
            
            if is_valid:
                valid_count += 1
        
        validity = valid_count / batch_size if batch_size > 0 else 0.0
        
        return {
            'validity': validity,
            'valid_count': valid_count,
            'total_count': batch_size,
            'valid_smiles': valid_smiles
        }
    
    def compute_uniqueness(self, smiles_list: List[str]) -> float:
        """Compute uniqueness (fraction of unique molecules)."""
        if not smiles_list:
            return 0.0
        
        unique = set(smiles_list)
        return len(unique) / len(smiles_list)
    
    def compute_novelty(self, smiles_list: List[str]) -> float:
        """Compute novelty (fraction not in training set)."""
        if not smiles_list or not self.training_smiles:
            return 1.0  # If no training set, assume all novel
        
        novel = [s for s in smiles_list if s not in self.training_smiles]
        return len(novel) / len(smiles_list)
    
    def compute_diversity(self, smiles_list: List[str]) -> float:
        """
        Compute internal diversity using Tanimoto similarity.
        """
        if not self.rdkit_available or len(smiles_list) < 2:
            # Use hash-based diversity estimate
            if len(smiles_list) < 2:
                return 1.0
            counter = Counter(smiles_list)
            entropy = -sum(
                (c / len(smiles_list)) * np.log(c / len(smiles_list) + 1e-10)
                for c in counter.values()
            )
            return entropy / np.log(len(smiles_list))
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        # Compute fingerprints
        fps = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fps.append(fp)
            except Exception:
                continue
        
        if len(fps) < 2:
            return 1.0
        
        # Compute pairwise Tanimoto
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        return 1.0 - np.mean(similarities)
    
    def compute_property_mae(
        self,
        pred_properties: torch.Tensor,
        true_properties: torch.Tensor,
        property_indices: List[int] = None
    ) -> Dict[str, float]:
        """
        Compute MAE for property predictions.
        
        Args:
            pred_properties: Predicted properties [batch, num_props]
            true_properties: True properties [batch, num_props]
            property_indices: Indices of properties to evaluate
        
        Returns:
            MAE for each property and overall
        """
        if property_indices is None:
            property_indices = list(range(pred_properties.size(1)))
        
        results = {}
        maes = []
        
        for idx in property_indices:
            mae = torch.abs(pred_properties[:, idx] - true_properties[:, idx]).mean().item()
            results[f'mae_property_{idx}'] = mae
            maes.append(mae)
        
        results['mae_overall'] = np.mean(maes)
        
        return results
    
    def evaluate_generation(
        self,
        model,
        num_samples: int = 1000,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Full evaluation of generation quality.
        
        Args:
            model: VAE model
            num_samples: Number of samples to generate
            device: Device to use
        
        Returns:
            Dictionary with all metrics
        """
        model.eval()
        model.to(device)
        
        # Generate samples
        with torch.no_grad():
            samples = model.sample(num_samples, device)
        
        # Compute validity
        validity_results = self.compute_validity(
            samples['atom_types'],
            samples['bond_types'].float(),
            samples['mol_sizes']
        )
        
        # Compute uniqueness and novelty
        valid_smiles = validity_results['valid_smiles']
        uniqueness = self.compute_uniqueness(valid_smiles)
        novelty = self.compute_novelty(valid_smiles)
        diversity = self.compute_diversity(valid_smiles)
        
        return {
            'validity': validity_results['validity'],
            'uniqueness': uniqueness,
            'novelty': novelty,
            'diversity': diversity,
            'valid_count': validity_results['valid_count'],
            'total_samples': num_samples
        }


class PropertyMetrics:
    """Metrics for property prediction evaluation."""
    
    @staticmethod
    def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return torch.abs(pred - target).mean().item()
    
    @staticmethod
    def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Root Mean Squared Error."""
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    @staticmethod
    def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
        """R-squared coefficient."""
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2)
        return 1 - (ss_res / ss_tot).item()
    
    @staticmethod
    def evaluate_predictions(
        pred: torch.Tensor,
        target: torch.Tensor,
        property_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate property predictions.
        
        Returns metrics for each property.
        """
        num_props = pred.size(1) if len(pred.shape) > 1 else 1
        
        if property_names is None:
            property_names = [f'property_{i}' for i in range(num_props)]
        
        results = {}
        
        for i, name in enumerate(property_names):
            if len(pred.shape) > 1:
                p, t = pred[:, i], target[:, i]
            else:
                p, t = pred, target
            
            results[name] = {
                'mae': PropertyMetrics.compute_mae(p, t),
                'rmse': PropertyMetrics.compute_rmse(p, t),
                'r2': PropertyMetrics.compute_r2(p, t)
            }
        
        # Overall metrics
        if len(pred.shape) > 1:
            results['overall'] = {
                'mae': PropertyMetrics.compute_mae(pred, target),
                'rmse': PropertyMetrics.compute_rmse(pred, target),
                'r2': PropertyMetrics.compute_r2(pred.flatten(), target.flatten())
            }
        
        return results


def evaluate_model(
    model,
    test_loader,
    config,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    Complete model evaluation.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    model.to(device)
    
    # Generation metrics
    mol_metrics = MolecularMetrics(
        num_atom_types=config.data.num_atom_types,
        max_atoms=config.data.max_atoms
    )
    
    gen_metrics = mol_metrics.evaluate_generation(
        model,
        num_samples=config.evaluation.num_generation_samples,
        device=device
    )
    
    # Property prediction metrics
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch)
            all_preds.append(outputs['pred_properties'].cpu())
            all_targets.append(batch['y'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    prop_metrics = PropertyMetrics.evaluate_predictions(
        all_preds,
        all_targets,
        property_names=[f'property_{i}' for i in range(all_preds.size(1))]
    )
    
    return {
        'generation': gen_metrics,
        'property_prediction': prop_metrics
    }


if __name__ == "__main__":
    # Test metrics
    import sys
    sys.path.append("..")
    from config import get_config
    from models.vae import build_model
    
    config = get_config(debug=True)
    model = build_model(config)
    
    # Test generation metrics
    metrics = MolecularMetrics()
    
    # Fake data
    atom_types = torch.randint(0, 5, (10, 29))
    adjacency = torch.zeros(10, 29, 29)
    for i in range(10):
        n = np.random.randint(3, 10)
        for j in range(n - 1):
            adjacency[i, j, j + 1] = 1
            adjacency[i, j + 1, j] = 1
    mol_sizes = torch.randint(3, 10, (10,))
    
    validity_results = metrics.compute_validity(atom_types, adjacency, mol_sizes)
    print(f"\nValidity results: {validity_results}")
