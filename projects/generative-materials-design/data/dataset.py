"""
Dataset module for QM9 molecular dataset.

QM9 Dataset Properties:
----------------------
- ~134k small organic molecules (up to 9 heavy atoms)
- 12 computed quantum mechanical properties
- Properties include: dipole moment, polarizability, HOMO, LUMO, gap, etc.

References:
----------
- Ramakrishnan et al. (2014): QM9 dataset
- Wu et al. (2018): MoleculeNet benchmark
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# QM9 property names and units
QM9_PROPERTIES = {
    0: ("mu", "D", "Dipole moment"),
    1: ("alpha", "a_0^3", "Isotropic polarizability"),
    2: ("homo", "eV", "HOMO energy"),
    3: ("lumo", "eV", "LUMO energy"),
    4: ("gap", "eV", "HOMO-LUMO gap"),
    5: ("r2", "a_0^2", "Electronic spatial extent"),
    6: ("zpve", "eV", "Zero point vibrational energy"),
    7: ("u0", "eV", "Internal energy at 0K"),
    8: ("u", "eV", "Internal energy at 298.15K"),
    9: ("h", "eV", "Enthalpy at 298.15K"),
    10: ("g", "eV", "Free energy at 298.15K"),
    11: ("cv", "cal/(mol·K)", "Heat capacity at 298.15K"),
}

# Atom types in QM9
ATOM_TYPES = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}
ATOM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


class QM9Dataset(Dataset):
    """QM9 molecular dataset wrapper."""
    
    def __init__(
        self,
        root: str = "./data/qm9",
        split: str = "train",
        transform=None,
        max_atoms: int = 29,
        debug_samples: Optional[int] = None,
    ):
        """
        Initialize QM9 dataset.
        
        Args:
            root: Data directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            max_atoms: Maximum number of atoms (for padding)
            debug_samples: If set, use only this many samples
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.max_atoms = max_atoms
        self.debug_samples = debug_samples
        
        self._load_data()
    
    def _load_data(self):
        """Load QM9 data using torch_geometric if available, otherwise create synthetic."""
        try:
            from torch_geometric.datasets import QM9 as QM9_TG
            from torch_geometric.loader import DataLoader as TGDataLoader
            
            logger.info(f"Loading QM9 dataset from torch_geometric to {self.root}")
            os.makedirs(self.root, exist_ok=True)
            
            # Load full dataset
            full_dataset = QM9_TG(root=self.root)
            
            # Split indices
            n_total = len(full_dataset)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            
            # Set random seed for reproducibility
            torch.manual_seed(42)
            perm = torch.randperm(n_total)
            
            if self.split == "train":
                indices = perm[:n_train].tolist()
            elif self.split == "val":
                indices = perm[n_train:n_train + n_val].tolist()
            else:  # test
                indices = perm[n_train + n_val:].tolist()
            
            # Apply debug subset if specified
            if self.debug_samples is not None:
                indices = indices[:self.debug_samples]
            
            self.data = [full_dataset[i] for i in indices]
            self.use_torch_geometric = True
            
            logger.info(f"Loaded {len(self.data)} molecules for {self.split} split")
            
        except ImportError:
            logger.warning("torch_geometric not available, using synthetic data")
            self._create_synthetic_data()
            self.use_torch_geometric = False
    
    def _create_synthetic_data(self):
        """Create synthetic molecular data for testing."""
        n_samples = {
            "train": 1000,
            "val": 100,
            "test": 100
        }.get(self.split, 100)
        
        if self.debug_samples is not None:
            n_samples = min(n_samples, self.debug_samples)
        
        self.data = []
        for i in range(n_samples):
            # Random molecule size
            n_atoms = np.random.randint(3, 15)
            
            # Random atom types
            atom_types = np.random.randint(0, 5, size=n_atoms)
            
            # Random adjacency (sparse, symmetric)
            adj = np.zeros((n_atoms, n_atoms))
            for j in range(n_atoms - 1):
                # Connect to next atom (ensures connectivity)
                adj[j, j + 1] = adj[j + 1, j] = np.random.randint(1, 4)
            
            # Random properties
            properties = np.random.randn(12).astype(np.float32)
            
            self.data.append({
                'x': torch.tensor(atom_types, dtype=torch.long),
                'adj': torch.tensor(adj, dtype=torch.float32),
                'y': torch.tensor(properties, dtype=torch.float32),
                'num_atoms': n_atoms
            })
        
        logger.info(f"Created {n_samples} synthetic molecules for {self.split}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single molecule."""
        if self.use_torch_geometric:
            data = self.data[idx]
            return self._process_tg_data(data)
        else:
            return self._pad_molecule(self.data[idx])
    
    def _process_tg_data(self, data) -> Dict[str, torch.Tensor]:
        """Process torch_geometric Data object."""
        # Get atom types (atomic numbers: H=1, C=6, N=7, O=8, F=9)
        atomic_nums = data.z.numpy()
        atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        atom_types = np.array([atom_map.get(z, 0) for z in atomic_nums])
        
        # Build adjacency matrix from edge_index
        n_atoms = len(atom_types)
        adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj[src, dst] = 1.0
        
        # Get properties (QM9 has different property indexing)
        # torch_geometric QM9 uses: y[:, 0:12] for properties
        if hasattr(data, 'y') and data.y is not None:
            properties = data.y.squeeze().numpy()
            if len(properties.shape) == 0:
                properties = np.array([properties])
            # Ensure 12 properties
            if len(properties) < 12:
                properties = np.pad(properties, (0, 12 - len(properties)))
        else:
            properties = np.zeros(12, dtype=np.float32)
        
        mol_dict = {
            'x': torch.tensor(atom_types, dtype=torch.long),
            'adj': torch.tensor(adj, dtype=torch.float32),
            'y': torch.tensor(properties, dtype=np.float32),
            'num_atoms': n_atoms
        }
        
        return self._pad_molecule(mol_dict)
    
    def _pad_molecule(self, mol: Dict) -> Dict[str, torch.Tensor]:
        """Pad molecule to max_atoms."""
        n_atoms = mol['num_atoms']
        
        # Pad atom types
        x_padded = torch.zeros(self.max_atoms, dtype=torch.long)
        x_padded[:n_atoms] = mol['x']
        
        # Pad adjacency
        adj_padded = torch.zeros(self.max_atoms, self.max_atoms)
        adj_padded[:n_atoms, :n_atoms] = mol['adj']
        
        # Atom mask
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        mask[:n_atoms] = True
        
        return {
            'x': x_padded,
            'adj': adj_padded,
            'y': mol['y'],
            'mask': mask,
            'num_atoms': torch.tensor(n_atoms, dtype=torch.long)
        }


def get_dataloaders(
    config,
    debug_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    
    if debug_samples is None and config.training.debug_mode:
        debug_samples = config.training.debug_samples
    
    train_dataset = QM9Dataset(
        root=config.data.data_root,
        split="train",
        max_atoms=config.data.max_atoms,
        debug_samples=debug_samples
    )
    
    val_dataset = QM9Dataset(
        root=config.data.data_root,
        split="val",
        max_atoms=config.data.max_atoms,
        debug_samples=debug_samples // 10 if debug_samples else None
    )
    
    test_dataset = QM9Dataset(
        root=config.data.data_root,
        split="test",
        max_atoms=config.data.max_atoms,
        debug_samples=debug_samples // 10 if debug_samples else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_loader, val_loader, test_loader


def get_property_stats(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of properties for normalization."""
    all_props = []
    for batch in loader:
        all_props.append(batch['y'])
    
    all_props = torch.cat(all_props, dim=0)
    mean = all_props.mean(dim=0)
    std = all_props.std(dim=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero
    
    return mean, std


if __name__ == "__main__":
    # Test dataset
    from config import get_config
    
    config = get_config(debug=True)
    train_loader, val_loader, test_loader = get_dataloaders(config, debug_samples=100)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape if hasattr(val, 'shape') else val}")
