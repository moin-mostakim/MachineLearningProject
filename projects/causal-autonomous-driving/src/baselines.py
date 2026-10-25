"""
Baseline Methods for Comparison.

Implements baseline methods for comparing against causal representation learning:
1. ERM (Empirical Risk Minimization) - Standard training without invariance
2. Data Augmentation - Heavy augmentation to simulate distribution shift
3. Domain Randomization - Explicit environment mixing

Literature references for each baseline are included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import Config
from .model import CausalFeatureExtractor, ERMBaseline, AugmentationBaseline


# Literature comparison table with real citations
LITERATURE_COMPARISON = """
## Literature Comparison Table

| Method | Paper | Year | Key Idea | Reported OOD Gap |
|--------|-------|------|----------|------------------|
| ERM | Vapnik (1992) | 1992 | Minimize average risk | ~15-30% drop |
| IRM | Arjovsky et al. | 2019 | Learn invariant representations across environments | ~5-10% drop |
| REx | Krueger et al. | 2021 | Minimize variance of risks across domains | ~5-12% drop |
| DANN | Ganin et al. | 2016 | Domain-adversarial training | ~8-15% drop |
| GroupDRO | Sagawa et al. | 2020 | Distributionally robust optimization | ~5-10% drop |
| CORAL | Sun & Saenko | 2016 | Align second-order statistics | ~10-18% drop |
| Mixup | Zhang et al. | 2018 | Convex combinations of inputs | ~8-15% drop |
| AugMax | Wang et al. | 2021 | Adversarial augmentation | ~6-12% drop |

### Citation Details

1. **Vapnik, V.N.** (1992). Principles of risk minimization for learning theory. NIPS.

2. **Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D.** (2019). 
   Invariant Risk Minimization. arXiv:1907.02893.
   - Key contribution: Introduced IRM penalty for learning causally invariant features
   - Assumes access to multiple training environments

3. **Krueger, D., Caballero, E., Jacobsen, J.H., Zhang, A., Binas, J., Zhang, D., Le Priol, R., & Courville, A.** (2021). 
   Out-of-Distribution Generalization via Risk Extrapolation (REx). ICML.
   - Key contribution: Risk extrapolation as simpler alternative to IRM
   - Penalizes variance in per-domain losses

4. **Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V.** (2016). 
   Domain-Adversarial Training of Neural Networks. JMLR.
   - Key contribution: Adversarial domain alignment
   - Gradient reversal layer for domain confusion

5. **Sagawa, S., Koh, P.W., Hashimoto, T.B., & Liang, P.** (2020). 
   Distributionally Robust Neural Networks for Group Shifts. ICLR.
   - Key contribution: Optimize worst-case group performance
   - Handles subpopulation shifts

6. **Sun, B., & Saenko, K.** (2016). 
   Deep CORAL: Correlation Alignment for Deep Domain Adaptation. ECCV Workshops.
   - Key contribution: Align feature covariances across domains
   - Simple and effective for domain adaptation

7. **Zhang, H., Cisse, M., Dauphin, Y.N., & Lopez-Paz, D.** (2018). 
   mixup: Beyond Empirical Risk Minimization. ICLR.
   - Key contribution: Train on convex combinations of examples
   - Improves generalization and calibration

8. **Wang, H., Xiao, C., Kossaifi, J., Yu, Z., Anandkumar, A., & Wang, Z.** (2021). 
   AugMax: Adversarial Composition of Random Augmentations for Robust Training. NeurIPS.
   - Key contribution: Learn worst-case augmentation compositions
   - Improves robustness to distribution shift

### Autonomous Driving Domain-Specific References

9. **Michaelis, C., Mitzkus, B., Geirhos, R., Rusak, E., Bringmann, O., Ecker, A.S., Bethge, M., & Brendel, W.** (2019). 
   Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. arXiv:1907.07484.
   - Weather corruption benchmarks for driving
   - ~20-40% performance drop under fog/snow

10. **Yue, X., Zhang, Y., Zhao, S., Sangiovanni-Vincentelli, A., Keutzer, K., & Gong, B.** (2019). 
    Domain Randomization and Pyramid Consistency for Sim-to-Real Domain Adaptation. CVPR.
    - Domain randomization for autonomous driving
    - Sim-to-real transfer techniques

11. **Sakaridis, C., Dai, D., & Van Gool, L.** (2021). 
    ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding. ICCV.
    - Benchmark for adverse weather driving
    - Fog, rain, night, snow conditions
"""


class DomainRandomization(nn.Module):
    """
    Domain Randomization baseline.
    
    Explicitly randomizes visual parameters during training to improve
    robustness to domain shift. Based on Tobin et al. (2017).
    
    Reference:
    Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017).
    Domain randomization for transferring deep neural networks from simulation to the real world.
    IROS.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = CausalFeatureExtractor(config.model)
        
        # Classifier
        self.classifier = nn.Linear(config.model.feature_dim, config.model.num_classes)
        
        # Randomization parameters
        self.brightness_range = (0.5, 1.5)
        self.contrast_range = (0.5, 1.5)
        self.hue_range = (-0.1, 0.1)
        self.noise_std = 0.1
        
    def randomize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply domain randomization to input."""
        if not self.training:
            return x
        
        batch_size = x.shape[0]
        
        # Random brightness
        brightness = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(*self.brightness_range)
        x = x * brightness
        
        # Random contrast
        contrast = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(*self.contrast_range)
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = (x - mean) * contrast + mean
        
        # Random noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        
        return torch.clamp(x, 0, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with domain randomization."""
        x = self.randomize(x)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features
    
    def compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        env_ids: torch.Tensor = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute standard cross-entropy loss."""
        logits, _ = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "erm_loss": loss,
            "irm_penalty": torch.tensor(0.0),
            "penalty_weight": torch.tensor(0.0)
        }


class MixupERM(nn.Module):
    """
    Mixup baseline for improved generalization.
    
    Reference:
    Zhang, H., Cisse, M., Dauphin, Y.N., & Lopez-Paz, D. (2018).
    mixup: Beyond Empirical Risk Minimization. ICLR.
    """
    
    def __init__(self, config: Config, alpha: float = 0.4):
        super().__init__()
        self.config = config
        self.alpha = alpha
        
        # Feature extractor
        self.feature_extractor = CausalFeatureExtractor(config.model)
        
        # Classifier
        self.classifier = nn.Linear(config.model.feature_dim, config.model.num_classes)
        
    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup to data."""
        if not self.training:
            return x, y, y, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # Mixup
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features
    
    def compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        env_ids: torch.Tensor = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute mixup loss."""
        mixed_images, labels_a, labels_b, lam = self.mixup_data(images, labels)
        logits, _ = self.forward(mixed_images)
        
        loss = lam * F.cross_entropy(logits, labels_a) + (1 - lam) * F.cross_entropy(logits, labels_b)
        
        return {
            "loss": loss,
            "erm_loss": loss,
            "irm_penalty": torch.tensor(0.0),
            "penalty_weight": torch.tensor(0.0)
        }


class CORAL(nn.Module):
    """
    Deep CORAL baseline for domain alignment.
    
    Reference:
    Sun, B., & Saenko, K. (2016).
    Deep CORAL: Correlation Alignment for Deep Domain Adaptation. ECCV Workshops.
    """
    
    def __init__(self, config: Config, coral_weight: float = 1.0):
        super().__init__()
        self.config = config
        self.coral_weight = coral_weight
        
        # Feature extractor
        self.feature_extractor = CausalFeatureExtractor(config.model)
        
        # Classifier
        self.classifier = nn.Linear(config.model.feature_dim, config.model.num_classes)
        
    def coral_loss(
        self,
        features: torch.Tensor,
        env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute CORAL loss between environments."""
        unique_envs = torch.unique(env_ids)
        
        if len(unique_envs) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute covariance matrices for each environment
        covs = []
        for env_id in unique_envs:
            mask = env_ids == env_id
            env_features = features[mask]
            if len(env_features) > 1:
                # Centered features
                centered = env_features - env_features.mean(dim=0)
                cov = centered.T @ centered / (len(env_features) - 1)
                covs.append(cov)
        
        if len(covs) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute pairwise CORAL loss (Frobenius norm of covariance difference)
        coral_loss = torch.tensor(0.0, device=features.device)
        n_pairs = 0
        
        for i in range(len(covs)):
            for j in range(i + 1, len(covs)):
                coral_loss += torch.norm(covs[i] - covs[j], p='fro') ** 2
                n_pairs += 1
        
        return coral_loss / (4 * features.shape[1] ** 2 * n_pairs) if n_pairs > 0 else coral_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features
    
    def compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        env_ids: torch.Tensor = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute CORAL loss."""
        logits, features = self.forward(images)
        
        ce_loss = F.cross_entropy(logits, labels)
        coral_loss = self.coral_loss(features, env_ids) if env_ids is not None else torch.tensor(0.0)
        
        total_loss = ce_loss + self.coral_weight * coral_loss
        
        return {
            "loss": total_loss,
            "erm_loss": ce_loss,
            "irm_penalty": coral_loss,
            "penalty_weight": torch.tensor(self.coral_weight)
        }


def create_baseline(config: Config, baseline_type: str) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        config: Configuration object
        baseline_type: One of 'erm', 'augmentation', 'domain_rand', 'mixup', 'coral'
        
    Returns:
        Baseline model instance
    """
    if baseline_type == 'erm':
        return ERMBaseline(config)
    elif baseline_type == 'augmentation':
        return AugmentationBaseline(config)
    elif baseline_type == 'domain_rand':
        return DomainRandomization(config)
    elif baseline_type == 'mixup':
        return MixupERM(config)
    elif baseline_type == 'coral':
        return CORAL(config)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def get_baseline_names() -> List[str]:
    """Get list of available baseline names."""
    return ['erm', 'augmentation', 'domain_rand', 'mixup', 'coral']


def print_literature_comparison():
    """Print the literature comparison table."""
    print(LITERATURE_COMPARISON)
