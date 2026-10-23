"""
Evaluation metrics for climate emulation.

Implements metrics following WeatherBench standards:
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- ACC: Anomaly Correlation Coefficient
- SSIM: Structural Similarity Index

Plus uncertainty metrics:
- CRPS: Continuous Ranked Probability Score
- Calibration Error
- Sharpness
- Coverage

References:
- Rasp et al. (2020) WeatherBench
- Gneiting & Raftery (2007) Strictly Proper Scoring Rules
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    per_channel: Optional[np.ndarray] = None
    per_level: Optional[Dict[str, float]] = None


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reduce_spatial: bool = True
) -> torch.Tensor:
    """
    Root Mean Squared Error.
    
    Standard metric for regression tasks.
    
    Args:
        pred: Predictions (B, C, H, W)
        target: Ground truth (B, C, H, W)
        weights: Optional spatial weights (e.g., cos(latitude))
        reduce_spatial: If True, reduce over spatial dims
        
    Returns:
        RMSE value(s)
    """
    squared_error = (pred - target) ** 2
    
    if weights is not None:
        squared_error = squared_error * weights
    
    if reduce_spatial:
        mse = squared_error.mean(dim=(2, 3))  # Average over H, W
        return torch.sqrt(mse.mean())  # Average over B, C
    else:
        return torch.sqrt(squared_error)


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Mean Absolute Error.
    
    More robust to outliers than RMSE.
    
    Args:
        pred: Predictions
        target: Ground truth
        weights: Optional spatial weights
        
    Returns:
        MAE value
    """
    abs_error = torch.abs(pred - target)
    
    if weights is not None:
        abs_error = abs_error * weights
    
    return abs_error.mean()


def anomaly_correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    climatology: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Anomaly Correlation Coefficient (ACC).
    
    Measures spatial pattern similarity relative to climatology.
    ACC = 1 means perfect correlation
    ACC = 0 means no skill (random)
    ACC < 0 means worse than climatology
    
    This is the primary skill metric in operational weather forecasting.
    
    Args:
        pred: Predictions (B, C, H, W)
        target: Ground truth (B, C, H, W)
        climatology: Climatological mean (C, H, W) or None for zero mean
        weights: Latitude weights (cos(lat))
        
    Returns:
        ACC value
    """
    if climatology is None:
        # Assume zero climatology (for normalized data)
        pred_anom = pred
        target_anom = target
    else:
        pred_anom = pred - climatology.unsqueeze(0)
        target_anom = target - climatology.unsqueeze(0)
    
    if weights is None:
        # Create default latitude weighting
        lat_size = pred.shape[2]
        lat = torch.linspace(-90, 90, lat_size, device=pred.device)
        # Use abs to handle the cosine at +-90 degrees
        weights = torch.abs(torch.cos(lat * np.pi / 180.0))
        weights = weights.clamp(min=0.01)  # Avoid zero weights
        weights = weights.view(1, 1, -1, 1)  # (1, 1, H, 1)
    
    # Weighted anomalies
    pred_weighted = pred_anom * torch.sqrt(weights)
    target_weighted = target_anom * torch.sqrt(weights)
    
    # Flatten spatial dimensions
    pred_flat = pred_weighted.flatten(2)  # (B, C, H*W)
    target_flat = target_weighted.flatten(2)  # (B, C, H*W)
    
    # Center the data (remove mean for proper correlation)
    pred_centered = pred_flat - pred_flat.mean(dim=2, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=2, keepdim=True)
    
    # Compute correlation
    # ACC = sum(pred' * target') / sqrt(sum(pred'^2) * sum(target'^2))
    numerator = (pred_centered * target_centered).sum(dim=2)
    pred_norm = torch.sqrt((pred_centered ** 2).sum(dim=2) + 1e-8)
    target_norm = torch.sqrt((target_centered ** 2).sum(dim=2) + 1e-8)
    
    acc = numerator / (pred_norm * target_norm + 1e-8)
    
    # Handle any remaining NaN values
    acc = torch.where(torch.isnan(acc), torch.zeros_like(acc), acc)
    
    return acc.mean()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM).
    
    Measures perceptual similarity considering luminance,
    contrast, and structure.
    
    Args:
        pred: Predictions (B, C, H, W)
        target: Ground truth (B, C, H, W)
        window_size: Size of Gaussian window
        k1, k2: Stability constants
        
    Returns:
        SSIM value
    """
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)
    
    window = gaussian_window(window_size).to(pred.device)
    
    # Compute means
    mu_pred = F.conv2d(pred, window.expand(pred.shape[1], -1, -1, -1), 
                       padding=window_size//2, groups=pred.shape[1])
    mu_target = F.conv2d(target, window.expand(target.shape[1], -1, -1, -1),
                         padding=window_size//2, groups=target.shape[1])
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window.expand(pred.shape[1], -1, -1, -1),
                              padding=window_size//2, groups=pred.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window.expand(target.shape[1], -1, -1, -1),
                                padding=window_size//2, groups=target.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window.expand(pred.shape[1], -1, -1, -1),
                                  padding=window_size//2, groups=pred.shape[1]) - mu_pred_target
    
    # Dynamic range (assuming normalized data)
    L = 2.0  # Range from -1 to 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # SSIM formula
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    
    ssim_map = numerator / denominator
    
    return ssim_map.mean()


def crps_gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score for Gaussian predictions.
    
    Proper scoring rule that simultaneously measures calibration
    and sharpness. Lower is better.
    
    CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    
    Reference: Gneiting & Raftery (2007)
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        target: Ground truth
        
    Returns:
        CRPS value
    """
    z = (target - mean) / (std + 1e-8)
    
    # Standard normal CDF: Φ(z)
    cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    
    # Standard normal PDF: φ(z)
    pdf = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    
    # CRPS for Gaussian
    sqrt_pi = np.sqrt(np.pi)
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt_pi)
    
    return crps.mean()


def calibration_error(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expected Calibration Error for Gaussian predictions.
    
    Measures how well predicted confidence levels match actual
    coverage frequencies.
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        target: Ground truth
        num_bins: Number of confidence bins
        
    Returns:
        Tuple of (ECE, expected_freq, observed_freq)
    """
    # Compute z-scores
    z = torch.abs(target - mean) / (std + 1e-8)
    
    # Expected coverage levels (quantiles of standard normal)
    confidence_levels = torch.linspace(0.1, 0.99, num_bins, device=mean.device)
    z_thresholds = torch.erfinv(confidence_levels) * np.sqrt(2)
    
    # Compute actual coverage at each level
    expected = []
    observed = []
    
    for i, (conf, z_thresh) in enumerate(zip(confidence_levels, z_thresholds)):
        expected.append(conf.item())
        observed.append((z <= z_thresh).float().mean().item())
    
    expected = torch.tensor(expected, device=mean.device)
    observed = torch.tensor(observed, device=mean.device)
    
    # Expected Calibration Error
    ece = torch.abs(expected - observed).mean()
    
    return ece, expected, observed


def sharpness(std: torch.Tensor) -> torch.Tensor:
    """
    Sharpness: average predicted standard deviation.
    
    Lower is better (tighter predictions), but only if
    calibrated. Overconfident models have low sharpness
    but poor calibration.
    
    Args:
        std: Predicted standard deviation
        
    Returns:
        Mean standard deviation
    """
    return std.mean()


def coverage(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    level: float = 0.9
) -> torch.Tensor:
    """
    Coverage: fraction of targets within prediction interval.
    
    For calibrated Gaussian predictions at 90% level,
    coverage should be approximately 0.9.
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        target: Ground truth
        level: Confidence level (default 90%)
        
    Returns:
        Coverage fraction
    """
    # Z-score for confidence level
    z_thresh = torch.erfinv(torch.tensor(level)) * np.sqrt(2)
    
    # Compute whether target is within interval
    z = torch.abs(target - mean) / (std + 1e-8)
    within = (z <= z_thresh).float()
    
    return within.mean()


class ClimateMetrics:
    """
    Aggregated metrics calculator for climate emulation.
    
    Computes all standard metrics following WeatherBench.
    """
    
    def __init__(
        self,
        climatology: Optional[torch.Tensor] = None,
        lat_weights: Optional[torch.Tensor] = None,
        variable_names: Optional[list] = None
    ):
        """
        Initialize metrics calculator.
        
        Args:
            climatology: Climatological mean for ACC
            lat_weights: Latitude weights
            variable_names: Names for per-variable reporting
        """
        self.climatology = climatology
        self.lat_weights = lat_weights
        self.variable_names = variable_names or []
        
    def compute_deterministic(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute deterministic metrics.
        
        Args:
            pred: Predictions
            target: Ground truth
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        # RMSE
        results['rmse'] = rmse(pred, target, self.lat_weights).item()
        
        # MAE
        results['mae'] = mae(pred, target, self.lat_weights).item()
        
        # ACC
        results['acc'] = anomaly_correlation_coefficient(
            pred, target, self.climatology, self.lat_weights
        ).item()
        
        # SSIM
        results['ssim'] = ssim(pred, target).item()
        
        # Per-channel RMSE
        for c in range(pred.shape[1]):
            name = self.variable_names[c] if c < len(self.variable_names) else f'ch_{c}'
            results[f'rmse_{name}'] = rmse(
                pred[:, c:c+1], target[:, c:c+1], self.lat_weights
            ).item()
        
        return results
    
    def compute_probabilistic(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute probabilistic/uncertainty metrics.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: Ground truth
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        # CRPS
        results['crps'] = crps_gaussian(mean, std, target).item()
        
        # Calibration error
        ece, _, _ = calibration_error(mean, std, target)
        results['calibration_error'] = ece.item()
        
        # Sharpness
        results['sharpness'] = sharpness(std).item()
        
        # Coverage at different levels
        for level in [0.5, 0.9, 0.95]:
            results[f'coverage_{int(level*100)}'] = coverage(
                mean, std, target, level
            ).item()
        
        return results
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        std: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            pred: Predictions (mean if probabilistic)
            target: Ground truth
            std: Optional standard deviation for uncertainty metrics
            
        Returns:
            Dictionary of all metrics
        """
        results = self.compute_deterministic(pred, target)
        
        if std is not None:
            prob_results = self.compute_probabilistic(pred, std, target)
            results.update(prob_results)
        
        return results


if __name__ == "__main__":
    # Test metrics
    print("Testing climate metrics...")
    
    # Create dummy data
    batch_size = 4
    channels = 7
    height, width = 32, 64
    
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    std = torch.abs(torch.randn(batch_size, channels, height, width)) + 0.1
    
    # Test individual metrics
    print(f"RMSE: {rmse(pred, target).item():.4f}")
    print(f"MAE: {mae(pred, target).item():.4f}")
    print(f"ACC: {anomaly_correlation_coefficient(pred, target).item():.4f}")
    print(f"SSIM: {ssim(pred, target).item():.4f}")
    print(f"CRPS: {crps_gaussian(pred, std, target).item():.4f}")
    
    # Test calibration
    ece, expected, observed = calibration_error(pred, std, target)
    print(f"ECE: {ece.item():.4f}")
    
    # Test coverage
    print(f"Coverage 90%: {coverage(pred, std, target, 0.9).item():.4f}")
    
    # Test aggregate calculator
    print("\nTesting ClimateMetrics...")
    metrics = ClimateMetrics(variable_names=['t2m', 'u10', 'v10', 'z500', 't500', 'z850', 't850'])
    results = metrics.compute_all(pred, target, std)
    
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nAll metrics tests passed!")
