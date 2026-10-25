"""Model architectures for federated learning."""

from .cnn import SimpleCNN, CIFAR10CNN
from .resnet import ResNet18
from .mlp import MLP

__all__ = ["SimpleCNN", "CIFAR10CNN", "ResNet18", "MLP", "get_model"]


def get_model(model_name: str, num_classes: int = 10, **kwargs):
    """Factory function to get model by name."""
    models = {
        "cnn": SimpleCNN,
        "cifar10_cnn": CIFAR10CNN,
        "resnet18": ResNet18,
        "mlp": MLP,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, **kwargs)
