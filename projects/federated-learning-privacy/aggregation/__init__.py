"""Aggregation strategies for federated learning."""

from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .personalized import PersonalizedAggregator

__all__ = [
    "FedAvgAggregator",
    "FedProxAggregator", 
    "PersonalizedAggregator",
    "get_aggregator"
]


def get_aggregator(name: str, **kwargs):
    """Factory function to get aggregator by name."""
    aggregators = {
        "fedavg": FedAvgAggregator,
        "fedprox": FedProxAggregator,
        "personalized": PersonalizedAggregator,
    }
    
    if name not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(aggregators.keys())}")
    
    return aggregators[name](**kwargs)
