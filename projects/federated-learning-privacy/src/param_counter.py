"""Parameter counting utilities."""

import torch.nn as nn
from typing import Dict, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ModelAnalysis:
    model_name: str
    total_params: int
    trainable_params: int
    size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParameterCounter:
    @staticmethod
    def count(model: nn.Module) -> tuple:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    @staticmethod
    def get_size_mb(model: nn.Module) -> float:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    def analyze(self, model: nn.Module, name: str = "Model") -> ModelAnalysis:
        total, trainable = self.count(model)
        return ModelAnalysis(name, total, trainable, self.get_size_mb(model))


def generate_comparison_table(analyses: Dict[str, ModelAnalysis]) -> str:
    lines = [
        "## Model Parameter Comparison",
        "| Model | Total Params | Trainable | Size (MB) |",
        "|-------|-------------|-----------|-----------|"
    ]
    for name, a in analyses.items():
        lines.append(f"| {name} | {a.total_params:,} | {a.trainable_params:,} | {a.size_mb:.2f} |")
    return "\n".join(lines)


def save_param_analysis(analyses: Dict[str, ModelAnalysis], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({n: a.to_dict() for n, a in analyses.items()}, f, indent=2)
