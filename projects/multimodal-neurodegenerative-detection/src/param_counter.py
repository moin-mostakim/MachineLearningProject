"""Parameter counting utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentParams:
    """Parameter counts for a component."""
    name: str
    total_params: int
    trainable_params: int
    size_mb: float


@dataclass
class ModelAnalysis:
    """Model parameter analysis."""
    model_name: str
    total_params: int
    trainable_params: int
    total_size_mb: float
    components: List[ComponentParams]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['components'] = [asdict(c) for c in self.components]
        return result
    
    def to_markdown_table(self) -> str:
        lines = [
            f"### {self.model_name}",
            "",
            "| Component | Total Params | Trainable | Size (MB) |",
            "|-----------|-------------|-----------|-----------|"
        ]
        
        for comp in self.components:
            lines.append(
                f"| {comp.name} | {comp.total_params:,} | "
                f"{comp.trainable_params:,} | {comp.size_mb:.3f} |"
            )
        
        lines.append(
            f"| **Total** | **{self.total_params:,}** | "
            f"**{self.trainable_params:,}** | **{self.total_size_mb:.3f}** |"
        )
        
        return "\n".join(lines)


class ParameterCounter:
    """Count model parameters."""
    
    @staticmethod
    def count_parameters(module: nn.Module) -> tuple:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    @staticmethod
    def get_size_mb(module: nn.Module) -> float:
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        return param_size / (1024 * 1024)
    
    def analyze_multimodal_model(
        self,
        model: nn.Module,
        model_name: str = "Model"
    ) -> ModelAnalysis:
        """Analyze multimodal model."""
        components = []
        
        component_mapping = {
            "Image Encoder": "image_encoder",
            "Clinical Encoder": "clinical_encoder",
            "Fusion Module": "fusion",
            "Classifier": "classifier"
        }
        
        for comp_name, attr_name in component_mapping.items():
            if hasattr(model, attr_name):
                module = getattr(model, attr_name)
                total, trainable = self.count_parameters(module)
                size = self.get_size_mb(module)
                components.append(ComponentParams(
                    name=comp_name,
                    total_params=total,
                    trainable_params=trainable,
                    size_mb=size
                ))
        
        total_params, trainable_params = self.count_parameters(model)
        
        return ModelAnalysis(
            model_name=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            total_size_mb=self.get_size_mb(model),
            components=components
        )


def compare_models(models: Dict[str, nn.Module]) -> Dict[str, ModelAnalysis]:
    """Compare multiple models."""
    counter = ParameterCounter()
    return {name: counter.analyze_multimodal_model(model, name) 
            for name, model in models.items()}


def generate_comparison_table(analyses: Dict[str, ModelAnalysis]) -> str:
    """Generate comparison table."""
    lines = [
        "## Model Parameter Comparison",
        "",
        "| Model | Total Params | Trainable | Size (MB) | % Trainable |",
        "|-------|-------------|-----------|-----------|-------------|"
    ]
    
    for name, analysis in analyses.items():
        pct = 100 * analysis.trainable_params / analysis.total_params if analysis.total_params > 0 else 0
        lines.append(
            f"| {name} | {analysis.total_params:,} | {analysis.trainable_params:,} | "
            f"{analysis.total_size_mb:.3f} | {pct:.2f}% |"
        )
    
    return "\n".join(lines)


def save_param_analysis(analyses: Dict[str, ModelAnalysis], output_path: str):
    """Save analysis to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {name: analysis.to_dict() for name, analysis in analyses.items()}
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
