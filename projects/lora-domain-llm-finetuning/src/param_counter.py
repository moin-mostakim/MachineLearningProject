"""Parameter counting and model analysis utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentParams:
    """Parameter counts for a model component."""
    name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    size_mb: float
    
    @property
    def trainable_pct(self) -> float:
        return 100 * self.trainable_params / self.total_params if self.total_params > 0 else 0.0


@dataclass  
class ModelAnalysis:
    """Complete model parameter analysis."""
    model_name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    total_size_mb: float
    components: List[ComponentParams]
    flops_estimate: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['components'] = [asdict(c) for c in self.components]
        return result
    
    def to_markdown_table(self) -> str:
        """Generate markdown table of parameter breakdown."""
        lines = [
            f"### {self.model_name}",
            "",
            "| Component | Total Params | Trainable | Frozen | Size (MB) | % Trainable |",
            "|-----------|-------------|-----------|--------|-----------|-------------|"
        ]
        
        for comp in self.components:
            lines.append(
                f"| {comp.name} | {comp.total_params:,} | {comp.trainable_params:,} | "
                f"{comp.frozen_params:,} | {comp.size_mb:.2f} | {comp.trainable_pct:.1f}% |"
            )
        
        lines.extend([
            f"| **Total** | **{self.total_params:,}** | **{self.trainable_params:,}** | "
            f"**{self.frozen_params:,}** | **{self.total_size_mb:.2f}** | "
            f"**{100*self.trainable_params/self.total_params:.1f}%** |"
        ])
        
        if self.flops_estimate:
            lines.append(f"\n**Estimated FLOPs:** {self.flops_estimate:,}")
        
        return "\n".join(lines)


class ParameterCounter:
    """Count and analyze model parameters."""
    
    @staticmethod
    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    @staticmethod
    def get_size_mb(module: nn.Module) -> float:
        """Get module size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def analyze_transformer_lm(
        self,
        model: nn.Module,
        model_name: str = "Model"
    ) -> ModelAnalysis:
        """Analyze a transformer language model (GPT-2 style)."""
        components = []
        
        component_patterns = [
            ("Embeddings", ["wte", "wpe", "embed"]),
            ("Attention", ["attn", "attention", "c_attn"]),
            ("MLP/FFN", ["mlp", "ffn", "c_fc", "c_proj"]),
            ("LayerNorm", ["ln", "layer_norm", "norm"]),
            ("LM Head", ["lm_head", "head", "classifier"]),
        ]
        
        counted_params = set()
        
        for comp_name, patterns in component_patterns:
            comp_total = 0
            comp_trainable = 0
            comp_size = 0.0
            
            for name, module in model.named_modules():
                if any(p in name.lower() for p in patterns):
                    for pname, param in module.named_parameters(recurse=False):
                        full_name = f"{name}.{pname}"
                        if full_name not in counted_params:
                            counted_params.add(full_name)
                            comp_total += param.numel()
                            if param.requires_grad:
                                comp_trainable += param.numel()
                            comp_size += param.numel() * param.element_size() / (1024 * 1024)
            
            if comp_total > 0:
                components.append(ComponentParams(
                    name=comp_name,
                    total_params=comp_total,
                    trainable_params=comp_trainable,
                    frozen_params=comp_total - comp_trainable,
                    size_mb=comp_size
                ))
        
        other_total = 0
        other_trainable = 0
        other_size = 0.0
        
        for name, param in model.named_parameters():
            if name not in counted_params:
                other_total += param.numel()
                if param.requires_grad:
                    other_trainable += param.numel()
                other_size += param.numel() * param.element_size() / (1024 * 1024)
        
        if other_total > 0:
            components.append(ComponentParams(
                name="Other",
                total_params=other_total,
                trainable_params=other_trainable,
                frozen_params=other_total - other_trainable,
                size_mb=other_size
            ))
        
        total_params, trainable_params = self.count_parameters(model)
        
        return ModelAnalysis(
            model_name=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=total_params - trainable_params,
            total_size_mb=self.get_size_mb(model),
            components=components,
            flops_estimate=self._estimate_flops_transformer(model)
        )
    
    def analyze_lora_model(
        self,
        model: nn.Module,
        model_name: str = "LoRA Model"
    ) -> ModelAnalysis:
        """Analyze a PEFT/LoRA model specifically."""
        components = []
        
        lora_total = 0
        lora_trainable = 0
        base_total = 0
        base_trainable = 0
        
        for name, param in model.named_parameters():
            if "lora_" in name.lower():
                lora_total += param.numel()
                if param.requires_grad:
                    lora_trainable += param.numel()
            else:
                base_total += param.numel()
                if param.requires_grad:
                    base_trainable += param.numel()
        
        if base_total > 0:
            components.append(ComponentParams(
                name="Base Model (Frozen)",
                total_params=base_total,
                trainable_params=base_trainable,
                frozen_params=base_total - base_trainable,
                size_mb=base_total * 4 / (1024 * 1024)
            ))
        
        if lora_total > 0:
            components.append(ComponentParams(
                name="LoRA Adapters",
                total_params=lora_total,
                trainable_params=lora_trainable,
                frozen_params=lora_total - lora_trainable,
                size_mb=lora_total * 4 / (1024 * 1024)
            ))
        
        total_params, trainable_params = self.count_parameters(model)
        
        return ModelAnalysis(
            model_name=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=total_params - trainable_params,
            total_size_mb=self.get_size_mb(model),
            components=components,
            flops_estimate=self._estimate_flops_transformer(model)
        )
    
    def _estimate_flops_transformer(self, model: nn.Module, seq_len: int = 128) -> Optional[int]:
        """Rough FLOPs estimate for transformer forward pass."""
        try:
            total_params, _ = self.count_parameters(model)
            flops = 2 * total_params * seq_len
            return flops
        except Exception:
            return None


def compare_models(
    models: Dict[str, nn.Module],
    analyzer: Optional[ParameterCounter] = None
) -> Dict[str, ModelAnalysis]:
    """Compare multiple models."""
    if analyzer is None:
        analyzer = ParameterCounter()
    
    results = {}
    for name, model in models.items():
        if hasattr(model, 'peft_config') or any('lora' in n.lower() for n, _ in model.named_parameters()):
            results[name] = analyzer.analyze_lora_model(model, name)
        else:
            results[name] = analyzer.analyze_transformer_lm(model, name)
    
    return results


def generate_comparison_table(analyses: Dict[str, ModelAnalysis]) -> str:
    """Generate markdown comparison table for multiple models."""
    lines = [
        "## Model Parameter Comparison",
        "",
        "| Model | Total Params | Trainable | Frozen | Size (MB) | % Trainable |",
        "|-------|-------------|-----------|--------|-----------|-------------|"
    ]
    
    for name, analysis in analyses.items():
        pct = 100 * analysis.trainable_params / analysis.total_params if analysis.total_params > 0 else 0
        lines.append(
            f"| {name} | {analysis.total_params:,} | {analysis.trainable_params:,} | "
            f"{analysis.frozen_params:,} | {analysis.total_size_mb:.2f} | {pct:.2f}% |"
        )
    
    return "\n".join(lines)


def save_param_analysis(
    analyses: Dict[str, ModelAnalysis],
    output_path: str
):
    """Save parameter analysis to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {name: analysis.to_dict() for name, analysis in analyses.items()}
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Parameter analysis saved to {output_path}")
    return output_path
