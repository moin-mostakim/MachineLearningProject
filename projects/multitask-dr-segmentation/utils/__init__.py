"""Utilities package for Multi-Task DR Learning."""

from utils.losses import DiceLoss, DiceBCELoss, FocalLoss, MultiTaskLoss
from utils.metrics import ClassificationMetrics, SegmentationMetrics, MultiTaskMetrics
from utils.monitoring import ResourceMonitor, TrainingLogger
from utils.quantization import (
    apply_dynamic_quantization, 
    compare_quantization, 
    run_quantization_analysis
)
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    save_segmentation_examples,
    create_literature_comparison_table,
    plot_model_comparison,
    generate_all_figures
)

__all__ = [
    'DiceLoss', 'DiceBCELoss', 'FocalLoss', 'MultiTaskLoss',
    'ClassificationMetrics', 'SegmentationMetrics', 'MultiTaskMetrics',
    'ResourceMonitor', 'TrainingLogger',
    'apply_dynamic_quantization', 'compare_quantization', 'run_quantization_analysis',
    'plot_training_curves', 'plot_confusion_matrix', 'save_segmentation_examples',
    'create_literature_comparison_table', 'plot_model_comparison', 'generate_all_figures'
]
