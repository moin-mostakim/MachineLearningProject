"""Utility functions for climate emulation."""

from .metrics import (
    rmse,
    mae,
    anomaly_correlation_coefficient,
    ssim,
    crps_gaussian,
    calibration_error,
    sharpness,
    coverage,
    ClimateMetrics,
    MetricResult
)

from .resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    TrainingStats,
    Timer,
    count_parameters,
    estimate_model_size,
    estimate_flops,
    get_parameter_table,
    ExperimentLogger
)

from .quantization import (
    dynamic_quantize,
    prepare_static_quantization,
    calibrate_and_convert,
    compare_models,
    quantize_for_deployment,
    print_quantization_report,
    QuantizationWrapper
)

__all__ = [
    # Metrics
    'rmse',
    'mae',
    'anomaly_correlation_coefficient',
    'ssim',
    'crps_gaussian',
    'calibration_error',
    'sharpness',
    'coverage',
    'ClimateMetrics',
    'MetricResult',
    # Resource monitoring
    'ResourceMonitor',
    'ResourceSnapshot',
    'TrainingStats',
    'Timer',
    'count_parameters',
    'estimate_model_size',
    'estimate_flops',
    'get_parameter_table',
    'ExperimentLogger',
    # Quantization
    'dynamic_quantize',
    'prepare_static_quantization',
    'calibrate_and_convert',
    'compare_models',
    'quantize_for_deployment',
    'print_quantization_report',
    'QuantizationWrapper'
]
