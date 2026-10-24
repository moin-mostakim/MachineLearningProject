"""Neural network models for climate emulation."""

from .baselines import (
    PersistenceBaseline,
    ClimatologyBaseline,
    LinearRegressionBaseline,
    WeightedClimatologyBaseline,
    get_baseline_models
)

from .unet import (
    UNet,
    UNetWithUncertainty,
    ConvBlock,
    AttentionGate
)

from .convlstm import (
    ConvLSTM,
    ConvLSTMCell,
    ConvLSTMPredictor,
    ConvGRUCell
)

from .ensemble import (
    DeepEnsemble,
    EnsembleTrainer,
    NegativeLogLikelihoodLoss,
    CRPS
)

__all__ = [
    # Baselines
    'PersistenceBaseline',
    'ClimatologyBaseline', 
    'LinearRegressionBaseline',
    'WeightedClimatologyBaseline',
    'get_baseline_models',
    # U-Net
    'UNet',
    'UNetWithUncertainty',
    'ConvBlock',
    'AttentionGate',
    # ConvLSTM
    'ConvLSTM',
    'ConvLSTMCell',
    'ConvLSTMPredictor',
    'ConvGRUCell',
    # Ensemble
    'DeepEnsemble',
    'EnsembleTrainer',
    'NegativeLogLikelihoodLoss',
    'CRPS'
]
