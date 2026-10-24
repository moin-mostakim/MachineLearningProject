"""Quantization modules for edge AI efficiency."""

from .qat import QuantizationAwareTraining, prepare_qat, convert_to_quantized
from .ptq import PostTrainingQuantization
