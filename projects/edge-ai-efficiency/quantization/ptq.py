"""
Post-Training Quantization (PTQ) implementation.

Simpler alternative to QAT that quantizes a pre-trained model
using calibration data to determine quantization ranges.

Supports:
- INT8 quantization (standard, good accuracy)
- INT4 quantization (aggressive, some accuracy loss)
"""

import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    default_dynamic_qconfig
)
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import copy


class PostTrainingQuantization:
    """
    Post-Training Quantization manager.
    
    Quantizes a pre-trained model without retraining.
    Uses calibration data to determine optimal quantization ranges.
    """
    
    def __init__(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        num_calibration_batches: int = 100,
        backend: str = 'x86',
        device: str = 'cpu'
    ):
        """
        Initialize PTQ manager.
        
        Args:
            model: Pre-trained model to quantize
            calibration_loader: Data loader for calibration
            num_calibration_batches: Number of batches for calibration
            backend: Quantization backend ('x86' or 'qnnpack')
            device: Device for calibration
        """
        self.original_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        self.calibration_loader = calibration_loader
        self.num_calibration_batches = num_calibration_batches
        self.backend = backend
        self.device = device
        
        self.quantized_model: Optional[nn.Module] = None
    
    def calibrate(self, model: nn.Module):
        """
        Run calibration to determine quantization ranges.
        
        Args:
            model: Model to calibrate (should be prepared for quantization)
        """
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.calibration_loader):
                if batch_idx >= self.num_calibration_batches:
                    break
                
                images = images.to(self.device)
                model(images)
    
    def quantize_static(self) -> nn.Module:
        """
        Apply static post-training quantization.
        
        Returns:
            Quantized model
        """
        model = copy.deepcopy(self.model)
        model.eval()
        model.to(self.device)
        
        # Set quantization config
        model.qconfig = get_default_qconfig(self.backend)
        
        # Prepare for quantization
        try:
            prepare(model, inplace=True)
        except Exception as e:
            print(f"Warning: Static quantization preparation failed: {e}")
            return model
        
        # Calibrate
        print("Running calibration...")
        self.calibrate(model)
        
        # Convert to quantized
        try:
            convert(model, inplace=True)
        except Exception as e:
            print(f"Warning: Static quantization conversion failed: {e}")
            return model
        
        self.quantized_model = model
        return model
    
    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization (weights only).
        
        Returns:
            Dynamically quantized model
        """
        model = copy.deepcopy(self.model)
        model.eval()
        model.to('cpu')  # Dynamic quantization works on CPU
        
        # Apply dynamic quantization to Linear and LSTM layers
        self.quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return self.quantized_model
    
    def simulate_int4_quantization(self) -> nn.Module:
        """
        Simulate INT4 quantization by aggressive scaling.
        
        Note: True INT4 requires specialized hardware support.
        This is a simulation that shows the accuracy impact.
        
        Returns:
            Model with simulated INT4 quantization
        """
        model = copy.deepcopy(self.model)
        model.eval()
        
        # Simulate INT4 by quantizing weights to 16 levels
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    # Calculate scale for 4-bit range (-8 to 7)
                    abs_max = param.abs().max()
                    scale = abs_max / 7.0 if abs_max > 0 else 1.0
                    
                    # Quantize to INT4 range and dequantize
                    quantized = torch.clamp(torch.round(param / scale), -8, 7)
                    param.copy_(quantized * scale)
        
        return model
    
    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """
        Evaluate a model.
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def compare_quantization_methods(
        self,
        val_loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different quantization methods.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary comparing float, INT8, and INT4 performance
        """
        results = {}
        
        # Float32 baseline
        print("Evaluating Float32 baseline...")
        results['float32'] = self.evaluate(self.original_model, val_loader)
        results['float32']['bits'] = 32
        
        # INT8 dynamic quantization
        print("Applying INT8 dynamic quantization...")
        int8_model = self.quantize_dynamic()
        results['int8_dynamic'] = self.evaluate(int8_model, val_loader)
        results['int8_dynamic']['bits'] = 8
        
        # INT4 simulation
        print("Simulating INT4 quantization...")
        int4_model = self.simulate_int4_quantization()
        results['int4_simulated'] = self.evaluate(int4_model, val_loader)
        results['int4_simulated']['bits'] = 4
        
        return results
    
    def get_model_size_reduction(self) -> Dict[str, float]:
        """
        Calculate model size reduction from quantization.
        
        Returns:
            Dictionary with size information
        """
        def get_size(model):
            param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 ** 2)  # MB
        
        float_size = get_size(self.original_model)
        
        results = {
            'float32_size_mb': float_size,
            'int8_size_mb': float_size / 4,  # Theoretical 4x reduction
            'int4_size_mb': float_size / 8,  # Theoretical 8x reduction
        }
        
        results['int8_reduction'] = 1 - results['int8_size_mb'] / float_size
        results['int4_reduction'] = 1 - results['int4_size_mb'] / float_size
        
        return results


def create_ptq(
    model: nn.Module,
    calibration_loader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> PostTrainingQuantization:
    """Factory function for PTQ."""
    return PostTrainingQuantization(
        model=model,
        calibration_loader=calibration_loader,
        num_calibration_batches=config.get('observer_batches', 100),
        backend=config.get('backend', 'x86'),
        device=device
    )
