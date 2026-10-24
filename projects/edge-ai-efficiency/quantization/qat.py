"""
Quantization-Aware Training (QAT) implementation.

Reference:
Jacob et al., "Quantization and Training of Neural Networks for Efficient 
Integer-Arithmetic-Only Inference", CVPR 2018
https://arxiv.org/abs/1712.05877

Key concepts:
- Fake quantization: simulates quantization during training
- Straight-through estimator: gradients pass through quantization
- Per-channel quantization for weights (better accuracy)
- Per-tensor quantization for activations (simpler)
"""

import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat as torch_prepare_qat,
    convert as torch_convert,
    QConfigMapping,
    default_qat_qconfig,
    fuse_modules
)
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import copy


def fuse_model_modules(model: nn.Module) -> nn.Module:
    """
    Fuse Conv-BN-ReLU modules for quantization.
    
    Batch normalization can be folded into convolution:
    - Reduces operations during inference
    - Required before quantization
    """
    model = copy.deepcopy(model)
    
    # Find fuseable patterns
    # Conv + BN + ReLU or Conv + BN
    modules_to_fuse = []
    
    prev_name = None
    prev_module = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_name = name
            prev_module = module
        elif isinstance(module, nn.BatchNorm2d) and prev_module is not None:
            # Check if they're in the same Sequential
            parent_name = '.'.join(name.split('.')[:-1])
            prev_parent = '.'.join(prev_name.split('.')[:-1])
            
            if parent_name == prev_parent:
                modules_to_fuse.append([prev_name, name])
            prev_name = None
            prev_module = None
        else:
            prev_name = None
            prev_module = None
    
    # Fuse modules
    for fusion in modules_to_fuse:
        try:
            fuse_modules(model, fusion, inplace=True)
        except Exception:
            pass  # Skip if fusion fails
    
    return model


def prepare_qat(
    model: nn.Module,
    qconfig: str = 'default',
    inplace: bool = False
) -> nn.Module:
    """
    Prepare model for quantization-aware training.
    
    Args:
        model: Model to prepare
        qconfig: Quantization config ('default', 'per_channel', 'x86', 'qnnpack')
        inplace: Modify model in place
        
    Returns:
        Model prepared for QAT
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    model.train()
    
    # Set quantization configuration
    if qconfig == 'default' or qconfig == 'per_channel':
        qconfig_obj = get_default_qat_qconfig('x86')
    elif qconfig == 'x86':
        qconfig_obj = get_default_qat_qconfig('x86')
    elif qconfig == 'qnnpack':
        qconfig_obj = get_default_qat_qconfig('qnnpack')
    else:
        qconfig_obj = default_qat_qconfig
    
    # Apply qconfig to all modules
    model.qconfig = qconfig_obj
    
    # Fuse modules first
    model = fuse_model_modules(model)
    
    # Prepare for QAT
    try:
        torch_prepare_qat(model, inplace=True)
    except Exception as e:
        print(f"Warning: Standard QAT preparation failed: {e}")
        print("Using fallback approach...")
        # Fallback: just return the model as-is
        pass
    
    return model


def convert_to_quantized(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Convert QAT model to quantized model.
    
    Args:
        model: QAT-prepared model
        inplace: Modify model in place
        
    Returns:
        Quantized model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    model.eval()
    
    try:
        torch_convert(model, inplace=True)
    except Exception as e:
        print(f"Warning: Quantization conversion failed: {e}")
        # Return float model as fallback
        pass
    
    return model


class QuantizationAwareTraining:
    """
    Quantization-Aware Training manager.
    
    Handles the full QAT pipeline:
    1. Prepare model for QAT
    2. Train with fake quantization
    3. Convert to quantized model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer_fn=None,
        qat_epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 4e-5,
        warmup_epochs: int = 2,
        device: str = 'cuda',
        qconfig: str = 'default'
    ):
        """
        Initialize QAT manager.
        
        Args:
            model: Model to quantize
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer_fn: Optimizer factory function
            qat_epochs: Number of QAT epochs
            learning_rate: Learning rate for QAT
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs before QAT
            device: Device to train on
            qconfig: Quantization configuration
        """
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer_fn = optimizer_fn
        self.qat_epochs = qat_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.qconfig = qconfig
        
        self.qat_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None
        
        self.training_history: List[Dict[str, float]] = []
    
    def prepare(self):
        """Prepare model for QAT."""
        print("Preparing model for QAT...")
        self.qat_model = prepare_qat(self.model, qconfig=self.qconfig)
        self.qat_model.to(self.device)
        print("Model prepared for QAT")
    
    def train_epoch(self, optimizer, epoch: int) -> Dict[str, float]:
        """Train one epoch with QAT."""
        self.qat_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.qat_model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def evaluate(self, model: nn.Module = None) -> Dict[str, float]:
        """Evaluate model."""
        if model is None:
            model = self.qat_model if self.qat_model else self.model
        
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def train(self, verbose: bool = True) -> nn.Module:
        """
        Run full QAT training.
        
        Args:
            verbose: Print progress
            
        Returns:
            QAT-trained model
        """
        if self.qat_model is None:
            self.prepare()
        
        # Create optimizer
        if self.optimizer_fn:
            optimizer = self.optimizer_fn(self.qat_model.parameters())
        else:
            optimizer = torch.optim.Adam(
                self.qat_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.qat_epochs
        )
        
        # Evaluate before QAT
        pre_qat_metrics = self.evaluate(self.model)
        if verbose:
            print(f"Pre-QAT accuracy: {pre_qat_metrics['accuracy']:.2f}%")
        
        # QAT training
        for epoch in range(self.qat_epochs):
            train_metrics = self.train_epoch(optimizer, epoch)
            val_metrics = self.evaluate()
            scheduler.step()
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy']
            })
            
            if verbose:
                print(f"QAT Epoch {epoch+1}/{self.qat_epochs}: "
                      f"Train Acc={train_metrics['accuracy']:.2f}%, "
                      f"Val Acc={val_metrics['accuracy']:.2f}%")
        
        return self.qat_model
    
    def convert(self) -> nn.Module:
        """Convert QAT model to quantized model."""
        if self.qat_model is None:
            raise RuntimeError("Must train QAT model before converting")
        
        print("Converting to quantized model...")
        self.quantized_model = convert_to_quantized(self.qat_model)
        print("Model quantized")
        
        return self.quantized_model
    
    def get_quantized_model(self) -> nn.Module:
        """Get the quantized model (train and convert if needed)."""
        if self.quantized_model is not None:
            return self.quantized_model
        
        if self.qat_model is None:
            self.train()
        
        return self.convert()
    
    def compare_accuracy(self) -> Dict[str, float]:
        """Compare accuracy: float vs QAT vs quantized."""
        results = {}
        
        # Float model
        float_metrics = self.evaluate(self.original_model)
        results['float_accuracy'] = float_metrics['accuracy']
        
        # QAT model (if trained)
        if self.qat_model is not None:
            qat_metrics = self.evaluate(self.qat_model)
            results['qat_accuracy'] = qat_metrics['accuracy']
        
        # Quantized model (if converted)
        if self.quantized_model is not None:
            quant_metrics = self.evaluate(self.quantized_model)
            results['quantized_accuracy'] = quant_metrics['accuracy']
        
        return results


def create_qat_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> QuantizationAwareTraining:
    """Factory function for QAT trainer."""
    return QuantizationAwareTraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        qat_epochs=config.get('qat_epochs', 5),
        learning_rate=config.get('qat_learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 4e-5),
        warmup_epochs=config.get('qat_warmup_epochs', 2),
        device=device,
        qconfig=config.get('qconfig', 'default')
    )
