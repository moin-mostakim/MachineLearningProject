"""
Data module for Diabetic Retinopathy project.
"""

from .dataset import DRDataset, get_dataloaders

__all__ = ['DRDataset', 'get_dataloaders']
