#!/usr/bin/env python3
"""
Multi-Task Learning Framework for Joint Lesion Segmentation and DR Grading

Main entry point for training and evaluation.

This framework implements:
1. Classification-only baseline (DR grading)
2. Segmentation-only baseline (U-Net for lesion segmentation)  
3. Multi-task model with shared encoder (our method)

Dataset: Synthetic retinal data mimicking IDRiD dataset characteristics.

References:
- Porwal et al. (2018) "IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge"
- Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
- Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing"

Usage:
    python main.py                  # Run full pipeline
    python main.py --train-only     # Training only
    python main.py --eval-only      # Evaluation only
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from train import run_all_experiments, set_seed
from evaluate import run_full_evaluation, generate_parameter_table
from data.synthetic_data import create_synthetic_dataset
from utils.quantization import run_quantization_analysis
from models.multitask_model import print_model_comparison


def print_header():
    """Print project header."""
    print("\n" + "=" * 80)
    print("MULTI-TASK LEARNING FRAMEWORK")
    print("Joint Lesion Segmentation and Diabetic Retinopathy Grading")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Debug Mode: {config.DEBUG_MODE}")
    print(f"  Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  Batch Size: {config.get_effective_config()['batch_size']}")
    print(f"  Epochs: {config.get_effective_config()['num_epochs']}")
    print(f"  Task Weighting: {config.DEFAULT_WEIGHTING}")
    print(f"  Seed: {config.SEED}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Task Learning for DR Grading and Lesion Segmentation'
    )
    parser.add_argument('--train-only', action='store_true',
                       help='Run training only')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic data only')
    parser.add_argument('--param-table', action='store_true',
                       help='Print parameter table only')
    parser.add_argument('--quantize', action='store_true',
                       help='Run quantization analysis only')
    args = parser.parse_args()
    
    print_header()
    
    start_time = time.time()
    
    if args.generate_data:
        print("\nGenerating synthetic dataset...")
        set_seed(config.SEED)
        create_synthetic_dataset()
        print("Done!")
        return
    
    if args.param_table:
        print_model_comparison()
        return
    
    if args.quantize:
        run_quantization_analysis()
        return
    
    if args.eval_only:
        print("\nRunning evaluation only...")
        results = run_full_evaluation()
    elif args.train_only:
        print("\nRunning training only...")
        results = run_all_experiments()
    else:
        # Full pipeline
        print("\nRunning full training and evaluation pipeline...")
        
        # Training
        train_results = run_all_experiments()
        
        # Evaluation
        print("\n" + "=" * 80)
        print("STARTING EVALUATION PHASE")
        print("=" * 80)
        eval_results = run_full_evaluation()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"\nOutputs saved to:")
    print(f"  Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print(f"  Experiments: {config.EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
