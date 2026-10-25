#!/usr/bin/env python3
"""
Parameter-Efficient Fine-Tuning of LLMs via Adaptive Low-Rank Adaptation
for Domain Specialization

Main entry point for training, evaluation, and benchmarking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')

from src.config import ExperimentConfig
from src.data_loader import DomainDataLoader
from src.model import AdaptiveLoRAModel
from src.trainer import LoRATrainer
from src.baselines import BaselineModels, LiteratureResults
from src.benchmark import BenchmarkEvaluator, compare_methods_on_benchmark
from src.evaluation import ModelEvaluator
from src.visualization import (
    plot_training_curves, plot_lora_analysis, plot_method_comparison,
    plot_parameter_efficiency, create_summary_report
)
from src.quantization import ModelQuantizer, generate_quantization_report
from src.param_counter import ParameterCounter, compare_models, generate_comparison_table, save_param_analysis
from src.resource_monitor import (
    HardwareInfo, ResourceMonitor, setup_cpu_optimization, 
    save_resource_results, get_model_size_mb
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Domain Fine-Tuning Experiment"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "train", "eval", "benchmark", "quantize"],
        help="Execution mode"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Use small debug configuration"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-threads", type=int, default=None,
        help="Number of CPU threads (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    thread_config = setup_cpu_optimization(args.num_threads)
    logger.info(f"CPU optimization: {thread_config}")
    
    hardware_info = HardwareInfo.collect()
    logger.info(f"Hardware: {hardware_info.cpu_model}, {hardware_info.cpu_cores_physical} cores, "
               f"{hardware_info.ram_total_gb:.1f}GB RAM")
    
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.debug:
        config = ExperimentConfig.small_debug()
    else:
        config = ExperimentConfig.default()
    
    config.save(output_dir / "config.json")
    
    all_results = {
        "hardware": hardware_info.__dict__,
        "config": config.to_dict()
    }
    
    logger.info("=" * 60)
    logger.info("Starting LoRA Fine-Tuning Experiment")
    logger.info("=" * 60)
    
    logger.info("Loading data...")
    data_loader = DomainDataLoader(
        model_name=config.model.model_name,
        max_length=config.model.max_length,
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config
    )
    
    processed_data = data_loader.load_dataset(
        max_train_samples=config.data.max_train_samples,
        max_val_samples=config.data.max_val_samples,
        max_test_samples=config.data.max_test_samples
    )
    
    dataloaders = data_loader.create_dataloaders(
        processed_data,
        batch_size=config.training.batch_size
    )
    
    tokenizer = data_loader.get_tokenizer()
    
    logger.info("Initializing baselines...")
    baselines = BaselineModels(config.model.model_name, device="cpu")
    
    baseline_models = {
        "Frozen Baseline": baselines.get_frozen_baseline(),
        "Full Fine-tuning": baselines.get_full_finetuning_model(),
        "Vanilla LoRA (r=8)": baselines.get_vanilla_lora_model(rank=8),
        "Adaptive LoRA (Ours)": baselines.get_adaptive_lora_model(
            rank=config.lora.r,
            alpha_multiplier=config.lora.lora_alpha / config.lora.r,
            dropout=config.lora.lora_dropout
        )
    }
    
    logger.info("Analyzing model parameters...")
    param_counter = ParameterCounter()
    param_analyses = compare_models(baseline_models, param_counter)
    
    all_results["parameter_analysis"] = {
        name: analysis.to_dict() for name, analysis in param_analyses.items()
    }
    
    save_param_analysis(param_analyses, output_dir / "parameter_analysis.json")
    
    param_table = generate_comparison_table(param_analyses)
    logger.info("\n" + param_table)
    
    methods_params = {
        name: {
            "trainable": analysis.trainable_params,
            "total": analysis.total_params
        }
        for name, analysis in param_analyses.items()
    }
    plot_parameter_efficiency(methods_params, save_path=figures_dir / "parameter_efficiency.png")
    
    if args.mode in ["full", "train"]:
        logger.info("Training Adaptive LoRA model...")
        
        adaptive_model = baseline_models["Adaptive LoRA (Ours)"]
        
        monitor = ResourceMonitor()
        monitor.start()
        
        trainer = LoRATrainer(
            model=adaptive_model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders.get("validation"),
            learning_rate=config.training.optimizer.learning_rate,
            num_epochs=config.training.num_epochs,
            warmup_steps=config.training.scheduler.warmup_steps,
            weight_decay=config.training.optimizer.weight_decay,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            output_dir=str(output_dir / "checkpoints"),
            logging_steps=config.training.logging_steps
        )
        
        training_results = trainer.train()
        
        training_resources = monitor.stop()
        all_results["training_resources"] = training_resources.to_dict()
        
        logger.info(f"Training completed in {training_resources.wall_time_seconds:.1f}s, "
                   f"Peak memory: {training_resources.memory_peak_mb:.1f}MB")
        
        all_results["training"] = training_results
        
        if training_results["train_losses"]:
            plot_training_curves(
                train_losses=training_results["train_losses"],
                val_losses=training_results.get("val_losses"),
                learning_rates=training_results.get("learning_rates"),
                save_path=figures_dir / "training_curves.png"
            )
        
        lora_stats = {
            "rank": config.lora.r,
            "alpha": config.lora.lora_alpha,
            "dropout": config.lora.lora_dropout,
            "trainable_params": param_analyses["Adaptive LoRA (Ours)"].trainable_params,
            "total_params": param_analyses["Adaptive LoRA (Ours)"].total_params
        }
        plot_lora_analysis(lora_stats, save_path=figures_dir / "lora_analysis.png")
    
    if args.mode in ["full", "benchmark"]:
        logger.info("Running benchmark evaluation...")
        
        benchmark_results = {}
        for method_name, model in baseline_models.items():
            logger.info(f"Evaluating {method_name} on WikiText-2...")
            evaluator = BenchmarkEvaluator(model, tokenizer, device="cpu")
            benchmark_results[method_name] = evaluator.evaluate_wikitext2(
                max_samples=config.data.max_test_samples
            )
        
        all_results["benchmark"] = benchmark_results
        
        published = {"GPT-2 Small (Radford 2019)": 29.41}
        plot_method_comparison(
            benchmark_results,
            metric_name="Perplexity",
            lower_is_better=True,
            title="WikiText-2 Perplexity Comparison",
            save_path=figures_dir / "benchmark_comparison.png",
            published_results=published
        )
    
    if args.mode in ["full", "quantize"]:
        logger.info("Running quantization comparison...")
        
        # Use frozen baseline for quantization (PEFT models have issues with dynamic quantization)
        model_to_quantize = baseline_models["Frozen Baseline"]
        quantizer = ModelQuantizer(model_to_quantize)
        
        def make_input():
            return {
                "input_ids": torch.randint(0, 1000, (1, 64)),
                "attention_mask": torch.ones(1, 64, dtype=torch.long)
            }
        
        def evaluate_perplexity(model):
            model.eval()
            evaluator = BenchmarkEvaluator(model, tokenizer, device="cpu")
            result = evaluator.evaluate_wikitext2(max_samples=100)
            return result["value"]
        
        quant_comparison = quantizer.full_comparison(
            input_fn=make_input,
            evaluate_fn=evaluate_perplexity,
            num_latency_runs=30
        )
        
        all_results["quantization"] = quant_comparison
        
        quant_report = generate_quantization_report(quant_comparison)
        logger.info("\n" + quant_report)
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(['FP32', 'INT8'], 
                   [quant_comparison['size']['fp32_size_mb'], 
                    quant_comparison['size']['int8_size_mb']],
                   color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('Size (MB)')
        axes[0].set_title('Model Size Comparison')
        
        axes[1].bar(['FP32', 'INT8'],
                   [quant_comparison['latency']['fp32']['mean_ms'],
                    quant_comparison['latency']['int8']['mean_ms']],
                   color=['#3498db', '#e74c3c'])
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title(f'Inference Latency (Speedup: {quant_comparison["latency"]["speedup"]:.2f}x)')
        
        if quant_comparison.get('accuracy'):
            axes[2].bar(['FP32', 'INT8'],
                       [quant_comparison['accuracy'].get('fp32_metric', 0),
                        quant_comparison['accuracy'].get('int8_metric', 0)],
                       color=['#3498db', '#e74c3c'])
            axes[2].set_ylabel('Perplexity (lower is better)')
            axes[2].set_title('Accuracy Comparison')
        
        plt.tight_layout()
        plt.savefig(figures_dir / "quantization_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("Experiment Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if "benchmark" in all_results:
        print("\nBenchmark Results (WikiText-2 Perplexity):")
        for method, result in all_results["benchmark"].items():
            if isinstance(result, dict) and "value" in result:
                print(f"  {method}: {result['value']:.2f}")
    
    if "quantization" in all_results:
        q = all_results["quantization"]
        print(f"\nQuantization Results:")
        print(f"  Size reduction: {q['size']['size_reduction_pct']:.1f}%")
        print(f"  Speedup: {q['latency']['speedup']:.2f}x")
    
    if "training_resources" in all_results:
        r = all_results["training_resources"]
        print(f"\nTraining Resources:")
        print(f"  Wall time: {r['wall_time_seconds']:.1f}s")
        print(f"  Peak memory: {r['memory_peak_mb']:.1f}MB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
