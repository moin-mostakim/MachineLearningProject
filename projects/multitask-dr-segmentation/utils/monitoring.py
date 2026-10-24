"""
Resource Monitoring Utilities for Multi-Task DR Learning.

Tracks:
- GPU memory usage
- CPU utilization
- Training time per epoch/batch
- Model throughput

References:
- Standard MLOps practices for resource tracking
"""

import time
import psutil
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ResourceMonitor:
    """Monitor computational resources during training."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
        self.reset()
    
    def reset(self):
        self.epoch_times = []
        self.batch_times = []
        self.gpu_memory = []
        self.cpu_memory = []
        self.samples_processed = 0
        self.start_time = None
    
    def start_epoch(self):
        """Start timing an epoch."""
        self.epoch_start = time.time()
        self.batch_times_epoch = []
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    def end_epoch(self, num_samples):
        """End timing an epoch."""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        self.samples_processed += num_samples
        
        # GPU memory
        if self.has_gpu:
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.gpu_memory.append(peak_memory)
        
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 ** 3)
        self.cpu_memory.append(cpu_memory)
        
        return epoch_time
    
    def start_batch(self):
        """Start timing a batch."""
        self.batch_start = time.time()
    
    def end_batch(self):
        """End timing a batch."""
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        self.batch_times_epoch.append(batch_time)
        return batch_time
    
    def get_current_stats(self):
        """Get current resource statistics."""
        stats = {
            'device': str(self.device),
            'cpu_percent': psutil.cpu_percent(),
            'cpu_memory_gb': psutil.Process().memory_info().rss / (1024 ** 3),
            'cpu_memory_percent': psutil.virtual_memory().percent,
        }
        
        if self.has_gpu:
            stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['gpu_memory_cached_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
            stats['gpu_name'] = torch.cuda.get_device_name(0)
        
        return stats
    
    def get_summary(self):
        """Get summary of resource usage."""
        summary = {
            'device': str(self.device),
            'total_epochs': len(self.epoch_times),
            'total_samples': self.samples_processed,
        }
        
        if self.epoch_times:
            summary['avg_epoch_time_sec'] = sum(self.epoch_times) / len(self.epoch_times)
            summary['total_time_sec'] = sum(self.epoch_times)
            summary['throughput_samples_per_sec'] = (
                self.samples_processed / sum(self.epoch_times) if sum(self.epoch_times) > 0 else 0
            )
        
        if self.batch_times:
            summary['avg_batch_time_ms'] = 1000 * sum(self.batch_times) / len(self.batch_times)
        
        if self.gpu_memory:
            summary['peak_gpu_memory_gb'] = max(self.gpu_memory)
            summary['avg_gpu_memory_gb'] = sum(self.gpu_memory) / len(self.gpu_memory)
        
        if self.cpu_memory:
            summary['peak_cpu_memory_gb'] = max(self.cpu_memory)
        
        return summary
    
    def print_summary(self):
        """Print formatted resource summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("RESOURCE MONITORING SUMMARY")
        print("=" * 60)
        
        print(f"\nDevice: {summary['device']}")
        
        if 'total_epochs' in summary:
            print(f"\nTraining Statistics:")
            print(f"  Total epochs: {summary['total_epochs']}")
            print(f"  Total samples: {summary['total_samples']:,}")
            
            if 'avg_epoch_time_sec' in summary:
                print(f"  Avg epoch time: {summary['avg_epoch_time_sec']:.2f}s")
                print(f"  Total time: {summary['total_time_sec']:.2f}s")
                print(f"  Throughput: {summary['throughput_samples_per_sec']:.1f} samples/sec")
            
            if 'avg_batch_time_ms' in summary:
                print(f"  Avg batch time: {summary['avg_batch_time_ms']:.1f}ms")
        
        if 'peak_gpu_memory_gb' in summary:
            print(f"\nGPU Memory:")
            print(f"  Peak: {summary['peak_gpu_memory_gb']:.2f} GB")
            print(f"  Average: {summary['avg_gpu_memory_gb']:.2f} GB")
        
        if 'peak_cpu_memory_gb' in summary:
            print(f"\nCPU Memory:")
            print(f"  Peak: {summary['peak_cpu_memory_gb']:.2f} GB")


class TrainingLogger:
    """Log training progress and metrics."""
    
    def __init__(self, log_dir=None, experiment_name=None):
        self.log_dir = log_dir or config.EXPERIMENTS_DIR
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': [],
            'task_weights': [],
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics, 
                  lr=None, task_weights=None):
        """Log epoch results."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        
        if lr is not None:
            self.history['lr'].append(lr)
        if task_weights is not None:
            self.history['task_weights'].append(task_weights)
    
    def print_epoch(self, epoch, train_loss, val_loss, train_metrics=None, 
                    val_metrics=None, lr=None):
        """Print epoch summary."""
        print(f"\nEpoch {epoch+1}")
        print("-" * 40)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        if lr is not None:
            print(f"  Learning Rate: {lr:.6f}")
        
        if val_metrics:
            if 'cls_kappa' in val_metrics:
                print(f"  Val Kappa:  {val_metrics['cls_kappa']:.4f}")
            if 'cls_accuracy' in val_metrics:
                print(f"  Val Acc:    {val_metrics['cls_accuracy']:.4f}")
            if 'seg_dice_mean' in val_metrics:
                print(f"  Val Dice:   {val_metrics['seg_dice_mean']:.4f}")
    
    def save_history(self, filename=None):
        """Save training history to file."""
        import json
        import numpy as np
        
        if filename is None:
            filename = os.path.join(self.log_dir, f"{self.experiment_name}_history.json")
        
        # Convert numpy types to python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif hasattr(obj, 'item') and hasattr(obj, 'shape') and obj.shape == ():
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj
        
        history_serializable = convert(self.history)
        
        with open(filename, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"History saved to {filename}")


if __name__ == "__main__":
    # Test resource monitor
    monitor = ResourceMonitor()
    
    print("Current system stats:")
    stats = monitor.get_current_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Simulate training loop
    monitor.start_epoch()
    for _ in range(10):
        monitor.start_batch()
        time.sleep(0.05)  # Simulate processing
        monitor.end_batch()
    monitor.end_epoch(100)
    
    monitor.print_summary()
