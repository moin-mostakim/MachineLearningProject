"""Main federated learning trainer orchestrating clients and server."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import copy
import time
import numpy as np
import json
from pathlib import Path

from trainers.client_trainer import ClientTrainer
from aggregation import FedAvgAggregator, FedProxAggregator, PersonalizedAggregator
from privacy.privacy_accountant import RDPAccountant


class FederatedTrainer:
    """
    Main federated learning trainer.
    
    Orchestrates:
    - Client selection
    - Local training dispatch
    - Model aggregation
    - Global evaluation
    - Privacy accounting
    """
    
    def __init__(
        self,
        model: nn.Module,
        federated_dataset,  # FederatedDataset
        config,  # Config dataclass
        device: str = "cpu"
    ):
        """
        Initialize federated trainer.
        
        Args:
            model: Global model
            federated_dataset: FederatedDataset with client data splits
            config: Configuration object
            device: Training device
        """
        self.global_model = model.to(device)
        self.federated_dataset = federated_dataset
        self.config = config
        self.device = device
        
        # Initialize aggregator based on method
        self._init_aggregator()
        
        # Privacy accounting
        self.privacy_accountant = RDPAccountant()
        self.privacy_budget = config.dp.epsilon
        self.privacy_delta = config.dp.delta
        
        # Training state
        self.current_round = 0
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
            "privacy_epsilon": [],
            "round_times": [],
            "client_metrics": []
        }
        
        # Create clients
        self.clients = self._create_clients()
        
    def _init_aggregator(self):
        """Initialize aggregator based on configuration."""
        if self.config.personalization.enabled:
            self.aggregator = PersonalizedAggregator(
                personalized_layers=self.config.personalization.local_layers,
                mixing_weight=self.config.personalization.mixing_weight,
                dp_enabled=self.config.dp.enabled,
                noise_multiplier=self.config.dp.noise_multiplier,
                max_grad_norm=self.config.dp.max_grad_norm,
                device=self.device
            )
            self.training_mode = "personalized"
        elif hasattr(self.config, 'fedprox') and self.config.fedprox.mu > 0:
            self.aggregator = FedProxAggregator(
                mu=self.config.fedprox.mu,
                device=self.device
            )
            self.training_mode = "fedprox"
        else:
            self.aggregator = FedAvgAggregator(device=self.device)
            self.training_mode = "fedavg"
    
    def _create_clients(self) -> Dict[int, ClientTrainer]:
        """Create client trainers."""
        clients = {}
        
        # Determine training mode
        if self.config.dp.enabled:
            mode = "dp"
        elif self.training_mode == "fedprox":
            mode = "fedprox"
        else:
            mode = "fedavg"
        
        # Client configuration
        client_config = {
            "learning_rate": self.config.training.learning_rate,
            "momentum": self.config.training.momentum,
            "weight_decay": self.config.training.weight_decay,
            "local_epochs": self.config.training.local_epochs,
            "mu": self.config.fedprox.mu if hasattr(self.config, 'fedprox') else 0.01,
            "noise_multiplier": self.config.dp.noise_multiplier,
            "max_grad_norm": self.config.dp.max_grad_norm,
        }
        
        for client_id in range(self.federated_dataset.num_clients):
            dataloader = self.federated_dataset.get_client_dataloader(
                client_id=client_id,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.experiment.num_workers
            )
            
            # Create local model copy
            local_model = copy.deepcopy(self.global_model)
            
            clients[client_id] = ClientTrainer(
                client_id=client_id,
                model=local_model,
                dataloader=dataloader,
                device=self.device,
                mode=mode,
                config=client_config
            )
        
        return clients
    
    def select_clients(self, num_clients: int = None) -> List[int]:
        """Select clients for current round."""
        if num_clients is None:
            num_clients = int(
                self.federated_dataset.num_clients * 
                self.config.data.participation_rate
            )
        
        num_clients = max(1, min(num_clients, self.federated_dataset.num_clients))
        
        selected = np.random.choice(
            self.federated_dataset.num_clients,
            size=num_clients,
            replace=False
        ).tolist()
        
        return selected
    
    def train_round(self, selected_clients: List[int] = None) -> Dict[str, float]:
        """
        Execute one round of federated training.
        
        Args:
            selected_clients: List of client IDs to train
            
        Returns:
            Round metrics
        """
        start_time = time.time()
        
        if selected_clients is None:
            selected_clients = self.select_clients()
        
        # Get current global model state
        global_state = self.global_model.state_dict()
        self.aggregator.set_global_model(global_state)
        
        # Collect client updates
        client_updates = []
        round_metrics = {"client_losses": [], "client_accuracies": []}
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Send global model to client
            client.set_global_model(copy.deepcopy(global_state))
            
            # Local training
            update, metrics = client.train()
            
            if self.training_mode == "personalized":
                client_updates.append((
                    client_id,
                    update,
                    metrics["num_samples"]
                ))
            else:
                client_updates.append((
                    update,
                    metrics["num_samples"]
                ))
            
            round_metrics["client_losses"].append(metrics["loss"])
            if "accuracy" in metrics:
                round_metrics["client_accuracies"].append(metrics["accuracy"])
        
        # Aggregate updates
        if self.training_mode == "personalized":
            new_global_state = self.aggregator.aggregate(client_updates)
        else:
            new_global_state = self.aggregator.aggregate(client_updates)
        
        # Update global model
        self.global_model.load_state_dict(new_global_state)
        
        # Update privacy accounting
        if self.config.dp.enabled:
            batch_size = self.config.data.batch_size
            # Approximate dataset size
            total_samples = sum(
                len(self.federated_dataset.client_indices[cid])
                for cid in selected_clients
            )
            sampling_rate = batch_size / (total_samples / len(selected_clients))
            num_steps = self.config.training.local_epochs * (total_samples // batch_size)
            
            self.privacy_accountant.accumulate(
                noise_multiplier=self.config.dp.noise_multiplier,
                sampling_rate=sampling_rate,
                num_steps=num_steps
            )
        
        round_time = time.time() - start_time
        
        # Compute round summary
        round_summary = {
            "round": self.current_round,
            "avg_loss": np.mean(round_metrics["client_losses"]),
            "avg_accuracy": np.mean(round_metrics["client_accuracies"]) if round_metrics["client_accuracies"] else 0,
            "num_clients": len(selected_clients),
            "round_time": round_time,
            "epsilon_spent": self.privacy_accountant.get_epsilon(self.privacy_delta) if self.config.dp.enabled else 0
        }
        
        self.history["train_loss"].append(round_summary["avg_loss"])
        self.history["round_times"].append(round_time)
        self.history["client_metrics"].append(round_metrics)
        
        if self.config.dp.enabled:
            self.history["privacy_epsilon"].append(round_summary["epsilon_spent"])
        
        self.current_round += 1
        
        return round_summary
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate global model on test set."""
        test_loader = self.federated_dataset.get_test_dataloader(
            batch_size=self.config.data.test_batch_size,
            num_workers=self.config.experiment.num_workers
        )
        
        self.global_model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += len(data)
        
        metrics = {
            "test_loss": total_loss / total_samples,
            "test_accuracy": 100.0 * correct / total_samples,
            "test_samples": total_samples
        }
        
        self.history["test_loss"].append(metrics["test_loss"])
        self.history["test_accuracy"].append(metrics["test_accuracy"])
        
        return metrics
    
    def train(
        self,
        num_rounds: int = None,
        eval_every: int = None,
        callback: Callable = None
    ) -> Dict:
        """
        Run full federated training.
        
        Args:
            num_rounds: Number of communication rounds
            eval_every: Evaluate every N rounds
            callback: Optional callback function(round, metrics)
            
        Returns:
            Training history
        """
        if num_rounds is None:
            num_rounds = self.config.training.num_rounds
        if eval_every is None:
            eval_every = self.config.experiment.eval_interval
        
        print(f"\nStarting Federated Learning Training")
        print(f"Method: {self.training_mode.upper()}")
        print(f"DP Enabled: {self.config.dp.enabled}")
        print(f"Clients: {self.federated_dataset.num_clients}")
        print(f"Rounds: {num_rounds}")
        print("-" * 50)
        
        for round_idx in range(num_rounds):
            # Training round
            round_metrics = self.train_round()
            
            # Evaluation
            if (round_idx + 1) % eval_every == 0 or round_idx == num_rounds - 1:
                eval_metrics = self.evaluate()
                
                print(
                    f"Round {round_idx + 1:3d}/{num_rounds} | "
                    f"Train Loss: {round_metrics['avg_loss']:.4f} | "
                    f"Test Acc: {eval_metrics['test_accuracy']:.2f}% | "
                    f"Time: {round_metrics['round_time']:.2f}s"
                    + (f" | ε: {round_metrics['epsilon_spent']:.2f}" if self.config.dp.enabled else "")
                )
            
            # Check privacy budget
            if self.config.dp.enabled:
                current_epsilon = self.privacy_accountant.get_epsilon(self.privacy_delta)
                if current_epsilon > self.privacy_budget:
                    print(f"\nPrivacy budget exhausted at round {round_idx + 1}")
                    print(f"Epsilon spent: {current_epsilon:.2f} > {self.privacy_budget}")
                    break
            
            # Callback
            if callback is not None:
                callback(round_idx, round_metrics)
        
        # Final evaluation
        final_metrics = self.evaluate()
        print("\n" + "=" * 50)
        print(f"Final Test Accuracy: {final_metrics['test_accuracy']:.2f}%")
        print(f"Final Test Loss: {final_metrics['test_loss']:.4f}")
        
        if self.config.dp.enabled:
            final_epsilon = self.privacy_accountant.get_epsilon(self.privacy_delta)
            print(f"Total Privacy Spent: (ε={final_epsilon:.2f}, δ={self.privacy_delta})")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state": self.global_model.state_dict(),
            "round": self.current_round,
            "history": self.history,
            "config": self.config.to_dict()
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint["model_state"])
        self.current_round = checkpoint["round"]
        self.history = checkpoint["history"]
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from round {self.current_round}")
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        if self.config.dp.enabled:
            epsilon = self.privacy_accountant.get_epsilon(self.privacy_delta)
            return epsilon, self.privacy_delta
        return 0.0, 0.0
