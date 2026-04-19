"""
Google Colab trainer for LTS Memory Controller
Designed to run on Colab T4 GPU for training the memory controller
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from controller import MemoryController
from rl_trainer import TrainingExample, ControllerDataset

class ColabTrainer:
    """Google Colab optimized trainer for memory controller"""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Colab trainer.
        
        Args:
            device: Device to use ('cuda' for GPU, 'cpu' for CPU)
        """
        self.device = device
        self.controller = None
        self.training_history = []
        
        # Check if GPU is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_trajectories(self, trajectories_path: str) -> List[Dict[str, Any]]:
        """
        Load trajectories from local execution.
        
        Args:
            trajectories_path: Path to trajectories.pt file
            
        Returns:
            List of trajectory dictionaries
        """
        if not os.path.exists(trajectories_path):
            raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")
        
        trajectories = torch.load(trajectories_path, map_location=self.device, weights_only=False)
        print(f"Loaded {len(trajectories)} trajectories")
        
        return trajectories
    
    def create_controller(self, embedding_dim: int = 384, hidden_dim: int = 256, 
                        num_layers: int = 2) -> MemoryController:
        """
        Create a new memory controller.
        
        Args:
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            
        Returns:
            Created controller
        """
        self.controller = MemoryController(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Move to device
        self.controller.to(self.device)
        
        print(f"Created controller with {self.controller.get_parameters_count():,} parameters")
        return self.controller
    
    def load_controller(self, controller_path: str) -> MemoryController:
        """
        Load existing controller.
        
        Args:
            controller_path: Path to controller.pt file
            
        Returns:
            Loaded controller
        """
        self.controller = MemoryController.load(controller_path)
        self.controller.to(self.device)
        
        print(f"Loaded controller from {controller_path}")
        print(f"Controller has {self.controller.get_parameters_count():,} parameters")
        
        return self.controller
    
    def prepare_training_data(self, trajectories: List[Dict[str, Any]], 
                            embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                            sample_rate: float = 1.0) -> List[TrainingExample]:
        """
        Prepare training data from trajectories.
        
        Args:
            trajectories: List of trajectories
            embedder_model: Embedding model to use
            sample_rate: Fraction of trajectories to sample (for faster training)
            
        Returns:
            List of training examples
        """
        # Sample trajectories if needed
        if sample_rate < 1.0:
            num_samples = int(len(trajectories) * sample_rate)
            trajectories = np.random.choice(trajectories, num_samples, replace=False)
            print(f"Sampled {len(trajectories)} trajectories for training")
        
        # Initialize embedder
        from embedder import Embedder
        embedder = Embedder(embedder_model)
        
        # Extract training examples
        examples = []
        for trajectory in trajectories:
            trajectory_examples = self._extract_examples_from_trajectory(trajectory, embedder)
            examples.extend(trajectory_examples)
        
        print(f"Extracted {len(examples)} training examples")
        return examples
    
    def _extract_examples_from_trajectory(self, trajectory: Dict[str, Any], 
                                        embedder) -> List[TrainingExample]:
        """Extract training examples from a single trajectory"""
        examples = []
        
        # Get task information
        task_info = trajectory.get("task", {})
        question = task_info.get("question", "")
        
        if not question:
            return examples
        
        # Generate query embedding
        query_embedding = embedder.embed_single(question)
        
        # Process trajectory steps
        trajectory_steps = trajectory.get("trajectory", [])
        final_reward = trajectory.get("reward", 0.0)
        
        for step in trajectory_steps:
            if step.get("step") == "memory_decision":
                decision_details = step.get("decision_details", [])
                
                for decision in decision_details:
                    memory_content = decision.get("memory", {}).get("content", "")
                    used_memory = decision.get("decision", False)
                    probability = decision.get("probability", 0.5)
                    
                    if not memory_content:
                        continue
                    
                    # Generate memory embedding
                    memory_embedding = embedder.embed_single(memory_content)
                    
                    # Determine label and reward
                    if used_memory:
                        label = 1 if final_reward > 0.5 else 0
                        reward = final_reward * probability
                    else:
                        label = 0
                        reward = (1.0 - final_reward) * (1.0 - probability)
                    
                    example = TrainingExample(
                        query_embedding=query_embedding,
                        memory_embedding=memory_embedding,
                        label=label,
                        reward=reward
                    )
                    
                    examples.append(example)
        
        return examples
    
    def train(self, train_examples: List[TrainingExample], 
              val_examples: List[TrainingExample] = None,
              learning_rate: float = 1e-4,
              batch_size: int = 64,
              epochs: int = 20,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the memory controller on Colab GPU.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            validation_split: Fraction of training data for validation
            
        Returns:
            Training results
        """
        if self.controller is None:
            raise ValueError("Controller not initialized. Call create_controller() or load_controller() first")
        
        # Split training data if no validation examples provided
        if not val_examples and validation_split > 0:
            num_val = int(len(train_examples) * validation_split)
            val_examples = train_examples[-num_val:]
            train_examples = train_examples[:-num_val]
            print(f"Split data: {len(train_examples)} train, {len(val_examples)} validation")
        
        # Create datasets and dataloaders
        train_dataset = ControllerDataset(train_examples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        val_loader = None
        if val_examples:
            val_dataset = ControllerDataset(val_examples)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.controller.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        print(f"Starting training on {self.device}")
        print(f"Train examples: {len(train_examples)}, Batch size: {batch_size}, Epochs: {epochs}")
        
        training_history = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.controller.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move data to device
                query_emb = batch["query_embedding"].to(self.device)
                memory_emb = batch["memory_embedding"].to(self.device)
                labels = batch["label"].squeeze().to(self.device)
                rewards = batch["reward"].squeeze().to(self.device)
                
                # Forward pass
                probabilities = self.controller(query_emb, memory_emb)
                
                # Ensure tensors have same shape
                probabilities = probabilities.squeeze()
                labels = labels.squeeze()
                
                # Compute weighted loss
                loss = criterion(probabilities, labels)
                weighted_loss = loss * rewards
                mean_loss = weighted_loss.mean()
                
                # Backward pass
                mean_loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += mean_loss.item()
                predictions = (probabilities > 0.5).float()
                
                # Handle scalar tensors
                if labels.dim() == 0:
                    train_correct += (predictions == labels).item()
                    train_total += 1
                else:
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                self.controller.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        query_emb = batch["query_embedding"].to(self.device)
                        memory_emb = batch["memory_embedding"].to(self.device)
                        labels = batch["label"].squeeze().to(self.device)
                        
                        probabilities = self.controller(query_emb, memory_emb)
                        
                        # Ensure tensors have same shape
                        probabilities = probabilities.squeeze()
                        labels = labels.squeeze()
                        
                        loss = criterion(probabilities, labels)
                        val_loss += loss.item()
                        
                        predictions = (probabilities > 0.5).float()
                        
                        # Handle scalar tensors
                        if labels.dim() == 0:
                            val_correct += (predictions == labels).item()
                            val_total += 1
                        else:
                            val_correct += (predictions == labels).sum().item()
                            val_total += labels.size(0)
                
                val_metrics = {
                    "val_loss": val_loss / len(val_loader),
                    "val_accuracy": val_correct / val_total if val_total > 0 else 0.0
                }
                
                # Save best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_controller("best_controller.pt")
            
            # Record epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_correct / train_total if train_total > 0 else 0.0,
                **val_metrics
            }
            training_history.append(epoch_metrics)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}, "
                  f"Accuracy: {epoch_metrics['train_accuracy']:.3f}")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Accuracy: {val_metrics['val_accuracy']:.3f}")
        
        self.training_history = training_history
        
        # Final evaluation
        final_metrics = self._evaluate_on_all_data(train_examples + (val_examples or []))
        
        print(f"\nTraining completed!")
        print(f"Final metrics: {final_metrics}")
        
        return {
            "history": training_history,
            "final_metrics": final_metrics
        }
    
    def _evaluate_on_all_data(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate controller on all examples"""
        if not examples:
            return {}
        
        self.controller.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for example in examples:
                query_tensor = torch.FloatTensor(example.query_embedding).unsqueeze(0).to(self.device)
                memory_tensor = torch.FloatTensor(example.memory_embedding).unsqueeze(0).to(self.device)
                label_tensor = torch.FloatTensor([example.label]).to(self.device)
                
                probability = self.controller(query_tensor, memory_tensor)
                loss = criterion(probability, label_tensor)
                
                total_loss += loss.item()
                prediction = (probability > 0.5).float()
                if prediction.item() == example.label:
                    correct += 1
                total += 1
        
        return {
            "loss": total_loss / len(examples),
            "accuracy": correct / total if total > 0 else 0.0,
            "total_examples": total
        }
    
    def save_controller(self, path: str):
        """Save the trained controller"""
        if self.controller is None:
            raise ValueError("No controller to save")
        
        # Move to CPU for saving
        controller_cpu = MemoryController(
            embedding_dim=self.controller.embedding_dim,
            hidden_dim=self.controller.hidden_dim
        )
        controller_cpu.load_state_dict(self.controller.state_dict())
        controller_cpu.save(path)
        
        print(f"Controller saved to {path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            print("No training history to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_history)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        if 'val_loss' in df.columns:
            axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', marker='o')
        if 'val_accuracy' in df.columns:
            axes[1].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_training_report(self) -> str:
        """Generate a training report"""
        if not self.training_history:
            return "No training history available"
        
        final_metrics = self.training_history[-1]
        initial_metrics = self.training_history[0]
        
        report = f"""
# Colab Training Report

## Training Configuration
- Device: {self.device}
- Total Epochs: {len(self.training_history)}
- Controller Parameters: {self.controller.get_parameters_count():,}

## Final Performance
- Final Training Loss: {final_metrics['train_loss']:.4f}
- Final Training Accuracy: {final_metrics['train_accuracy']:.3f}
"""
        
        if 'val_loss' in final_metrics:
            report += f"""
- Final Validation Loss: {final_metrics['val_loss']:.4f}
- Final Validation Accuracy: {final_metrics['val_accuracy']:.3f}
"""
        
        report += f"""
## Learning Progress
- Loss Reduction: {initial_metrics['train_loss'] - final_metrics['train_loss']:.4f}
- Accuracy Improvement: {final_metrics['train_accuracy'] - initial_metrics['train_accuracy']:.3f}
"""
        
        return report

# Example usage for Colab
def colab_training_example():
    """Example of how to use ColabTrainer in Google Colab"""
    
    # Initialize trainer
    trainer = ColabTrainer(device="cuda")
    
    # Load trajectories from local execution
    trajectories = trainer.load_trajectories("trajectories.pt")
    
    # Prepare training data
    train_examples = trainer.prepare_training_data(trajectories, sample_rate=0.5)
    
    # Create or load controller
    controller = trainer.create_controller(embedding_dim=384, hidden_dim=256)
    
    # Train the controller
    results = trainer.train(
        train_examples=train_examples,
        learning_rate=1e-4,
        batch_size=64,
        epochs=20
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Generate report
    report = trainer.generate_training_report()
    print(report)
    
    # Save trained controller
    trainer.save_controller("trained_controller.pt")
    
    return results
