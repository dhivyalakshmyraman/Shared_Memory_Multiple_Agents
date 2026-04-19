import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json
from dataclasses import dataclass

from controller import MemoryController
from rewards import compute_memory_utility_reward
from embedder import Embedder

@dataclass
class TrainingExample:
    """Single training example for the memory controller"""
    query_embedding: np.ndarray
    memory_embedding: np.ndarray
    label: int  # 1 if memory should be used, 0 otherwise
    reward: float  # Reward received for this decision

class ControllerDataset(Dataset):
    """Dataset for training the memory controller"""
    
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "query_embedding": torch.FloatTensor(example.query_embedding),
            "memory_embedding": torch.FloatTensor(example.memory_embedding),
            "label": torch.FloatTensor([example.label]),
            "reward": torch.FloatTensor([example.reward])
        }

class RLTrainer:
    """Reinforcement Learning trainer for the memory controller"""
    
    def __init__(self, controller: MemoryController, embedder: Embedder, config: Dict[str, Any]):
        """
        Initialize RL trainer.
        
        Args:
            controller: Memory controller to train
            embedder: Text embedder for generating embeddings
            config: Training configuration
        """
        self.controller = controller
        self.embedder = embedder
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 10)
        
        # Optimizer
        self.optimizer = optim.Adam(self.controller.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = []
        
    def collect_training_data(self, trajectories: List[Dict[str, Any]]) -> List[TrainingExample]:
        """
        Collect training examples from execution trajectories.
        
        Args:
            trajectories: List of execution trajectories
            
        Returns:
            List of training examples
        """
        examples = []
        
        for trajectory in trajectories:
            examples.extend(self._extract_examples_from_trajectory(trajectory))
        
        return examples
    
    def _extract_examples_from_trajectory(self, trajectory: Dict[str, Any]) -> List[TrainingExample]:
        """Extract training examples from a single trajectory"""
        examples = []
        
        # Get task information
        task_info = trajectory.get("task", {})
        question = task_info.get("question", "")
        
        if not question:
            return examples
        
        # Generate query embedding
        query_embedding = self.embedder.embed_single(question)
        
        # Process memory decisions
        memory_decisions = trajectory.get("memory_decisions", [])
        final_reward = trajectory.get("reward", 0.0)
        
        for decision in memory_decisions:
            memory_content = decision.get("memory_content", "")
            used_memory = decision.get("used_memory", False)
            decision_probability = decision.get("probability", 0.5)
            
            if not memory_content:
                continue
            
            # Generate memory embedding
            memory_embedding = self.embedder.embed_single(memory_content)
            
            # Determine label based on success
            label = 1 if used_memory and final_reward > 0.5 else 0
            
            # Adjust reward based on decision quality
            if used_memory:
                reward = final_reward * decision_probability
            else:
                reward = (1.0 - final_reward) * (1.0 - decision_probability)
            
            example = TrainingExample(
                query_embedding=query_embedding,
                memory_embedding=memory_embedding,
                label=label,
                reward=reward
            )
            
            examples.append(example)
        
        return examples
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.controller.train()
        total_loss = 0.0
        total_reward = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Get batch data
            query_embeddings = batch["query_embedding"]
            memory_embeddings = batch["memory_embedding"]
            labels = batch["label"].squeeze()
            rewards = batch["reward"].squeeze()
            
            # Forward pass
            probabilities = self.controller(query_embeddings, memory_embeddings)
            
            # Compute loss (weighted by rewards)
            loss = self.criterion(probabilities, labels)
            weighted_loss = loss * rewards
            mean_loss = weighted_loss.mean()
            
            # Backward pass
            mean_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += mean_loss.item()
            total_reward += rewards.mean().item()
            
            # Accuracy (threshold at 0.5)
            predictions = (probabilities > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "reward": avg_reward,
            "accuracy": accuracy
        }
    
    def validate(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """
        Validate the controller on validation data.
        
        Args:
            examples: Validation examples
            
        Returns:
            Validation metrics
        """
        if not examples:
            return {"loss": 0.0, "accuracy": 0.0, "reward": 0.0}
        
        self.controller.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for example in examples:
                # Convert to tensors
                query_tensor = torch.FloatTensor(example.query_embedding).unsqueeze(0)
                memory_tensor = torch.FloatTensor(example.memory_embedding).unsqueeze(0)
                label_tensor = torch.FloatTensor([example.label])
                
                # Forward pass
                probability = self.controller(query_tensor, memory_tensor)
                
                # Compute loss
                loss = self.criterion(probability, label_tensor)
                total_loss += loss.item()
                
                # Accuracy
                prediction = (probability > 0.5).float()
                if prediction.item() == example.label:
                    correct_predictions += 1
                total_predictions += 1
        
        avg_loss = total_loss / len(examples)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_reward = np.mean([ex.reward for ex in examples])
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "reward": avg_reward
        }
    
    def train(self, train_examples: List[TrainingExample], 
              val_examples: List[TrainingExample] = None) -> Dict[str, Any]:
        """
        Train the memory controller.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples (optional)
            
        Returns:
            Training results
        """
        if not train_examples:
            print("No training examples available")
            return {"history": [], "final_metrics": {}}
        
        # Create datasets and dataloaders
        train_dataset = ControllerDataset(train_examples)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if val_examples:
            val_dataset = ControllerDataset(val_examples)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Starting training with {len(train_examples)} examples")
        print(f"Batch size: {self.batch_size}, Epochs: {self.epochs}")
        
        training_history = []
        
        for epoch in range(self.epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate if validation data available
            val_metrics = {}
            if val_examples:
                val_metrics = self.validate(val_examples)
            
            # Record history
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_reward": train_metrics["reward"],
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            training_history.append(epoch_record)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.3f}, "
                  f"Reward: {train_metrics['reward']:.3f}")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Accuracy: {val_metrics['accuracy']:.3f}")
        
        # Final validation
        final_metrics = self.validate(train_examples + (val_examples or []))
        
        print(f"Training completed. Final metrics:")
        print(f"  Loss: {final_metrics['loss']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Reward: {final_metrics['reward']:.3f}")
        
        return {
            "history": training_history,
            "final_metrics": final_metrics
        }
    
    def save_training_data(self, examples: List[TrainingExample], path: str):
        """Save training examples to file"""
        data = []
        for example in examples:
            data.append({
                "query_embedding": example.query_embedding.tolist(),
                "memory_embedding": example.memory_embedding.tolist(),
                "label": example.label,
                "reward": example.reward
            })
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(examples)} training examples to {path}")
    
    def load_training_data(self, path: str) -> List[TrainingExample]:
        """Load training examples from file"""
        if not os.path.exists(path):
            return []
        
        with open(path, "r") as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = TrainingExample(
                query_embedding=np.array(item["query_embedding"]),
                memory_embedding=np.array(item["memory_embedding"]),
                label=item["label"],
                reward=item["reward"]
            )
            examples.append(example)
        
        print(f"Loaded {len(examples)} training examples from {path}")
        return examples
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process"""
        if not self.training_history:
            return {}
        
        # Compute summary statistics
        final_epoch = self.training_history[-1]
        initial_epoch = self.training_history[0]
        
        accuracy_improvement = final_epoch["train_accuracy"] - initial_epoch["train_accuracy"]
        loss_reduction = initial_epoch["train_loss"] - final_epoch["train_loss"]
        
        return {
            "total_epochs": len(self.training_history),
            "final_accuracy": final_epoch["train_accuracy"],
            "final_loss": final_epoch["train_loss"],
            "accuracy_improvement": accuracy_improvement,
            "loss_reduction": loss_reduction,
            "training_examples": len(self.training_history) * self.batch_size
        }
