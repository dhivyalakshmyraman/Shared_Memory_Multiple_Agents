import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any
import os

class MemoryController(nn.Module):
    """Neural controller for memory sharing decisions"""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256, num_layers: int = 2):
        """
        Initialize the memory controller.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(MemoryController, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Query + Memory embeddings
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer for binary decision (YES/NO)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, query_embedding: torch.Tensor, 
                memory_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to decide whether to use memory.
        
        Args:
            query_embedding: Query task embedding
            memory_embedding: Memory entry embedding
            
        Returns:
            Probability of using the memory (0-1)
        """
        # Concatenate embeddings
        combined = torch.cat([query_embedding, memory_embedding], dim=-1)
        
        # Pass through network
        logits = self.network(combined)
        
        # Apply sigmoid for probability
        probability = torch.sigmoid(logits)
        
        return probability.squeeze(-1)
    
    def predict(self, query_embedding: np.ndarray, 
                memory_embedding: np.ndarray, 
                bootstrap: bool = False) -> float:
        """
        Predict probability of memory usage (for training).
        
        Args:
            query_embedding: Query task embedding
            memory_embedding: Memory entry embedding
            bootstrap: Use random predictions for untrained controller
            
        Returns:
            Probability of using the memory (0-1)
        """
        if bootstrap:
            # Bootstrap with random probability
            import random
            return random.random()
        
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0)
            memory_tensor = torch.FloatTensor(memory_embedding).unsqueeze(0)
            
            # Get probability
            probability = self.forward(query_tensor, memory_tensor)
            
            return probability.item()
    
    def decide(self, query_embedding: np.ndarray, 
               memory_embedding: np.ndarray, 
               threshold: float = 0.5,
               bootstrap: bool = False) -> Tuple[bool, float]:
        """
        Make a binary decision about memory usage.
        
        Args:
            query_embedding: Query task embedding
            memory_embedding: Memory entry embedding
            threshold: Decision threshold
            
        Returns:
            Tuple of (decision, probability)
        """
        if bootstrap:
            # Bootstrap with random decision
            import random
            probability_val = random.random()
            decision = probability_val >= threshold
            return decision, probability_val
        
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0)
            memory_tensor = torch.FloatTensor(memory_embedding).unsqueeze(0)
            
            # Get probability
            probability = self.forward(query_tensor, memory_tensor)
            probability_val = probability.item()
            
            # Make decision
            decision = probability_val >= threshold
            
            return decision, probability_val
    
    def batch_decide(self, query_embedding: np.ndarray, 
                    memory_embeddings: List[np.ndarray],
                    threshold: float = 0.5) -> List[Tuple[bool, float]]:
        """
        Make batch decisions for multiple memory entries.
        
        Args:
            query_embedding: Query task embedding
            memory_embeddings: List of memory entry embeddings
            threshold: Decision threshold
            
        Returns:
            List of (decision, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0)
            memory_tensors = torch.FloatTensor(np.array(memory_embeddings))
            
            # Expand query to match batch size
            query_batch = query_tensor.expand(len(memory_embeddings), -1)
            
            # Get probabilities
            probabilities = self.forward(query_batch, memory_tensors)
            probabilities_vals = probabilities.cpu().numpy()
            
            # Make decisions
            decisions = probabilities_vals >= threshold
            
            return list(zip(decisions, probabilities_vals))
    
    def save(self, path: str):
        """Save the controller model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "MemoryController":
        """Load a saved controller model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Controller file not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        
        controller = cls(
            embedding_dim=config.get('embedding_dim', 384),
            hidden_dim=config.get('hidden_dim', 256)
        )
        
        controller.load_state_dict(checkpoint['model_state_dict'])
        controller.is_trained = True  # Mark as trained
        return controller
    
    def get_parameters_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
