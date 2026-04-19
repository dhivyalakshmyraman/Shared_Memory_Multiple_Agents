import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LTSConfig:
    """Configuration for Learning to Share system"""
    
    # Model configurations
    orchestrator_model: str = "groq/llama-3.3-70b"
    worker_models: list = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # System configurations
    max_parallel_teams: int = 4  # NUM_TEAMS = 4
    memory_bank_size: int = 1000
    embedding_dim: int = 384
    max_steps_per_team: int = 20  # MAX_STEPS_PER_TEAM = 20
    
    # Task configurations
    num_tasks: int = 30  # NUM_TASKS = 30
    
    # Training configurations
    controller_hidden_dim: int = 256
    controller_layers: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    
    # File paths
    dataset_path: str = "data/tasks.json"
    controller_path: str = "controller.pt"
    trajectories_path: str = "trajectories.pt"
    
    # API configurations
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    
    def __post_init__(self):
        if self.worker_models is None:
            self.worker_models = ["qwen2.5:7b", "llama3.2:3b"]
        
        # Load from environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LTSConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "orchestrator_model": self.orchestrator_model,
            "worker_models": self.worker_models,
            "embedding_model": self.embedding_model,
            "max_parallel_teams": self.max_parallel_teams,
            "memory_bank_size": self.memory_bank_size,
            "embedding_dim": self.embedding_dim,
            "controller_hidden_dim": self.controller_hidden_dim,
            "controller_layers": self.controller_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dataset_path": self.dataset_path,
            "controller_path": self.controller_path,
            "trajectories_path": self.trajectories_path,
            "groq_api_key": self.groq_api_key,
            "ollama_base_url": self.ollama_base_url
        }

# Global configuration instance
config = LTSConfig()
