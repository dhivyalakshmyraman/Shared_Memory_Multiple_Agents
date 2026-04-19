import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    """Text embedder using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM-L6-v2
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Using random embeddings as fallback")
            self.model = None
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Embedding vectors as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=8,
                    show_progress_bar=False
                )
                return embeddings
            except Exception as e:
                print(f"Warning: Embedding failed: {e}")
                print("Using random embeddings as fallback")
        
        # Fallback to random embeddings
        return self._random_embeddings(len(texts))
    
    def _random_embeddings(self, num_texts: int) -> np.ndarray:
        """Generate random embeddings as fallback"""
        return np.random.randn(num_texts, self.embedding_dim).astype(np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Single embedding vector
        """
        embedding = self.embed(text)
        return embedding[0] if len(embedding.shape) > 1 else embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are 1D
        if len(embedding1.shape) > 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1:
            embedding2 = embedding2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[int]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            Indices of most similar embeddings
        """
        # Ensure query is 1D
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
        
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            if len(candidate.shape) > 1:
                candidate = candidate.flatten()
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k indices
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:top_k]]
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded successfully"""
        return self.model is not None
