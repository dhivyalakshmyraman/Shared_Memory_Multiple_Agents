import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class MemoryEntry:
    """A single memory entry in the shared memory bank"""
    content: str
    embedding: np.ndarray
    task_id: str
    team_id: str
    timestamp: datetime
    utility_score: float = 0.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "embedding": self.embedding.tolist(),
            "task_id": self.task_id,
            "team_id": self.team_id,
            "timestamp": self.timestamp.isoformat(),
            "utility_score": self.utility_score,
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data["content"],
            embedding=np.array(data["embedding"]),
            task_id=data["task_id"],
            team_id=data["team_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            utility_score=data["utility_score"],
            usage_count=data["usage_count"]
        )

class MemoryBank:
    """Thread-safe shared memory bank for cross-team memory sharing"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: List[MemoryEntry] = []
        self.lock = threading.RLock()
        self.task_index: Dict[str, List[int]] = defaultdict(list)
        self.team_index: Dict[str, List[int]] = defaultdict(list)
    
    def add_entry(self, content: str, embedding: np.ndarray, task_id: str, team_id: str) -> int:
        """
        Add a new memory entry to the bank.
        
        Args:
            content: The memory content
            embedding: The embedding vector
            task_id: ID of the task that generated this memory
            team_id: ID of the team that generated this memory
            
        Returns:
            Index of the added entry
        """
        with self.lock:
            # Remove oldest entry if at capacity
            if len(self.entries) >= self.max_size:
                self._remove_oldest()
            
            entry = MemoryEntry(
                content=content,
                embedding=embedding,
                task_id=task_id,
                team_id=team_id,
                timestamp=datetime.now()
            )
            
            index = len(self.entries)
            self.entries.append(entry)
            
            # Update indices
            self.task_index[task_id].append(index)
            self.team_index[team_id].append(index)
            
            return index
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               exclude_team: Optional[str] = None) -> List[MemoryEntry]:
        """
        Search for memories similar to the query embedding.
        
        Args:
            query_embedding: Embedding to search for
            top_k: Number of top results to return
            exclude_team: Optional team ID to exclude from results
            
        Returns:
            List of similar memory entries
        """
        with self.lock:
            if not self.entries:
                return []
            
            # Compute similarities
            similarities = []
            for i, entry in enumerate(self.entries):
                if exclude_team and entry.team_id == exclude_team:
                    continue
                
                similarity = np.dot(entry.embedding, query_embedding) / (
                    np.linalg.norm(entry.embedding) * np.linalg.norm(query_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_entries = [self.entries[i] for i, _ in similarities[:top_k]]
            
            # Update usage counts
            for entry in top_entries:
                entry.usage_count += 1
            
            return top_entries
    
    def get_by_task(self, task_id: str) -> List[MemoryEntry]:
        """Get all memories for a specific task"""
        with self.lock:
            indices = self.task_index.get(task_id, [])
            return [self.entries[i] for i in indices]
    
    def get_by_team(self, team_id: str) -> List[MemoryEntry]:
        """Get all memories for a specific team"""
        with self.lock:
            indices = self.team_index.get(team_id, [])
            return [self.entries[i] for i in indices]
    
    def update_utility(self, index: int, utility_score: float):
        """Update utility score for a memory entry"""
        with self.lock:
            if 0 <= index < len(self.entries):
                self.entries[index].utility_score = utility_score
    
    def _remove_oldest(self):
        """Remove the oldest entry from the bank"""
        if self.entries:
            oldest_entry = self.entries[0]
            oldest_task_id = oldest_entry.task_id
            oldest_team_id = oldest_entry.team_id
            
            # Remove from main list
            self.entries.pop(0)
            
            # Update indices
            self.task_index[oldest_task_id] = [
                i - 1 for i in self.task_index[oldest_task_id] if i > 0
            ]
            self.team_index[oldest_team_id] = [
                i - 1 for i in self.team_index[oldest_team_id] if i > 0
            ]
            
            # Clean up empty indices
            if not self.task_index[oldest_task_id]:
                del self.task_index[oldest_task_id]
            if not self.team_index[oldest_team_id]:
                del self.team_index[oldest_team_id]
            
            # Shift all other indices
            for task_id in self.task_index:
                self.task_index[task_id] = [i - 1 for i in self.task_index[task_id] if i > 0]
            for team_id in self.team_index:
                self.team_index[team_id] = [i - 1 for i in self.team_index[team_id] if i > 0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        with self.lock:
            return {
                "total_entries": len(self.entries),
                "unique_tasks": len(self.task_index),
                "unique_teams": len(self.team_index),
                "avg_utility": np.mean([e.utility_score for e in self.entries]) if self.entries else 0,
                "avg_usage": np.mean([e.usage_count for e in self.entries]) if self.entries else 0
            }
    
    def clear(self):
        """Clear all entries from the memory bank"""
        with self.lock:
            self.entries.clear()
            self.task_index.clear()
            self.team_index.clear()
