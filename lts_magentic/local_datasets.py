import json
import os
from typing import List, Dict, Any

def load_all_tasks(path: str = "data/tasks.json") -> List[Dict[str, Any]]:
    """
    Load all tasks from the local JSON dataset with new multi-step format.
    
    Args:
        path: Path to the tasks JSON file
        
    Returns:
        List of task dictionaries with subtasks and shared_entities
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for item in data:
        tasks.append({
            "id": item["id"],
            "question": item["question"],
            "subtasks": item.get("subtasks", []),
            "shared_entities": item.get("shared_entities", []),
            "ground_truth": item["expected_answer"],
            "type": item.get("type", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
            "source": item.get("source", "unknown")
        })

    return tasks

def get_tasks_by_source(tasks: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Filter tasks by source dataset.
    
    Args:
        tasks: List of all tasks
        source: Source to filter by (e.g., "gsm8k", "hotpotqa", "math")
        
    Returns:
        List of tasks from the specified source
    """
    return [task for task in tasks if task["source"] == source]

def expand_into_steps(task):
    """
    Converts task into explicit reasoning steps for agents.
    
    Args:
        task: Task dictionary with subtasks
        
    Returns:
        List of step strings
    """
    steps = []

    if task["subtasks"]:
        for i, step in enumerate(task["subtasks"]):
            steps.append(f"Step {i+1}: {step}")
    else:
        steps.append(task["question"])

    return steps

def get_task_stats(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get statistics about the task dataset.
    
    Args:
        tasks: List of tasks
        
    Returns:
        Dictionary with task counts by source
    """
    stats = {}
    for task in tasks:
        source = task["source"]
        stats[source] = stats.get(source, 0) + 1
    
    stats["total"] = len(tasks)
    return stats
