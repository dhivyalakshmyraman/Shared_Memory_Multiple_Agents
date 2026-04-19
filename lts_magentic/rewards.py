import re
from typing import Dict, Any, List
import numpy as np

def compute_ragg(predicted: str, ground_truth: str, task_source: str = "unknown") -> float:
    """
    Compute reward score for a predicted answer against ground truth.
    Robust matching that handles different dataset types.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        task_source: Source dataset type (gsm8k, hotpotqa, math)
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    if not predicted or not ground_truth:
        return 0.0
    
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    
    # Exact match (highest reward)
    if pred == gt:
        return 1.0
    
    # Substring match for QA datasets
    if task_source in ["hotpotqa", "unknown"]:
        if gt in pred:
            return 0.8
        if pred in gt:
            return 0.7
    
    # Numeric matching for math datasets
    if task_source in ["gsm8k", "math"]:
        numeric_reward = _compute_numeric_reward(pred, gt)
        if numeric_reward > 0:
            return numeric_reward
    
    # Fuzzy string matching
    fuzzy_reward = _compute_fuzzy_reward(pred, gt)
    return fuzzy_reward

def _compute_numeric_reward(pred: str, gt: str) -> float:
    """Compute numeric reward for mathematical answers"""
    # Extract numbers from both strings
    pred_numbers = re.findall(r'-?\d+\.?\d*', pred)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt)
    
    if not pred_numbers or not gt_numbers:
        return 0.0
    
    # Try to match first numbers
    try:
        pred_num = float(pred_numbers[0])
        gt_num = float(gt_numbers[0])
        
        # Exact match
        if abs(pred_num - gt_num) < 1e-6:
            return 1.0
        
        # Close match (within tolerance)
        tolerance = max(abs(gt_num) * 0.01, 0.1)  # 1% or 0.1 whichever is larger
        if abs(pred_num - gt_num) <= tolerance:
            return 0.9
        
        # Order of magnitude check
        if abs(np.log10(abs(pred_num) + 1e-6) - np.log10(abs(gt_num) + 1e-6)) < 0.5:
            return 0.5
            
    except (ValueError, ZeroDivisionError):
        pass
    
    return 0.0

def _compute_fuzzy_reward(pred: str, gt: str) -> float:
    """Compute fuzzy string matching reward"""
    # Word-level overlap
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    
    if not pred_words or not gt_words:
        return 0.0
    
    # Jaccard similarity
    intersection = pred_words.intersection(gt_words)
    union = pred_words.union(gt_words)
    
    if not union:
        return 0.0
    
    jaccard = len(intersection) / len(union)
    
    # Scale Jaccard similarity to reward
    if jaccard >= 0.8:
        return 0.6
    elif jaccard >= 0.6:
        return 0.4
    elif jaccard >= 0.4:
        return 0.2
    else:
        return 0.0

def compute_memory_utility_reward(memory_usage: List[bool], 
                                 task_success: List[float],
                                 memory_efficiency: float = 0.1) -> float:
    """
    Compute reward for memory controller based on usage patterns.
    
    Args:
        memory_usage: List of whether memory was used for each task
        task_success: List of task success scores
        memory_efficiency: Weight for memory efficiency
        
    Returns:
        Controller reward score
    """
    if not memory_usage or not task_success:
        return 0.0
    
    # Reward successful memory usage
    usage_success = []
    for used, success in zip(memory_usage, task_success):
        if used:
            usage_success.append(success)
        else:
            usage_success.append(0.0)
    
    # Base reward from successful memory usage
    base_reward = np.mean(usage_success) if usage_success else 0.0
    
    # Efficiency penalty for overuse
    usage_rate = np.mean(memory_usage)
    efficiency_penalty = 0.0
    
    if usage_rate > 0.8:  # Too much memory usage
        efficiency_penalty = (usage_rate - 0.8) * 2.0
    elif usage_rate < 0.1:  # Too little memory usage
        efficiency_penalty = (0.1 - usage_rate) * 1.0
    
    # Final reward
    final_reward = base_reward - memory_efficiency * efficiency_penalty
    return max(0.0, min(1.0, final_reward))

def compute_team_diversity_reward(team_results: List[Dict[str, Any]]) -> float:
    """
    Compute reward for team diversity based on different approaches.
    
    Args:
        team_results: List of results from different teams
        
    Returns:
        Diversity reward score
    """
    if len(team_results) < 2:
        return 0.0
    
    # Extract final answers from teams
    answers = [result.get("answer", "") for result in team_results]
    
    # Compute pairwise diversity
    diversity_scores = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            score = _compute_answer_diversity(answers[i], answers[j])
            diversity_scores.append(score)
    
    # Average diversity
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    return avg_diversity

def _compute_answer_diversity(answer1: str, answer2: str) -> float:
    """Compute diversity between two answers"""
    if not answer1 or not answer2:
        return 0.0
    
    # Simple diversity based on string difference
    if answer1.lower().strip() == answer2.lower().strip():
        return 0.0
    
    # Word-level diversity
    words1 = set(answer1.lower().split())
    words2 = set(answer2.lower().split())
    
    if not words1 or not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 1.0
    
    diversity = 1.0 - (len(intersection) / len(union))
    return diversity

def evaluate_batch(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate a batch of task results.
    
    Args:
        results: List of task result dictionaries
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not results:
        return {"accuracy": 0.0, "avg_reward": 0.0, "success_rate": 0.0}
    
    rewards = []
    successes = []
    memory_usage = []
    
    for result in results:
        predicted = result.get("answer", "")
        ground_truth = result.get("ground_truth", "")
        task_source = result.get("source", "unknown")
        
        # Compute task reward
        reward = compute_ragg(predicted, ground_truth, task_source)
        rewards.append(reward)
        
        # Success threshold
        successes.append(reward >= 0.8)
        
        # Memory usage
        memory_usage.append(result.get("used_memory", False))
    
    return {
        "accuracy": np.mean(successes),
        "avg_reward": np.mean(rewards),
        "success_rate": np.mean(successes),
        "memory_usage_rate": np.mean(memory_usage),
        "total_tasks": len(results)
    }
