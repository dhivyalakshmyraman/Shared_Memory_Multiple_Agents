from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass

from parallel_runner import ParallelExecutionResult
from rewards import compute_ragg, compute_team_diversity_reward, evaluate_batch

@dataclass
class AggregatedResult:
    """Result from aggregating multiple parallel executions"""
    total_tasks: int
    successful_tasks: int
    accuracy: float
    avg_reward: float
    memory_usage_rate: float
    team_diversity: float
    avg_execution_time: float
    memory_bank_efficiency: float
    per_task_results: List[Dict[str, Any]]

class ResultAggregator:
    """Aggregates results from parallel team executions"""
    
    def __init__(self):
        self.results_history = []
    
    def aggregate_batch_results(self, batch_results: List[ParallelExecutionResult], 
                               tasks: List[Dict[str, Any]]) -> AggregatedResult:
        """
        Aggregate results from a batch of task executions.
        
        Args:
            batch_results: List of ParallelExecutionResult
            tasks: Original task list
            
        Returns:
            AggregatedResult with comprehensive metrics
        """
        if len(batch_results) != len(tasks):
            raise ValueError("Number of results must match number of tasks")
        
        per_task_results = []
        successful_tasks = 0
        total_execution_time = 0.0
        memory_usage_counts = []
        rewards = []
        
        # Process each task result
        for result, task in zip(batch_results, tasks):
            task_result = self._process_single_task_result(result, task)
            per_task_results.append(task_result)
            
            if task_result["success"]:
                successful_tasks += 1
            
            total_execution_time += result.execution_time
            memory_usage_counts.append(task_result["memory_used"])
            rewards.append(task_result["reward"])
        
        # Compute aggregate metrics
        total_tasks = len(tasks)
        accuracy = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        avg_reward = np.mean(rewards) if rewards else 0.0
        memory_usage_rate = np.mean(memory_usage_counts) if memory_usage_counts else 0.0
        avg_execution_time = total_execution_time / total_tasks if total_tasks > 0 else 0.0
        
        # Compute team diversity
        team_diversity = self._compute_overall_team_diversity(batch_results)
        
        # Compute memory bank efficiency
        memory_bank_efficiency = self._compute_memory_bank_efficiency(batch_results)
        
        aggregated = AggregatedResult(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            accuracy=accuracy,
            avg_reward=avg_reward,
            memory_usage_rate=memory_usage_rate,
            team_diversity=team_diversity,
            avg_execution_time=avg_execution_time,
            memory_bank_efficiency=memory_bank_efficiency,
            per_task_results=per_task_results
        )
        
        # Store in history
        self.results_history.append(aggregated)
        
        return aggregated
    
    def _process_single_task_result(self, result: ParallelExecutionResult, 
                                  task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task execution result"""
        # Get ground truth
        ground_truth = task.get("ground_truth", "")
        task_source = task.get("source", "unknown")
        
        # Evaluate aggregated answer
        reward = compute_ragg(result.aggregated_answer, ground_truth, task_source)
        success = reward >= 0.8
        
        # Check if any team used memory
        memory_used = any(team_result.memory_used for team_result in result.team_results)
        
        return {
            "task_id": result.task_id,
            "question": task.get("question", ""),
            "ground_truth": ground_truth,
            "aggregated_answer": result.aggregated_answer,
            "reward": reward,
            "success": success,
            "memory_used": memory_used,
            "execution_time": result.execution_time,
            "num_successful_teams": sum(1 for r in result.team_results if r.success),
            "team_answers": [r.answer for r in result.team_results],
            "memory_bank_size": result.memory_bank_stats.get("total_entries", 0)
        }
    
    def _compute_overall_team_diversity(self, batch_results: List[ParallelExecutionResult]) -> float:
        """Compute team diversity across all tasks"""
        all_team_diversities = []
        
        for result in batch_results:
            if len(result.team_results) >= 2:
                # Convert team results to expected format
                team_results_data = [
                    {"answer": r.answer} for r in result.team_results
                ]
                diversity = compute_team_diversity_reward(team_results_data)
                all_team_diversities.append(diversity)
        
        return np.mean(all_team_diversities) if all_team_diversities else 0.0
    
    def _compute_memory_bank_efficiency(self, batch_results: List[ParallelExecutionResult]) -> float:
        """Compute memory bank efficiency metrics"""
        if not batch_results:
            return 0.0
        
        # Track memory bank growth and usage
        initial_size = batch_results[0].memory_bank_stats.get("total_entries", 0)
        final_size = batch_results[-1].memory_bank_stats.get("total_entries", 0)
        
        # Growth rate
        growth_rate = (final_size - initial_size) / max(initial_size, 1)
        
        # Average utility score
        utilities = []
        for result in batch_results:
            stats = result.memory_bank_stats
            if "avg_utility" in stats:
                utilities.append(stats["avg_utility"])
        
        avg_utility = np.mean(utilities) if utilities else 0.0
        
        # Combine growth and utility
        efficiency = 0.7 * min(growth_rate, 1.0) + 0.3 * avg_utility
        return min(efficiency, 1.0)
    
    def compare_results(self, baseline_results: AggregatedResult, 
                        new_results: AggregatedResult) -> Dict[str, float]:
        """
        Compare two aggregated results.
        
        Args:
            baseline_results: Baseline aggregated results
            new_results: New aggregated results
            
        Returns:
            Dictionary of improvement metrics
        """
        return {
            "accuracy_improvement": new_results.accuracy - baseline_results.accuracy,
            "reward_improvement": new_results.avg_reward - baseline_results.avg_reward,
            "memory_efficiency_change": new_results.memory_usage_rate - baseline_results.memory_usage_rate,
            "diversity_change": new_results.team_diversity - baseline_results.team_diversity,
            "execution_time_change": new_results.avg_execution_time - baseline_results.avg_execution_time,
            "memory_bank_efficiency_change": new_results.memory_bank_efficiency - baseline_results.memory_bank_efficiency
        }
    
    def get_learning_curve(self) -> List[Dict[str, float]]:
        """Get learning curve data from results history"""
        if not self.results_history:
            return []
        
        curve_data = []
        for i, result in enumerate(self.results_history):
            curve_data.append({
                "batch": i + 1,
                "accuracy": result.accuracy,
                "avg_reward": result.avg_reward,
                "memory_usage_rate": result.memory_usage_rate,
                "team_diversity": result.team_diversity,
                "memory_bank_efficiency": result.memory_bank_efficiency
            })
        
        return curve_data
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze common failure patterns"""
        if not self.results_history:
            return {}
        
        failure_analysis = {
            "common_failure_sources": Counter(),
            "memory_vs_no_memory": {"memory_used": 0, "no_memory": 0},
            "execution_time_outliers": []
        }
        
        total_failures = 0
        total_memory_failures = 0
        total_no_memory_failures = 0
        
        for result in self.results_history:
            for task_result in result.per_task_results:
                if not task_result["success"]:
                    total_failures += 1
                    
                    # Extract source from question (simplified)
                    question = task_result["question"].lower()
                    if "math" in question or "calculate" in question:
                        failure_analysis["common_failure_sources"]["math"] += 1
                    elif "who" in question or "what" in question or "where" in question:
                        failure_analysis["common_failure_sources"]["qa"] += 1
                    else:
                        failure_analysis["common_failure_sources"]["other"] += 1
                    
                    # Track memory usage in failures
                    if task_result["memory_used"]:
                        total_memory_failures += 1
                    else:
                        total_no_memory_failures += 1
                    
                    # Track execution time outliers
                    if task_result["execution_time"] > 10.0:  # 10 seconds threshold
                        failure_analysis["execution_time_outliers"].append({
                            "task_id": task_result["task_id"],
                            "execution_time": task_result["execution_time"]
                        })
        
        if total_failures > 0:
            failure_analysis["memory_vs_no_memory"]["memory_used"] = total_memory_failures / total_failures
            failure_analysis["memory_vs_no_memory"]["no_memory"] = total_no_memory_failures / total_failures
        
        return failure_analysis
    
    def generate_report(self, latest_results: AggregatedResult) -> str:
        """Generate a comprehensive performance report"""
        report_lines = [
            "# LTS Performance Report",
            "",
            f"## Overall Metrics",
            f"- Total Tasks: {latest_results.total_tasks}",
            f"- Successful Tasks: {latest_results.successful_tasks}",
            f"- Accuracy: {latest_results.accuracy:.2%}",
            f"- Average Reward: {latest_results.avg_reward:.3f}",
            f"- Memory Usage Rate: {latest_results.memory_usage_rate:.2%}",
            f"- Team Diversity: {latest_results.team_diversity:.3f}",
            f"- Average Execution Time: {latest_results.avg_execution_time:.2f}s",
            f"- Memory Bank Efficiency: {latest_results.memory_bank_efficiency:.3f}",
            ""
        ]
        
        # Add learning progress if we have history
        if len(self.results_history) > 1:
            baseline = self.results_history[0]
            improvements = self.compare_results(baseline, latest_results)
            
            report_lines.extend([
                "## Learning Progress",
                f"- Accuracy Improvement: {improvements['accuracy_improvement']:+.2%}",
                f"- Reward Improvement: {improvements['reward_improvement']:+.3f}",
                f"- Memory Efficiency Change: {improvements['memory_efficiency_change']:+.2%}",
                f"- Diversity Change: {improvements['diversity_change']:+.3f}",
                f"- Execution Time Change: {improvements['execution_time_change']:+.2f}s",
                ""
            ])
        
        # Add failure analysis
        failure_analysis = self.analyze_failure_patterns()
        if failure_analysis:
            report_lines.extend([
                "## Failure Analysis",
                f"- Common Failure Sources: {dict(failure_analysis['common_failure_sources'])}",
                f"- Memory vs No Memory Failures: {failure_analysis['memory_vs_no_memory']}",
                ""
            ])
        
        return "\n".join(report_lines)
