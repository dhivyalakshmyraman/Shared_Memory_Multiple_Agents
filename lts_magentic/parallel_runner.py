import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from lts_team import LTSTeam, TeamResult
from memory_bank import MemoryBank
from embedder import Embedder
from controller import MemoryController
from config import LTSConfig

@dataclass
class ParallelExecutionResult:
    """Result from parallel execution of multiple teams"""
    task_id: str
    team_results: List[TeamResult]
    aggregated_answer: str
    execution_time: float
    memory_bank_stats: Dict[str, Any]
    reuse_metrics: Dict[str, Any] = None

class ParallelRunner:
    """Manages parallel execution of multiple LTS teams"""
    
    def __init__(self, config: LTSConfig):
        """
        Initialize parallel runner.
        
        Args:
            config: LTS configuration
        """
        self.config = config
        self.max_teams = config.max_parallel_teams
        
        # Initialize shared components
        self.memory_bank = MemoryBank(max_size=config.memory_bank_size)
        self.embedder = Embedder(model_name=config.embedding_model)
        self.controller = self._load_or_create_controller()
        
        # Initialize teams
        self.teams = self._create_teams()
        
        # Execution state
        self.executor = ThreadPoolExecutor(max_workers=self.max_teams)
        
    def _load_or_create_controller(self) -> MemoryController:
        """Load existing controller or create new one"""
        try:
            # Try to load existing controller
            controller = MemoryController.load(self.config.controller_path)
            print(f"Loaded existing controller from {self.config.controller_path}")
            return controller
        except FileNotFoundError:
            # Create new controller
            controller = MemoryController(
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.controller_hidden_dim,
                num_layers=2
            )
            print("Created new memory controller")
            return controller
    
    def _create_teams(self) -> List[LTSTeam]:
        """Create multiple LTS teams"""
        teams = []
        
        for i in range(self.max_teams):
            team_id = f"team_{i}"
            
            team = LTSTeam(
                team_id=team_id,
                config=self.config,
                memory_bank=self.memory_bank,
                embedder=self.embedder,
                controller=self.controller
            )
            
            teams.append(team)
        
        return teams
    
    async def run_task_parallel(self, task: Dict[str, Any]) -> ParallelExecutionResult:
        """
        Run a single task on all teams in parallel.
        
        Args:
            task: Task dictionary with question, ground_truth, source
            
        Returns:
            ParallelExecutionResult with all team results
        """
        task_id = task.get("id", str(uuid.uuid4()))
        start_time = datetime.now()
        
        # Add task ID to task for tracking
        task["id"] = task_id
        
        try:
            # Run task on all teams in parallel
            team_results = await self._execute_teams_parallel(task)
            
            # Aggregate results
            aggregated_answer = self._aggregate_answers(team_results)
            
            # Calculate reuse metrics across all teams
            total_reuse_count = sum(
                r.metadata.get("reuse_count", 0) 
                for r in team_results 
                if hasattr(r, 'metadata') and r.metadata
            )
            total_steps = sum(
                r.metadata.get("subtasks_completed", 1)
                for r in team_results
                if hasattr(r, 'metadata') and r.metadata
            )
            cross_team_reuse = self._calculate_cross_team_reuse(team_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Get memory bank stats
            memory_stats = self.memory_bank.get_stats()
            
            return ParallelExecutionResult(
                task_id=task_id,
                team_results=team_results,
                aggregated_answer=aggregated_answer,
                execution_time=execution_time,
                memory_bank_stats=memory_stats,
                reuse_metrics={
                    "total_reuse_count": total_reuse_count,
                    "total_steps": total_steps,
                    "reuse_rate": total_reuse_count / total_steps if total_steps > 0 else 0,
                    "cross_team_reuse": cross_team_reuse
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Return error result
            return ParallelExecutionResult(
                task_id=task_id,
                team_results=[],
                aggregated_answer=f"Error: {str(e)}",
                execution_time=execution_time,
                memory_bank_stats=self.memory_bank.get_stats(),
                reuse_metrics=None
            )
    
    async def _execute_teams_parallel(self, task: Dict[str, Any]) -> List[TeamResult]:
        """Execute task on all teams in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create tasks for each team
        futures = []
        for team in self.teams:
            future = loop.run_in_executor(
                self.executor, 
                asyncio.run, 
                team.solve_task(task)
            )
            futures.append(future)
        
        # Wait for all teams to complete
        team_results = []
        for future in futures:
            try:
                result = await future
                team_results.append(result)
            except Exception as e:
                # Create error result for failed team
                error_result = TeamResult(
                    team_id="unknown",
                    task_id=task.get("id", "unknown"),
                    answer=f"Team execution error: {str(e)}",
                    reasoning="Execution failed",
                    memory_used=False,
                    memory_entries=[],
                    execution_time=0.0,
                    success=False,
                    trajectory=[],
                    metadata=None
                )
                team_results.append(error_result)
        
        return team_results
    
    def _aggregate_answers(self, team_results: List[TeamResult]) -> str:
        """
        Aggregate answers from multiple teams.
        
        Args:
            team_results: Results from all teams
            
        Returns:
            Aggregated answer
        """
        if not team_results:
            return "No results available"
        
        # Filter successful results
        successful_results = [r for r in team_results if r.success]
        
        if not successful_results:
            # If no successful results, return most common answer
            answers = [r.answer for r in team_results]
            return self._most_common_answer(answers)
        
        # Use successful results for aggregation
        answers = [r.answer for r in successful_results]
        
        # Simple majority voting
        most_common = self._most_common_answer(answers)
        
        # If there's a clear majority, use it
        if answers.count(most_common) > len(answers) / 2:
            return most_common
        
        # Otherwise, use the answer from the most successful team
        best_result = max(successful_results, key=lambda r: len(r.reasoning))
        return best_result.answer
    
    def _most_common_answer(self, answers: List[str]) -> str:
        """Find the most common answer in a list"""
        if not answers:
            return ""
        
        # Count occurrences
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Return answer with highest count
        return max(answer_counts, key=answer_counts.get)
    
    def _calculate_cross_team_reuse(self, team_results: List[TeamResult]) -> int:
        """Calculate how many times memories were reused across different teams"""
        if not team_results:
            return 0
        
        # Track which teams accessed which memories
        memory_accesses = {}
        
        for result in team_results:
            if not hasattr(result, 'metadata') or not result.metadata:
                continue
            
            reuse_count = result.metadata.get("reuse_count", 0)
            team_id = result.team_id
            
            # Count accesses per team
            if team_id not in memory_accesses:
                memory_accesses[team_id] = 0
            memory_accesses[team_id] += reuse_count
        
        # Cross-team reuse is the minimum reuse across different teams
        # (indicating memories shared between teams)
        if len(memory_accesses) <= 1:
            return 0
        
        # Return average reuse across teams as proxy for cross-team sharing
        return int(sum(memory_accesses.values()) / len(memory_accesses))
    
    async def run_batch_parallel(self, tasks: List[Dict[str, Any]]) -> List[ParallelExecutionResult]:
        """
        Run a batch of tasks in parallel.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            List of ParallelExecutionResult
        """
        results = []
        
        # Process tasks sequentially (but each task runs on all teams in parallel)
        for task in tasks:
            result = await self.run_task_parallel(task)
            results.append(result)
            
            # Optional: Add delay between tasks to avoid overwhelming resources
            await asyncio.sleep(0.1)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            "num_teams": len(self.teams),
            "memory_bank": self.memory_bank.get_stats(),
            "embedder_loaded": self.embedder.is_loaded(),
            "controller_parameters": self.controller.get_parameters_count(),
            "config": {
                "max_teams": self.max_teams,
                "embedding_dim": self.config.embedding_dim,
                "memory_bank_size": self.config.memory_bank_size
            }
        }
    
    def save_controller(self):
        """Save the current controller state"""
        self.controller.save(self.config.controller_path)
        print(f"Controller saved to {self.config.controller_path}")
    
    def reset_memory_bank(self):
        """Clear the memory bank"""
        self.memory_bank.clear()
        print("Memory bank cleared")
    
    def shutdown(self):
        """Shutdown the parallel runner"""
        self.executor.shutdown(wait=True)
        print("Parallel runner shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass
