#!/usr/bin/env python3
"""
Compute Comparison: No-Memory vs Shared Memory Systems
Detailed analysis of computational costs, resource usage, and performance
"""

import asyncio
import time
import psutil
import torch
import numpy as np
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Import LTS components
from config import LTSConfig
from local_datasets import load_all_tasks
from parallel_runner import ParallelRunner
from baseline_no_memory import BaselineTester, NoMemoryMultiAgentSystem

@dataclass
class ComputeMetrics:
    """Compute resource metrics"""
    cpu_percent: float
    memory_mb: float
    execution_time: float
    gpu_memory_mb: float = 0
    tokens_processed: int = 0
    embeddings_computed: int = 0
    memory_operations: int = 0

class ComputeProfiler:
    """Profile computational resources"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_metrics = None
        self.end_metrics = None
    
    def start_profiling(self):
        """Start profiling"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.start_metrics = self._get_current_metrics()
    
    def stop_profiling(self):
        """Stop profiling"""
        self.end_metrics = self._get_current_metrics()
    
    def _get_current_metrics(self) -> ComputeMetrics:
        """Get current system metrics"""
        # CPU and Memory
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # GPU Memory if available
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return ComputeMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            execution_time=0
        )
    
    def get_delta(self, execution_time: float, 
                  tokens_processed: int = 0,
                  embeddings_computed: int = 0,
                  memory_operations: int = 0) -> Dict[str, Any]:
        """Get resource usage delta"""
        if not self.start_metrics or not self.end_metrics:
            return {}
        
        return {
            "cpu_usage": {
                "peak": max(self.start_metrics.cpu_percent, self.end_metrics.cpu_percent),
                "average": (self.start_metrics.cpu_percent + self.end_metrics.cpu_percent) / 2
            },
            "memory_usage": {
                "peak_mb": max(self.start_metrics.memory_mb, self.end_metrics.memory_mb),
                "delta_mb": self.end_metrics.memory_mb - self.start_metrics.memory_mb
            },
            "gpu_memory": {
                "peak_mb": max(self.start_metrics.gpu_memory_mb, self.end_metrics.gpu_memory_mb),
                "delta_mb": self.end_metrics.gpu_memory_mb - self.start_metrics.gpu_memory_mb
            },
            "execution_time": execution_time,
            "tokens_processed": tokens_processed,
            "embeddings_computed": embeddings_computed,
            "memory_operations": memory_operations,
            "compute_efficiency": {
                "tokens_per_second": tokens_processed / execution_time if execution_time > 0 else 0,
                "embeddings_per_second": embeddings_computed / execution_time if execution_time > 0 else 0,
                "memory_ops_per_second": memory_operations / execution_time if execution_time > 0 else 0
            }
        }

class SharedMemorySystemAnalyzer:
    """Analyze shared memory system compute usage"""
    
    def __init__(self):
        self.profiler = ComputeProfiler()
        self.config = LTSConfig()
    
    async def analyze_shared_memory(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze shared memory system"""
        print("🧠 Analyzing Shared Memory System...")
        
        # Initialize components
        self.profiler.start_profiling()
        start_time = time.time()
        
        # Create parallel runner (this includes memory bank, embedder, controller)
        runner = ParallelRunner(self.config)
        
        # Track compute operations
        embeddings_computed = 0
        memory_operations = 0
        tokens_processed = 0
        
        # Run tasks and track operations
        results = []
        for task in tasks:
            task_start = time.time()
            
            # Simulate embedding computation
            embeddings_computed += 1  # One embedding per task
            tokens_processed += len(task["question"].split())
            
            # Simulate memory operations
            memory_operations += 3  # Search + Decision + Store
            
            result = await runner.run_task_parallel(task)
            results.append(result)
            
            task_time = time.time() - task_start
            print(f"   Task completed in {task_time:.3f}s")
        
        total_time = time.time() - start_time
        self.profiler.stop_profiling()
        
        # Get compute metrics
        compute_metrics = self.profiler.get_delta(
            execution_time=total_time,
            tokens_processed=tokens_processed,
            embeddings_computed=embeddings_computed,
            memory_operations=memory_operations
        )
        
        # Add performance metrics
        # For parallel execution, we need to determine success from team results
        successful_tasks = 0
        for result in results:
            # Check if any team was successful
            team_success = any(
                team_result.success if hasattr(team_result, 'success') else False
                for team_result in result.team_results
            )
            if team_success:
                successful_tasks += 1
        
        compute_metrics["performance"] = {
            "total_tasks": len(tasks),
            "successful_tasks": successful_tasks,
            "accuracy": successful_tasks / len(tasks) if tasks else 0,
            "avg_time_per_task": total_time / len(tasks) if tasks else 0,
            "tasks_per_second": len(tasks) / total_time if total_time > 0 else 0
        }
        
        # Add system component costs
        compute_metrics["component_costs"] = {
            "memory_bank_size_mb": self._estimate_memory_bank_size(runner.memory_bank),
            "controller_parameters": runner.controller.get_parameters_count() if hasattr(runner.controller, 'get_parameters_count') else 0,
            "embedder_model_size_mb": self._estimate_embedder_size(),
            "teams_count": len(runner.teams)
        }
        
        runner.shutdown()
        return compute_metrics
    
    def _estimate_memory_bank_size(self, memory_bank) -> float:
        """Estimate memory bank size in MB"""
        try:
            total_entries = len(memory_bank.memories) if hasattr(memory_bank, 'memories') else 0
            # Rough estimate: each entry ~1KB
            return total_entries * 1 / 1024
        except:
            return 0.0
    
    def _estimate_embedder_size(self) -> float:
        """Estimate embedder model size in MB"""
        # SentenceTransformer models are typically 100-500MB
        return 420.0  # Estimated for all-MiniLM-L6-v2

class NoMemorySystemAnalyzer:
    """Analyze no-memory system compute usage"""
    
    def __init__(self):
        self.profiler = ComputeProfiler()
    
    async def analyze_no_memory(self, tasks: List[Dict[str, Any]], num_agents: int = 4) -> Dict[str, Any]:
        """Analyze no-memory system"""
        print(f"🚀 Analyzing No-Memory System ({num_agents} agents)...")
        
        self.profiler.start_profiling()
        start_time = time.time()
        
        # Create system
        system = NoMemoryMultiAgentSystem(num_agents)
        
        # Track operations
        tokens_processed = 0
        agent_operations = 0
        
        # Run tasks
        results = []
        for task in tasks:
            tokens_processed += len(task["question"].split())
            agent_operations += num_agents  # Each agent processes the task
            
            result = await system.solve_task(task)
            results.append(result)
        
        total_time = time.time() - start_time
        self.profiler.stop_profiling()
        
        # Get compute metrics
        compute_metrics = self.profiler.get_delta(
            execution_time=total_time,
            tokens_processed=tokens_processed,
            embeddings_computed=0,  # No embeddings in no-memory system
            memory_operations=0  # No memory operations
        )
        
        # Add performance metrics
        successful_tasks = sum(1 for r in results if r.success)
        compute_metrics["performance"] = {
            "total_tasks": len(tasks),
            "successful_tasks": successful_tasks,
            "accuracy": successful_tasks / len(tasks) if tasks else 0,
            "avg_time_per_task": total_time / len(tasks) if tasks else 0,
            "tasks_per_second": len(tasks) / total_time if total_time > 0 else 0
        }
        
        # Add system component costs
        compute_metrics["component_costs"] = {
            "memory_bank_size_mb": 0,  # No memory bank
            "controller_parameters": 0,  # No controller
            "embedder_model_size_mb": 0,  # No embedder
            "agents_count": num_agents
        }
        
        return compute_metrics

class ComputeComparisonReport:
    """Generate comprehensive compute comparison report"""
    
    def __init__(self):
        self.shared_analyzer = SharedMemorySystemAnalyzer()
        self.no_memory_analyzer = NoMemorySystemAnalyzer()
    
    async def run_comparison(self, tasks: List[Dict[str, Any]]):
        """Run complete comparison"""
        print("🔬 Running Compute Comparison: No-Memory vs Shared Memory")
        print("=" * 70)
        
        # Test both systems
        shared_metrics = await self.shared_analyzer.analyze_shared_memory(tasks)
        no_memory_metrics = await self.no_memory_analyzer.analyze_no_memory(tasks)
        
        # Generate comparison
        comparison = self._compare_metrics(shared_metrics, no_memory_metrics)
        
        # Generate report
        self._generate_detailed_report(shared_metrics, no_memory_metrics, comparison)
        
        return shared_metrics, no_memory_metrics, comparison
    
    def _compare_metrics(self, shared: Dict[str, Any], no_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between systems"""
        comparison = {}
        
        # Performance comparison
        comparison["performance"] = {
            "accuracy_improvement": shared["performance"]["accuracy"] - no_memory["performance"]["accuracy"],
            "speed_ratio": no_memory["performance"]["tasks_per_second"] / shared["performance"]["tasks_per_second"],
            "time_per_task_ratio": shared["performance"]["avg_time_per_task"] / no_memory["performance"]["avg_time_per_task"]
        }
        
        # Resource comparison
        comparison["resources"] = {
            "memory_overhead_mb": shared["memory_usage"]["peak_mb"] - no_memory["memory_usage"]["peak_mb"],
            "cpu_overhead": shared["cpu_usage"]["average"] - no_memory["cpu_usage"]["average"],
            "gpu_memory_mb": shared["gpu_memory"]["peak_mb"] - no_memory["gpu_memory"]["peak_mb"]
        }
        
        # Efficiency comparison
        comparison["efficiency"] = {
            "compute_per_accuracy": {
                "shared": shared["execution_time"] / shared["performance"]["accuracy"] if shared["performance"]["accuracy"] > 0 else float('inf'),
                "no_memory": no_memory["execution_time"] / no_memory["performance"]["accuracy"] if no_memory["performance"]["accuracy"] > 0 else float('inf')
            },
            "memory_per_task": {
                "shared": shared["memory_usage"]["peak_mb"] / shared["performance"]["total_tasks"],
                "no_memory": no_memory["memory_usage"]["peak_mb"] / no_memory["performance"]["total_tasks"]
            }
        }
        
        return comparison
    
    def _generate_detailed_report(self, shared: Dict[str, Any], no_memory: Dict[str, Any], comparison: Dict[str, Any]):
        """Generate detailed comparison report"""
        report = []
        report.append("# Compute Comparison Report: No-Memory vs Shared Memory")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        report.append(f"- **Shared Memory Accuracy**: {shared['performance']['accuracy']:.1%}")
        report.append(f"- **No-Memory Accuracy**: {no_memory['performance']['accuracy']:.1%}")
        report.append(f"- **Accuracy Improvement**: {comparison['performance']['accuracy_improvement']:.1%}")
        report.append(f"- **Speed Advantage (No-Memory)**: {comparison['performance']['speed_ratio']:.1f}x faster")
        report.append(f"- **Memory Overhead**: {comparison['resources']['memory_overhead_mb']:.1f} MB")
        report.append("")
        
        # Performance Comparison
        report.append("## 📊 Performance Comparison")
        report.append("")
        report.append("| Metric | Shared Memory | No Memory | Ratio |")
        report.append("|--------|---------------|-----------|-------|")
        
        # Handle zero accuracy case
        shared_acc = shared['performance']['accuracy']
        no_mem_acc = no_memory['performance']['accuracy']
        if no_mem_acc > 0:
            acc_ratio = f"{shared_acc/no_mem_acc:.2f}x"
        else:
            acc_ratio = "N/A (baseline 0%)"
        
        report.append(f"| Accuracy | {shared_acc:.1%} | {no_mem_acc:.1%} | {acc_ratio} |")
        report.append(f"| Tasks/Second | {shared['performance']['tasks_per_second']:.1f} | {no_memory['performance']['tasks_per_second']:.1f} | {comparison['performance']['speed_ratio']:.1f}x |")
        report.append(f"| Avg Time/Task | {shared['performance']['avg_time_per_task']:.3f}s | {no_memory['performance']['avg_time_per_task']:.3f}s | {comparison['performance']['time_per_task_ratio']:.1f}x |")
        report.append("")
        
        # Resource Usage
        report.append("## 💻 Resource Usage")
        report.append("")
        report.append("### Shared Memory System")
        report.append(f"- **Peak Memory**: {shared['memory_usage']['peak_mb']:.1f} MB")
        report.append(f"- **CPU Usage**: {shared['cpu_usage']['average']:.1f}%")
        report.append(f"- **GPU Memory**: {shared['gpu_memory']['peak_mb']:.1f} MB")
        report.append(f"- **Embeddings Computed**: {shared['embeddings_computed']}")
        report.append(f"- **Memory Operations**: {shared['memory_operations']}")
        report.append("")
        
        report.append("### No-Memory System")
        report.append(f"- **Peak Memory**: {no_memory['memory_usage']['peak_mb']:.1f} MB")
        report.append(f"- **CPU Usage**: {no_memory['cpu_usage']['average']:.1f}%")
        report.append(f"- **GPU Memory**: {no_memory['gpu_memory']['peak_mb']:.1f} MB")
        report.append("")
        
        # Component Analysis
        report.append("## 🧩 Component Analysis")
        report.append("")
        report.append("### Shared Memory Components")
        report.append(f"- **Memory Bank**: {shared['component_costs']['memory_bank_size_mb']:.1f} MB")
        report.append(f"- **Controller Parameters**: {shared['component_costs']['controller_parameters']:,}")
        report.append(f"- **Embedder Model**: {shared['component_costs']['embedder_model_size_mb']:.1f} MB")
        report.append(f"- **Active Teams**: {shared['component_costs']['teams_count']}")
        report.append("")
        
        report.append("### No-Memory Components")
        report.append(f"- **Active Agents**: {no_memory['component_costs']['agents_count']}")
        report.append(f"- **Memory Bank**: 0 MB")
        report.append(f"- **Controller Parameters**: 0")
        report.append(f"- **Embedder Model**: 0 MB")
        report.append("")
        
        # Efficiency Analysis
        report.append("## ⚡ Efficiency Analysis")
        report.append("")
        report.append("### Compute per Accuracy")
        shared_eff = comparison['efficiency']['compute_per_accuracy']['shared']
        no_mem_eff = comparison['efficiency']['compute_per_accuracy']['no_memory']
        report.append(f"- **Shared Memory**: {shared_eff:.3f}s per 1% accuracy")
        report.append(f"- **No Memory**: {no_mem_eff:.3f}s per 1% accuracy")
        report.append(f"- **Efficiency Ratio**: {no_mem_eff/shared_eff:.2f}x")
        report.append("")
        
        report.append("### Memory per Task")
        shared_mem_task = comparison['efficiency']['memory_per_task']['shared']
        no_mem_task = comparison['efficiency']['memory_per_task']['no_memory']
        report.append(f"- **Shared Memory**: {shared_mem_task:.2f} MB per task")
        report.append(f"- **No Memory**: {no_mem_task:.2f} MB per task")
        report.append("")
        
        # Trade-offs Analysis
        report.append("## ⚖️ Trade-offs Analysis")
        report.append("")
        report.append("### When to Use Shared Memory")
        report.append("✅ **Pros**:")
        report.append(f"- Higher accuracy (+{comparison['performance']['accuracy_improvement']:.1%})")
        report.append("- Learning and adaptation capabilities")
        report.append("- Knowledge accumulation over time")
        report.append("- Better performance on complex tasks")
        report.append("")
        report.append("❌ **Cons**:")
        report.append(f"- {comparison['performance']['speed_ratio']:.1f}x slower")
        report.append(f"- {comparison['resources']['memory_overhead_mb']:.1f} MB memory overhead")
        report.append("- More complex system")
        report.append("- Requires training and maintenance")
        report.append("")
        
        report.append("### When to Use No-Memory")
        report.append("✅ **Pros**:")
        report.append(f"- {comparison['performance']['speed_ratio']:.1f}x faster")
        report.append("- Minimal resource usage")
        report.append("- Simple and reliable")
        report.append("- No training required")
        report.append("")
        report.append("❌ **Cons**:")
        report.append(f"- Lower accuracy (-{comparison['performance']['accuracy_improvement']:.1%})")
        report.append("- No learning capability")
        report.append("- Limited to simple tasks")
        report.append("- Cannot improve over time")
        report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        report.append("")
        
        acc_improvement = comparison['performance']['accuracy_improvement']
        speed_ratio = comparison['performance']['speed_ratio']
        
        if acc_improvement > 0.1 and speed_ratio < 100:
            report.append("🧠 **Use Shared Memory when**:")
            report.append("- Accuracy is critical (>10% improvement needed)")
            report.append("- Tasks are complex or require domain knowledge")
            report.append("- System can afford the computational overhead")
            report.append("- Long-term learning benefits outweigh initial costs")
            report.append("")
        
        if speed_ratio > 10:
            report.append("🚀 **Use No-Memory when**:")
            report.append("- Speed is the primary concern")
            report.append("- Tasks are simple and well-defined")
            report.append("- Resources are limited")
            report.append("- One-shot tasks with no learning requirement")
            report.append("")
        
        # Save detailed metrics
        report.append("## 📈 Detailed Metrics")
        report.append("")
        report.append("### Shared Memory System")
        report.append("```json")
        report.append(json.dumps(shared, indent=2))
        report.append("```")
        report.append("")
        
        report.append("### No-Memory System")
        report.append("```json")
        report.append(json.dumps(no_memory, indent=2))
        report.append("```")
        
        # Save report
        report_content = "\n".join(report)
        with open("compute_comparison_report.md", "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print("📄 Detailed report saved to compute_comparison_report.md")
        
        # Save raw metrics
        metrics_data = {
            "shared_memory": shared,
            "no_memory": no_memory,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("compute_metrics.json", "w", encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
        
        print("💾 Raw metrics saved to compute_metrics.json")

async def main():
    """Main comparison function"""
    # Load test tasks
    tasks = load_all_tasks("data/tasks.json")
    test_tasks = tasks[:6]  # Use subset for comparison
    
    print(f"🔬 Running compute comparison on {len(test_tasks)} tasks...")
    
    # Run comparison
    reporter = ComputeComparisonReport()
    shared_metrics, no_memory_metrics, comparison = await reporter.run_comparison(test_tasks)
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 COMPUTE COMPARISON SUMMARY")
    print("="*70)
    print(f"📊 Accuracy: Shared {shared_metrics['performance']['accuracy']:.1%} vs No-Memory {no_memory_metrics['performance']['accuracy']:.1%}")
    print(f"⚡ Speed: No-Memory {comparison['performance']['speed_ratio']:.1f}x faster")
    print(f"💾 Memory: Shared uses {comparison['resources']['memory_overhead_mb']:.1f} MB more")
    print(f"🎪 Efficiency: {comparison['efficiency']['compute_per_accuracy']['no_memory']/comparison['efficiency']['compute_per_accuracy']['shared']:.2f}x compute per accuracy")
    print("\n📄 See compute_comparison_report.md for detailed analysis!")

if __name__ == "__main__":
    asyncio.run(main())
