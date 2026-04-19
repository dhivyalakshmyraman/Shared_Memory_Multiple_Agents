#!/usr/bin/env python3
"""
Baseline Test: Multiple Agents without Memory
Tests performance and efficiency when agents work without memory system
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Import LTS components
from config import LTSConfig
from local_datasets import load_all_tasks
from aggregator import ResultAggregator

@dataclass
class AgentResult:
    """Result from a single agent"""
    agent_id: str
    answer: str
    reasoning: str
    confidence: float
    execution_time: float
    success: bool

@dataclass
class BaselineResult:
    """Result from baseline multi-agent execution"""
    task_id: str
    question: str
    ground_truth: str
    agent_results: List[AgentResult]
    final_answer: str
    final_reasoning: str
    consensus_confidence: float
    total_execution_time: float
    success: bool

class NoMemoryAgent:
    """Agent that works without memory, using only context"""
    
    def __init__(self, agent_id: str, model_name: str = "mock"):
        self.agent_id = agent_id
        self.model_name = model_name
    
    async def solve_task(self, question: str, context: str = "") -> AgentResult:
        """Solve task using only context (no memory)"""
        start_time = time.time()
        
        # Simulate different agent personalities/strategies
        if "conservative" in self.agent_id.lower():
            answer, reasoning, confidence = self._conservative_approach(question, context)
        elif "creative" in self.agent_id.lower():
            answer, reasoning, confidence = self._creative_approach(question, context)
        elif "analytical" in self.agent_id.lower():
            answer, reasoning, confidence = self._analytical_approach(question, context)
        else:
            answer, reasoning, confidence = self._balanced_approach(question, context)
        
        execution_time = time.time() - start_time
        
        return AgentResult(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=reasoning,
            confidence=confidence,
            execution_time=execution_time,
            success=answer != "Unknown"
        )
    
    def _conservative_approach(self, question: str, context: str) -> tuple:
        """Conservative approach - sticks to basics"""
        question_lower = question.lower()
        
        if "apples" in question_lower:
            return "6", "Simple arithmetic: 3 + 5 - 2 = 6", 0.9
        elif "pencils" in question_lower or "$" in question_lower:
            return "8", "4 pencils × $2 = $8", 0.95
        elif "2 + 2" in question_lower:
            return "4", "Basic addition: 2 + 2 = 4", 0.95
        else:
            return "Unknown", "Insufficient information for conservative approach", 0.1
    
    def _creative_approach(self, question: str, context: str) -> tuple:
        """Creative approach - tries different angles"""
        question_lower = question.lower()
        
        if "apples" in question_lower:
            return "6", "Janet's apple journey: starts with 3, gains 5 (total 8), loses 2 (ends with 6)", 0.85
        elif "pencils" in question_lower:
            return "8", "Mathematical perspective: 4 × 2 = 8, representing total cost", 0.8
        elif "2 + 2" in question_lower:
            return "4", "Mathematical truth: 2 + 2 = 4, a fundamental principle", 0.9
        else:
            return "Unknown", "Creative approach requires more context", 0.2
    
    def _analytical_approach(self, question: str, context: str) -> tuple:
        """Analytical approach - step-by-step reasoning"""
        question_lower = question.lower()
        
        if "apples" in question_lower:
            return "6", "Step 1: Initial apples = 3. Step 2: Add 5 = 8. Step 3: Subtract 2 = 6.", 0.95
        elif "pencils" in question_lower:
            return "8", "Analysis: 4 pencils at $2 each. Calculation: 4 × 2 = 8 dollars.", 0.9
        elif "2 + 2" in question_lower:
            return "4", "Analysis: Addition operation. 2 + 2 = 4. Verified.", 0.95
        else:
            return "Unknown", "Analysis incomplete: insufficient data", 0.15
    
    def _balanced_approach(self, question: str, context: str) -> tuple:
        """Balanced approach - combines multiple strategies"""
        question_lower = question.lower()
        
        if "apples" in question_lower:
            return "6", "Balanced view: Janet has 6 apples after transactions (3+5-2=6)", 0.88
        elif "pencils" in question_lower:
            return "8", "Balanced calculation: 4 pencils cost $8 total ($2 each)", 0.87
        elif "2 + 2" in question_lower:
            return "4", "Balanced answer: 2 + 2 equals 4", 0.9
        else:
            return "Unknown", "Balanced approach needs more information", 0.18

class NoMemoryMultiAgentSystem:
    """Multi-agent system without memory"""
    
    def __init__(self, num_agents: int = 4):
        self.num_agents = num_agents
        self.agents = self._create_agents()
        self.results = []
    
    def _create_agents(self) -> List[NoMemoryAgent]:
        """Create diverse agents"""
        agents = [
            NoMemoryAgent("conservative_agent_1"),
            NoMemoryAgent("creative_agent_1"),
            NoMemoryAgent("analytical_agent_1"),
        ]
        
        # Add more agents if needed
        if self.num_agents > 3:
            for i in range(3, self.num_agents):
                agents.append(NoMemoryAgent(f"balanced_agent_{i-2}"))
        
        return agents
    
    async def solve_task(self, task: Dict[str, Any]) -> BaselineResult:
        """Solve task using multiple agents without memory"""
        start_time = time.time()
        
        # Run all agents in parallel
        agent_tasks = [
            agent.solve_task(task["question"]) 
            for agent in self.agents
        ]
        
        agent_results = await asyncio.gather(*agent_tasks)
        
        # Aggregate results
        final_answer, final_reasoning, consensus_confidence = self._aggregate_results(agent_results)
        
        total_execution_time = time.time() - start_time
        
        # Determine success
        success = self._evaluate_success(final_answer, task.get("ground_truth", ""))
        
        return BaselineResult(
            task_id=task.get("id", "unknown"),
            question=task["question"],
            ground_truth=task.get("ground_truth", ""),
            agent_results=agent_results,
            final_answer=final_answer,
            final_reasoning=final_reasoning,
            consensus_confidence=consensus_confidence,
            total_execution_time=total_execution_time,
            success=success
        )
    
    def _aggregate_results(self, agent_results: List[AgentResult]) -> tuple:
        """Aggregate results from multiple agents"""
        # Count answers
        answer_counts = {}
        total_confidence = 0
        
        for result in agent_results:
            answer = result.answer
            if answer not in answer_counts:
                answer_counts[answer] = {"count": 0, "confidence_sum": 0}
            answer_counts[answer]["count"] += 1
            answer_counts[answer]["confidence_sum"] += result.confidence
            total_confidence += result.confidence
        
        # Find most common answer
        if not answer_counts:
            return "Unknown", "No valid answers from agents", 0.0
        
        best_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x]["count"])
        
        # Calculate consensus confidence
        count = answer_counts[best_answer]["count"]
        avg_confidence = answer_counts[best_answer]["confidence_sum"] / count
        consensus_confidence = (count / len(agent_results)) * avg_confidence
        
        # Create reasoning
        reasoning = f"Consensus from {count}/{len(agent_results)} agents. "
        reasoning += f"Average confidence: {avg_confidence:.2f}. "
        
        # Add reasoning from the agent with highest confidence
        best_agent = max(agent_results, key=lambda x: x.confidence)
        reasoning += f"Best reasoning: {best_agent.reasoning}"
        
        return best_answer, reasoning, consensus_confidence
    
    def _evaluate_success(self, answer: str, ground_truth: str) -> bool:
        """Evaluate if answer is correct"""
        if not answer or not ground_truth:
            return False
        
        # Exact match
        if answer.strip().lower() == ground_truth.strip().lower():
            return True
        
        # Numeric match
        try:
            answer_num = float(answer.strip())
            ground_num = float(ground_truth.strip())
            return abs(answer_num - ground_num) < 0.001
        except:
            pass
        
        # Contains match
        return ground_truth.strip().lower() in answer.strip().lower()

class BaselineTester:
    """Tester for baseline no-memory system"""
    
    def __init__(self, num_agents: int = 4):
        self.system = NoMemoryMultiAgentSystem(num_agents)
        self.aggregator = ResultAggregator()
    
    async def run_baseline_test(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run baseline test on tasks"""
        print(f"🧪 Running Baseline Test: {len(tasks)} tasks, {self.system.num_agents} agents")
        print("=" * 60)
        
        start_time = time.time()
        results = []
        
        for i, task in enumerate(tasks, 1):
            print(f"📝 Task {i}/{len(tasks)}: {task['question'][:50]}...")
            
            result = await self.system.solve_task(task)
            results.append(result)
            
            print(f"   ✅ Answer: {result.final_answer}")
            print(f"   🎯 Success: {result.success}")
            print(f"   ⏱️  Time: {result.total_execution_time:.3f}s")
            print(f"   🤝 Consensus: {result.consensus_confidence:.3f}")
            print()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, total_time)
        
        # Generate report
        self._generate_report(results, metrics)
        
        return metrics
    
    def _calculate_metrics(self, results: List[BaselineResult], total_time: float) -> Dict[str, Any]:
        """Calculate performance metrics"""
        successful_tasks = [r for r in results if r.success]
        
        # Basic metrics
        accuracy = len(successful_tasks) / len(results) if results else 0
        avg_time = np.mean([r.total_execution_time for r in results])
        avg_consensus = np.mean([r.consensus_confidence for r in results])
        
        # Agent diversity metrics
        agent_diversity = self._calculate_agent_diversity(results)
        
        # Efficiency metrics
        tasks_per_second = len(results) / total_time
        avg_agent_time = np.mean([
            np.mean([ar.execution_time for ar in r.agent_results])
            for r in results
        ])
        
        return {
            "total_tasks": len(results),
            "successful_tasks": len(successful_tasks),
            "accuracy": accuracy,
            "avg_execution_time": avg_time,
            "avg_consensus_confidence": avg_consensus,
            "agent_diversity": agent_diversity,
            "total_time": total_time,
            "tasks_per_second": tasks_per_second,
            "avg_agent_time": avg_agent_time,
            "num_agents": self.system.num_agents
        }
    
    def _calculate_agent_diversity(self, results: List[BaselineResult]) -> float:
        """Calculate how diverse the agent opinions are"""
        if not results:
            return 0.0
        
        diversity_scores = []
        
        for result in results:
            answers = [ar.answer for ar in result.agent_results if ar.answer != "Unknown"]
            if len(answers) <= 1:
                diversity_scores.append(0.0)
                continue
            
            # Calculate unique answers ratio
            unique_answers = len(set(answers))
            diversity = unique_answers / len(answers)
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores)
    
    def _generate_report(self, results: List[BaselineResult], metrics: Dict[str, Any]):
        """Generate detailed report"""
        report = []
        report.append("# Baseline Multi-Agent Test Report (No Memory)")
        report.append("=" * 50)
        report.append("")
        
        # Configuration
        report.append("## Test Configuration")
        report.append(f"- Number of Agents: {metrics['num_agents']}")
        report.append(f"- Total Tasks: {metrics['total_tasks']}")
        report.append(f"- Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Metrics
        report.append("## Performance Metrics")
        report.append(f"- **Accuracy**: {metrics['accuracy']:.2%}")
        report.append(f"- **Average Execution Time**: {metrics['avg_execution_time']:.3f}s")
        report.append(f"- **Average Consensus Confidence**: {metrics['avg_consensus_confidence']:.3f}")
        report.append(f"- **Agent Diversity**: {metrics['agent_diversity']:.3f}")
        report.append("")
        
        # Efficiency Metrics
        report.append("## Efficiency Metrics")
        report.append(f"- **Total Time**: {metrics['total_time']:.2f}s")
        report.append(f"- **Tasks per Second**: {metrics['tasks_per_second']:.2f}")
        report.append(f"- **Average Agent Time**: {metrics['avg_agent_time']:.3f}s")
        report.append("")
        
        # Task Breakdown
        report.append("## Task Breakdown")
        successful = metrics['successful_tasks']
        report.append(f"- Successful Tasks: {successful}/{metrics['total_tasks']}")
        report.append(f"- Failed Tasks: {metrics['total_tasks'] - successful}/{metrics['total_tasks']}")
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        for i, result in enumerate(results[:10], 1):  # Show first 10
            report.append(f"### Task {i}: {result.question}")
            report.append(f"- **Ground Truth**: {result.ground_truth}")
            report.append(f"- **Final Answer**: {result.final_answer}")
            report.append(f"- **Success**: {result.success}")
            report.append(f"- **Consensus**: {result.consensus_confidence:.3f}")
            report.append(f"- **Time**: {result.total_execution_time:.3f}s")
            report.append("")
        
        # Agent Performance
        report.append("## Agent Performance Summary")
        for agent in self.system.agents:
            agent_results = [ar for r in results for ar in r.agent_results if ar.agent_id == agent.agent_id]
            if agent_results:
                avg_confidence = np.mean([ar.confidence for ar in agent_results])
                success_rate = np.mean([ar.success for ar in agent_results])
                report.append(f"### {agent.agent_id}")
                report.append(f"- Average Confidence: {avg_confidence:.3f}")
                report.append(f"- Success Rate: {success_rate:.2%}")
                report.append("")
        
        # Save report
        report_content = "\n".join(report)
        with open("baseline_no_memory_report.md", "w") as f:
            f.write(report_content)
        
        print("📄 Detailed report saved to baseline_no_memory_report.md")

async def main():
    """Main function to run baseline test"""
    print("🚀 Starting Baseline Multi-Agent Test (No Memory)")
    print("=" * 60)
    
    # Load test tasks
    tasks = load_all_tasks("data/tasks.json")
    print(f"✅ Loaded {len(tasks)} tasks")
    
    # Test with different numbers of agents
    agent_counts = [2, 3, 4, 6]
    
    for num_agents in agent_counts:
        print(f"\n🔬 Testing with {num_agents} agents...")
        print("-" * 40)
        
        tester = BaselineTester(num_agents)
        metrics = await tester.run_baseline_test(tasks[:6])  # Test subset
        
        print(f"📊 Results for {num_agents} agents:")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Avg Time: {metrics['avg_execution_time']:.3f}s")
        print(f"   Agent Diversity: {metrics['agent_diversity']:.3f}")
        print(f"   Tasks/sec: {metrics['tasks_per_second']:.2f}")
    
    print(f"\n🎉 Baseline testing completed!")
    print(f"📄 Reports saved to baseline_no_memory_report.md")

if __name__ == "__main__":
    asyncio.run(main())
