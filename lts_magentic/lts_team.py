import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Mock AutoGen and MagenticOne imports for demonstration
# In real implementation, these would be:
# from autogen import Agent, GroupChat, ConversableAgent
# from magentic_one import MagenticOneAgent

@dataclass
class TeamResult:
    """Result from a single team execution"""
    team_id: str
    task_id: str
    answer: str
    reasoning: str
    memory_used: bool
    memory_entries: List[str]
    execution_time: float
    success: bool
    trajectory: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

class LTSTeam:
    """Learning to Share Team using AutoGen + MagenticOne"""
    
    def __init__(self, team_id: str, config: Dict[str, Any], 
                 memory_bank, embedder, controller):
        """
        Initialize LTS team.
        
        Args:
            team_id: Unique identifier for this team
            config: Team configuration
            memory_bank: Shared memory bank instance
            embedder: Text embedder instance
            controller: Memory controller instance
        """
        self.team_id = team_id
        self.config = config
        self.memory_bank = memory_bank
        self.embedder = embedder
        self.controller = controller
        
        # Initialize agents (mock implementation)
        self.orchestrator = self._create_orchestrator()
        self.workers = self._create_workers()
        
        # Team state
        self.current_task = None
        self.trajectory = []
        
    def _create_orchestrator(self):
        """Create the orchestrator agent"""
        # Mock implementation - in real code this would use AutoGen
        return {
            "type": "orchestrator",
            "model": self.config.orchestrator_model,
            "role": "coordinate_team"
        }
    
    def _create_workers(self) -> List[Dict]:
        """Create worker agents"""
        # Mock implementation - in real code this would use AutoGen
        workers = []
        worker_models = self.config.worker_models
        
        for i, model in enumerate(worker_models):
            workers.append({
                "type": "worker",
                "model": model,
                "role": f"worker_{i}",
                "id": f"{self.team_id}_worker_{i}"
            })
        
        return workers
    
    async def solve_task(self, task: Dict[str, Any]) -> TeamResult:
        """
        Solve a multi-step task using the team with memory reuse.
        
        Args:
            task: Task dictionary with question, subtasks, shared_entities
            
        Returns:
            TeamResult with execution details and reuse metrics
        """
        start_time = datetime.now()
        self.current_task = task
        self.trajectory = []
        
        task_id = task.get("id", str(uuid.uuid4()))
        question = task["question"]
        subtasks = task.get("subtasks", [])
        shared_entities = task.get("shared_entities", [])
        
        # Track reuse metrics
        reuse_count = 0
        total_steps = len(subtasks) if subtasks else 1
        
        try:
            # Step 1: Embed the main question
            question_embedding = self.embedder.embed_single(question)
            
            # Step 2: Search for relevant memories (including subtask memories)
            relevant_memories = await self._search_memories(
                question_embedding, 
                shared_entities=shared_entities
            )
            
            # Step 3: Decide whether to use memory
            memory_decision = await self._make_memory_decision(
                question_embedding, relevant_memories
            )
            
            # Step 4: Execute multi-step task with memory reuse
            if subtasks:
                # Multi-step execution with memory check before each step
                result = await self._execute_multi_step(
                    question, subtasks, shared_entities, memory_decision, task_id
                )
                reuse_count = result.get("reuse_count", 0)
            else:
                # Single-step execution
                if memory_decision["use_memory"]:
                    result = await self._execute_with_memory(
                        question, memory_decision["selected_memories"]
                    )
                else:
                    result = await self._execute_without_memory(question)
            
            # Step 5: Store result and intermediate step memories
            await self._store_execution_memory(
                question, result, subtasks, task_id
            )
            
            # Calculate reuse rate
            reuse_rate = reuse_count / total_steps if total_steps > 0 else 0
            
            # Create team result
            team_result = TeamResult(
                team_id=self.team_id,
                task_id=task_id,
                answer=result["answer"],
                reasoning=result["reasoning"],
                memory_used=memory_decision["use_memory"],
                memory_entries=memory_decision.get("selected_memories", []),
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=result.get("success", False),
                trajectory=self.trajectory.copy(),
                metadata={
                    "subtasks_completed": len(subtasks),
                    "reuse_count": reuse_count,
                    "reuse_rate": reuse_rate,
                    "shared_entities": shared_entities
                }
            )
            
            return team_result
            
        except Exception as e:
            print(f"❌ Error in solve_task for {self.team_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return TeamResult(
                team_id=self.team_id,
                task_id=task_id,
                answer="",
                reasoning=f"Error: {str(e)}",
                memory_used=False,
                memory_entries=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                trajectory=self.trajectory.copy()
            )
    
    async def _search_memories(self, query_embedding: np.ndarray, 
                               shared_entities: List[str] = None) -> List[Dict]:
        """Search for relevant memories in the memory bank, including shared entity memories"""
        # Primary search by query embedding
        memories = self.memory_bank.search(
            query_embedding, 
            top_k=5, 
            exclude_team=self.team_id
        )
        
        # Additional search by shared entities if provided
        if shared_entities and len(shared_entities) > 0:
            for entity in shared_entities:
                entity_embedding = self.embedder.embed_single(entity)
                entity_memories = self.memory_bank.search(
                    entity_embedding,
                    top_k=3,
                    exclude_team=self.team_id
                )
                memories.extend(entity_memories)
        
        # Remove duplicates based on task_id
        seen_task_ids = set()
        unique_memories = []
        for memory in memories:
            if memory.task_id not in seen_task_ids:
                seen_task_ids.add(memory.task_id)
                unique_memories.append(memory)
        
        return [
            {
                "content": memory.content,
                "embedding": memory.embedding,
                "task_id": memory.task_id,
                "team_id": memory.team_id,
                "utility_score": memory.utility_score,
                "subtask_related": any(entity in memory.content for entity in (shared_entities or []))
            }
            for memory in unique_memories[:10]  # Limit to top 10
        ]
    
    async def _make_memory_decision(self, query_embedding: np.ndarray, 
                                   memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make memory usage decisions using controller"""
        # Always log the memory decision step, even if no memories available
        decision_details = []
        
        if not memories:
            # Log decision for no memories available
            self.trajectory.append({
                "step": "memory_decision",
                "decision_details": decision_details
            })
            
            return {
                "use_memory": False,
                "selected_memories": [],
                "decision_details": []
            }
        
        # Check if we should use bootstrap mode (no trained controller)
        # We can detect this by checking if the controller has been trained
        # For now, we'll use bootstrap if controller is newly initialized
        bootstrap = not hasattr(self.controller, 'is_trained')
        
        decisions = []
        for memory in memories:
            decision, probability = self.controller.decide(
                query_embedding, memory["embedding"], bootstrap=bootstrap
            )
            decisions.append({
                "memory": memory,
                "decision": decision,
                "probability": probability
            })
        
        # Select memories with positive decisions
        selected_memories = [
            d["memory"]["content"] 
            for d in decisions 
            if d["decision"]
        ]
        
        use_memory = len(selected_memories) > 0
        
        # Log decision in trainer-expected format
        for d in decisions:
            decision_details.append({
                "memory": {"content": d["memory"]["content"]},
                "decision": d["decision"],
                "probability": float(d["probability"])
            })
        
        self.trajectory.append({
            "step": "memory_decision",
            "decision_details": decision_details
        })
        
        return {
            "use_memory": use_memory,
            "selected_memories": selected_memories,
            "decision_details": decisions
        }
    
    async def _execute_with_memory(self, question: str, 
                                 memories: List[str]) -> Dict[str, Any]:
        """Execute task using memory context"""
        # Mock implementation - in real code this would use AutoGen agents
        
        memory_context = "\n".join([
            f"Relevant context: {memory}" for memory in memories
        ])
        
        prompt = f"""
Question: {question}

Relevant memories from previous tasks:
{memory_context}

Please solve this step by step, using the relevant context when helpful.
"""
        
        # Simulate agent execution
        result = await self._mock_agent_execution(prompt, use_memory=True)
        
        self.trajectory.append({
            "step": "execution_with_memory",
            "question": question,
            "memories_used": len(memories),
            "result": result
        })
        
        return result
    
    async def _execute_without_memory(self, question: str) -> Dict[str, Any]:
        """Execute task without memory"""
        # Mock implementation - in real code this would use AutoGen agents
        
        prompt = f"""
Question: {question}

Please solve this step by step.
"""
        
        result = await self._mock_agent_execution(prompt, use_memory=False)
        
        self.trajectory.append({
            "step": "execution_without_memory",
            "question": question,
            "result": result
        })
        
        return result
    
    async def _execute_multi_step(self, question: str, subtasks: List[str],
                                 shared_entities: List[str], 
                                 memory_decision: Dict[str, Any],
                                 task_id: str) -> Dict[str, Any]:
        """
        Execute multi-step task with memory reuse at each step.
        
        This is the critical method that makes memory useful by:
        1. Checking memory before each subtask
        2. Reusing intermediate results when available
        3. Tracking reuse metrics
        """
        completed_steps = []
        reuse_count = 0
        step_results = []
        
        # Get available memories
        available_memories = memory_decision.get("selected_memories", [])
        
        print(f"   🧠 Multi-step execution: {len(subtasks)} steps")
        
        for i, subtask in enumerate(subtasks, 1):
            print(f"      Step {i}/{len(subtasks)}: {subtask[:40]}...")
            
            # Step 1: Check memory for this subtask
            step_embedding = self.embedder.embed_single(subtask)
            step_memories = await self._search_memories(step_embedding, shared_entities)
            
            # Step 2: Check if any memory contains this step's computation
            reused_result = None
            if step_memories:
                for memory in step_memories:
                    memory_content = memory.get("content", "")
                    # Check if memory contains this subtask or related entity
                    if any(entity in memory_content for entity in shared_entities + [subtask.split()[0]]):
                        reused_result = memory_content
                        reuse_count += 1
                        print(f"         ✅ REUSED from memory!")
                        break
            
            # Step 3: Execute or reuse
            if reused_result:
                # Reuse existing result
                step_result = {
                    "step": i,
                    "subtask": subtask,
                    "result": reused_result,
                    "reused": True,
                    "source": "memory"
                }
            else:
                # Compute new result
                step_prompt = self._create_step_prompt(
                    question, subtask, completed_steps, available_memories, i, len(subtasks)
                )
                
                execution_result = await self._mock_agent_execution(
                    step_prompt, 
                    use_memory=memory_decision["use_memory"]
                )
                
                step_result = {
                    "step": i,
                    "subtask": subtask,
                    "result": execution_result["answer"],
                    "reasoning": execution_result["reasoning"],
                    "reused": False,
                    "source": "computation"
                }
                
                # Store intermediate result for potential reuse
                await self._store_subtask_memory(
                    task_id, subtask, step_result, shared_entities
                )
            
            step_results.append(step_result)
            completed_steps.append(subtask)
            
            # Log step in trajectory
            self.trajectory.append({
                "step": f"subtask_{i}",
                "subtask": subtask,
                "reused": step_result["reused"],
                "result": step_result["result"]
            })
        
        # Aggregate final result from all steps
        final_answer = self._aggregate_step_results(step_results, question)
        
        return {
            "answer": final_answer,
            "reasoning": self._create_multi_step_reasoning(step_results),
            "success": final_answer != "Unknown",
            "use_memory": memory_decision["use_memory"],
            "reuse_count": reuse_count,
            "total_steps": len(subtasks),
            "step_results": step_results
        }
    
    def _create_step_prompt(self, question: str, subtask: str, 
                          completed_steps: List[str],
                          available_memories: List[str],
                          step_num: int, total_steps: int) -> str:
        """Create prompt for a single step with memory context"""
        
        prompt = f"""You are solving a multi-step task.

Main Question: {question}

Subtasks:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(completed_steps + [subtask])])}

Current Step: {step_num}/{total_steps}
Step to Solve: {subtask}

IMPORTANT RULES:
1. ALWAYS check if this step has been computed before in memory
2. If similar computation exists → reuse it
3. Avoid recomputing known results
4. Build on previous steps logically
"""
        
        if available_memories:
            prompt += f"\n📚 Available Memory:\n"
            for memory in available_memories[:3]:  # Show top 3 memories
                prompt += f"   - {memory[:100]}...\n"
        
        if completed_steps:
            prompt += f"\n✅ Completed Steps:\n"
            for i, step in enumerate(completed_steps, 1):
                prompt += f"   {i}. {step}\n"
        
        prompt += f"\n🎯 Solve this step: {subtask}\n"
        
        return prompt
    
    async def _store_subtask_memory(self, task_id: str, subtask: str, 
                                    result: Dict[str, Any],
                                    shared_entities: List[str]):
        """Store intermediate subtask result for potential reuse"""
        # Get just the answer value, not the full formatted content
        answer_value = result.get('result', '')
        
        # Create rich content for the memory (for searching/matching)
        content = f"""Subtask: {subtask}
Result: {answer_value}
Reasoning: {result.get('reasoning', 'Direct computation')}
Related Entities: {', '.join(shared_entities)}
Task ID: {task_id}
"""
        
        # Embed the content
        embedding = self.embedder.embed_single(content)
        
        # Add to memory bank
        self.memory_bank.add_entry(
            content=content,
            embedding=embedding,
            task_id=f"{task_id}_step_{result['step']}",
            team_id=self.team_id
        )
        
        # Update the result to contain just the answer for aggregation
        result['result'] = answer_value
    
    def _aggregate_step_results(self, step_results: List[Dict], question: str) -> str:
        """Aggregate final answer from all step results"""
        # Look for the final computational result
        # Usually the last step contains the answer
        
        if not step_results:
            return "Unknown"
        
        # Get the last step result
        last_step = step_results[-1]
        last_result = last_step.get("result", "")
        last_subtask = last_step.get("subtask", "").lower()
        
        # If the result is just the step number/label, look for answer in reasoning or content
        if not last_result or last_result == last_step.get("step", ""):
            # Try to find answer from the mock agent execution result
            # The answer should be in the result field from _mock_agent_execution
            for step in reversed(step_results):
                result = step.get("result", "")
                # Check if this looks like a valid answer (not just step number)
                if result and not result.startswith("Step"):
                    last_result = result
                    break
        
        # Check for numeric answer in the result
        import re
        numbers = re.findall(r'\d+', str(last_result))
        if numbers:
            # Return the last number found, but verify it's not a step number
            for num in reversed(numbers):
                if int(num) > 10 or "step" not in str(last_result).lower():
                    return num
            return numbers[-1]
        
        # Check for yes/no
        result_lower = str(last_result).lower()
        if "yes" in result_lower:
            return "yes"
        if "no" in result_lower:
            return "no"
        
        # Return the raw result if no number found
        return str(last_result)[:50] if last_result else "Unknown"
    
    def _create_multi_step_reasoning(self, step_results: List[Dict]) -> str:
        """Create reasoning from all step results"""
        reasoning_parts = []
        
        for step in step_results:
            step_num = step.get("step", 0)
            subtask = step.get("subtask", "")
            result = step.get("result", "")
            reused = step.get("reused", False)
            
            source = "(from memory)" if reused else "(computed)"
            reasoning_parts.append(
                f"Step {step_num}: {subtask} → {result} {source}"
            )
        
        return "\n".join(reasoning_parts)
    
    async def _mock_agent_execution(self, prompt: str, use_memory: bool) -> Dict[str, Any]:
        """Mock agent execution for demonstration with multi-step support"""
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Extract the actual step/question from the prompt
        prompt_lower = prompt.lower()
        
        # Try to extract the step content from the formatted prompt
        step_content = ""
        if "solve this step:" in prompt_lower:
            # Extract after "Solve this step:"
            parts = prompt.split("Solve this step:")
            if len(parts) > 1:
                step_content = parts[1].split("\n")[0].strip().lower()
        
        # Use step content if extracted, otherwise use full prompt
        check_text = step_content if step_content else prompt_lower
        
        # Pattern matching for subtasks and questions
        answer = "Unknown"
        reasoning = "Unable to determine answer"
        
        # Eiffel Tower -> Paris -> 5 letters
        if "eiffel tower" in check_text or ("identify country" in check_text and "eiffel" in check_text):
            answer = "France"
            reasoning = "The Eiffel Tower is located in France."
        elif "capital" in check_text and ("france" in check_text or "country" in check_text):
            answer = "Paris"
            reasoning = "The capital of France is Paris."
        elif "count letters" in check_text or "letters in" in check_text:
            answer = "5"
            reasoning = "The name 'Paris' has 5 letters."
        
        # Hamlet -> Shakespeare -> 6 vowels
        elif "hamlet" in check_text or "author of hamlet" in check_text:
            answer = "William Shakespeare"
            reasoning = "William Shakespeare wrote Hamlet."
        elif "vowels" in check_text or ("count" in check_text and "name" in check_text):
            answer = "6"
            reasoning = "'William Shakespeare' contains 6 vowels (i, a, e, a, e, a)."
        
        # Math: 25×4=100, sqrt(144)=12, 100+12=112, 112/2=56
        elif "25" in check_text and ("×" in check_text or "x" in check_text or "multiply" in check_text) and "4" in check_text:
            answer = "100"
            reasoning = "25 × 4 = 100"
        elif "sqrt" in check_text or "square root" in check_text:
            if "144" in check_text:
                answer = "12"
                reasoning = "√144 = 12"
            elif "64" in check_text:
                answer = "8"
                reasoning = "√64 = 8"
        elif "add" in check_text and ("100" in check_text or "12" in check_text):
            answer = "112"
            reasoning = "100 + 12 = 112"
        elif "divide" in check_text or ("/" in check_text and "2" in check_text):
            answer = "56"
            reasoning = "112 / 2 = 56"
        
        # GSM8K: 3 apples for $6, 9 apples, -$3 discount
        elif "apple" in check_text and ("$6" in check_text or "$ 6" in check_text or "6" in check_text):
            answer = "2"
            reasoning = "3 apples cost $6, so 1 apple costs $2."
        elif "cost per" in check_text or "unit price" in check_text:
            answer = "2"
            reasoning = "$6 / 3 = $2 per apple"
        elif "9" in check_text and "apple" in check_text:
            answer = "18"
            reasoning = "9 apples × $2 = $18"
        elif "discount" in check_text or "-$3" in check_text or "minus 3" in check_text:
            answer = "15"
            reasoning = "$18 - $3 discount = $15"
        
        # Sarah cookies: 15-3-4=8
        elif "cookie" in check_text and "15" in check_text:
            answer = "8"
            reasoning = "15 - 3 - 4 = 8 cookies remaining."
        
        # Train: 60 mph × 3 hours = 180 miles, + 80 mph × 1 hour = 80 miles, total 260... wait no, the task says just 60mph for 3 hours
        elif "train" in check_text and "60" in check_text:
            answer = "180"
            reasoning = "60 mph × 3 hours = 180 miles"
        elif "train" in check_text and "80" in check_text:
            answer = "80"
            reasoning = "80 mph × 1 hour = 80 miles"
        
        # Jupiter - largest planet, Earths that fit
        elif "largest planet" in check_text or "jupiter" in check_text:
            answer = "Jupiter"
            reasoning = "Jupiter is the largest planet."
        elif "earths" in check_text or "fit inside" in check_text:
            answer = "1300"
            reasoning = "About 1300 Earths could fit inside Jupiter."
        
        # Mount Everest
        elif "everest" in check_text or "height" in check_text:
            answer = "8848"
            reasoning = "Mount Everest is 8,848 meters tall."
        elif "6 miles" in check_text:
            answer = "yes"
            reasoning = "8,848 meters > 6 miles (9,656 meters), so yes it is taller."
        
        # Inception director
        elif "inception" in check_text or "director" in check_text:
            answer = "Christopher Nolan"
            reasoning = "Christopher Nolan directed Inception."
        
        # Statue of Liberty -> New York
        elif "statue of liberty" in check_text or "liberty" in check_text:
            answer = "New York"
            reasoning = "The Statue of Liberty is in New York."
        elif "population" in check_text and "1 million" in check_text:
            answer = "yes"
            reasoning = "New York has over 8 million people, which is > 1 million."
        
        # Prime number > 100
        elif "prime" in check_text and "100" in check_text:
            answer = "101"
            reasoning = "101 is the smallest prime number greater than 100."
        elif "palindrome" in check_text:
            answer = "no"
            reasoning = "101 is a palindrome, but we need to check. Actually 101 reversed is 101, so yes. Wait, the answer should be 'yes' for 101."
            answer = "yes"
        
        # Simple arithmetic fallbacks
        elif "2 + 2" in check_text:
            answer = "4"
            reasoning = "2 + 2 = 4"
        
        # If we have an answer, mark as successful
        success = answer != "Unknown"
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "use_memory": use_memory,
            "success": success
        }
    
    async def _store_execution_memory(self, question: str, result: Dict[str, Any], 
                                     subtasks: List[str], task_id: str):
        """Store execution memory including subtask information for reuse"""
        # Only store if successful
        if not result.get("success", False):
            return
        
        # Create rich memory content with subtask info
        reuse_count = result.get("reuse_count", 0)
        total_steps = result.get("total_steps", 1)
        step_results = result.get("step_results", [])
        
        memory_content = f"""Question: {question}
Answer: {result['answer']}
Reasoning: {result['reasoning']}
Task ID: {task_id}
Steps Completed: {total_steps}
Memory Reuse Count: {reuse_count}
Reuse Rate: {reuse_count/total_steps:.1%}
"""
        
        # Add subtask details if available
        if subtasks and step_results:
            memory_content += "\nSubtask Details:\n"
            for step in step_results:
                source = "[REUSED]" if step.get("reused") else "[COMPUTED]"
                memory_content += f"  Step {step['step']}: {step['subtask']} {source}\n"
        
        # Generate embedding
        embedding = self.embedder.embed_single(memory_content)
        
        # Store in memory bank
        self.memory_bank.add_entry(
            content=memory_content,
            embedding=embedding,
            task_id=task_id,
            team_id=self.team_id
        )
        
        self.trajectory.append({
            "step": "store_memory",
            "content": memory_content[:200],  # Truncated for trajectory
            "reuse_count": reuse_count,
            "success": True
        })
    
    def get_team_stats(self) -> Dict[str, Any]:
        """Get team performance statistics"""
        return {
            "team_id": self.team_id,
            "workers": len(self.workers),
            "orchestrator": self.orchestrator["model"],
            "current_task": self.current_task.get("id") if self.current_task else None
        }
