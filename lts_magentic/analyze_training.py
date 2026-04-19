#!/usr/bin/env python3
"""
Analyze training results: accuracy and ground truth comparison
"""

import torch
import json
from typing import Dict, List, Any

def analyze_trajectories():
    """Analyze trajectories for accuracy and ground truth matching"""
    print("🔍 Analyzing Training Results")
    print("=" * 60)
    
    # Load trajectories
    try:
        trajectories = torch.load("trajectories.pt", weights_only=False)
        print(f"✅ Loaded {len(trajectories)} trajectories")
    except Exception as e:
        print(f"❌ Error loading trajectories: {e}")
        return
    
    # Load training metadata
    try:
        with open("training_metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"✅ Loaded training metadata")
    except:
        metadata = {}
    
    # Statistics
    total_tasks = len(trajectories)
    successful_tasks = 0
    correct_answers = 0
    total_reuse_count = 0
    total_steps = 0
    
    task_results = []
    
    for i, traj in enumerate(trajectories, 1):
        task = traj.get("task", {})
        question = task.get("question", "N/A")[:50]
        ground_truth = str(task.get("ground_truth", "")).strip().lower()
        answer = str(traj.get("answer", "")).strip().lower()
        success = traj.get("success", False)
        
        # Check if answer matches ground truth
        is_correct = False
        if ground_truth and answer:
            # Exact match
            if answer == ground_truth:
                is_correct = True
            # Numeric match
            try:
                ans_num = float(answer)
                gt_num = float(ground_truth)
                if abs(ans_num - gt_num) < 0.001:
                    is_correct = True
            except:
                pass
            # Contains match
            if ground_truth in answer or answer in ground_truth:
                is_correct = True
        
        if success:
            successful_tasks += 1
        if is_correct:
            correct_answers += 1
        
        # Get reuse metrics
        reuse_count = traj.get("reuse_count", 0)
        subtasks_completed = traj.get("subtasks_completed", 1)
        total_reuse_count += reuse_count
        total_steps += subtasks_completed
        
        task_results.append({
            "task_id": task.get("id", f"task_{i}"),
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "success": success,
            "correct": is_correct,
            "reuse_count": reuse_count,
            "subtasks": subtasks_completed
        })
    
    # Calculate metrics
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    accuracy = correct_answers / total_tasks if total_tasks > 0 else 0
    reuse_rate = total_reuse_count / total_steps if total_steps > 0 else 0
    
    print(f"\n📊 OVERALL METRICS")
    print("-" * 60)
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful Executions: {successful_tasks} ({success_rate:.1%})")
    print(f"Correct Answers (vs Ground Truth): {correct_answers} ({accuracy:.1%})")
    print(f"Total Memory Reuses: {total_reuse_count}")
    print(f"Total Steps: {total_steps}")
    print(f"Overall Reuse Rate: {reuse_rate:.1%}")
    
    # Per-task breakdown
    print(f"\n📋 TASK BREAKDOWN (showing first 10)")
    print("-" * 60)
    print(f"{'Task ID':<12} {'Ground Truth':<15} {'Answer':<15} {'Correct':<8} {'Reuse':<6}")
    print("-" * 60)
    
    for result in task_results[:10]:
        task_id = result["task_id"][:11]
        gt = str(result["ground_truth"])[:14]
        ans = str(result["answer"])[:14]
        correct = "✅" if result["correct"] else "❌"
        reuse = f"{result['reuse_count']}/{result['subtasks']}"
        
        print(f"{task_id:<12} {gt:<15} {ans:<15} {correct:<8} {reuse:<6}")
    
    # Show reuse examples
    print(f"\n🔄 MEMORY REUSE EXAMPLES")
    print("-" * 60)
    
    high_reuse_tasks = [t for t in task_results if t["reuse_count"] > 0]
    if high_reuse_tasks:
        print(f"Tasks with memory reuse: {len(high_reuse_tasks)}/{total_tasks}")
        for result in high_reuse_tasks[:5]:
            print(f"  • {result['task_id']}: {result['reuse_count']} reuses / {result['subtasks']} steps")
    else:
        print("No memory reuse recorded in this training run.")
    
    # Show metadata info if available
    if metadata:
        print(f"\n📁 TRAINING METADATA")
        print("-" * 60)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    print(f"Training Accuracy: {accuracy:.1%} ({correct_answers}/{total_tasks})")
    print(f"Success Rate: {success_rate:.1%} ({successful_tasks}/{total_tasks})")
    print(f"Memory Reuse: {reuse_rate:.1%} ({total_reuse_count}/{total_steps} steps)")
    
    if accuracy >= 0.5:
        print("✅ Good accuracy - trajectories ready for Colab training!")
    else:
        print("⚠️ Lower accuracy - may need more training data")
    
    return {
        "total_tasks": total_tasks,
        "accuracy": accuracy,
        "success_rate": success_rate,
        "reuse_rate": reuse_rate,
        "correct_answers": correct_answers
    }

if __name__ == "__main__":
    results = analyze_trajectories()
    
    # Save detailed results
    with open("training_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed analysis saved to training_analysis.json")
