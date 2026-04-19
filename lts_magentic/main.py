#!/usr/bin/env python3
"""
Learning to Share (LTS) - Hybrid Local + Colab Implementation
Main entry point for the LTS system

Usage:
    python main.py --mode train    # Train mode: collect trajectories and train controller
    python main.py --mode eval     # Eval mode: evaluate with trained controller
    python main.py --mode demo     # Demo mode: run on sample tasks
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

# Import LTS modules
from local_datasets import load_all_tasks, get_task_stats
from config import LTSConfig
from parallel_runner import ParallelRunner
from aggregator import ResultAggregator
from rl_trainer import RLTrainer
from rewards import evaluate_batch

def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lts.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_system_info():
    """Print system information"""
    print("=" * 60)
    print("Learning to Share (LTS) - Hybrid Implementation")
    print("=" * 60)
    
    # Python and system info
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    # Check GPU availability
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: Available ({torch.cuda.get_device_name()})")
        else:
            print("CUDA: Not available")
    except ImportError:
        print("PyTorch: Not installed")
    
    print("-" * 60)

async def train_mode(config: LTSConfig, logger):
    """Training mode: collect trajectories and prepare for Colab training"""
    print("\n🚀 Starting TRAIN MODE")
    print("This will:")
    print("1. Load tasks from local dataset")
    print("2. Run parallel teams to collect trajectories")
    print("3. Save trajectories for Colab training")
    print("4. Generate Colab training script")
    
    # Load tasks
    print(f"\n📁 Loading tasks from {config.dataset_path}")
    try:
        tasks = load_all_tasks(config.dataset_path)
        stats = get_task_stats(tasks)
        print(f"✅ Loaded {len(tasks)} tasks:")
        for source, count in stats.items():
            if source != "total":
                print(f"   - {source}: {count} tasks")
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        return
    
    # Initialize parallel runner
    print(f"\n🤖 Initializing parallel runner with {config.max_parallel_teams} teams")
    try:
        runner = ParallelRunner(config)
        print(f"✅ System ready:")
        system_stats = runner.get_system_stats()
        for key, value in system_stats.items():
            if key != "config":
                print(f"   - {key}: {value}")
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return
    
    # Run tasks and collect trajectories
    print(f"\n🏃 Running {len(tasks)} tasks on parallel teams...")
    start_time = time.time()
    
    try:
        batch_results = await runner.run_batch_parallel(tasks)
        execution_time = time.time() - start_time
        
        print(f"✅ Completed in {execution_time:.2f} seconds")
        print(f"   - Avg time per task: {execution_time/len(tasks):.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to run tasks: {e}")
        return
    
    # Aggregate results
    print(f"\n📊 Aggregating results...")
    try:
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate_batch_results(batch_results, tasks)
        
        print(f"✅ Results:")
        print(f"   - Accuracy: {aggregated.accuracy:.2%}")
        print(f"   - Avg Reward: {aggregated.avg_reward:.3f}")
        print(f"   - Memory Usage: {aggregated.memory_usage_rate:.2%}")
        print(f"   - Team Diversity: {aggregated.team_diversity:.3f}")
        print(f"   - Memory Bank Efficiency: {aggregated.memory_bank_efficiency:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        return
    
    # Extract trajectories for training
    print(f"\n🧠 Extracting trajectories for training...")
    try:
        trajectories = []
        
        for result in batch_results:
            for team_result in result.team_results:
                # Find the corresponding task
                task = next((t for t in tasks if t.get("id") == result.task_id), {})
                
                # Create trajectory structure
                trajectory = {
                    "task": {
                        "id": result.task_id,
                        "question": task.get("question", ""),
                        "ground_truth": task.get("ground_truth", ""),
                        "source": task.get("source", "unknown")
                    },
                    "team_id": team_result.team_id,
                    "answer": team_result.answer,
                    "reasoning": team_result.reasoning,
                    "memory_used": team_result.memory_used,
                    "memory_entries": team_result.memory_entries,
                    "execution_time": team_result.execution_time,
                    "success": team_result.success,
                    "trajectory": team_result.trajectory,
                    "metadata": team_result.metadata if hasattr(team_result, 'metadata') and team_result.metadata else {},
                    "reuse_count": team_result.metadata.get("reuse_count", 0) if hasattr(team_result, 'metadata') and team_result.metadata else 0,
                    "subtasks_completed": team_result.metadata.get("subtasks_completed", 0) if hasattr(team_result, 'metadata') and team_result.metadata else 0,
                    "reward": evaluate_batch([{
                        "answer": team_result.answer,
                        "ground_truth": task.get("ground_truth", ""),
                        "source": task.get("source", "unknown")
                    }])["avg_reward"]
                }
                trajectories.append(trajectory)
        
        print(f"✅ Extracted {len(trajectories)} trajectories")
        
    except Exception as e:
        logger.error(f"Failed to extract trajectories: {e}")
        return
    
    # Save trajectories
    print(f"\n💾 Saving trajectories...")
    try:
        import torch
        torch.save(trajectories, config.trajectories_path)
        print(f"✅ Saved trajectories to {config.trajectories_path}")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_trajectories": len(trajectories),
            "num_tasks": len(tasks),
            "config": config.to_dict(),
            "aggregated_results": {
                "accuracy": aggregated.accuracy,
                "avg_reward": aggregated.avg_reward,
                "memory_usage_rate": aggregated.memory_usage_rate,
                "team_diversity": aggregated.team_diversity,
                "memory_bank_efficiency": aggregated.memory_bank_efficiency
            }
        }
        
        with open("training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to training_metadata.json")
        
    except Exception as e:
        logger.error(f"Failed to save trajectories: {e}")
        return
    
    # Generate Colab training script
    print(f"\n📝 Generating Colab training script...")
    try:
        colab_script = generate_colab_script(config)
        with open("colab_train.py", "w", encoding='utf-8') as f:
            f.write(colab_script)
        print(f"✅ Generated colab_train.py")
        
    except Exception as e:
        logger.error(f"Failed to generate Colab script: {e}")
        print(f"⚠️ Colab script generation failed, but continuing with controller save...")
    
    # Save controller
    print(f"\n💾 Saving current controller...")
    try:
        runner.save_controller()
        print(f"✅ Controller saved to {config.controller_path}")
        
        # Verify file was created
        if os.path.exists(config.controller_path):
            file_size = os.path.getsize(config.controller_path)
            print(f"✅ Controller file verified: {file_size} bytes")
        else:
            print(f"❌ Controller file not found after save!")
        
    except Exception as e:
        logger.error(f"Failed to save controller: {e}")
        print(f"❌ Error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    runner.shutdown()
    
    print(f"\n🎉 TRAIN MODE COMPLETED!")
    print(f"Next steps:")
    print(f"1. Upload {config.trajectories_path} to Google Colab")
    print(f"2. Run colab_train.py on Colab T4 GPU")
    print(f"3. Download trained_controller.pt from Colab")
    print(f"4. Place it in this directory for EVAL MODE")

async def eval_mode(config: LTSConfig, logger):
    """Evaluation mode: evaluate with trained controller"""
    print("\n🎯 Starting EVAL MODE")
    print("This will:")
    print("1. Load trained controller")
    print("2. Load tasks from local dataset")
    print("3. Evaluate performance with trained controller")
    
    # Check if controller exists
    if not os.path.exists(config.controller_path):
        print(f"❌ Controller not found: {config.controller_path}")
        print("Please run TRAIN MODE first or place trained_controller.pt here")
        return
    
    # Load tasks
    print(f"\n📁 Loading tasks from {config.dataset_path}")
    try:
        tasks = load_all_tasks(config.dataset_path)
        stats = get_task_stats(tasks)
        print(f"✅ Loaded {len(tasks)} tasks:")
        for source, count in stats.items():
            if source != "total":
                print(f"   - {source}: {count} tasks")
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        return
    
    # Initialize parallel runner (will load trained controller)
    print(f"\n🤖 Initializing parallel runner with trained controller...")
    try:
        runner = ParallelRunner(config)
        print(f"✅ Loaded trained controller")
        
        system_stats = runner.get_system_stats()
        print(f"Controller parameters: {system_stats['controller_parameters']:,}")
        
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return
    
    # Run evaluation
    print(f"\n🏃 Running evaluation on {len(tasks)} tasks...")
    start_time = time.time()
    
    try:
        batch_results = await runner.run_batch_parallel(tasks)
        execution_time = time.time() - start_time
        
        print(f"✅ Completed in {execution_time:.2f} seconds")
        print(f"   - Avg time per task: {execution_time/len(tasks):.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        return
    
    # Aggregate results
    print(f"\n📊 Analyzing evaluation results...")
    try:
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate_batch_results(batch_results, tasks)
        
        print(f"\n🎯 EVALUATION RESULTS:")
        print(f"=" * 50)
        print(f"Total Tasks: {aggregated.total_tasks}")
        print(f"Successful Tasks: {aggregated.successful_tasks}")
        print(f"Accuracy: {aggregated.accuracy:.2%}")
        print(f"Average Reward: {aggregated.avg_reward:.3f}")
        print(f"Memory Usage Rate: {aggregated.memory_usage_rate:.2%}")
        print(f"Team Diversity: {aggregated.team_diversity:.3f}")
        print(f"Average Execution Time: {aggregated.avg_execution_time:.2f}s")
        print(f"Memory Bank Efficiency: {aggregated.memory_bank_efficiency:.3f}")
        print(f"=" * 50)
        
        # Generate detailed report
        report = aggregator.generate_report(aggregated)
        with open("evaluation_report.md", "w") as f:
            f.write(report)
        print(f"📄 Detailed report saved to evaluation_report.md")
        
    except Exception as e:
        logger.error(f"Failed to analyze results: {e}")
        return
    
    # Cleanup
    runner.shutdown()
    
    print(f"\n🎉 EVAL MODE COMPLETED!")

async def demo_mode(config: LTSConfig, logger):
    """Demo mode: run on a few sample tasks"""
    print("\n🎪 Starting DEMO MODE")
    print("This will run a quick demo on 3 sample tasks")
    
    # Load a few sample tasks
    print(f"\n📁 Loading sample tasks...")
    try:
        all_tasks = load_all_tasks(config.dataset_path)
        tasks = all_tasks[:3]  # Just 3 tasks for demo
        
        print(f"✅ Selected {len(tasks)} demo tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task['question'][:50]}...")
            
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        return
    
    # Initialize runner
    print(f"\n🤖 Initializing runner...")
    try:
        runner = ParallelRunner(config)
        print(f"✅ System ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return
    
    # Run demo
    print(f"\n🏃 Running demo...")
    start_time = time.time()
    
    try:
        batch_results = await runner.run_batch_parallel(tasks)
        execution_time = time.time() - start_time
        
        print(f"✅ Demo completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to run demo: {e}")
        return
    
    # Show results
    print(f"\n📊 Demo Results:")
    print(f"=" * 60)
    
    for i, (result, task) in enumerate(zip(batch_results, tasks), 1):
        print(f"\nTask {i}: {task['question']}")
        print(f"Ground Truth: {task['ground_truth']}")
        print(f"LTS Answer: {result.aggregated_answer}")
        
        # Check if any team got it right
        correct_teams = []
        for team_result in result.team_results:
            if team_result.success:
                correct_teams.append(team_result.team_id)
        
        if correct_teams:
            print(f"✅ Correct teams: {', '.join(correct_teams)}")
        else:
            print(f"❌ No teams got it right")
        
        print(f"Memory used: {any(r.memory_used for r in result.team_results)}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print("-" * 40)
    
    # Cleanup
    runner.shutdown()
    
    print(f"\n🎉 DEMO MODE COMPLETED!")

def generate_colab_script(config: LTSConfig) -> str:
    """Generate Colab training script"""
    return f'''"""
Google Colab Training Script for LTS Memory Controller
Upload this to Google Colab along with {config.trajectories_path}
"""

# Install required packages
!pip install torch sentence-transformers matplotlib pandas

# Import Colab trainer
from colab_trainer import ColabTrainer

# Initialize trainer on GPU
trainer = ColabTrainer(device="cuda")

# Load trajectories from local execution
trajectories = trainer.load_trajectories("{config.trajectories_path}")

# Prepare training data (sample 50% for faster training)
train_examples = trainer.prepare_training_data(trajectories, sample_rate=0.5)

# Create new controller
controller = trainer.create_controller(
    embedding_dim={config.embedding_dim},
    hidden_dim={config.controller_hidden_dim}
)

# Train the controller
results = trainer.train(
    train_examples=train_examples,
    learning_rate={config.learning_rate},
    batch_size={config.batch_size},
    epochs={config.epochs}
)

# Plot training progress
trainer.plot_training_history()

# Generate training report
report = trainer.generate_training_report()
print(report)

# Save trained controller
trainer.save_controller("trained_controller.pt")

print("\\n🎉 Training completed! Download trained_controller.pt and place it in your local directory.")
'''

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Learning to Share (LTS) - Hybrid Implementation")
    parser.add_argument("--mode", choices=["train", "eval", "demo"], required=True,
                       help="Mode to run in")
    parser.add_argument("--config", default="config.json",
                       help="Configuration file path")
    parser.add_argument("--dataset", default="data/tasks.json",
                       help="Dataset file path")
    
    args = parser.parse_args()
    
    # Setup
    print_system_info()
    logger = setup_logging()
    
    # Load configuration
    config = LTSConfig()
    config.dataset_path = args.dataset
    
    print(f"\n⚙️ Configuration:")
    print(f"   - Mode: {args.mode}")
    print(f"   - Dataset: {config.dataset_path}")
    print(f"   - Max Teams: {config.max_parallel_teams}")
    print(f"   - Embedding Model: {config.embedding_model}")
    
    # Run appropriate mode
    try:
        if args.mode == "train":
            asyncio.run(train_mode(config, logger))
        elif args.mode == "eval":
            asyncio.run(eval_mode(config, logger))
        elif args.mode == "demo":
            asyncio.run(demo_mode(config, logger))
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}")
    
    print(f"\n👋 Goodbye!")

if __name__ == "__main__":
    main()
