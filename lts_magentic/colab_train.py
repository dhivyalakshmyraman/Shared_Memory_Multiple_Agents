"""
Google Colab Training Script for LTS Memory Controller
Upload this to Google Colab along with trajectories.pt
"""

# Install required packages
#!pip install torch sentence-transformers matplotlib pandas

# Import Colab trainer
from colab_trainer import ColabTrainer

# Initialize trainer on GPU
trainer = ColabTrainer(device="cuda")

# Load trajectories from local execution
trajectories = trainer.load_trajectories("trajectories.pt")

# Prepare training data (sample 50% for faster training)
train_examples = trainer.prepare_training_data(trajectories, sample_rate=0.5)

# Create new controller
controller = trainer.create_controller(
    embedding_dim=384,
    hidden_dim=256
)

# Train the controller
results = trainer.train(
    train_examples=train_examples,
    learning_rate=0.0001,
    batch_size=32,
    epochs=10
)

# Plot training progress
trainer.plot_training_history()

# Generate training report
report = trainer.generate_training_report()
print(report)

# Save trained controller
trainer.save_controller("trained_controller.pt")

print("\n🎉 Training completed! Download trained_controller.pt and place it in your local directory.")
