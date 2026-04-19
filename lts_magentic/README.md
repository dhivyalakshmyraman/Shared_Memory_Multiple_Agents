# Learning to Share (LTS) - Hybrid Local + Colab Implementation

A complete implementation of the "Learning to Share" (LTS) paper using Microsoft AutoGen + MagenticOne with a hybrid architecture:

- 💻 **Local (Cursor)** → execution engine
- ☁️ **Google Colab (T4 GPU)** → controller training

## 🚀 Key Features

- **Zero-download**: Prepackaged dataset samples for maximum stability
- **Hybrid architecture**: Local execution + Colab GPU training
- **Multi-agent orchestration**: AutoGen + MagenticOne framework
- **Parallel execution**: K=3 teams running simultaneously
- **Shared MemoryBank**: Cross-team memory reuse
- **Neural controller**: Learns when to use shared memory
- **Robust evaluation**: Handles GSM8K, HotpotQA, and Math datasets

## 📁 Project Structure

```
lts_magentic/
├── main.py              # Main entry point
├── config.py            # Configuration management
├── memory_bank.py       # Shared memory bank
├── controller.py        # Neural memory controller
├── embedder.py          # Text embeddings
├── lts_team.py          # Multi-agent team
├── parallel_runner.py   # Parallel execution engine
├── aggregator.py        # Result aggregation
├── rewards.py           # Reward functions
├── rl_trainer.py        # Local RL trainer
├── colab_trainer.py     # Colab GPU trainer
├── datasets.py          # Local dataset loading
├── data/
│   └── tasks.json       # Prepackaged dataset
├── requirements.txt     # Dependencies
├── .env.example         # Environment variables
└── README.md           # This file
```

## 🛠️ Installation

1. **Clone and setup:**
```bash
cd lts_magentic
pip install -r requirements.txt
cp .env.example .env
```

2. **Optional: Setup Ollama for local models**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull worker models
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
```

## 🎯 Usage

### Train Mode (Local Execution)
```bash
python main.py --mode train
```

This will:
1. Load 12 prepackaged tasks from `data/tasks.json`
2. Run 3 parallel teams on all tasks
3. Collect trajectories and save to `trajectories.pt`
4. Generate `colab_train.py` for GPU training
5. Save initial controller to `controller.pt`

### Colab Training
1. Upload `trajectories.pt` to Google Colab
2. Run the generated `colab_train.py` script
3. Download `trained_controller.pt`
4. Place it in the local directory

### Eval Mode (with trained controller)
```bash
python main.py --mode eval
```

### Demo Mode (quick test)
```bash
python main.py --mode demo
```

## 📊 Dataset

The system uses a prepackaged dataset with 12 tasks across 3 domains:

- **GSM8K** (4 tasks): Arithmetic reasoning
- **HotpotQA** (4 tasks): Multi-hop factual reasoning  
- **Math** (4 tasks): Symbolic reasoning

Sample tasks:
```json
{
  "source": "gsm8k",
  "question": "Janet has 3 apples and buys 5 more. She gives 2 away. How many apples does she have?",
  "ground_truth": "6"
}
```

## 🧠 Architecture

### Local Execution Engine
- **Multi-agent orchestration**: AutoGen + MagenticOne
- **Parallel teams**: K=3 teams running simultaneously
- **Shared MemoryBank**: Cross-team memory reuse
- **Embedding-based search**: Find relevant memories
- **Neural controller**: Decide when to use memory

### Colab Training Engine
- **GPU acceleration**: T4 GPU training
- **Trajectory learning**: Learn from execution data
- **Neural controller**: MLP for memory decisions
- **Visualization**: Training progress plots

## 🔄 Workflow

1. **Local Training Phase**:
   - Run parallel teams on tasks
   - Collect trajectories with memory decisions
   - Save for GPU training

2. **Colab Training Phase**:
   - Load trajectories
   - Train neural controller on GPU
   - Save improved controller

3. **Evaluation Phase**:
   - Load trained controller
   - Run evaluation on same tasks
   - Measure improvement

## 📈 Expected Behavior

- **GSM8K**: Arithmetic reasoning with memory reuse
- **HotpotQA**: Multi-hop factual reasoning
- **Math**: Symbolic problem solving

Memory sharing should:
- Help reuse intermediate results
- Reduce redundant computation
- Improve final answer selection

## 🎯 Success Criteria

✅ Loads 12 tasks from JSON  
✅ Runs 3 parallel teams  
✅ Memory entries shared across teams  
✅ trajectories.pt generated  
✅ controller.pt improves after training  

## 🔧 Configuration

Key settings in `config.py`:

```python
# System settings
max_parallel_teams = 3
memory_bank_size = 1000
embedding_dim = 384

# Training settings
learning_rate = 1e-4
batch_size = 32
epochs = 10

# Models
orchestrator_model = "groq/llama-3.3-70b"
worker_models = ["qwen2.5:7b", "llama3.2:3b"]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

## 🐛 Troubleshooting

**Common Issues:**

1. **Missing dependencies**: `pip install -r requirements.txt`
2. **CUDA not available**: System will fall back to CPU
3. **Memory issues**: Reduce `memory_bank_size` or `batch_size`
4. **Ollama not running**: Start with `ollama serve`

**Debug mode:**
```bash
python main.py --mode demo  # Quick test
```

## 📝 Logs and Outputs

- `lts.log`: Detailed execution logs
- `trajectories.pt`: Training data for Colab
- `controller.pt`: Trained memory controller
- `evaluation_report.md`: Performance analysis
- `training_metadata.json`: Training session info

## 🤝 Contributing

This is a research implementation. Key areas for improvement:

1. **Real AutoGen integration**: Replace mock agents
2. **Larger datasets**: Add more tasks
3. **Advanced reward functions**: Better learning signals
4. **Memory efficiency**: Optimize memory bank usage

## 📄 License

MIT License - feel free to use for research and development.

## 🔗 References

- Original LTS paper
- Microsoft AutoGen framework
- MagenticOne architecture
- Sentence Transformers
