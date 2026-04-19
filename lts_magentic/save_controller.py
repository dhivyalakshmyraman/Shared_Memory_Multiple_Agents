import torch
from controller import MemoryController

# Create a new controller
controller = MemoryController(embedding_dim=384, hidden_dim=256, num_layers=2)
controller.save('controller.pt')
print('Controller saved successfully')
