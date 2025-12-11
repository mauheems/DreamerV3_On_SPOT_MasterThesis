# dreamer4/config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(torch.cuda.get_device_name(0))  # prints your GTX model)

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
LATENT_SIZE = 256
IMAGINATION_HORIZON = 15
