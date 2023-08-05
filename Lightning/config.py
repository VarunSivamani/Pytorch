import torch

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 512
MAX_EPOCHS = 24

CLASSES = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
