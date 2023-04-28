import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 100
SPARSITY_MULTIPLE = 1
