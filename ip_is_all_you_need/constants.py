import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 20
SPARSITY_MULTIPLE = 1
