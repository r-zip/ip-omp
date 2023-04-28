import torch


def recall(estimated: list[int], true: list[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(true)


def precision(estimated: list[int], true: list[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(estimated)


def iou(set_1: list[int], set_2: list[int]) -> float:
    return len(set(set_1).intersection(set_2)) / len(set(set_1).union(set_2))


def mse(estimated: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((estimated - true) ** 2).item()


def mutual_coherence(Phi: torch.Tensor) -> float:
    return torch.max(torch.abs(torch.triu(Phi.T @ Phi, diagonal=1))).item()
