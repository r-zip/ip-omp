import torch


def recall(estimated: list[int], true: list[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(true)


def precision(estimated: list[int], true: list[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(estimated)


def iou(estimated: list[int], true: list[int]) -> float:
    return len(set(estimated).intersection(true)) / len(set(estimated).union(true))


def mse(estimated: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((estimated - true) ** 2).item()


def mutual_coherence(Phi: torch.Tensor) -> float:
    return torch.max(torch.abs(torch.triu(Phi.T @ Phi, diagonal=1))).item()
