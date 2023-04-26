from copy import copy
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def projection(Phi_t: torch.Tensor, perp: bool = False) -> torch.Tensor:
    U, *_ = torch.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.T

    if perp:
        return torch.eye(P.shape[0]).to(device) - P

    return P


def omp_estimate_y(Phi: torch.Tensor, y: torch.Tensor, indices):
    Phi_t = Phi[:, indices]
    return projection(Phi_t, perp=False) @ y


def ip_estimate_y(Phi: torch.Tensor, y: torch.Tensor, indices):
    return omp_estimate_y(Phi, y, indices)


def omp_estimate_x(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    Phi_t = Phi[:, indices]
    x_hat = torch.zeros((Phi.shape[1])).to(device)
    x_hat[indices] = torch.linalg.pinv(Phi_t) @ y
    return x_hat


def ip_estimate_x(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    return omp_estimate_x(Phi, y, indices)


def mutual_coherence(Phi: np.ndarray) -> float:
    plt.hist(np.abs(np.triu(Phi.T @ Phi, k=1)))
    plt.savefig("useless.png")
    return np.max(np.abs(np.triu(Phi.T @ Phi, k=1))).item()


def ip_objective(Phi: torch.Tensor, y: torch.Tensor, indices) -> torch.Tensor:
    P = projection(Phi[:, indices], perp=True)

    Phi_projected = P @ Phi
    Phi_projected_normalized = Phi_projected / torch.linalg.norm(
        Phi_projected, axis=0
    ).reshape(1, -1)

    objective = torch.absolute(Phi_projected_normalized.T @ y)
    objective[indices] = -np.inf
    return objective


def omp(Phi: np.ndarray, y: torch.Tensor, tol: float = 1e-6) -> dict:
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        P = projection(Phi[:, indices], perp=True)
        residual = P @ y

        # TODO: rethink termination criterion
        squared_error = residual.T @ residual
        if squared_error < tol or k == Phi.shape[1] - 1:
            break

        objective = torch.absolute(Phi.T @ residual)
        log["objective"].append(objective.max().item())
        indices.append(torch.argmax(objective).item())
        log["indices"] = copy(indices)
        # y_hat = omp_estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = omp_estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)


def ip(Phi: torch.Tensor, y: torch.Tensor, tol: float = 1e-6, num_iterations=None):
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        objective = ip_objective(Phi, y, indices=indices)
        max_objective = objective.max()
        log["objective"].append(max_objective.item())

        # TODO: rethink termination criterion
        if num_iterations is None and (
            torch.absolute(max_objective) < tol or k == Phi.shape[1] - 1
        ):
            break
        elif k == num_iterations:
            break

        indices.append(torch.argmax(objective).item())
        log["indices"] = copy(indices)
        # y_hat = ip_estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = ip_estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)


def mse(estimated: torch.Tensor, true: torch.Tensor):
    return torch.mean((estimated - true) ** 2).item()
