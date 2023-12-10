import logging
from collections import defaultdict

import numpy as np
import torch

from .constants import DEVICE

logger = logging.getLogger(__name__)


def omp(
    Phi: torch.Tensor,
    y: torch.Tensor,
    num_iterations: int | None = None,
    device: str | torch.device = DEVICE,
) -> dict:
    log = defaultdict(list)
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    y_hat = 0
    for k in range(num_iterations):
        logger.debug(f"OMP iteration {k} / {num_iterations}")
        residual = y - y_hat

        objective = torch.abs(Phi.transpose(1, 2) @ residual)
        log["objective"].append(objective.max(dim=1).values.ravel().tolist())
        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        log["indices"].append(curr_indices.ravel().tolist())
        x_hat = estimate_x_lsq(Phi, y, columns)
        log["x_hat"].append(x_hat)
        y_hat = Phi @ x_hat
        log["y_hat"].append(y_hat)

    return dict(log)


def estimate_x_lsq(
    Phi: torch.Tensor,
    y: torch.Tensor,
    indices,
    device=DEVICE,
    eps: float = 1e-12,
) -> torch.Tensor:
    if indices is None:
        return torch.zeros(Phi.shape[0], Phi.shape[2], 1, device=device)

    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape((-1, 1))

    Phi_t = Phi[batches, :, indices].transpose(1, 2)

    if len(y.size()) == 2:
        y = y.unsqueeze(2)

    x_hat = torch.zeros((Phi.shape[0], Phi.shape[2], y.size(2)), device=device)

    A = Phi_t.transpose(1, 2) @ Phi_t
    A += eps * torch.eye(A.shape[1], device=device)[None, :]

    B = Phi_t.transpose(1, 2) @ y

    L = torch.linalg.cholesky(A)
    coeff = torch.cholesky_inverse(L) @ B

    x_hat[batches, indices] = coeff

    return x_hat


def ip_objective(
    Phi: torch.Tensor,
    y: torch.Tensor,
    columns: torch.Tensor,
    batches: torch.Tensor,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    if columns is None:
        columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    if batches is None:
        batches = batches or torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape(-1, 1)

    if columns.size(1) == 0:
        Phi_projected = Phi.clone()  # this is project onto orthogonal complement

    else:
        Phi_coeff = estimate_x_lsq(Phi, Phi, columns)
        Phi_projected = Phi - (Phi @ Phi_coeff)  # this is projection onto orthogonal complement

    Phi_projected_normalized = Phi_projected / torch.linalg.norm(Phi_projected, dim=1)[:, None, :]
    objective = torch.abs(Phi_projected_normalized.transpose(1, 2) @ y)

    objective[batches, columns] = -np.inf
    return objective


def ip(
    Phi: torch.Tensor,
    y: torch.Tensor,
    num_iterations: int | None = None,
    device: str | torch.device = DEVICE,
    compute_xhat=True,
) -> dict:
    log = defaultdict(list)
    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape((-1, 1))
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    for k in range(num_iterations):
        logger.debug(f"IP iteration {k} / {num_iterations}")
        objective = ip_objective(Phi, y, batches=batches, columns=columns, device=device)
        max_objective = objective.max(dim=1).values
        logger.debug(f"IP iteration {k}, max objective = {max_objective.max().item()}")

        log["objective"].append(max_objective.ravel().tolist())

        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        log["indices"].append(curr_indices.ravel().tolist())
        log["columns"] = columns.clone()

        if compute_xhat:
            x_hat = estimate_x_lsq(Phi, y, columns)
            log["x_hat"].append(x_hat)
            log["y_hat"].append(Phi @ x_hat)

    return dict(log)
