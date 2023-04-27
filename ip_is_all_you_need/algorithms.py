from collections import defaultdict
from copy import copy

import numpy as np
import torch

from .constants import DEVICE


def projection(Phi_t: torch.Tensor, perp: bool = False) -> torch.Tensor:
    if Phi_t.numel() == 0 and perp:
        return torch.eye(Phi_t.shape[1]).repeat(Phi_t.shape[0], 1, 1)
    elif Phi_t.numel() == 0 and not perp:
        return torch.zeros((Phi_t.shape[0], Phi_t.shape[1], Phi_t.shape[1]))

    U, *_ = torch.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.transpose(1, 2)

    if perp:
        return torch.eye(P.shape[1]).to(DEVICE)[None, :, :] - P

    return P


def estimate_y(Phi: torch.Tensor, y: torch.Tensor, indices):
    Phi_t = Phi[:, :, indices]
    return projection(Phi_t, perp=False) @ y


def estimate_x(Phi: torch.Tensor, y: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros(Phi.shape[0], Phi.shape[2], 1).to(DEVICE)

    Phi_t = Phi[:, :, indices]
    x_hat = torch.zeros((Phi.shape[0], Phi.shape[2], 1)).to(DEVICE)
    x_hat[:, indices] = torch.linalg.pinv(Phi_t) @ y
    return x_hat


def ip_objective(
    Phi: torch.Tensor,
    y: torch.Tensor,
    columns: torch.Tensor | None = None,
    batches: torch.Tensor | None = None,
) -> torch.Tensor:
    if columns is None:
        columns = torch.empty(Phi.shape[0], 0, dtype=torch.long)
    if batches is None:
        batches = batches or torch.arange(Phi.shape[0], dtype=torch.long).reshape(-1, 1)

    P = projection(Phi[batches, :, columns].transpose(1, 2), perp=True)

    Phi_projected = P @ Phi
    Phi_projected_normalized = (
        Phi_projected / torch.linalg.norm(Phi_projected, dim=1)[:, None, :]
    )

    objective = torch.abs(Phi_projected_normalized.transpose(1, 2) @ y)
    objective[:, columns.ravel()] = -np.inf
    return objective


def update_index_array(
    index_array: torch.Tensor, new_indices: torch.Tensor
) -> torch.Tensor:
    pass


def omp(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    num_iterations: int | None = None,
) -> dict:
    log = defaultdict(list)
    batches = torch.arange(Phi.shape[0], dtype=torch.long).reshape((-1, 1))
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long)
    k = 0
    while k < Phi.shape[2]:
        P = projection(Phi[batches, :, columns].transpose(1, 2), perp=True)
        residual = P @ y

        # TODO: rethink termination criterion
        squared_error = residual.transpose(1, 2) @ residual
        if num_iterations is None and (
            (squared_error < tol).all() or k == Phi.shape[1] - 1
        ):
            break
        elif k == num_iterations:
            break

        objective = torch.absolute(Phi.transpose(1, 2) @ residual)
        log["objective"].append(objective.max().item())
        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        log["indices"].append(curr_indices.ravel().tolist())
        # y_hat = estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)


def ip(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    num_iterations: int | None = None,
) -> dict:
    log = defaultdict(list)
    batches = torch.arange(Phi.shape[0], dtype=torch.long).reshape((-1, 1))
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long)
    k = 0
    while k < Phi.shape[2]:
        objective = ip_objective(Phi, y, batches=batches, columns=columns)
        max_objective = objective.max()

        # TODO: rethink termination criterion
        if num_iterations is None and (
            torch.absolute(max_objective) < tol or k == Phi.shape[1] - 1
        ):
            break
        elif k == num_iterations:
            break

        log["objective"].append(max_objective.item())

        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        # TODO: fix summarize code to take this as input (no longer list of lists)
        log["indices"].append(curr_indices.ravel().tolist())
        # y_hat = estimate_y(Phi, y, indices)
        # log["y_hat"].append(y_hat)
        # x_hat = estimate_x(Phi, y, indices)
        # log["x_hat"].append(x_hat)
        k += 1

    return dict(log)
