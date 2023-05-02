import logging
from collections import defaultdict

import numpy as np
import torch

from .constants import DEVICE

logger = logging.getLogger(__name__)


def projection(
    Phi_t: torch.Tensor, perp: bool = False, device: str | torch.device = DEVICE
) -> torch.Tensor:
    if Phi_t.numel() == 0 and perp:
        return torch.eye(Phi_t.shape[1], device=device).repeat(Phi_t.shape[0], 1, 1)
    elif Phi_t.numel() == 0 and not perp:
        return torch.zeros(
            (Phi_t.shape[0], Phi_t.shape[1], Phi_t.shape[1]), device=device
        )

    U, *_ = torch.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.transpose(1, 2)

    if perp:
        return torch.eye(P.shape[1], device=device)[None, :, :] - P

    return P


def estimate_y(
    Phi: torch.Tensor, y: torch.Tensor, indices, device: str | torch.device = DEVICE
) -> torch.Tensor:
    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape(
        (-1, 1)
    )
    Phi_t = Phi[batches, :, indices].transpose(1, 2)
    P = projection(Phi_t, perp=False, device=device)
    return P @ y


def estimate_x_and_y(
    Phi: torch.Tensor, y: torch.Tensor, indices, device: str | torch.device = DEVICE
) -> torch.Tensor:
    if indices is None:
        return torch.zeros(Phi.shape[0], Phi.shape[2], 1, device=device)

    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape(
        (-1, 1)
    )

    Phi_t = Phi[batches, :, indices].transpose(1, 2)
    x_hat = torch.zeros((Phi.shape[0], Phi.shape[2], 1), device=device)

    # equivalent to left-multiplying y by pseudoinverse of Phi_t
    x_hat[batches, indices] = torch.linalg.lstsq(Phi_t, y).solution
    return x_hat, Phi @ x_hat


def ip_objective(
    Phi: torch.Tensor,
    y: torch.Tensor,
    columns: torch.Tensor | None = None,
    batches: torch.Tensor | None = None,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    if columns is None:
        columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    if batches is None:
        batches = batches or torch.arange(
            Phi.shape[0], dtype=torch.long, device=device
        ).reshape(-1, 1)

    P = projection(Phi[batches, :, columns].transpose(1, 2), perp=True, device=device)

    Phi_projected = P @ Phi
    Phi_projected_normalized = (
        Phi_projected / torch.linalg.norm(Phi_projected, dim=1)[:, None, :]
    )

    objective = torch.abs(Phi_projected_normalized.transpose(1, 2) @ y)
    objective[batches, columns] = -np.inf
    return objective


def omp(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    num_iterations: int | None = None,
    device: str | torch.device = DEVICE,
) -> dict:
    log = defaultdict(list)
    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape(
        (-1, 1)
    )
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    k = 0
    while k < Phi.shape[2]:
        logger.debug(f"OMP iteration {k} / {num_iterations}")
        P = projection(
            Phi[batches, :, columns].transpose(1, 2), perp=True, device=device
        )
        residual = P @ y

        # TODO: rethink termination criterion
        squared_error = residual.transpose(1, 2) @ residual
        if num_iterations is None and (
            (squared_error < tol).all() or k == Phi.shape[2] - 1
        ):
            break
        elif k == num_iterations:
            break

        objective = torch.abs(Phi.transpose(1, 2) @ residual)
        log["objective"].append(objective.max(dim=1).values.ravel().tolist())
        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        log["indices"].append(curr_indices.ravel().tolist())
        x_hat, y_hat = estimate_x_and_y(Phi, y, columns, device=device)
        log["x_hat"].append(x_hat)
        log["y_hat"].append(y_hat)
        k += 1

    return dict(log)


def ip(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    num_iterations: int | None = None,
    device: str | torch.device = DEVICE,
) -> dict:
    log = defaultdict(list)
    batches = torch.arange(Phi.shape[0], dtype=torch.long, device=device).reshape(
        (-1, 1)
    )
    columns = torch.empty(Phi.shape[0], 0, dtype=torch.long, device=device)
    k = 0
    while k < Phi.shape[2]:
        logger.debug(f"IP iteration {k} / {num_iterations}")
        objective = ip_objective(
            Phi, y, batches=batches, columns=columns, device=device
        )
        max_objective = objective.max(dim=1).values

        # TODO: rethink termination criterion
        if num_iterations is None and (
            (max_objective < tol).all() or k == Phi.shape[2] - 1
        ):
            break
        elif k == num_iterations:
            break

        log["objective"].append(max_objective.ravel().tolist())

        curr_indices = objective.argmax(dim=1)
        columns = torch.cat((columns, curr_indices), dim=1)

        # TODO: fix summarize code to take this as input (no longer list of lists)
        log["indices"].append(curr_indices.ravel().tolist())
        x_hat, y_hat = estimate_x_and_y(Phi, y, columns, device=device)
        log["x_hat"].append(x_hat)
        log["y_hat"].append(y_hat)
        k += 1

    return dict(log)
