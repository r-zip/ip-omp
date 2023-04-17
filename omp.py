import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import typer
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "Helvetica",
#     }
# )

device_: str | torch.device = "cpu"
log_frequency_: int | None = None


def gen_dictionary(m: int, n: int) -> torch.Tensor:
    Phi = torch.randn(m, n, device=device_)
    return Phi / torch.linalg.norm(Phi, dim=0)


def projection(Phi_t: torch.Tensor, perp: bool = False) -> torch.Tensor:
    U, *_ = torch.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.T

    if perp:
        return torch.eye(P.shape[0], device=device_) - P

    return P


def generate_measurements_and_coeffs(
    Phi: torch.Tensor, p: float = 0.01, noise_std: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = Phi.shape
    supp = torch.rand(n, device=device_) <= p
    x = torch.zeros(n, device=device_)
    x[supp] = torch.randn(int(torch.sum(supp)), device=device_)
    return (Phi @ x + noise_std * torch.randn(m, device=device_)).reshape(
        -1, 1
    ), x.reshape(-1, 1)


class Log:
    def __init__(self, debug: list[str] | None = None) -> None:
        self.debug = set(debug or [])
        self._keys = []

    def log(self, key: str, value: Any, context: dict | str | None = None) -> None:
        if not hasattr(self, key):
            self.keys().append(key)
            setattr(self, key, [])

        if self.debug and context:
            print(f"[{context}] {key}: {value}")
        elif key in self.debug:
            print(f"{key}: {value}")

        getattr(self, key).append(deepcopy(value))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}

    def keys(self) -> list[str]:
        return self._keys

    def __repr__(self) -> str:
        return f"Log({self.keys()})"


def ip_objective(
    Phi: torch.Tensor, y: torch.Tensor, indices: list[int] | None = None
) -> torch.Tensor:
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    Phi_projected_normalized = Phi_projected / torch.linalg.norm(
        Phi_projected, dim=0
    ).reshape(1, -1)
    objective = torch.abs(Phi_projected_normalized.T @ y)
    objective[indices] = -torch.inf
    return objective


def omp_objective(
    Phi: torch.Tensor, y: torch.Tensor, indices: list[int] | None = None
) -> torch.Tensor:
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    return torch.abs(Phi_projected.T @ y)


def omp_estimate_y(
    Phi: torch.Tensor, indices: list[int], y: torch.Tensor
) -> torch.Tensor:
    Phi_t = Phi[:, indices]
    return projection(Phi_t, perp=False) @ y


def ip_estimate_y(
    Phi: torch.Tensor, indices: list[int], y: torch.Tensor
) -> torch.Tensor:
    return omp_estimate_y(Phi, indices, y)


def omp_estimate_x(
    Phi: torch.Tensor, indices: list[int], y: torch.Tensor
) -> torch.Tensor:
    Phi_t = Phi[:, indices]
    x_hat = torch.zeros((Phi.shape[1], 1), device=device_)
    x_hat[indices] = torch.linalg.pinv(Phi_t) @ y
    return x_hat


def ip_estimate_x(
    Phi: torch.Tensor, indices: list[int], y: torch.Tensor
) -> torch.Tensor:
    return omp_estimate_x(Phi, indices, y)


def ip(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    debug: list[str] | None = None,
) -> Log:
    log = Log(debug=debug)
    indices = []
    k = 0
    while True:
        if log_frequency_ is not None and (k + 1) % log_frequency_ == 0:
            logger.debug(f"Starting OMP iteration {k}.")
        objective = ip_objective(Phi, y, indices=indices)
        max_objective = objective.max()
        log.log("objective", max_objective.item())
        if torch.abs(max_objective) < tol:
            break
        indices.append(torch.argmax(objective).item())
        log.log("indices", indices)
        y_hat = ip_estimate_y(Phi, indices, y)
        log.log("y_hat", y_hat)
        x_hat = ip_estimate_x(Phi, indices, y)
        log.log("x_hat", x_hat)
        k += 1

    return log


def omp(
    Phi: torch.Tensor,
    y: torch.Tensor,
    tol: float = 1e-6,
    debug: list[str] | None = None,
) -> Log:
    log = Log(debug=debug)
    indices = []
    k = 0
    while True:
        if log_frequency_ is not None and (k + 1) % log_frequency_ == 0:
            logger.info(f"Starting OMP iteration {k}.")
        P = projection(Phi[:, indices], perp=True)
        residual = P @ y
        squared_error = residual.T @ residual
        if squared_error < tol:
            break
        objective = torch.abs(Phi.T @ residual)
        log.log("objective", objective.max().item())
        indices.append(torch.argmax(objective).item())
        log.log("indices", indices)
        y_hat = ip_estimate_y(Phi, indices, y)
        log.log("y_hat", y_hat)
        x_hat = omp_estimate_x(Phi, indices, y)
        log.log("x_hat", x_hat)
        k += 1

    return log


def recall(estimated: list[int] | set[int], true: list[int] | set[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(true)


def precision(estimated: list[int] | set[int], true: list[int] | set[int]) -> float:
    return len(set(estimated).intersection(set(true))) / len(estimated)


def mse(estimated: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((estimated - true) ** 2).item()


def main(
    m: int,
    n: int,
    s: float,
    output_dir: Path,
    device: str | None = None,
    plot: bool = False,
    log_frequency: int | None = None,
):
    global device_
    global log_frequency_
    log_frequency_ = log_frequency
    if log_frequency_:
        logger.setLevel(logging.DEBUG)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Using device {device}.")

    with torch.no_grad():
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Saving to {output_dir}.")

        sns.set()

        logger.info(f"Generating dictionary of size {(m, n)}")
        Phi = gen_dictionary(m, n)
        y, x = generate_measurements_and_coeffs(Phi, p=s)
        true_support = set(torch.where(x.ravel() != 0)[0])
        y = y.reshape(-1, 1)

        logger.info("Running IP.")
        log_ip = ip(Phi, y, debug=None)
        logger.info("Running OMP.")
        log_omp = omp(Phi, y, debug=None)

        ip_recall = []
        omp_recall = []
        ip_precision = []
        omp_precision = []
        ip_mse_y = []
        omp_mse_y = []
        ip_mse_x = []
        omp_mse_x = []

        logger.info("Summarizing IP results.")
        for indices, x_hat, y_hat in zip(log_ip.indices, log_ip.x_hat, log_ip.y_hat):
            ip_recall.append(recall(indices, true_support))
            ip_precision.append(precision(indices, true_support))
            ip_mse_x.append(mse(x_hat, x))
            ip_mse_y.append(mse(y_hat, y))

        logger.info("Summarizing OMP results.")
        for indices, x_hat, y_hat in zip(log_omp.indices, log_omp.x_hat, log_omp.y_hat):
            omp_recall.append(recall(indices, true_support))
            omp_precision.append(precision(indices, true_support))
            omp_mse_x.append(mse(x_hat, x))
            omp_mse_y.append(mse(y_hat, y))

        if plot:
            logger.info("Generating single-run plots.")
            plt.plot(omp_recall)
            plt.plot(ip_recall, "--")
            plt.legend(["OMP", "IP"])
            plt.xlabel("Iteration")
            plt.ylabel(
                r"$\frac{|\mathrm{supp}(\widehat{x}) \, \cap \, \mathrm{supp}(x)|}{|\mathrm{supp}(x)|}$",
                fontsize="x-large",
            )
            plt.title("Recall of estimated support")
            plt.savefig(output_dir / "recall.png", dpi=300)
            plt.close()

            plt.plot(omp_precision)
            plt.plot(ip_precision, "--")
            plt.legend(["OMP", "IP"])
            plt.xlabel("Iteration")
            plt.ylabel(
                r"$\frac{|\mathrm{supp}(\widehat{x}) \, \cap \, \mathrm{supp}(x)|}{|\mathrm{supp}(\widehat{x})|}$",
                fontsize="x-large",
            )
            plt.title("Precision of estimated support")
            plt.savefig(output_dir / "precision.png", dpi=300)
            plt.close()

            plt.plot(omp_mse_x)
            plt.plot(ip_mse_x, "--")
            plt.legend(["OMP", "IP"])
            plt.xlabel("Iteration")
            plt.ylabel(r"$\|x - \widehat{x}\|_2^2$")
            plt.title("MSE of Sparse Code Estimate")
            plt.savefig(output_dir / "mse_x.png", dpi=300)
            plt.close()

            plt.plot(omp_mse_y)
            plt.plot(ip_mse_y, "--")
            plt.legend(["OMP", "IP"])
            plt.xlabel("Iteration")
            plt.ylabel(r"$\|y - \widehat{y}\|_2^2$")
            plt.title("MSE of Measurement Estimate")
            plt.savefig(output_dir / "mse_y.png", dpi=300)
            plt.close()

        results = {
            "precision_ip": ip_precision,
            "precision_omp": omp_precision,
            "recall_ip": ip_recall,
            "recall_omp": omp_recall,
            "mse_x_ip": ip_mse_x,
            "mse_x_omp": omp_mse_x,
            "mse_y_ip": ip_mse_y,
            "mse_y_omp": omp_mse_y,
            "iters_ip": len(log_ip.indices),
            "iters_omp": len(log_omp.indices),
            "max_objective_ip": log_ip.objective,
            "max_objective_omp": log_omp.objective,
        }

        logger.info(f"Saving results to {output_dir / 'results.json'}")
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    typer.run(main)
