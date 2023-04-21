import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from copy import copy
from itertools import chain, product, repeat
from multiprocessing import cpu_count
from pathlib import Path

import GPUtil
import typer
from rich.logging import RichHandler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()

TRIALS = 20
SPARSITY_MULTIPLE = 1


# fmt: off
SETTINGS = {
    "dimensions": [
        # (500, 500),
        (500, 750),
        # (500, 1000),
        # (500, 1250),
        # (500, 1500),
    ],
    "sparsity": [
        0.01,
        0.05,
        0.1,
        # 0.2,
        # 0.3,
        # 0.4,
    ],
    "noise_std": [
        0.0,
        # 0.01,
        # 0.1,
    ],
}
# fmt: on


try:
    import cupy as np

    gpu = True
except ModuleNotFoundError:
    gpu = False
    logger.info("cupy installation not found. Falling back to numpy.")
    import numpy as np


def gen_dictionary(m, n):
    Phi = np.random.randn(m, n)
    return Phi / np.linalg.norm(Phi, axis=0)


def projection(Phi_t, perp=False):
    logger.debug(f"Computing projection for {Phi_t.shape[1]} columns of Phi")
    U, *_ = np.linalg.svd(Phi_t, full_matrices=False)
    P = U @ U.T

    if perp:
        return np.eye(P.shape[0]) - P

    return P


def generate_measurements_and_coeffs(Phi, p=0.01, noise_std=0.0):
    m, n = Phi.shape
    supp = np.random.rand(n) <= p
    x = np.zeros(n)
    x[supp] = np.random.randn(int(np.sum(supp)))
    return (Phi @ x + noise_std * np.random.randn(m)).reshape(-1, 1), x.reshape(-1, 1)


def ip_objective(Phi, y, indices=None):
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    Phi_projected_normalized = Phi_projected / np.linalg.norm(
        Phi_projected, axis=0
    ).reshape(1, -1)
    objective = np.abs(Phi_projected_normalized.T @ y)
    objective[indices] = -np.inf
    return objective


def omp_objective(Phi, y, indices=None):
    P = projection(Phi[:, indices], perp=True)
    Phi_projected = P @ Phi
    return np.abs(Phi_projected.T @ y)


def omp_estimate_y(Phi, indices, y):
    Phi_t = Phi[:, indices]
    return projection(Phi_t, perp=False) @ y


def ip_estimate_y(Phi, indices, y):
    return omp_estimate_y(Phi, indices, y)


def omp_estimate_x(Phi, indices, y):
    Phi_t = Phi[:, indices]
    x_hat = np.zeros((Phi.shape[1], 1))
    logger.debug("Computing pseudo-inverse estimate of x_hat")
    x_hat[indices] = np.linalg.pinv(Phi_t) @ y
    return x_hat


def ip_estimate_x(Phi, indices, y):
    return omp_estimate_x(Phi, indices, y)


def ip(Phi, y, sparsity, tol=1e-6):
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        if k >= SPARSITY_MULTIPLE * sparsity:
            break
        logger.debug(f"Starting iteration {k} of IP")
        objective = ip_objective(Phi, y, indices=indices)
        max_objective = objective.max()
        log["objective"].append(max_objective.item())

        # TODO: rethink termination criterion
        # if np.abs(max_objective) < tol:
        #     break

        indices.append(np.argmax(objective).item())
        log["indices"].append(copy(indices))
        y_hat = ip_estimate_y(Phi, indices, y)
        log["y_hat"].append(y_hat)
        x_hat = ip_estimate_x(Phi, indices, y)
        log["x_hat"].append(x_hat)
        k += 1

    return log


def omp(Phi, y, sparsity, tol=1e-6):
    log = defaultdict(list)
    indices = []
    k = 0
    while True:
        if k >= SPARSITY_MULTIPLE * sparsity:
            break
        logger.debug(f"Starting iteration {k} of OMP")
        P = projection(Phi[:, indices], perp=True)
        residual = P @ y

        # TODO: rethink termination criterion
        # squared_error = residual.T @ residual
        # if squared_error < tol:
        #     break

        objective = np.abs(Phi.T @ residual)
        log["objective"].append(objective.max().item())
        indices.append(np.argmax(objective).item())
        log["indices"].append(copy(indices))
        y_hat = ip_estimate_y(Phi, indices, y)
        log["y_hat"].append(y_hat)
        x_hat = omp_estimate_x(Phi, indices, y)
        log["x_hat"].append(x_hat)
        k += 1

    return log


def recall(estimated, true):
    return len(set(estimated).intersection(set(true))) / len(true)


def precision(estimated, true):
    return len(set(estimated).intersection(set(true))) / len(estimated)


def iou(estimated, true):
    return len(set(estimated).intersection(true)) / len(set(estimated).union(true))


def mse(estimated, true):
    return np.mean((estimated - true) ** 2).item()


def mutual_coherence(Phi):
    return np.max(np.abs(np.triu(Phi.T @ Phi))).item()


def run_experiment(
    experiment_number: int,
    m: int,
    n: int,
    s: float,
    noise_std: float,
    output_dir: Path,
    device: int | None = None,
):
    if gpu and device is not None:
        # get device
        context = np.cuda.Device(device)
    elif gpu:
        context = np.cuda.Device(
            GPUtil.getFirstAvailable(order="load", maxLoad=1.0, maxMemory=1.0)[0]
        )
    else:
        # dummy context manager
        context = nullcontext()

    with context:
        output_dir.mkdir(exist_ok=True)
        experiment_results_dir = output_dir / str(experiment_number)
        experiment_results_dir.mkdir(exist_ok=True)

        settings = {
            "experiment_number": experiment_number,
            "m": m,
            "n": n,
            "measurement_rate": m / n,
            "sparsity": s,
            "noise_std": noise_std,
            "output_dir": str(output_dir),
        }

        records = []
        for trial in range(TRIALS):
            logger.info(f"Starting trial {trial + 1} / {TRIALS}")
            logger.info(
                f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}"
            )
            Phi = gen_dictionary(m, n)
            y, x = generate_measurements_and_coeffs(Phi, p=s, noise_std=noise_std)

            true_support = set(np.where(x.ravel() != 0)[0].tolist())
            y = y.reshape(-1, 1)
            logger.info("Running IP")
            log_ip = ip(Phi, y, sparsity=np.count_nonzero(x))
            logger.info("Running OMP")
            log_omp = omp(Phi, y, sparsity=np.count_nonzero(x))

            ip_recall = []
            omp_recall = []
            ip_precision = []
            omp_precision = []
            ip_mse_y = []
            omp_mse_y = []
            ip_mse_x = []
            omp_mse_x = []

            logger.info("Generating metrics for IP")
            for indices, x_hat, y_hat in zip(
                log_ip["indices"], log_ip["x_hat"], log_ip["y_hat"]
            ):
                ip_recall.append(recall(indices, true_support))
                ip_precision.append(precision(indices, true_support))
                ip_mse_x.append(mse(x_hat, x))
                ip_mse_y.append(mse(y_hat, y))

            logger.info("Generating metrics for OMP")
            for indices, x_hat, y_hat in zip(
                log_omp["indices"], log_omp["x_hat"], log_omp["y_hat"]
            ):
                omp_recall.append(recall(indices, true_support))
                omp_precision.append(precision(indices, true_support))
                omp_mse_x.append(mse(x_hat, x))
                omp_mse_y.append(mse(y_hat, y))

            results = {
                "coherence": mutual_coherence(Phi),
                "precision_ip": ip_precision,
                "precision_omp": omp_precision,
                "recall_ip": ip_recall,
                "recall_omp": omp_recall,
                "mse_x_ip": ip_mse_x,
                "mse_x_omp": omp_mse_x,
                "mse_y_ip": ip_mse_y,
                "mse_y_omp": omp_mse_y,
                "iters_ip": len(log_ip["indices"]),
                "iters_omp": len(log_omp["indices"]),
                "max_objective_ip": log_ip["objective"],
                "max_objective_omp": log_omp["objective"],
            }

            with open(experiment_results_dir / f"results_{trial}.json", "w") as f:
                json.dump(results, f)

            records.append(results)

        logger.info(f"Saving metrics to {experiment_results_dir / 'results.json'}")
        with open(experiment_results_dir / "results.json", "w") as f:
            json.dump({**settings, "results": records}, f)


def main(
    results_dir: Path,
    overwrite: bool = False,
    jobs: int = 1,
):
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    if jobs > 1:
        if gpu:
            gpu_list = [
                g for g in GPUtil.getAvailable(maxLoad=0.2, maxMemory=0.2, limit=jobs)
            ]
            workers = len(gpu_list)
        else:
            gpu_list = []
            workers = min(jobs, cpu_count())

        if workers < jobs:
            logger.info(
                f"Running {workers} jobs; {jobs} was too many for system resources"
            )

        pool = ProcessPoolExecutor(max_workers=workers)

        gpus = chain(gpu_list, repeat(None))
        futures = []
        for k, ((m, n), s, noise_std) in enumerate(
            product(SETTINGS["dimensions"], SETTINGS["sparsity"], SETTINGS["noise_std"])
        ):
            futures.append(
                pool.submit(
                    run_experiment,
                    k,
                    m,
                    n,
                    s,
                    output_dir=results_dir,
                    noise_std=noise_std,
                    device=next(gpus),
                )
            )
        # progress bar
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    else:
        for k, ((m, n), s, noise_std) in enumerate(
            tqdm(
                product(
                    SETTINGS["dimensions"], SETTINGS["sparsity"], SETTINGS["noise_std"]
                ),
                total=len(SETTINGS["dimensions"])
                * len(SETTINGS["sparsity"])
                * len(SETTINGS["noise_std"]),
            )
        ):
            run_experiment(
                k,
                m,
                n,
                s,
                output_dir=results_dir,
                noise_std=noise_std,
            )

    logger.info("Done!")


if __name__ == "__main__":
    typer.run(main)
