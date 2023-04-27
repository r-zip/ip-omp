import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain, product, repeat
from multiprocessing import cpu_count
from pathlib import Path

import GPUtil
import torch
import typer
from rich.logging import RichHandler
from tqdm import tqdm

from .algorithms import ip, omp
from .constants import DEVICE, TRIALS
from .metrics import iou, mse, mutual_coherence, precision, recall

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()

# fmt: off
SETTINGS = {
    "dimensions": [
        # (500, 500),
        # (500, 750),
        # (500, 1000),
        # (500, 1250),
        (500, 1500),
    ],
    "sparsity": [
        # 0.01,
        # 0.05,
        # 0.1,
        0.2,
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

NUM_SETTINGS = len(list(product(*list(SETTINGS.values()))))


def gen_dictionary(batch_size: int, m: int, n: int) -> torch.Tensor:
    Phi = torch.randn(batch_size, m, n).to(DEVICE)
    return Phi / torch.linalg.norm(Phi, dim=1)[:, None, :]


def generate_measurements_and_coeffs(
    Phi: torch.Tensor, p: float = 0.01, noise_std: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, m, n = Phi.shape
    supp = torch.rand(batch_size, n, 1).to(DEVICE) <= p
    x = torch.zeros(batch_size, n, 1).to(DEVICE)
    x[supp] = torch.randn(int(supp.sum().item())).to(DEVICE)
    return (Phi @ x + noise_std * torch.randn(batch_size, m, 1).to(DEVICE)), x


def get_true_support(x: torch.Tensor) -> list[set[int]]:
    nonzeros = torch.nonzero(x.squeeze()).tolist()

    support_sets = defaultdict(set)
    for k, v in nonzeros:
        support_sets[k].add(v)

    return [support_sets[k] for k in sorted(support_sets.keys())]


def run_experiment(
    experiment_number: int,
    m: int,
    n: int,
    s: float,
    noise_std: float,
    output_dir: Path,
) -> None:
    # handle directory creation
    output_dir.mkdir(exist_ok=True)
    experiment_results_dir = output_dir / str(experiment_number)
    experiment_results_dir.mkdir(exist_ok=True)

    # simulation settings to be recorded
    settings = {
        "experiment_number": experiment_number,
        "m": m,
        "n": n,
        "measurement_rate": m / n,
        "sparsity": s,
        "noise_std": noise_std,
        "output_dir": str(output_dir),
    }

    with torch.no_grad():
        records = []
        intermediate_results_files = []

        logger.info(
            f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}"
        )
        Phi = gen_dictionary(TRIALS, m, n)
        y, x = generate_measurements_and_coeffs(Phi, p=s, noise_std=noise_std)
        nnz = torch.count_nonzero(x, axis=1)

        true_support = get_true_support(x)
        logger.info("Running IP")
        log_ip = ip(Phi, y, num_iterations=nnz.max().item())
        logger.info("Running OMP")
        log_omp = omp(Phi, y, num_iterations=nnz.max().item())

        breakpoint()
        ip_precision = []
        ip_recall = []
        ip_mse_x = []
        ip_mse_y = []
        logger.info("Generating metrics for IP")
        for indices, x_hat, y_hat in zip(
            log_ip["indices"], log_ip["x_hat"], log_ip["y_hat"]
        ):
            ip_recall.append(recall(indices, true_support))
            ip_precision.append(precision(indices, true_support))
            ip_mse_x.append(mse(x_hat, x))
            ip_mse_y.append(mse(y_hat, y))

        omp_precision = []
        omp_recall = []
        omp_mse_x = []
        omp_mse_y = []
        logger.info("Generating metrics for OMP")
        for indices, x_hat, y_hat in zip(
            log_omp["indices"], log_omp["x_hat"], log_omp["y_hat"]
        ):
            omp_recall.append(recall(indices, true_support))
            omp_precision.append(precision(indices, true_support))
            omp_mse_x.append(mse(x_hat, x))
            omp_mse_y.append(mse(y_hat, y))

        ious = [
            iou(ind_ip, ind_omp)
            for ind_ip, ind_omp in zip(log_ip["indices"], log_omp["indices"])
        ]

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
            "iou": ious,
            "nnz": nnz,
            "norm_x": torch.linalg.norm(x).item(),
            "norm_y": torch.linalg.norm(y).item(),
        }

        intermediate_results_file = experiment_results_dir / f"results_{trial}.json"
        with open(intermediate_results_file, "w") as f:
            json.dump(results, f)

        records.append(results)

        logger.info(f"Saving metrics to {experiment_results_dir / 'results.json'}")
        with open(experiment_results_dir / "results.json", "w") as f:
            json.dump({**settings, "results": records}, f)

        # clean up intermediate results files
        for path in intermediate_results_files:
            if path.stem.endswith(tuple("0123456789")):
                path.unlink()


def main(
    results_dir: Path,
    overwrite: bool = False,
    jobs: int = typer.Option(default=1, min=1, max=NUM_SETTINGS),
):
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    if jobs > 1:
        if DEVICE == "cuda":
            gpu_list = [
                g for g in GPUtil.getAvailable(maxLoad=0.2, maxMemory=0.2, limit=jobs)
            ]
            workers = min(jobs, len(gpu_list))
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
                    gpu_number=next(gpus),
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
