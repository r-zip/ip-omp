import logging
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from itertools import product
from math import floor
from multiprocessing import cpu_count
from pathlib import Path
from time import sleep

import GPUtil
import polars as pl
import torch
import torch.multiprocessing as mp
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
        # (500, 750),
        (500, 1000),
        # (500, 1250),
        # (500, 1500),
    ],
    "sparsity": [
        # 0.025,
        0.05,
        # 0.075,
        # 0.1,
        # 0.125,
        # 0.15,
        # 0.175,
        # 0.2,
        # 0.225,
        # 0.25,
        # 0.275,
        # 0.3,
        # 0.325,
        # 0.35,
        # 0.375,
        # 0.4,
    ],
    "noise_std": [
        0.0,
        0.01,
        0.1,
    ],
}
# fmt: on

NUM_SETTINGS = len(list(product(*list(SETTINGS.values()))))


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


def get_gpus():
    return GPUtil.getAvailable(
        maxLoad=0.1, maxMemory=0.15, limit=float("inf"), order="memory"
    )


def gen_dictionary(
    batch_size: int, m: int, n: int, device: str | torch.device = DEVICE
) -> torch.Tensor:
    Phi = torch.randn(batch_size, m, n, device=device)
    return Phi / torch.linalg.norm(Phi, dim=1)[:, None, :]


def generate_measurements_and_coeffs(
    Phi: torch.Tensor,
    p: float = 0.01,
    noise_std: float = 0.0,
    device: str | torch.device = DEVICE,
    coeff_distribution: str = "sparse_gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, m, n = Phi.shape
    x = torch.zeros(batch_size, n, 1, device=device)

    if coeff_distribution == "bernoulli_gaussian":
        supp = torch.rand(batch_size, n, 1, device=device) <= p
        x[supp] = torch.randn(int(supp.sum().item()), device=device)
    elif coeff_distribution == "sparse_gaussian":
        s = floor(p * n)
        bool_index = torch.hstack(
            [
                torch.ones(s, device=device, dtype=torch.bool),
                torch.zeros(n - s, device=device, dtype=torch.bool),
            ]
        )
        supp = torch.vstack(
            [
                bool_index[torch.randperm(n, device=device)].clone()
                for _ in range(batch_size)
            ]
        )[:, :, None]
        values = torch.randn(batch_size * s, device=device)
        x[supp] = values
    else:
        raise ValueError(f"coeff_distribution {coeff_distribution} not understood.")

    return (Phi @ x + noise_std * torch.randn(batch_size, m, 1, device=device)), x


def get_true_support(x: torch.Tensor) -> list[set[int]]:
    nonzeros = torch.nonzero(x.squeeze()).tolist()

    support_sets = defaultdict(set)
    for k, v in nonzeros:
        support_sets[k].add(v)

    return [sorted(support_sets[k]) for k in sorted(support_sets.keys())]


def transpose_log(log: dict[str, list]) -> dict[str, list]:
    transposed = {}
    for key, value in log.items():
        transposed[key] = [list(v) for v in zip(*value)]

    return transposed


def compute_metrics(
    log: dict[str, list[int]],
    true_support: list[list[int]],
    xs: torch.Tensor,
    ys: torch.Tensor,
    Phis: torch.Tensor,
    algorithm: str,
) -> list[dict[str, list[float]]]:
    metrics = []
    for trial, (indices, x_hat, y_hat, Phi, support, x, y) in enumerate(
        zip(log["indices"], log["x_hat"], log["y_hat"], Phis, true_support, xs, ys),
    ):
        coherence = mutual_coherence(Phi)
        nnz = len(support)
        norm_x = torch.linalg.norm(x).item()
        norm_y = torch.linalg.norm(y).item()
        for iter, (x_hat_t, y_hat_t) in enumerate(zip(x_hat, y_hat)):
            metrics_now = {
                "trial": trial,
                "iter": iter,
                "algorithm": algorithm,
                "recall": recall(indices[: iter + 1], support),
                "precision": precision(indices[: iter + 1], support),
                "mse_x": mse(x_hat_t, x),
                "mse_y": mse(y_hat_t, y),
                "estimated_support": indices[: iter + 1],
                "true_support": support,
                "x_hat": x_hat_t.ravel().tolist(),
                "y_hat": y_hat_t.ravel().tolist(),
                "norm_x": norm_x,
                "norm_y": norm_y,
                "nnz": nnz,
                "coherence": coherence,
                "x": x.ravel().tolist(),
                "y": y.ravel().tolist(),
            }
            metrics.append(metrics_now)

    return metrics


def run_experiment(
    experiment_number: int,
    m: int,
    n: int,
    s: float,
    noise_std: float,
    output_dir: Path,
    device_type: Device,
) -> None:
    if device_type == Device.cuda:
        while not (gpus := get_gpus()):
            sleep(0.01)

        device = f"cuda:{gpus[0]}"
    else:
        device = "cpu"

    # handle directory creation
    output_dir.mkdir(exist_ok=True)
    experiment_results_dir = output_dir / str(experiment_number)
    experiment_results_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        logger.info(
            f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}"
        )
        Phi = gen_dictionary(TRIALS, m, n, device=device)
        y, x = generate_measurements_and_coeffs(
            Phi,
            p=s,
            noise_std=noise_std,
            device=device,
            coeff_distribution="sparse_gaussian",
        )
        nnz = torch.count_nonzero(x, axis=1)

        true_support = get_true_support(x)
        logger.info("Running IP")
        log_ip = ip(Phi, y, num_iterations=nnz.max().item(), device=device)
        logger.info("Running OMP")
        log_omp = omp(Phi, y, num_iterations=nnz.max().item(), device=device)

        # tranpose logs
        log_ip = transpose_log(log_ip)
        log_omp = transpose_log(log_omp)

        logger.info("Generating metrics for IP")
        metrics_ip = compute_metrics(log_ip, true_support, x, y, Phi, "ip")

        logger.info("Generating metrics for OMP")
        metrics_omp = compute_metrics(log_omp, true_support, x, y, Phi, "omp")

        logger.info("Combining metrics")
        df = pl.concat(
            [pl.DataFrame(metrics_ip), pl.DataFrame(metrics_omp)], how="vertical"
        )

        # compute iou
        pivot_table = df.pivot(
            index=["trial", "iter"],
            columns="algorithm",
            values="estimated_support",
            aggregate_function="first",
        ).with_columns(
            pl.struct(["ip", "omp"])
            .apply(lambda x: iou(x["ip"], x["omp"]))
            .alias("iou")
        )

        # FIXME
        # above pivot table appears correct, but for some reason, iou collapses into single sequence over trials
        df = (
            df.join(pivot_table, on=["trial", "iter"])
            .drop(["ip", "omp"])
            .with_columns(
                [
                    pl.lit(experiment_number).alias("experiment_number"),
                    pl.lit(m).alias("m"),
                    pl.lit(n).alias("n"),
                    pl.lit(m / n).alias("measurement_rate"),
                    pl.lit(s).alias("mean_sparsity"),
                    pl.lit(noise_std).alias("noise_std"),
                    pl.lit(str(output_dir)).alias("output_dir"),
                ]
            )
        )

        results_file = experiment_results_dir / "results.parquet"
        logger.info(f"Writing results to {results_file}")
        df.write_parquet(results_file)


def aggregate_results(results_dir: Path) -> None:
    dfs = []
    for path in results_dir.iterdir():
        if path.is_dir() and (f := (path / "results.parquet")).exists():
            dfs.append(
                pl.read_parquet(f).select(
                    "experiment_number",
                    "m",
                    "n",
                    "measurement_rate",
                    "mean_sparsity",
                    "noise_std",
                    "output_dir",
                    "trial",
                    "nnz",
                    "norm_x",
                    "norm_y",
                    "coherence",
                    "iou",
                    "iter",
                    "algorithm",
                    "recall",
                    "precision",
                    "mse_x",
                    "mse_y",
                )
            )

    df = pl.concat(dfs, how="vertical")
    df.write_parquet(results_dir / "results.parquet")


def main(
    results_dir: Path,
    overwrite: bool = False,
    jobs: int = typer.Option(default=1, min=1, max=NUM_SETTINGS),
    device: Device = Device.cuda if DEVICE == "cuda" else Device.cpu,
):
    mp.set_start_method("spawn")
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    if device == Device.cuda:
        workers = min(jobs, len(get_gpus()), NUM_SETTINGS)
    else:
        workers = min(jobs, cpu_count(), NUM_SETTINGS)

    if workers < jobs:
        logger.info(f"Running {workers} jobs; {jobs} was too many for system resources")

    pool = ProcessPoolExecutor(max_workers=workers)

    futures = []
    for k, ((m, n), s, noise_std) in enumerate(
        product(SETTINGS["dimensions"], SETTINGS["sparsity"], SETTINGS["noise_std"])
    ):
        if s * n > m:
            continue

        if jobs > 1:
            futures.append(
                pool.submit(
                    run_experiment,
                    k,
                    m,
                    n,
                    s,
                    output_dir=results_dir,
                    noise_std=noise_std,
                    device_type=device,
                )
            )
        else:
            run_experiment(
                k,
                m,
                n,
                s,
                output_dir=results_dir,
                noise_std=noise_std,
                device_type=device,
            )

    for f in tqdm(as_completed(futures), total=len(futures)):
        if e := f.exception():
            logger.exception(e)
            traceback.print_exc(e)

    aggregate_results(results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    typer.run(main)
