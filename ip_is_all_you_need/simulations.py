import logging
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from itertools import product
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
        # (500, 500),
        (500, 750),
        # (500, 1000),
        # (500, 1250),
        # (500, 1500),
    ],
    "sparsity": [
        0.025,
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
        # 0.01,
        # 0.1,
    ],
}
# fmt: on

NUM_SETTINGS = len(list(product(*list(SETTINGS.values()))))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, m, n = Phi.shape
    supp = torch.rand(batch_size, n, 1, device=device) <= p
    x = torch.zeros(batch_size, n, 1, device=device)
    x[supp] = torch.randn(int(supp.sum().item()), device=device)
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
        zip(log["indices"], log["x_hat"], log["y_hat"], Phis, true_support, xs, ys)
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
    device: str | torch.device = DEVICE,
) -> None:
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
            Phi, p=s, noise_std=noise_std, device=device
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


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


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

    if jobs > 1:
        if device == Device.cuda:
            workers = min(jobs, len(get_gpus()))
        else:
            workers = min(jobs, cpu_count())

        if workers < jobs:
            logger.info(
                f"Running {workers} jobs; {jobs} was too many for system resources"
            )

        pool = ProcessPoolExecutor(max_workers=workers)

        futures = []
        finished = 0
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
                    device=get_gpus()[0] if device == "cuda" else "cpu",
                )
            )

            for future in futures:
                if future.done():
                    finished += 1
                    logger.info(f"Finished {finished} / {NUM_SETTINGS} jobs")
                    futures.remove(future)
                elif e := future.exception():
                    logger.exception(f"Exception raised by subprocess: {e}")
                    traceback.print_exception(e)

            if device == "cuda":
                sleep(0.1)

            while (device == "cuda") and not get_gpus():
                sleep(0.1)

    else:
        futures = []
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

    for f in as_completed(futures):
        pass

    aggregate_results(results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    typer.run(main)
