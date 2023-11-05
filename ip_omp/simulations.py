import logging
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import cpu_count
from operator import attrgetter
from pathlib import Path
from time import sleep

import numpy as np
import polars as pl
import torch
import torch.multiprocessing as mp
import typer
from gpustat.core import GPUStatCollection
from rich.logging import RichHandler
from tqdm import tqdm

from .algorithms import ip, omp
from .constants import (
    DELTA_M,
    DELTA_S,
    DEVICE,
    MAX_M,
    MAX_S,
    MIN_M,
    MIN_S,
    SNR_GRID,
    TRIALS,
    CoeffDistribution,
    Device,
    NoiseSetting,
    OrderBy,
    ProblemSize,
)
from .metrics import iou, mse, mutual_coherence, precision, recall
from .util import db_to_ratio

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger()

DTYPE = torch.float64


def get_gpus(
    utilization: float = 0.25,
    memory_usage: float = 0.25,
    order_by: OrderBy = OrderBy.utilization,
) -> list[int]:
    gpus = GPUStatCollection.new_query()
    free_gpus = [
        g for g in gpus if (g.utilization < utilization * 100) and (g.memory_used / g.memory_total < memory_usage)
    ]

    free_gpus = sorted(free_gpus, key=attrgetter(str(order_by)))
    return [g.index for g in free_gpus]


def gen_dictionary(batch_size: int, m: int, n: int, device: str | torch.device = DEVICE) -> torch.Tensor:
    Phi = torch.randn(batch_size, m, n, device=device)
    return Phi / torch.linalg.norm(Phi, dim=1)[:, None, :]


def generate_measurements_and_coeffs(
    Phi: torch.Tensor,
    s: int,
    snr: float,
    device: str | torch.device = DEVICE,
    coeff_distribution: CoeffDistribution = CoeffDistribution.sparse_const,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate coefficients x and measurements y = Phi @ x + noise.

    Args:
        Phi: dictionaries
        s: sparsity level
        snr: signal-to-noise ratio (dB)
        device: torch device
        coeff_distribution: distribution of the coefficients x

    Returns:
        noisy measurements, coefficients
    """
    err_msg = f"coeff_distribution {coeff_distribution} not understood."

    batch_size, m, n = Phi.shape
    x = torch.zeros(batch_size, n, 1, device=device)

    if np.isinf(snr):
        # noiseless case
        noise_std = 0
    else:
        snr_ratio = db_to_ratio(snr, square=True)
        noise_std = np.sqrt(s / (snr_ratio * (m - 2)))

    bool_index = torch.hstack(
        [
            torch.ones(s, device=device, dtype=torch.bool),
            torch.zeros(n - s, device=device, dtype=torch.bool),
        ]
    )
    supp = torch.vstack([bool_index[torch.randperm(n, device=device)].clone() for _ in range(batch_size)])[:, :, None]
    if coeff_distribution == CoeffDistribution.sparse_gaussian:
        values = torch.randn(batch_size * s, device=device)
    elif coeff_distribution == CoeffDistribution.sparse_const:
        values = torch.ones(batch_size * s, device=device)
    else:
        raise ValueError(err_msg)
    x[supp] = values

    return (Phi @ x + noise_std * torch.randn(batch_size, m, 1, device=device)), x


def get_true_support(x: torch.Tensor) -> list[list[int]]:
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
    for trial, (objective, indices, x_hat, y_hat, Phi, support, x, y) in enumerate(
        zip(
            log["objective"],
            log["indices"],
            log["x_hat"],
            log["y_hat"],
            Phis,
            true_support,
            xs,
            ys,
        ),
    ):
        coherence = mutual_coherence(Phi)
        nnz = len(support)
        norm_x = torch.linalg.norm(x).item()
        norm_y = torch.linalg.norm(y).item()
        for iter, (objective_t, x_hat_t, y_hat_t) in enumerate(zip(objective, x_hat, y_hat)):
            metrics_now = {
                "trial": trial,
                "iter": iter,
                "algorithm": algorithm,
                "objective": objective_t,
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
    s: int,
    snr: float,
    output_dir: Path,
    device_type: Device,
    utilization: float,
    memory_usage: float,
    order_by: str,
    coeff_distribution: CoeffDistribution,
) -> None:
    torch.random.manual_seed(12345)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)

    if device_type == Device.cuda:
        while not (gpus := get_gpus(utilization=utilization, memory_usage=memory_usage, order_by=order_by)):
            logger.info("Waiting for available GPU...")
            sleep(1)

        device = f"cuda:{gpus[0]}"
    else:
        device = "cpu"

    # handle directory creation
    output_dir.mkdir(exist_ok=True)
    experiment_results_dir = output_dir / str(experiment_number)
    experiment_results_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        logger.info(f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}")
        Phi = gen_dictionary(TRIALS, m, n, device=device)

        torch.save(Phi, experiment_results_dir / "Phi.pt")

        y, x = generate_measurements_and_coeffs(
            Phi,
            s=s,
            snr=snr,
            device=device,
            coeff_distribution=coeff_distribution,
        )

        torch.save(y, experiment_results_dir / "y.pt")
        torch.save(x, experiment_results_dir / "x.pt")

        true_support = get_true_support(x)

        # run for num iterations = sparsity (as in Tropp/Gilbert paper)
        logger.info("Running IP")
        log_ip = ip(Phi, y, num_iterations=s, device=device)
        logger.info("Running OMP")
        log_omp = omp(Phi, y, num_iterations=s, device=device)

        # tranpose logs
        log_ip = transpose_log(log_ip)
        log_omp = transpose_log(log_omp)

        logger.info("Generating metrics for IP")
        metrics_ip = compute_metrics(log_ip, true_support, x, y, Phi, "ip")

        logger.info("Generating metrics for OMP")
        metrics_omp = compute_metrics(log_omp, true_support, x, y, Phi, "omp")

        logger.info("Combining metrics")
        df = pl.concat([pl.DataFrame(metrics_ip), pl.DataFrame(metrics_omp)], how="vertical")

        # compute iou
        pivot_table = df.pivot(
            index=["trial", "iter"],
            columns="algorithm",
            values="estimated_support",
            aggregate_function="first",
        ).with_columns(pl.struct(["ip", "omp"]).apply(lambda x: iou(x["ip"], x["omp"])).alias("iou"))

        df = (
            df.join(pivot_table, on=["trial", "iter"])
            .drop(["ip", "omp"])
            .with_columns(
                [
                    pl.lit(experiment_number).alias("experiment_number"),
                    pl.lit(m).alias("m"),
                    pl.lit(n).alias("n"),
                    pl.lit(m / n).alias("measurement_rate"),
                    pl.lit(s).alias("sparsity"),
                    pl.lit(snr).alias("snr"),
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
                pl.read_parquet(f)
                .select(
                    "experiment_number",
                    "m",
                    "n",
                    "measurement_rate",
                    "sparsity",
                    "snr",
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
                .cast(
                    {
                        "snr": float,
                    }
                )
            )

    df = pl.concat(dfs, how="vertical")
    df.write_parquet(results_dir / "results.parquet")


def get_settings(
    problem_size: ProblemSize,
    coeff_distribution: CoeffDistribution,
    noise_setting: NoiseSetting,
) -> dict[str, list[tuple[int, int]] | list[float]]:
    """
    Get dictionary of settings (dimensions, sparsities, SNRs) for problem size (small n/large n), coefficient
    distribution (gaussian/constant), and noise setting (noisy/noiseless).
    """
    n = 256 if problem_size == ProblemSize.small else 1024
    min_m = MIN_M[problem_size]
    max_m = MAX_M[(problem_size, noise_setting, coeff_distribution)]
    delta_m = DELTA_M[problem_size]
    min_s = MIN_S[problem_size]
    max_s = MAX_S[problem_size]
    delta_s = DELTA_S[problem_size]

    settings = dict()
    settings["dimensions"] = [(m, n) for m in range(min_m, max_m + delta_m, delta_m)]
    settings["sparsity"] = list(range(min_s, max_s + delta_s, delta_s))
    settings["snr"] = SNR_GRID[noise_setting]

    num_settings = len(list(product(*list(settings.values()))))
    experiment_numbers = list(range(num_settings))
    return settings, num_settings, experiment_numbers


def main(
    results_dir: Path,
    problem_size: ProblemSize = ProblemSize.small,
    coeff_distribution: CoeffDistribution = CoeffDistribution.sparse_gaussian,
    noise_setting: NoiseSetting = NoiseSetting.noiseless,
    overwrite: bool = False,
    jobs: int = typer.Option(default=1, min=1),
    device: Device = Device.cuda if DEVICE == "cuda" else Device.cpu,
    memory_usage: float = typer.Option(default=0.75, min=0.0, max=1.0),
    utilization: float = typer.Option(default=0.75, min=0.0, max=1.0),
    order_by: OrderBy = OrderBy.utilization,
) -> None:
    mp.set_start_method("spawn")
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    settings, num_settings, experiment_numbers = get_settings(problem_size, coeff_distribution, noise_setting)

    if device == Device.cuda:
        workers = min(
            jobs,
            len(
                get_gpus(
                    utilization=utilization,
                    memory_usage=memory_usage,
                    order_by=order_by,
                )
            ),
            num_settings,
        )
    else:
        workers = min(jobs, cpu_count(), num_settings)

    if workers < jobs:
        logger.info(f"Running {workers} jobs; {jobs} was too many for system resources")

    pool = ProcessPoolExecutor(max_workers=workers)

    futures = []
    for k, ((m, n), s, snr) in zip(
        experiment_numbers,
        product(settings["dimensions"], settings["sparsity"], settings["snr"]),
    ):
        # might dedupe this by specifying args, kwargs in one place
        if jobs > 1:
            futures.append(
                pool.submit(
                    run_experiment,
                    k,
                    m,
                    n,
                    s,
                    output_dir=results_dir,
                    snr=snr,
                    device_type=device,
                    utilization=utilization,
                    memory_usage=memory_usage,
                    order_by=order_by,
                    coeff_distribution=coeff_distribution,
                )
            )
        else:
            run_experiment(
                k,
                m,
                n,
                s,
                output_dir=results_dir,
                snr=snr,
                device_type=device,
                utilization=utilization,
                memory_usage=memory_usage,
                order_by=order_by,
                coeff_distribution=coeff_distribution,
            )

        if k < jobs:
            sleep(2)

    for f in tqdm(as_completed(futures), total=len(futures)):
        if e := f.exception():
            logger.exception(e)
            traceback.print_exc()

    aggregate_results(results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    typer.run(main)
