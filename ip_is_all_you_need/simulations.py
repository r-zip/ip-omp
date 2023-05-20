import logging
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from itertools import product
from multiprocessing import cpu_count
from operator import attrgetter
from pathlib import Path
from time import sleep

import polars as pl
import torch
import torch.multiprocessing as mp
import typer
from gpustat.core import GPUStatCollection
from rich.logging import RichHandler
from tqdm import tqdm

from .algorithms import ip, omp
from .constants import DEVICE
from .metrics import iou, mse, mutual_coherence, precision, recall

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()


class CoefficientDistribution(str, Enum):
    bernoulli_gaussian = "bernoulli_gaussian"
    sparse_gaussian = "sparse_gaussian"
    sparse_constant = "sparse_constant"


# fmt: off
TRIALS = 1_000
COEFFICIENT_DISTRIBUTION = CoefficientDistribution.sparse_constant
NUM_THREADS = 10

SMALL_SETTINGS = {
    "dimensions": [
        *[(m, 256) for m in range(4, 220, 12)],
    ],
    "sparsity": list(range(4, 42, 6)),
    "noise_std": [0.0],
}

LARGE_SETTINGS = {
    "dimensions": [
        *[(m, 1024) for m in range(5, 305, 5)],
    ],
    "sparsity": list(range(4, 18, 2)),
    "noise_std": [0.0],
}
# fmt: on

NUM_SETTINGS_SMALL = len(list(product(*list(SMALL_SETTINGS.values()))))
NUM_SETTINGS_LARGE = len(list(product(*list(LARGE_SETTINGS.values()))))
DTYPE = torch.float64


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


class Setting(str, Enum):
    small = "small"
    large = "large"


class OrderBy(str, Enum):
    utilization = "utilization"
    memory_usage = "memory_usage"

    def __str__(self) -> str:
        return self.value


def get_gpus(
    utilization: float = 0.25,
    memory_usage: float = 0.25,
    order_by: OrderBy = "utilization",
) -> list[int]:
    gpus = GPUStatCollection.new_query()
    free_gpus = [
        g
        for g in gpus
        if (g.utilization < utilization * 100)
        and (g.memory_used / g.memory_total < memory_usage)
    ]

    free_gpus = sorted(free_gpus, key=attrgetter(str(order_by)))
    return [g.index for g in free_gpus]


def gen_dictionary(
    batch_size: int, m: int, n: int, device: str | torch.device = DEVICE
) -> torch.Tensor:
    Phi = torch.randn(batch_size, m, n, device=device)
    return Phi / torch.linalg.norm(Phi, dim=1)[:, None, :]


def generate_measurements_and_coeffs(
    Phi: torch.Tensor,
    s: int,
    noise_std: float = 0.0,
    device: str | torch.device = DEVICE,
    coeff_distribution: CoefficientDistribution = CoefficientDistribution.sparse_gaussian,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, m, n = Phi.shape
    x = torch.zeros(batch_size, n, 1, device=device)

    if coeff_distribution == CoefficientDistribution.bernoulli_gaussian:
        p = s / n
        supp = torch.rand(batch_size, n, 1, device=device) <= p
        x[supp] = torch.randn(int(supp.sum().item()), device=device)
    elif coeff_distribution in [
        CoefficientDistribution.sparse_gaussian,
        CoefficientDistribution.sparse_constant,
    ]:
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
        if coeff_distribution == CoefficientDistribution.sparse_gaussian:
            values = torch.randn(batch_size * s, device=device)
        elif coeff_distribution == CoefficientDistribution.sparse_constant:
            values = torch.ones(batch_size * s, device=device)
        else:
            raise ValueError(f"coeff_distribution {coeff_distribution} not understood.")
        x[supp] = values
    else:
        raise ValueError(f"coeff_distribution {coeff_distribution} not understood.")

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
        for iter, (objective_t, x_hat_t, y_hat_t) in enumerate(
            zip(objective, x_hat, y_hat)
        ):
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
    noise_std: float,
    output_dir: Path,
    device_type: Device,
    utilization: float,
    memory_usage: float,
    order_by: str,
) -> None:
    torch.random.manual_seed(12345)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(NUM_THREADS)

    if device_type == Device.cuda:
        while not (
            gpus := get_gpus(
                utilization=utilization, memory_usage=memory_usage, order_by=order_by
            )
        ):
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
        logger.info(
            f"Generating dictionary, signal, and measurement with dimensions {m=}, {n=}"
        )
        Phi = gen_dictionary(TRIALS, m, n, device=device)

        torch.save(Phi, experiment_results_dir / "Phi.pt")

        y, x = generate_measurements_and_coeffs(
            Phi,
            s=s,
            noise_std=noise_std,
            device=device,
            coeff_distribution=COEFFICIENT_DISTRIBUTION,
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
                    pl.lit(s).alias("sparsity"),
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
                    "sparsity",
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

    torch.cuda.empty_cache()


def main(
    results_dir: Path,
    overwrite: bool = False,
    jobs: int = typer.Option(default=1, min=1, max=NUM_SETTINGS_LARGE),
    device: Device = Device.cuda if DEVICE == "cuda" else Device.cpu,
    setting: Setting = Setting.small,
    memory_usage: float = typer.Option(default=0.75, min=0.0, max=1.0),
    utilization: float = typer.Option(default=0.75, min=0.0, max=1.0),
    order_by: OrderBy = OrderBy.utilization,
):
    mp.set_start_method("spawn")
    if results_dir.exists() and not overwrite:
        FileExistsError(
            f"Results directory {results_dir.absolute()} exists. Please specify a different directory or --overwrite."
        )

    if setting == Setting.small:
        settings = SMALL_SETTINGS
        num_settings = NUM_SETTINGS_SMALL
    else:
        settings = LARGE_SETTINGS
        num_settings = NUM_SETTINGS_LARGE

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
    for k, ((m, n), s, noise_std) in enumerate(
        product(settings["dimensions"], settings["sparsity"], settings["noise_std"])
    ):
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
                    utilization=utilization,
                    memory_usage=memory_usage,
                    order_by=order_by,
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
                utilization=utilization,
                memory_usage=memory_usage,
                order_by=order_by,
            )

        if k < jobs:
            sleep(2)

    for f in tqdm(as_completed(futures), total=len(futures)):
        if e := f.exception():
            logger.exception(e)
            traceback.print_exc(e)

    aggregate_results(results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(NUM_THREADS)
    typer.run(main)
