from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

sns.set_context("notebook")
sns.set_palette("deep")

c = pl.col


def plot_trial(
    df: pl.DataFrame, experiment_number: int, algorithm: str, metric: str, trial: int, save_path: Path,
) -> None:
    df_filtered = df.filter(
        (c("experiment_number") == experiment_number)
        & (c("algorithm") == algorithm)
        & (c("trial") == trial)
    ).sort("iter")
    if metric.startswith("mse"):
        metric_values = 10 * df_filtered[metric].log10()
    else:
        metric_values = df_filtered[metric]
    plt.plot(df_filtered["iter"], metric_values)
    plt.vlines(
        [df_filtered["nnz"][0] - 1],
        metric_values.min(),
        metric_values.max(),
        linestyles=["--"],
    )
    plt.savefig(save_path, dpi=300)


def get_time_series_data(df: pl.DataFrame, experiment_number: int) -> pl.DataFrame:
    # "time series" data: average over trials
    df_filtered = df.filter(c("experiment_number") == experiment_number)

    # to avoid making buckets too fine-grained, use the min sparsity level
    # for this experiment
    num_buckets = df_filtered["nnz"].min()

    df_ts = (
        df_filtered.with_columns(
            ((c("iter") + 1) / c("nnz")).alias("rel_iter"),
        )
        .with_columns(
            (c("rel_iter") * num_buckets)
            .floor()
            .cast(pl.UInt32)
            .alias("rel_iter_bucket"),
            (c("mse_x") / c("norm_x")).alias("rel_mse_x"),
            (c("mse_y") / c("norm_y")).alias("rel_mse_y"),
        )
        .filter(c("rel_iter") <= 1.0)
        .groupby(["experiment_number", "rel_iter_bucket", "algorithm"])
        .agg(
            c("m").first(),
            c("n").first(),
            c("measurement_rate").first(),
            c("mean_sparsity").first(),
            c("noise_std").first(),
            c("nnz").first(),
            c("rel_iter").mean(),
            c("coherence").mean(),
            # mean
            c("mse_x").mean().alias("mse_x_mean"),
            c("mse_y").mean().alias("mse_y_mean"),
            c("rel_mse_x").mean().alias("rel_mse_x_mean"),
            c("rel_mse_y").mean().alias("rel_mse_y_mean"),
            c("precision").mean().alias("precision_mean"),
            c("recall").mean().alias("recall_mean"),
            c("iou").mean().alias("iou_mean"),
            # median
            c("mse_x").median().alias("mse_x_median"),
            c("mse_y").median().alias("mse_y_median"),
            c("rel_mse_x").median().alias("rel_mse_x_median"),
            c("rel_mse_y").median().alias("rel_mse_y_median"),
            c("precision").median().alias("precision_median"),
            c("recall").median().alias("recall_median"),
            c("iou").median().alias("iou_median"),
            # std
            c("mse_x").std().alias("mse_x_std"),
            c("mse_y").std().alias("mse_y_std"),
            c("rel_mse_x").std().alias("rel_mse_x_std"),
            c("rel_mse_y").std().alias("rel_mse_y_std"),
            c("precision").std().alias("precision_std"),
            c("recall").std().alias("recall_std"),
            c("iou").std().alias("iou_std"),
            # 1st quartile
            c("mse_x").quantile(0.25).alias("mse_x_lo"),
            c("mse_y").quantile(0.25).alias("mse_y_lo"),
            c("rel_mse_x").quantile(0.25).alias("rel_mse_x_lo"),
            c("rel_mse_y").quantile(0.25).alias("rel_mse_y_lo"),
            c("precision").quantile(0.25).alias("precision_lo"),
            c("recall").quantile(0.25).alias("recall_lo"),
            c("iou").quantile(0.25).alias("iou_lo"),
            # 3rd quartile
            c("mse_x").quantile(0.75).alias("mse_x_hi"),
            c("mse_y").quantile(0.75).alias("mse_y_hi"),
            c("rel_mse_x").quantile(0.75).alias("rel_mse_x_hi"),
            c("rel_mse_y").quantile(0.75).alias("rel_mse_y_hi"),
            c("precision").quantile(0.75).alias("precision_hi"),
            c("recall").quantile(0.75).alias("recall_hi"),
            c("iou").quantile(0.75).alias("iou_hi"),
            # min
            c("iou").min().alias("iou_min"),
            # max
            c("iou").max().alias("iou_max"),
        )
        .sort(["experiment_number", "algorithm", "rel_iter_bucket"])
    )

    return df_ts


def identity(x):
    return x


def plot_timeseries(
    df_ts: pl.DataFrame,
    experiment_number: int,
    metric: str,
    save_path: Path,
    central_tendency: str = "mean",
    error_bars: str = "std",
) -> None:
    xforms = defaultdict(lambda: identity)
    xforms["rel_mse_x"] = xforms["rel_mse_y"] = lambda x: x.log10() * 10
    xforms["mse_x"] = xforms["mse_y"] = lambda x: x.log10() * 10
    xform = xforms[metric]
    metric_to_ylabel = {
        "iou": "IOU",
        "mse_x": r"$\log_{10} ||x - \widehat{x}||_2^2$",
        "mse_y": r"$\log_{10} ||y - \widehat{y}||_2^2",
        "rel_mse_x": r"$10 \log_{10}(||x - \widehat{x}||_2^2\,/\,||x||_2^2)$",
        "rel_mse_y": r"$10 \log_{10}(||y - \widehat{y}||_2^2\,/\,||y||_2^2)$",
        "precision": "Precision",
        "recall": "Recall",
    }
    metric_to_title = {
        "iou": "IOU",
        "mse_x": r"MSE($\widehat{x}$) (log-scale)",
        "mse_y": r"MSE($\widehat{y}$) (log-scale)",
        "rel_mse_x": r"Rel. MSE($\widehat{x}$) (dB)",
        "rel_mse_y": r"Rel. MSE($\widehat{y}$) (dB)",
        "precision": "Precision",
        "recall": "Recall",
    }

    df_ts_omp = df_ts.filter(
        (c("algorithm") == "omp") & (c("experiment_number") == experiment_number)
    )
    df_ts_ip = df_ts.filter(
        (c("algorithm") == "ip") & (c("experiment_number") == experiment_number)
    )

    m = df_ts_ip["m"][0]
    n = df_ts_ip["n"][0]
    s = df_ts_ip["mean_sparsity"][0]
    noise_std = df_ts_ip["noise_std"][0]
    coherence = df_ts_ip["coherence"][0]

    rel_iter_ip = df_ts_ip["rel_iter"]
    rel_iter_omp = df_ts_omp["rel_iter"]
    center_ip = df_ts_ip[f"{metric}_{central_tendency}"]
    center_omp = df_ts_omp[f"{metric}_{central_tendency}"]
    plt.plot(rel_iter_ip, xform(center_ip))

    def plot_error_bars(rel_iter, center, df_ts):
        if error_bars == "iqr":
            plt.fill_between(
                rel_iter,
                xform(df_ts[f"{metric}_lo"]),
                xform(df_ts[f"{metric}_hi"]),
                alpha=0.3,
            )
        elif error_bars == "min_max":
            plt.fill_between(
                rel_iter_ip,
                xform(df_ts[f"{metric}_min"]),
                xform(df_ts[f"{metric}_max"]),
                alpha=0.3,
            )
        elif error_bars == "std":
            plt.fill_between(
                rel_iter_ip,
                xform(center - df_ts[f"{metric}_std"]),
                xform(center + df_ts[f"{metric}_std"]),
                alpha=0.3,
            )
        else:
            raise ValueError(f"error_bars value {error_bars} not understood.")

    plot_error_bars(rel_iter_ip, center_ip, df_ts_ip)

    if metric != "iou":
        plt.plot(
            rel_iter_omp,
            xform(center_omp),
            "--",
            marker="o",
            markerfacecolor="none",
        )
        plot_error_bars(rel_iter_omp, center_omp, df_ts_omp)
        plt.legend(["IP Mean", "IP IQR", "OMP Mean", "OMP IQR"])
    else:
        plt.legend(["Mean", "Min/Max"])
    plt.xlabel("Iteration / # Iterations")
    plt.ylabel(f"{metric_to_ylabel[metric]}")
    plt.title(
        f"{metric_to_title[metric]}, m={m}, n={n}, s={s}, $\\sigma_n={noise_std}$, $\\mu={coherence:0.2f}$"
    )
    plt.grid("on")

    plt.savefig(save_path, dpi=300)


def get_phase_transition_data(df: pl.DataFrame) -> pl.DataFrame:
    # support recovery data (phase transition)
    df_pt = (
        df.sort("iter")
        .groupby(["algorithm", "experiment_number", "trial"])
        .agg(
            c("m").first(),
            c("n").first(),
            c("measurement_rate").first(),
            c("mean_sparsity").first(),
            c("noise_std").first(),
            c("precision").last(),
            c("recall").last(),
            c("iou").last(),
            (c("mse_x") / c("norm_x")).last().alias("rel_mse_x"),
            (c("mse_y") / c("norm_y")).last().alias("rel_mse_y"),
        )
        .groupby("experiment_number")
        .agg(
            c("m").first(),
            c("n").first(),
            c("measurement_rate").first(),
            c("mean_sparsity").first(),
            c("noise_std").first(),
            c("precision").mean(),
            c("recall").mean(),
            c("iou").mean(),
            c("rel_mse_x").mean(),
            c("rel_mse_y").mean(),
        )
    )
    return df_pt


def main():
    pass


# %%
if __name__ == "__main__":
    typer.run(main)
