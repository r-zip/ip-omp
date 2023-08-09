from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

from ip_is_all_you_need.plots import get_phase_transition_data

matplotlib.rcParams.update({"font.size": 22})


c = pl.col
sns.set()
sns.set_context("talk")

SUCCESS_THRESHOLD = 1e-3
METRIC_NAME_LOOKUP = {
    "nmse_x_mean": "Mean NMSE",
    "nmse_x_median": "Median NMSE",
    "success_rate": "Probability of recovery",
}


def filter_df(
    df: pl.DataFrame, algorithm: Literal["ip", "omp", None] = None
) -> pl.DataFrame:
    if algorithm:
        df = df.filter(c("algorithm") == algorithm)
    df = df.with_columns(
        c("iter")
        .max()
        .over(["experiment_number", "trial", "algorithm"])
        .alias("max_iter")
    )
    return df.filter(c("iter") == c("max_iter"))


def early_termination(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            c("iter")
            .max()
            .over(["experiment_number", "trial", "algorithm"])
            .alias("max_iter")
        )
        .with_columns((c("max_iter") < c("sparsity") - 1).alias("early_term"))
        .groupby(["experiment_number", "trial", "algorithm"])
        .agg(
            c("m").first(),
            c("n").first(),
            c("sparsity").first(),
            c("early_term").mean(),
        )
    )


def get_phase_transition_data(
    df: pl.DataFrame, algorithm: Literal["ip", "omp"]
) -> pl.DataFrame:
    df_pt = (
        # filter to only the last iteration
        filter_df(df, algorithm=algorithm)
        .with_columns((c("n") * c("mse_x") / c("norm_x") ** 2).alias("nmse"))
        # define success as relative reconstruction error < eps
        .with_columns(
            (c("nmse") < SUCCESS_THRESHOLD).alias("success"),
        )
        # for each experiment
        .groupby("experiment_number")
        # record the settings, success rate, and iou statistics
        .agg(
            c("m").first(),
            c("n").first(),
            c("measurement_rate").first(),
            c("sparsity").first(),
            c("snr").first(),
            c("iou").mean(),
            c("iou").quantile(0.05).alias("iou_lo"),
            c("iou").quantile(0.95).alias("iou_hi"),
            c("success").mean().alias("success_rate"),
            c("nmse").mean().alias("nmse_x_mean"),
            c("nmse").median().alias("nmse_x_median"),
            c("nmse").quantile(0.05).alias("nmse_x_05p"),
            c("nmse").quantile(0.95).alias("nmse_x_95p"),
            c("nmse").quantile(0.25).alias("nmse_x_25p"),
            c("nmse").quantile(0.75).alias("nmse_x_75p"),
            c("nmse").std().alias("nmse_x_std"),
        )
        .with_columns(
            (c("m") / c("n")).alias("measurement_rate"),
            (c("sparsity") / c("m")).alias("sparsity_rate"),
        )
    )
    return df_pt


def plot_phase_transition(df: pl.DataFrame, algorithm: Literal["ip", "omp"]) -> None:
    n = df["n"][0]
    df_pt = get_phase_transition_data(df, algorithm)
    tbl = (
        df_pt.sort(by=["m", "sparsity"], descending=[True, False])
        .pivot(
            values="success_rate",
            index="m",
            columns="sparsity",
            aggregate_function="first",
        )
        .to_pandas()
    )
    tbl = tbl.set_index("m", drop=True)
    sns.heatmap(tbl)
    plt.xlabel("Sparsity $s$")
    plt.ylabel("Number of measurements $m$")

    plt.title(f"Phase Transition for {algorithm.upper()} (n={n})")


def plot_probability_curve(df: pl.DataFrame, save_file: Path | None = None) -> None:
    df_pt_omp = get_phase_transition_data(df, "omp")
    df_pt_ip = get_phase_transition_data(df, "ip")
    n = df_pt_omp["n"][0]

    labels = []
    lines = []
    plt.figure()
    for s in sorted(df_pt_omp["sparsity"].unique()):
        labels.append(f"$s$={s}")
        df_pt_at_s_omp = df_pt_omp.filter(c("sparsity") == s).sort("m")
        df_pt_at_s_ip = df_pt_ip.filter(c("sparsity") == s).sort("m")
        cur_lines = plt.plot(df_pt_at_s_omp["m"], df_pt_at_s_omp["success_rate"])
        lines.append(cur_lines[0])
        cur_lines = plt.plot(
            df_pt_at_s_ip["m"],
            df_pt_at_s_ip["success_rate"],
            "o",
            fillstyle="none",
            color=cur_lines[0].get_color(),
        )
        plt.xlabel("Number of measurements $m$")
        plt.ylabel("Probability of exact recovery")
        plt.title(f"Number of dictionary atoms $n$={n}")
        plt.grid("on")

    plt.legend(lines, labels, bbox_to_anchor=(1.32, 0.75))

    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.savefig(save_file, bbox_inches="tight")


def plot_metric_curves(
    df_small: pl.DataFrame,
    df_large: pl.DataFrame,
    save_file: Path | None = None,
    metric: str = "success_rate",
    semilogy: bool = False,
) -> None:
    _, axs = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for k, df in enumerate([df_small, df_large]):
        ax = axs[k]
        df_pt_omp = get_phase_transition_data(df, "omp")
        df_pt_ip = get_phase_transition_data(df, "ip")
        n = df_pt_omp["n"][0]

        labels = []
        lines = []
        for s in sorted(df_pt_omp["sparsity"].unique()):
            labels.append(f"$s$={s}")
            df_pt_at_s_omp = df_pt_omp.filter(c("sparsity") == s).sort("m")
            df_pt_at_s_ip = df_pt_ip.filter(c("sparsity") == s).sort("m")
            cur_lines = ax.plot(df_pt_at_s_omp["m"], df_pt_at_s_omp[metric])
            lines.append(cur_lines[0])
            cur_lines = ax.plot(
                df_pt_at_s_ip["m"],
                df_pt_at_s_ip[metric],
                "o",
                fillstyle="none",
                color=cur_lines[0].get_color(),
            )
            ax.set_xlabel("Number of measurements $m$")
            if k == 0:
                ax.set_ylabel(METRIC_NAME_LOOKUP.get(metric, metric))
            if semilogy:
                plt.yscale("log")
            ax.set_title(f"Number of dictionary atoms $n$={n}")
            ax.grid("on")

        ax.legend(lines, labels, loc="lower right")

    plt.subplots_adjust(wspace=0.05)

    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.savefig(save_file, bbox_inches="tight")


def main(
    small_result_path: Path,
    large_result_path: Path,
    save_file: Path,
    max_m_small: int | None = None,
    max_m_large: int | None = None,
    together: bool = False,
    snr: float | None = None,
    metric: str = "success_rate",
    semilogy: bool = False,
) -> None:
    df_small = pl.read_parquet(small_result_path)
    df_large = pl.read_parquet(large_result_path)

    if max_m_small is not None:
        df_small = df_small.filter(pl.col("m") <= max_m_small)

    if max_m_large is not None:
        df_large = df_large.filter(pl.col("m") <= max_m_large)

    if snr is not None:
        df_small = df_small.filter(pl.col("snr") == snr)
        df_large = df_large.filter(pl.col("snr") == snr)

    if together:
        plot_metric_curves(
            df_small, df_large, save_file=save_file, metric=metric, semilogy=semilogy
        )
    else:
        plot_probability_curve(
            df_small,
            save_file=save_file.parent / (f"{save_file.stem}_small" + save_file.suffix),
        )
        plot_probability_curve(
            df_large,
            save_file=save_file.parent / (f"{save_file.stem}_large" + save_file.suffix),
        )


if __name__ == "__main__":
    typer.run(main)
