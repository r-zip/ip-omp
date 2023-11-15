from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

from .constants import (
    MAX_M,
    CoeffDistribution,
    NoiseSetting,
    ProblemSize,
    SaveFileFormat,
    SNRs,
)

c = pl.col

SUCCESS_THRESHOLD = 1e-14
METRIC_NAME_LOOKUP = {
    "nmse_x_mean": "Mean NMSE",
    "nmse_x_median": "Median NMSE",
    "success_rate": "Probability of exact recovery",
}

app = typer.Typer()


def filter_df(df: pl.DataFrame, algorithm: Literal["ip", "omp", None] = None) -> pl.DataFrame:
    if algorithm:
        df = df.filter(c("algorithm") == algorithm)
    df = df.with_columns(c("iter").max().over(["experiment_number", "trial", "algorithm"]).alias("max_iter"))
    return df.filter(c("iter") == c("max_iter"))


def get_phase_transition_data(df: pl.DataFrame, algorithm: Literal["ip", "omp"]) -> pl.DataFrame:
    df_pt = (
        # filter to only the last iteration
        filter_df(df, algorithm=algorithm)
        .with_columns((c("n") * c("mse_x") / c("norm_x") ** 2).alias("nmse"))
        # define success as relative reconstruction error < eps
        .with_columns(
            (c("nmse") < SUCCESS_THRESHOLD).alias("success"),
        )
        # for each experiment
        .group_by("experiment_number")
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


def plot_single_noisy(
    ax: plt.Axes,
    df_pt_ip: pl.DataFrame,
    df_pt_omp: pl.DataFrame,
    metric: str,
    n: int,
    snr: int,
    semilogy: bool = False,
    legend: bool = False,
    ylabel: bool = False,
    font_size: int | None = None,
    legend_font_size: int | None = None,
):
    sparsities = sorted(df_pt_omp["sparsity"].unique())

    labels = []
    lines = []
    for s in sparsities:
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
        ax.set_xlabel("#Measurements $m$", fontsize=font_size)
        if ylabel:
            ax.set_ylabel(METRIC_NAME_LOOKUP.get(metric, metric), fontsize=font_size)
        ax.set_title(f"$n$={n}, E[SNR]={snr} dB", fontsize=font_size)
        if semilogy:
            ax.set_yscale("log")
        ax.grid("on")
        ax.tick_params(axis="both", which="major", labelsize=font_size - 2)
        ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)

    if legend:
        ax.legend(
            lines,
            labels,
            loc="center right",
            bbox_to_anchor=(1.38, 0.3, 0.1, 0.4),
            # bbox_to_anchor=(0.0, -0.2),
            # ncol=len(sparsities),
            # loc="lower center",
            fontsize=legend_font_size,
        )


def plot_metric_curves_noisy(
    df_small: pl.DataFrame,
    df_large: pl.DataFrame,
    save_file: Path | None = None,
    metric: str = "success_rate",
    semilogy: bool = False,
    dpi: int = 300,
    sharey: bool | str = False,
    font_size: int = 18,
    legend_font_size: int = 14,
    snrs: SNRs = SNRs.main,
) -> None:
    if snrs == SNRs.main:
        snr_list = [5, 20]
    else:
        snr_list = [10, 15]

    df_pt_omp_small = get_phase_transition_data(df_small, "omp")
    df_pt_ip_small = get_phase_transition_data(df_small, "ip")
    df_pt_omp_large = get_phase_transition_data(df_large, "omp")
    df_pt_ip_large = get_phase_transition_data(df_large, "ip")
    n_small = df_small["n"][0]
    n_large = df_large["n"][0]

    fig, ax = plt.subplots(2, 2, sharey=sharey, sharex=False, figsize=(9, 6))
    for k, snr in enumerate(snr_list):
        plot_single_noisy(
            ax[0, k],
            df_pt_ip_small.filter(c("snr") == snr),
            df_pt_omp_small.filter(c("snr") == snr),
            metric=metric,
            n=n_small,
            snr=snr,
            semilogy=semilogy,
            legend=True if k == 1 else False,
            ylabel=True if k == 0 else False,
            font_size=font_size,
            legend_font_size=legend_font_size,
        )
        ax[0, k].set(xlabel=None)

    for k, snr in enumerate(snr_list):
        plot_single_noisy(
            ax[1, k],
            df_pt_ip_large.filter(c("snr") == snr),
            df_pt_omp_large.filter(c("snr") == snr),
            metric=metric,
            n=n_large,
            snr=snr,
            semilogy=semilogy,
            legend=True if k == 1 else False,
            ylabel=True if k == 0 else False,
            font_size=font_size,
            legend_font_size=legend_font_size,
        )

    # fig.subplots_adjust(hspace=0.65, wspace=1.1)
    fig.tight_layout()
    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=dpi)
        plt.savefig(save_file, bbox_inches="tight")


@app.command()
def plot_noiseless(
    small_result_path: Path = typer.Argument(..., help="Path to the small (n=256) results file."),
    large_result_path: Path = typer.Argument(..., help="Path to the large (n=1024) results file."),
    coeff_distribution: CoeffDistribution = typer.Option(
        default=CoeffDistribution.sparse_gaussian,
        help="The distribution of the coefficients of the sparse code.",
    ),
    save_dir: Path = typer.Option(default=Path("."), help="Where to save resulting plots."),
    save_file_format: SaveFileFormat = typer.Option(default=SaveFileFormat.png, help="Format of the plot file."),
    together: bool = typer.Option(
        default=False,
        help="Whether to plot small (n=256) and large (n=1024) results in the same figure.",
    ),
) -> None:
    """Plot the results of the noiseless sparse recovery experiments."""
    max_m_small = MAX_M[(ProblemSize.small, NoiseSetting.noiseless, coeff_distribution)]
    max_m_large = MAX_M[(ProblemSize.large, NoiseSetting.noiseless, coeff_distribution)]

    sns.set()
    sns.set_context("talk")

    df_small = pl.read_parquet(small_result_path)
    df_large = pl.read_parquet(large_result_path)

    if max_m_small is not None:
        df_small = df_small.filter(c("m") <= max_m_small)

    if max_m_large is not None:
        df_large = df_large.filter(c("m") <= max_m_large)

    # for noiseless, filter to snr == inf
    df_small = df_small.filter(c("snr") == float("inf"))
    df_large = df_large.filter(c("snr") == float("inf"))

    if together:
        plot_metric_curves(
            df_small,
            df_large,
            save_file=save_dir / f"noiseless_recovery_{str(coeff_distribution)}.{str(save_file_format)}",
        )
    else:
        plot_probability_curve(
            df_small,
            save_file=save_dir / f"noiseless_recovery_{str(coeff_distribution)}_small.{str(save_file_format)}",
        )
        plot_probability_curve(
            df_large,
            save_file=save_dir / f"noiseless_recovery_{str(coeff_distribution)}_large.{str(save_file_format)}",
        )


@app.command()
def plot_noisy(
    small_result_path: Path = typer.Argument(..., help="Path to the small (n=256) results file."),
    large_result_path: Path = typer.Argument(..., help="Path to the large (n=1024) results file."),
    save_dir: Path = typer.Option(default=Path("."), help="Where to save resulting plots."),
    save_file_format: SaveFileFormat = typer.Option(default=SaveFileFormat.png, help="Format of the plot file."),
):
    """Plot the results of the noisy sparse recovery experiments."""
    sns.set()

    df_small = pl.read_parquet(small_result_path)
    df_large = pl.read_parquet(large_result_path)
    plot_metric_curves_noisy(
        df_small,
        df_large,
        save_dir / Path(f"noisy_recovery_main.{str(save_file_format)}"),
        metric="nmse_x_mean",
        semilogy=True,
        font_size=18,
        legend_font_size=14,
        snrs=SNRs.main,
    )
    plot_metric_curves_noisy(
        df_small,
        df_large,
        save_dir / Path(f"noisy_recovery_appendix.{str(save_file_format)}"),
        metric="nmse_x_mean",
        semilogy=True,
        font_size=18,
        legend_font_size=14,
        snrs=SNRs.appendix,
    )


if __name__ == "__main__":
    matplotlib.rcParams.update({"font.size": 22})
    app()
