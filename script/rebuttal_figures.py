from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ip_is_all_you_need.figures import METRIC_NAME_LOOKUP, get_phase_transition_data

c = pl.col


def plot_single(
    ax: plt.Axes,
    df_pt_ip: pl.DataFrame,
    df_pt_omp: pl.DataFrame,
    metric: str,
    n: int,
    snr: int,
    semilogy: bool = False,
    legend: bool = False,
    ylabel: bool = False,
):
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
        ax.set_xlabel("Measurements $m$")
        if ylabel:
            ax.set_ylabel(METRIC_NAME_LOOKUP.get(metric, metric))
        ax.set_title(f"Dictionary atoms $n$={n}, E[SNR]={snr} dB")
        if semilogy:
            ax.set_yscale("log")
        ax.grid("on")

    if legend:
        ax.legend(
            lines, labels, loc="center right", bbox_to_anchor=(1.3, 0.3, 0.1, 0.4)
        )


def plot_metric_curves(
    df_small: pl.DataFrame,
    df_large: pl.DataFrame,
    save_file: Path | None = None,
    metric: str = "success_rate",
    semilogy: bool = False,
    dpi: int = 300,
    sharey: bool | str = False,
) -> None:
    df_pt_omp_small = get_phase_transition_data(df_small, "omp")
    df_pt_ip_small = get_phase_transition_data(df_small, "ip")
    df_pt_omp_large = get_phase_transition_data(df_large, "omp")
    df_pt_ip_large = get_phase_transition_data(df_large, "ip")
    n_small = df_small["n"][0]
    n_large = df_large["n"][0]

    fig, ax = plt.subplots(2, 4, sharey=sharey, figsize=(16, 8))
    for k, snr in enumerate([5, 10, 15, 20]):
        plot_single(
            ax[0, k],
            df_pt_ip_small.filter(c("snr") == snr),
            df_pt_omp_small.filter(c("snr") == snr),
            metric=metric,
            n=n_small,
            snr=snr,
            semilogy=semilogy,
            legend=False if k < 3 else True,
            ylabel=True if k == 0 else False,
        )

    for k, snr in enumerate([5, 10, 15, 20]):
        plot_single(
            ax[1, k],
            df_pt_ip_large.filter(c("snr") == snr),
            df_pt_omp_large.filter(c("snr") == snr),
            metric=metric,
            n=n_large,
            snr=snr,
            semilogy=semilogy,
            legend=False if k < 3 else True,
            ylabel=True if k == 0 else False,
        )

    fig.subplots_adjust(hspace=0.5)
    if save_file:
        if save_file.suffix != ".eps":
            plt.savefig(save_file, bbox_inches="tight", dpi=dpi)
        plt.savefig(save_file, bbox_inches="tight")


files = dict()
files["small"] = Path("./rebuttal_final/small.parquet")
files["large"] = Path("./rebuttal_final/large.parquet")

dfs = dict()
dfs["small"] = pl.read_parquet(files["small"])
dfs["large"] = pl.read_parquet(files["large"])
plot_metric_curves(
    dfs["small"],
    dfs["large"],
    Path("./results_rebuttal.eps"),
    metric="nmse_x_mean",
    semilogy=True,
)

# for size in ["small", "large"]:
#     for snr in [5, 10, 15]:
#         plot_metric_curve(
#             dfs[size].filter(c("snr") == snr),
#             save_file=Path(f"results_size_{size}_snr_{snr}.png"),
#             metric="nmse_x_mean",
#             semilogy=True,
#             snr=snr,
#             legend=False,
#             dpi=600,
#         )
#     plot_metric_curve(
#         dfs[size].filter(c("snr") == 20),
#         save_file=Path(f"results_size_{size}_snr_20.png"),
#         metric="nmse_x_mean",
#         semilogy=True,
#         snr=snr,
#         legend=True,
#         dpi=600,
#     )
