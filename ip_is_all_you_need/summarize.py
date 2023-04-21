import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

sns.set()

SCALAR_KEYS = ["coherence", "iters_ip", "iters_omp", "sparsity"]


def interpolate(records: list[dict[str, list[float] | float]]) -> dict[str, np.ndarray]:
    max_length = max(len(r["precision_ip"]) for r in records)
    interpolated = {k: list() for k in records[0].keys() if k not in SCALAR_KEYS}

    # FIXME: error bars indicate recall of 1.0 at #iterations < sparsity, which is impossible
    # however, this phenomenon is not present in the raw data, indicating the bug is in this function
    x = np.arange(1, max_length + 1) / max_length
    for k in interpolated.keys():
        for record in records:
            n = len(record[k])
            xp = np.arange(1, n + 1) / n
            interpolated[k].append(np.interp(x, xp, record[k]))

        interpolated[k] = np.vstack(interpolated[k])

    return interpolated


def build_dataframe(
    records: list[dict[str, list[float] | float]],
    path: Path,
    settings: dict[str, float | int | str],
) -> pd.DataFrame:

    scalars = {k: list() for k in records[0].keys()}

    # NOTE: if the number of iterations for IP and OMP are the same, then the precision/recall values will be the same!
    # this is because the cardinality of the support and the estimated support are the same (see P/R equations)
    for k in scalars.keys():
        for record in records:
            if k in SCALAR_KEYS:
                scalars[k].append(record[k])
            else:
                scalars[k].append(record[k][-1])

    df = pd.DataFrame(scalars)

    for key, value in settings.items():
        df[key] = value

    df["trial"] = np.arange(len(df))

    df["setting_id"] = int(path.stem)

    return df


def load(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int | float | str]]:
    with open(path / "results.json") as f:
        results = json.load(f)

    settings = {k: v for k, v in results.items() if k != "results"}

    interpolated = interpolate(results["results"])
    df = build_dataframe(results["results"], path, settings)

    with open(path / "interpolated.pkl", "wb") as f:
        joblib.dump(interpolated, f)

    with open(path / "settings.json", "w") as f:
        json.dump(settings, f)

    return df, interpolated, settings


def key_to_ylabel(key: str) -> str:
    if key in ["precision", "recall"]:
        return key.title()

    elif key.startswith("mse_x"):
        return r"$||\widehat{x} - x||_2^2$"

    elif key.startswith("mse_y"):
        return r"$||\widehat{y} - y||_2^2$"

    return "Maximum Objective"


def plot(interpolated: dict[str, np.ndarray], settings: dict[str, float], path: Path):
    for key in ["precision", "recall", "mse_x", "mse_y"]:
        ip = interpolated[f"{key}_ip"]
        omp = interpolated[f"{key}_omp"]

        # hack to normalize (TODO: fix normalization in experiments)
        if key in ["mse_x", "mse_y"]:
            ip /= settings["n"] * settings["sparsity"]
            omp /= settings["n"] * settings["sparsity"]

        n = interpolated[f"{key}_ip"].shape[1]
        x = np.arange(1, n + 1) / n

        mean_ip = np.mean(ip, axis=0)
        lo_ip = np.quantile(ip, 0.25, axis=0)
        hi_ip = np.quantile(ip, 0.75, axis=0)
        mean_omp = np.mean(omp, axis=0)
        lo_omp = np.quantile(omp, 0.25, axis=0)
        hi_omp = np.quantile(omp, 0.75, axis=0)
        plt.figure()
        plt.plot(x, mean_omp)
        plt.fill_between(x, lo_omp, hi_omp, alpha=0.3)
        plt.plot(x, mean_ip, "--")
        plt.fill_between(x, lo_ip, hi_ip, alpha=0.3)
        plt.legend(["OMP Mean", "OMP IQR", "IP Mean", "IP IQR"])
        m, n, s = settings["m"], settings["n"], settings["sparsity"]
        plt.title(rf"{m=}, {n=}, $\mathbb{{E}}[s]={s}$ ($\rho={m/n:0.2f}$)")
        plt.xlabel("Fraction of Iterations Completed $k / s$")
        plt.ylabel(key_to_ylabel(key))
        if key in ["precision", "recall"]:
            plt.ylim((-0.025, 1.025))
        plt.tight_layout()
        plt.savefig(path / f"{key}.png", dpi=300)
        plt.close()


def main():
    dfs = []
    for path in Path("results").iterdir():
        if path.is_dir() and (path / "results.json").exists():
            df, interpolated, settings = load(path)
            plot(interpolated, settings, path)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df_long = pd.wide_to_long(
        df,
        ["precision", "recall", "mse_x", "mse_y", "max_objective", "iters"],
        i=["setting_id", "trial"],
        j="algorithm",
        sep="_",
        suffix="(ip|omp)",
    ).reset_index()

    # TODO: record norm of x, y so we can actually do dB calculations
    # df_long["mse_x_db"] = 10 * np.log10(df_long["mse_x"])
    # df_long["mse_y_db"] = 10 * np.log(df_long["mse_y"])
    df_long["rho"] = df_long["m"] / df_long["n"]
    df_long = df_long.assign(
        mse_x=df_long["mse_x"] / (df_long["n"] * df_long["sparsity"])
    )
    df_long = df_long.assign(
        mse_y=df_long["mse_y"] / (df_long["n"] * df_long["sparsity"])
    )

    (dfs[-1]["precision_ip"] == dfs[-1]["recall_ip"]).all()

    fig_path = Path("./figures")
    fig_path.mkdir(exist_ok=True)
    metrics = ["precision", "recall", "mse_x", "mse_y", "iou"]
    metric_to_label = {
        "precision": "Precision",
        "recall": "Recall",
        "mse_x_db": "Sparse Code MSE (dB)",
        "mse_y_db": "Reconstruction MSE (dB)",
        "mse_x": r"$||\widehat{x} - x||_2^2$",
        "mse_y": r"$||\widehat{y} - y||_2^2$",
        "iou": "IP/OMP Support IoU",
    }

    for metric in metrics:
        for _, row in df[["m", "n"]].drop_duplicates().iterrows():
            m = row["m"]
            n = row["n"]
            sub_df = df_long.loc[(df_long.m == m) & (df_long.n == n)]
            sub_df = sub_df.assign(Algorithm=sub_df["algorithm"].str.upper().copy())
            plt.figure()
            sns.boxplot(
                sub_df,
                x="sparsity",
                y=metric,
                hue="Algorithm" if metric != "iou" else None,
            )
            plt.xlabel(r"Average Sparsity $\mathbb{E}[s] = \Pr(x_i) \ne 0$")
            plt.ylabel(metric_to_label[metric])
            plt.title(f"{m=}, {n=}")
            plt.savefig(fig_path / f"{metric}_{m}_{n}.png", dpi=300)


if __name__ == "__main__":
    typer.run(main)
