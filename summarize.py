# %%
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.logging import RichHandler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger()


def plot():
    pass


def stats(results: list[dict[str, list[float] | float]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    pass



# %%
def load(results_dir: Path) -> pd.DataFrame:
    experiment_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    for path in tqdm(experiment_dirs):
        with open(path) as f:
            results = json.load(f)
    return results


def main(
    results_dir: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=True, file_okay=False
    )
):
    results = load(results_dir)


if __name__ == "__main__":
    typer.run(main)
