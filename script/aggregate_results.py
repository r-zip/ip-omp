from pathlib import Path

import typer

from ip_is_all_you_need.simulations import aggregate_results


def main(results_path: Path) -> None:
    aggregate_results(results_path)


if __name__ == "__main__":
    typer.run(main)
