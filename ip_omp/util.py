import numpy as np


def _get_scale(square: bool) -> float:
    return 10 if square else 20


def db_to_ratio(db: float, square: bool = True) -> float:
    return 10 ** (db / _get_scale(square))


def ratio_to_db(ratio: float, square: bool = True) -> float:
    return _get_scale(square) * np.log10(ratio)
