from ip_is_all_you_need.omp import (
    gen_dictionary,
    projection,
    generate_measurements_and_coeffs,
    ip_objective,
    omp_objective,
    ip_estimate_x,
    omp_estimate_x,
    ip_estimate_y,
    omp_estimate_y,
    ip,
    omp,
    recall,
    precision,
    mse,
    iou,
    mutual_coherence,
    run_experiment,
)

from ip_is_all_you_need.summarize import load, plot

import pytest
import numpy as np


def test_gen_dictionary():
    pass


def test_projection():
    m, n = 5, 10
    Phi = np.random.randn(m, n)
    Phi_t = Phi[:, []]

    P = projection(Phi_t, perp=True)
    assert np.allclose(P, np.eye(m))

    P = projection(Phi_t, perp=False)
    assert np.allclose(P, np.zeros((m, m)))

    # TODO: test normal cases


def test_generate_measurements_and_coeffs():
    pass


def test_ip_objective():
    pass


def test_omp_objective():
    pass


def test_ip_estimate_x():
    pass


def test_omp_estimate_x():
    pass


def test_ip_estimate_y():
    pass


def test_omp_estimate_y():
    pass


def test_ip():
    pass


def test_omp():
    pass


def test_recall():
    pass


def test_precision():
    pass


def test_mse():
    pass


def test_iou():
    pass


def test_mutual_coherence():
    pass


def test_run_experiment():
    pass


def test_load():
    pass


def test_plot():
    pass
