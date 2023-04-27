from math import sqrt

import pytest
import torch

from ip_is_all_you_need.algorithms import projection
from ip_is_all_you_need.metrics import mutual_coherence
from ip_is_all_you_need.simulations import (
    gen_dictionary,
    generate_measurements_and_coeffs,
)


@pytest.fixture
def sim_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.set_default_dtype(torch.double)
    m = 100
    n = 200
    p = 0.05
    batch_size = 10
    Phi = gen_dictionary(batch_size, m, n)
    y, x = generate_measurements_and_coeffs(Phi, p)
    return Phi, x, y


def test_sim_data(sim_data):
    Phi, x, y = sim_data
    batch_size, m, n = Phi.shape
    assert x.shape == (batch_size, n, 1)
    assert y.shape == (batch_size, m, 1)
    assert torch.allclose(Phi @ x, y)


def test_projection(sim_data):
    Phi, _, _ = sim_data
    eye = torch.eye(Phi.shape[1]).repeat((Phi.shape[0], 1, 1))
    assert torch.allclose(projection(Phi[:, :, []], perp=False), torch.tensor(0.0))
    assert torch.allclose(projection(Phi[:, :, []], perp=True), eye)
    assert torch.allclose(projection(Phi, perp=False), eye)
    assert torch.allclose(projection(Phi, perp=True), torch.tensor(0.0))


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


@pytest.mark.parametrize(
    "A,coherence",
    [
        (
            torch.arange(4).reshape(2, 2) / torch.Tensor([2.0, sqrt(10)]),
            0.9486833,
        ),
        (torch.eye(10), 0.0),
        (torch.ones((10, 10)) / sqrt(10), 1.0),
    ],
)
def test_mutual_coherence(A: torch.Tensor, coherence: float):
    assert torch.allclose(mutual_coherence(A), coherence)


def test_run_experiment():
    pass


def test_load():
    pass


def test_plot():
    pass
