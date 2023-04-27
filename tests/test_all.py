from math import sqrt

import pytest
import torch

from ip_is_all_you_need.algorithms import (
    estimate_x,
    estimate_y,
    ip,
    ip_objective,
    omp,
    projection,
)
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

    # just edge cases for now
    projection_shape = (Phi.shape[0], Phi.shape[1], Phi.shape[1])

    P = projection(Phi[:, :, []], perp=False)
    assert P.shape == projection_shape
    assert torch.allclose(P, torch.tensor(0.0))

    P = projection(Phi[:, :, []], perp=True)
    assert P.shape == projection_shape
    assert torch.allclose(P, eye)

    P = projection(Phi, perp=False)
    assert P.shape == projection_shape
    assert torch.allclose(P, eye)

    P = projection(Phi, perp=True)
    assert P.shape == projection_shape
    assert torch.allclose(P, torch.tensor(0.0))


def test_ip_objective(sim_data):
    Phi, _, y = sim_data

    obvals = ip_objective(Phi, y)
    assert obvals.shape == (Phi.shape[0], Phi.shape[2], 1)
    assert obvals.min() >= 0
    assert not torch.isnan(obvals).any()

    obvals = ip_objective(
        Phi,
        y,
        columns=torch.arange(2 * Phi.shape[0], dtype=torch.long).reshape(
            Phi.shape[0], 2
        ),
    )
    assert torch.isneginf(obvals[:, [1, 2], :]).all()


def test_estimate_x(sim_data):
    Phi, x, y = sim_data
    x_hat_empty = estimate_x(Phi, y, [])
    assert torch.allclose(x_hat_empty, torch.tensor(0.0))
    x_hat = estimate_x(Phi, y, [1, 2])
    assert x_hat.shape == x_hat_empty.shape == (Phi.shape[0], Phi.shape[2], 1)


def test_estimate_y(sim_data):
    Phi, x, y = sim_data
    y_hat_empty = estimate_y(Phi, y, [])
    assert torch.allclose(y_hat_empty, torch.tensor(0.0))
    y_hat = estimate_y(Phi, y, [1, 2])
    assert y_hat.shape == y_hat_empty.shape == (Phi.shape[0], Phi.shape[1], 1)


def test_ip(sim_data):
    Phi, x, y = sim_data
    log_ip = ip(Phi, y, num_iterations=10)
    assert "indices" in log_ip.keys()
    assert "objective" in log_ip.keys()
    assert len(log_ip["indices"]) == len(log_ip["objective"])
    assert all([isinstance(x, list) for x in log_ip["indices"]])
    assert all([isinstance(xi, int) for x in log_ip["indices"] for xi in x])
    assert all([len(x) == Phi.shape[0] for x in log_ip["indices"]])


def test_omp(sim_data):
    Phi, x, y = sim_data
    log_omp = omp(Phi, y)
    assert "indices" in log_omp.keys()
    assert "objective" in log_omp.keys()
    assert len(log_omp["indices"]) == len(log_omp["objective"])
    assert all([isinstance(x, list) for x in log_omp["indices"]])
    assert all([isinstance(xi, int) for x in log_omp["indices"] for xi in x])
    assert all([len(x) == Phi.shape[0] for x in log_omp["indices"]])


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
