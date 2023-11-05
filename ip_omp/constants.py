from enum import Enum

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 1_000


class BaseEnum(str, Enum):
    """Enum that plays nicely with __str__."""

    def __str__(self) -> str:
        return self.value


class CoeffDistribution(BaseEnum):
    """Distribution of the sparse code."""

    sparse_gaussian = "sparse_gaussian"  # sparse Gaussian coeff. distribution (normal on support)
    sparse_const = "sparse_const"  # sparse constant coeff. distribution (= 1 on support)


class Device(BaseEnum):
    """Device (CPU/GPU)."""

    cuda = "cuda"  # GPU
    cpu = "cpu"  # CPU


class ProblemSize(BaseEnum):
    """Problem size of the experiment (length n of sparse code)."""

    small = "small"  # small (n=256) experiment
    large = "large"  # large (n=1024) experiment


class OrderBy(BaseEnum):
    """How to order available GPUs when choosing one for a subprocess."""

    utilization = "utilization"  # order available GPUs by utilization (ascending)
    memory_usage = "memory_usage"  # order available GPUs by memory usage (ascending)


class SNRs(BaseEnum):
    """SNR values for figure generation."""

    main = "main"  # SNR values for plots in main paper
    appendix = "appendix"  # SNR values for plots in appendix


class SaveFileFormat(BaseEnum):
    """Save file format for figures."""

    eps = "eps"  # eps for paper
    png = "png"  # png for easy viewing


class NoiseSetting(BaseEnum):
    """Noiseless/Noisy"""

    noiseless = "noiseless"
    noisy = "noisy"


MIN_M = {
    ProblemSize.small: 4,
    ProblemSize.large: 5,
}

MAX_M = {
    (ProblemSize.small, NoiseSetting.noiseless, CoeffDistribution.sparse_const): 256,
    (ProblemSize.small, NoiseSetting.noiseless, CoeffDistribution.sparse_gaussian): 208,
    (ProblemSize.large, NoiseSetting.noiseless, CoeffDistribution.sparse_const): 205,
    (ProblemSize.large, NoiseSetting.noiseless, CoeffDistribution.sparse_gaussian): 115,
    (ProblemSize.small, NoiseSetting.noisy, CoeffDistribution.sparse_gaussian): 220,
    (ProblemSize.large, NoiseSetting.noisy, CoeffDistribution.sparse_gaussian): 120,
}

DELTA_M = {
    ProblemSize.small: 12,
    ProblemSize.large: 5,
}

MIN_S = {
    ProblemSize.small: 4,
    ProblemSize.large: 4,
}

MAX_S = {
    ProblemSize.small: 40,
    ProblemSize.large: 16,
}

DELTA_S = {
    ProblemSize.small: 6,
    ProblemSize.large: 2,
}

SNR_GRID = {
    NoiseSetting.noiseless: [float("inf")],
    NoiseSetting.noisy: list(range(5, 25, 5)),
}
