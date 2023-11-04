from enum import Enum

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEnum(str, Enum):
    """Enum that plays nicely with __str__."""

    def __str__(self) -> str:
        return self.value


class CoeffDistribution(BaseEnum):
    """Distribution of the sparse code."""

    sparse_gaussian = (
        "sparse_gaussian"  # sparse Gaussian coeff. distribution (normal on support)
    )
    sparse_const = (
        "sparse_const"  # sparse constant coeff. distribution (= 1 on support)
    )


class Device(BaseEnum):
    """Device (CPU/GPU)."""

    cuda = "cuda"  # GPU
    cpu = "cpu"  # CPU


class Setting(BaseEnum):
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
