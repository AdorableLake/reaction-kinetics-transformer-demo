"""Artificial dimensionless curves, defined without research inputs."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticData:
    features: np.ndarray
    targets: np.ndarray
    times: np.ndarray


def generate_synthetic(samples: int, time_points: int, seed: int) -> SyntheticData:
    if samples < 12 or time_points < 3:
        raise ValueError("Need at least 12 samples and 3 time points")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, 4)).astype(np.float32)
    times = np.linspace(0.0, 1.0, time_points, dtype=np.float32)
    rate = 0.4 + 0.3 * (x[:, 0] + 1.0) + 0.2 * (x[:, 1] + 1.0)
    plateau = 0.1 + 0.1 * (x[:, 2] + 1.0)
    # Feature 4 is an independent nuisance variable by construction.
    y = plateau[:, None] + (1.0 - plateau[:, None]) * np.exp(-rate[:, None] * times)
    return SyntheticData(x, y.astype(np.float32), times)


def split_indices(samples: int, train_fraction: float, validation_fraction: float, seed: int):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Fractions must leave nonempty train, validation, and test sets")
    n_train, n_val = int(samples * train_fraction), int(samples * validation_fraction)
    if min(n_train, n_val, samples - n_train - n_val) < 2:
        raise ValueError("Every split must contain at least two samples")
    indices = np.random.default_rng(seed).permutation(samples)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
