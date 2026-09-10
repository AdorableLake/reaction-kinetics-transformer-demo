"""Numeric preprocessing fitted exclusively to the training partition."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_features: np.ndarray):
        if training_features.ndim != 2 or len(training_features) == 0 or not np.isfinite(training_features).all():
            raise ValueError("Expected a nonempty finite feature matrix")
        scale = training_features.std(axis=0)
        return cls(training_features.mean(axis=0), np.where(scale > 1e-8, scale, 1.0))

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError("Invalid feature matrix")
        return ((features - self.mean) / self.scale).astype(np.float32)
