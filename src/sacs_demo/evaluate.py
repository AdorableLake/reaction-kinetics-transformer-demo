"""Metrics for synthetic demonstration targets only."""
import numpy as np
import torch
from torch import nn


def predict(model: nn.Module, standardized_features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(standardized_features)).cpu().numpy()


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("Targets and predictions must have matching nonempty shapes")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite arrays")
    error = predictions.astype(np.float64) - targets
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}
