"""Generic feature-token Transformer baseline, newly written for this demo."""
import torch
from torch import nn


class TransformerBaseline(nn.Module):
    def __init__(self, features: int, outputs: int, d_model: int = 16, heads: int = 2, layers: int = 1):
        super().__init__()
        if min(features, outputs, d_model, heads, layers) < 1 or d_model % heads:
            raise ValueError("Positive dimensions required; d_model must be divisible by heads")
        self.features = features
        self.value_embedding = nn.Linear(1, d_model)
        self.feature_embedding = nn.Embedding(features, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                           dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, outputs), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError("Expected batch by feature matrix")
        ids = torch.arange(self.features, device=x.device)
        tokens = self.value_embedding(x.unsqueeze(-1)) + self.feature_embedding(ids)
        return self.head(self.encoder(tokens).mean(dim=1))
