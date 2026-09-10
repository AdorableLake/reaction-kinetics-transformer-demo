"""CPU training; validation selects an in-memory model, test data stays outside."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .model import TransformerBaseline


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, config: dict):
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = TransformerBaseline(x_train.shape[1], y_train.shape[1], config["d_model"], config["heads"], config["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    xt, yt, xv, yv = [torch.from_numpy(a) for a in (x_train, y_train, x_val, y_val)]
    rng = torch.Generator().manual_seed(config["seed"])
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(xt), generator=rng)
        for idx in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(xv), yv))
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
        history.append(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_epoch": best_epoch, "validation_mse": best_loss, "validation_history": history}
