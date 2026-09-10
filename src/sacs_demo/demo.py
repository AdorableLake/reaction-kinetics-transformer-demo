"""Generate, preprocess, train, infer, and evaluate without writing artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .data import generate_synthetic, split_indices
from .preprocessing import Standardizer
from .train import train_model
from .evaluate import predict, metrics

KEYS = {"seed", "samples", "time_points", "train_fraction", "validation_fraction", "d_model",
        "heads", "layers", "epochs", "batch_size", "learning_rate"}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError("Configuration must contain exactly the documented demo keys")
    for key in ("seed", "samples", "time_points", "d_model", "heads", "layers", "epochs", "batch_size"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer setting: " + key)
    for key in ("learning_rate", "train_fraction", "validation_fraction"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid numeric setting: " + key)
    if config["d_model"] % config["heads"]:
        raise ValueError("d_model must be divisible by heads")
    split_indices(config["samples"], config["train_fraction"], config["validation_fraction"], config["seed"])
    return config


def run_demo(config: dict) -> dict:
    config = validate_config(config)
    data = generate_synthetic(config["samples"], config["time_points"], config["seed"])
    tr, va, te = split_indices(len(data.features), config["train_fraction"], config["validation_fraction"], config["seed"])
    scaler = Standardizer.fit(data.features[tr])
    x = scaler.transform(data.features)
    model, training = train_model(x[tr], data.targets[tr], x[va], data.targets[va], config)
    predictions = predict(model, x[te])
    mean_baseline = np.broadcast_to(data.targets[tr].mean(axis=0), predictions.shape)
    return {"scope": "synthetic demonstration only; not research performance", "seed": config["seed"],
            "splits": {"train": len(tr), "validation": len(va), "test": len(te)},
            "test_shape": list(predictions.shape), "training": training,
            "synthetic_test_metrics": metrics(data.targets[te], predictions),
            "synthetic_training_mean_baseline": metrics(data.targets[te], mean_baseline)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_demo(config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
