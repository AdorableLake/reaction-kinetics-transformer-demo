# Reaction Kinetics Transformer Demo

A reproducible machine-learning workflow for trajectory prediction using a
Transformer baseline, synthetic data, train-only preprocessing, model
selection, evaluation, and automated tests.

This repository is a public demonstration derived from my SACs
reaction-kinetics research workflow. Research-specific datasets,
unpublished model components, experimental results, and IP-sensitive
implementations are intentionally excluded.

## Research Project

The research project motivates the workflow of preparing tabular inputs, predicting trajectories, and evaluating predictions. Its scientific workspace remains the research source of truth. This repository does not reproduce its architecture, data, parameters, results, or validation protocol. This public repository does not represent the full scientific contribution or performance of the underlying research project.

## Public Demonstration

This repository contains an independently implemented, simplified Transformer
baseline for demonstrating the engineering workflow. It does not reproduce
the unpublished SACs research architecture, parameters, datasets, or results.

The demo generates artificial dimensionless curves, splits samples, fits
standardization only on training features, trains a feature-token Transformer,
selects its in-memory state using validation loss, and evaluates once on a
held-out synthetic test partition. A training-mean curve provides a simple
baseline for comparison.

All reported metrics describe only this synthetic demonstration and must not
be interpreted as research or publication results.

## What this demonstrates

- **Data pipeline:** Generates synthetic trajectories, creates disjoint train/validation/test splits, and fits feature standardization on training data only.
- **Model implementation:** Converts tabular features into tokens and maps them to a predicted trajectory with a simplified Transformer baseline.
- **Training and evaluation:** Selects the model state by validation loss, then evaluates on held-out synthetic test data against a training-mean curve baseline.
- **Reliability / testing:** Includes automated checks for split integrity, preprocessing boundaries, model gradients, validation-based state restoration, and invalid inputs.
- **Reproducibility:** Runs the workflow from a configuration file with fixed seeds and in-memory synthetic data, with an end-to-end repeatability test within one CPU software environment.

## Run from a fresh checkout

Use Python 3.9 or newer. CPU execution is the supported demo path. The direct dependency pins are in `requirements.lock`; platform-specific transitive dependencies are not fully locked.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m sacs_demo.demo --config configs/demo.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Run these commands from this repository root. No dataset download, CSV, scientific workspace, or saved model is needed. Synthetic features and targets are generated from scratch in memory on every run. Training and inference run together; the command prints JSON and does not save data, predictions, or model files. Inference is implemented by `sacs_demo.evaluate.predict` for a model trained in the current process.

Alternatively, install the package with `python -m pip install -e .` and run `sacs-demo --config configs/demo.yaml`.

## What to inspect

- `src/sacs_demo/data.py`: independently specified synthetic process and sample splits.
- `src/sacs_demo/preprocessing.py`: train-only standardization.
- `src/sacs_demo/model.py`: generic Transformer baseline.
- `src/sacs_demo/train.py`: training and validation selection.
- `src/sacs_demo/evaluate.py`: in-memory inference and metrics.
- `tests/test_demo.py`: data, leakage boundary, gradients, selection, and reproducibility checks.
- [Architecture](docs/architecture.md) and [limitations](docs/limitations.md).

Reproducibility is tested within one CPU software environment; bitwise agreement across operating systems, hardware, and dependency versions is not promised. No open-source license grant is included pending ownership and licensing review.
