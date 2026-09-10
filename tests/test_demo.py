import copy
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from sacs_demo.data import generate_synthetic, split_indices
from sacs_demo.preprocessing import Standardizer
from sacs_demo.model import TransformerBaseline
from sacs_demo.evaluate import metrics, predict
from sacs_demo.train import train_model
from sacs_demo.demo import run_demo, validate_config

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/demo.yaml").read_text())


class DemoTests(unittest.TestCase):
    def test_synthetic_determinism_and_definition(self):
        a, b = generate_synthetic(20, 8, 1), generate_synthetic(20, 8, 1)
        np.testing.assert_array_equal(a.targets, b.targets)
        self.assertFalse(np.array_equal(a.features, generate_synthetic(20, 8, 2).features))
        np.testing.assert_allclose(a.targets[:, 0], 1.0)
        self.assertTrue((np.diff(a.targets, axis=1) <= 0).all())
        self.assertTrue(((a.targets >= 0) & (a.targets <= 1)).all())

    def test_split_disjoint_complete(self):
        parts = split_indices(40, 0.6, 0.2, 3)
        self.assertEqual(len(set(np.concatenate(parts))), 40)
        self.assertEqual(sum(map(len, parts)), 40)

    def test_scaler_train_only_and_constant_feature(self):
        train = np.array([[1., 5.], [3., 5.]], dtype=np.float32)
        scaler = Standardizer.fit(train)
        np.testing.assert_allclose(scaler.mean, [2, 5])
        np.testing.assert_allclose(scaler.transform(train).mean(axis=0), [0, 0])
        scaler.transform(np.array([[10000., 8.]], dtype=np.float32))
        np.testing.assert_allclose(scaler.mean, [2, 5])
        self.assertTrue(np.isfinite(scaler.transform(train)).all())

    def test_forward_gradients_and_inference(self):
        torch.set_num_threads(1)
        model = TransformerBaseline(4, 8)
        x = torch.randn(3, 4)
        pred = model(x)
        self.assertEqual(tuple(pred.shape), (3, 8))
        self.assertTrue(torch.all((pred >= 0) & (pred <= 1)))
        pred.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        np.testing.assert_array_equal(predict(model, x.numpy()), predict(model, x.numpy()))

    def test_metrics_known_values_and_rejection(self):
        self.assertEqual(metrics(np.zeros((2, 3)), np.ones((2, 3))), {"mae": 1.0, "rmse": 1.0})
        with self.assertRaises(ValueError):
            metrics(np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            metrics(np.ones(2), np.array([np.nan, 0]))

    def test_invalid_config(self):
        for changes in ({"epochs": 0}, {"learning_rate": float("nan")}, {"heads": 3},
                        {"train_fraction": 0.9, "validation_fraction": 0.2}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                validate_config(dict(CONFIG, **changes))

    def test_training_updates_and_restores_validation_selection(self):
        cfg = dict(CONFIG, epochs=3, samples=24, time_points=8)
        data = generate_synthetic(24, 8, cfg["seed"])
        x = Standardizer.fit(data.features[:16]).transform(data.features)
        torch.manual_seed(cfg["seed"])
        before = TransformerBaseline(4, 8, cfg["d_model"], cfg["heads"], cfg["layers"])
        initial = copy.deepcopy(before.state_dict())
        model, info = train_model(x[:16], data.targets[:16], x[16:], data.targets[16:], cfg)
        self.assertTrue(any(not torch.equal(initial[k], model.state_dict()[k]) for k in initial))
        actual = np.square(predict(model, x[16:]) - data.targets[16:]).mean()
        self.assertAlmostEqual(float(actual), info["validation_mse"], places=6)
        self.assertEqual(info["validation_mse"], min(info["validation_history"]))

    def test_end_to_end_repeatable(self):
        cfg = dict(CONFIG, epochs=2, samples=24, time_points=8)
        first, second = run_demo(cfg), run_demo(cfg)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["splits"].values()), 24)
        self.assertTrue(np.isfinite(first["synthetic_test_metrics"]["rmse"]))


if __name__ == "__main__":
    unittest.main()
