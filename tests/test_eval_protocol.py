"""Tests for the matched evaluation protocol and paired statistics (CPU, no model)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_protocol as ep  # noqa: E402


class NoiseTests(unittest.TestCase):
    def test_noise_is_a_function_of_task_episode_step_only(self):
        a = ep.episode_noise(1000, "libero_spatial", 3, [1000, 1001, 1002], 7, 50, 32)
        b = ep.episode_noise(1000, "libero_spatial", 3, [1002, 1000], 7, 50, 32)  # different batch composition/order
        self.assertTrue(torch.equal(a[2], b[0]))
        self.assertTrue(torch.equal(a[0], b[1]))
        self.assertFalse(torch.equal(a[0], a[1]))
        self.assertFalse(torch.equal(a[0], ep.episode_noise(1000, "libero_spatial", 3, [1000], 8, 50, 32)[0]))
        self.assertFalse(torch.equal(a[0], ep.episode_noise(1000, "libero_spatial", 4, [1000], 7, 50, 32)[0]))
        self.assertFalse(torch.equal(a[0], ep.episode_noise(1001, "libero_spatial", 3, [1000], 7, 50, 32)[0]))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(tuple(a.shape), (3, 50, 32))

    def test_noise_is_standard_normal(self):
        big = ep.episode_noise(1000, "s", 0, list(range(1000, 1200)), 0, 50, 32)
        self.assertAlmostEqual(float(big.mean()), 0.0, places=2)
        self.assertAlmostEqual(float(big.std()), 1.0, places=2)

    def test_install_wraps_reset_and_select_action(self):
        calls = []

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("C", (), {"chunk_size": 4, "max_action_dim": 3})()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def select_action(self, observation, noise=None, **kwargs):
                calls.append(noise.clone())
                return torch.zeros(noise.shape[0], 3)

        class Env:
            def reset(self, *, seed=None, options=None):
                return {"seed": seed}, {}

        policy, env = Policy(), Env()
        info = ep.install_deterministic_noise(policy, {"suite": {5: env}}, base_seed=1000)
        self.assertTrue(info["enabled"])
        env.reset(seed=[1000, 1001])
        obs = {"observation.state": torch.zeros(2, 8)}
        policy.select_action(obs); policy.select_action(obs)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0], ep.episode_noise(1000, "suite", 5, [1000], 0, 4, 3)[0]))
        self.assertTrue(torch.equal(calls[1][1], ep.episode_noise(1000, "suite", 5, [1001], 1, 4, 3)[0]))
        env.reset(seed=[1002, 1003])  # new batch: step counter restarts
        policy.select_action(obs)
        self.assertTrue(torch.equal(calls[2][0], ep.episode_noise(1000, "suite", 5, [1002], 0, 4, 3)[0]))


class PairedStatsTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(ep.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(ep.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(ep.mcnemar_exact(0, 8), 2 / 256)
        self.assertAlmostEqual(ep.mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)

    def test_paired_stats_and_outcomes(self):
        a = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
        b = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
        s = ep.paired_stats(a, b, n_boot=2000, seed=1)
        self.assertEqual((s["wins"], s["losses"], s["ties"]), (4, 1, 5))
        self.assertAlmostEqual(s["delta_points"], 30.0)
        self.assertAlmostEqual(s["relative_delta_percent"], 75.0)
        lo, hi = s["paired_bootstrap_ci95_points"]
        self.assertLess(lo, 30.0); self.assertGreater(hi, 30.0)
        self.assertAlmostEqual(s["mcnemar_p"], ep.mcnemar_exact(1, 4))
        metrics = {"result": {"per_task": [{"task_id": 1, "episode_outcomes": [True, False]}, {"task_id": 0, "episode_outcomes": [False]}]}}
        keys, values = ep.episode_outcomes(metrics)
        self.assertEqual(keys, [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(values.tolist(), [0, 1, 0])
        rows = ep.per_task_deltas(keys, values, np.array([1, 1, 1]))
        self.assertEqual([(r["task_id"], r["delta"]) for r in rows], [(0, 1), (1, 1)])


if __name__ == "__main__":
    unittest.main()
