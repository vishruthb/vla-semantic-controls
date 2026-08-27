#!/usr/bin/env python3
"""Matched evaluation protocol and paired statistics for the A-D pilot.

Determinism model (per task, per episode index e = env seed - start seed):
  * LIBERO init state: the harness' LiberoEnv binds sub-env i of batch k to init state i + k*n_envs,
    i.e. to e, independent of batch size (envs/libero.py: init_state_id/_reset_stride).
  * Environment seed: start_seed + e (lerobot_eval.eval_policy), independent of batch size.
  * Flow-matching noise: normally drawn from the global CUDA RNG, so it depends on batch composition
    and call order. `install_deterministic_noise` passes the policy's public `select_action(noise=...)`
    argument a tensor whose slot i is drawn from a CPU generator seeded by
    sha256(base | suite | task | episode_seed | step) — identical for the same episode across
    checkpoints, batch sizes and orders, and still N(0, 1). No policy code is changed.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch


def noise_seed(base: int, suite: str, task_id: int, episode_seed: int, step: int) -> int:
    digest = hashlib.sha256(f"{base}|{suite}|{task_id}|{episode_seed}|{step}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def episode_noise(
    base: int, suite: str, task_id: int, episode_seeds: list[int], step: int, chunk_size: int, action_dim: int
) -> torch.Tensor:
    noise = torch.empty(len(episode_seeds), chunk_size, action_dim, dtype=torch.float32)
    for slot, episode_seed in enumerate(episode_seeds):
        generator = torch.Generator().manual_seed(noise_seed(base, suite, task_id, episode_seed, step))
        noise[slot] = torch.randn(chunk_size, action_dim, generator=generator)
    return noise


def install_deterministic_noise(policy, envs: dict[str, dict[int, Any]], base_seed: int) -> dict[str, Any]:
    """Wrap each task's vector-env `reset` (to learn task and episode seeds) and the policy's
    `select_action` (to supply the matched noise). Returns the protocol description for provenance."""
    state: dict[str, Any] = {"suite": None, "task": None, "seeds": None, "step": 0}
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.max_action_dim

    for suite, group in envs.items():
        for task_id, venv in group.items():
            original_reset = venv.reset

            def reset(*, seed=None, options=None, _orig=original_reset, _suite=suite, _task=task_id):
                seeds = list(seed) if seed is not None else None
                state.update(suite=_suite, task=_task, seeds=seeds, step=0)
                return _orig(seed=seed, options=options)

            venv.reset = reset

    original_select_action = policy.select_action

    def select_action(observation, noise=None, **kwargs):
        if noise is None and state["seeds"] is not None:
            tensors = [v for v in observation.values() if hasattr(v, "shape")] if isinstance(observation, dict) else []
            batch_size = int(tensors[0].shape[0]) if tensors else len(state["seeds"])
            seeds = state["seeds"][:batch_size]
            if len(seeds) < batch_size:  # more envs than seeds (unexpected): extend deterministically
                seeds = seeds + [seeds[-1] + 1 + i for i in range(batch_size - len(seeds))]
            device = next(policy.parameters()).device
            noise = episode_noise(base_seed, state["suite"], state["task"], seeds, state["step"], chunk_size, action_dim).to(device)
        state["step"] += 1
        return original_select_action(observation, noise=noise, **kwargs)

    policy.select_action = select_action
    return {
        "enabled": True,
        "base_seed": base_seed,
        "scheme": "slot i of select_action noise = randn(chunk, action_dim) from torch CPU generator seeded by "
        "sha256(base|suite|task|episode_seed|step); episode_seed = env reset seed = start_seed + episode index",
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "state": state,
    }


# --------------------------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------------------------


def episode_outcomes(metrics: dict[str, Any]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Flatten per-task episode outcomes in (task, episode) order; returns keys and a 0/1 vector."""
    keys, values = [], []
    for task in sorted(metrics["result"]["per_task"], key=lambda t: t["task_id"]):
        for index, outcome in enumerate(task["episode_outcomes"]):
            keys.append((task["task_id"], index))
            values.append(1 if outcome else 0)
    return keys, np.asarray(values, dtype=np.int64)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts (b: A wins, c: other wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_stats(a: np.ndarray, other: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    assert a.shape == other.shape
    diff = other - a
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        boot[i] = diff[sample].mean()
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    return {
        "n": int(n),
        "success_a": float(a.mean()),
        "success_other": float(other.mean()),
        "delta_points": float(100 * diff.mean()),
        "relative_delta_percent": float(100 * (other.mean() - a.mean()) / a.mean()) if a.mean() else None,
        "paired_bootstrap_ci95_points": [float(100 * np.percentile(boot, 2.5)), float(100 * np.percentile(boot, 97.5))],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mcnemar_p": mcnemar_exact(losses, wins),
        "sign_test_p": mcnemar_exact(losses, wins),
    }


def per_task_deltas(keys: list[tuple[int, int]], a: np.ndarray, other: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for task_id in sorted({k[0] for k in keys}):
        mask = np.asarray([k[0] == task_id for k in keys])
        rows.append({"task_id": task_id, "n": int(mask.sum()), "a": int(a[mask].sum()), "other": int(other[mask].sum()),
                     "delta": int(other[mask].sum() - a[mask].sum()),
                     "wins": int(((other - a)[mask] > 0).sum()), "losses": int(((other - a)[mask] < 0).sum())})
    return rows
