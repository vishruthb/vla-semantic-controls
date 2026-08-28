#!/usr/bin/env python3
"""Episode-clustered 2x2 factorial statistics for the matched A/B/C/D evaluation.

Every matched LIBERO episode i (same task, init state, env seed, flow-matching noise) has one
outcome under each configuration: (A_i, B_i, C_i, D_i). All resampling keeps those tuples intact.

Effects (per episode, then averaged):
  VLM-update main  = ((B - A) + (D - C)) / 2
  routing main     = ((C - A) + (D - B)) / 2
  interaction      = (D - C) - (B - A) = (D - B) - (C - A)

Primary interval: task-stratified episode bootstrap (within each task resample its episodes with
replacement; 20,000 replicates; percentile 95% CI). Sensitivity: task-cluster bootstrap (resample
the 10 tasks with replacement) — only 10 clusters, so a robustness check, not the primary interval.
Secondary p-values: paired sign-flip permutation test on the per-episode effect values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eval_protocol as ep  # noqa: E402

PRESETS = "ABCD"


def load_tuples(results_dir: Path, step: int, episodes: int, suffix: str) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """Return outcomes[task, episode, preset] (0/1), the task ids, and the episode keys."""
    keys_ref, columns = None, []
    for preset in PRESETS:
        metrics = json.loads((results_dir / f"{preset}_{step:05d}_e{episodes}{suffix}.json").read_text())
        if not metrics.get("semantic_control", {}).get("routing", {}).get("verified"):
            raise RuntimeError(f"{preset}: routing not verified; result invalid")
        keys, values = ep.episode_outcomes(metrics)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise RuntimeError(f"{preset}: episode keys differ from A; results are not matched")
        columns.append(values)
    flat = np.stack(columns, axis=1)  # (n_episodes, 4)
    task_ids = sorted({k[0] for k in keys_ref})
    per_task = [flat[[i for i, k in enumerate(keys_ref) if k[0] == t]] for t in task_ids]
    n_per_task = {len(x) for x in per_task}
    if len(n_per_task) != 1:
        raise RuntimeError(f"unequal episodes per task: {n_per_task}")
    return np.stack(per_task, axis=0), task_ids, keys_ref  # (tasks, episodes, 4)


def effects(t: np.ndarray) -> dict[str, float]:
    """Point estimates in percentage points from an array (..., 4) of A,B,C,D outcomes."""
    a, b, c, d = (t[..., i].astype(float) for i in range(4))
    return {
        "B_minus_A": 100 * (b - a).mean(),
        "D_minus_C": 100 * (d - c).mean(),
        "C_minus_A": 100 * (c - a).mean(),
        "D_minus_B": 100 * (d - b).mean(),
        "vlm_update_main": 100 * (((b - a) + (d - c)) / 2).mean(),
        "routing_main": 100 * (((c - a) + (d - b)) / 2).mean(),
        "interaction": 100 * ((d - c) - (b - a)).mean(),
        "success_A": 100 * a.mean(), "success_B": 100 * b.mean(), "success_C": 100 * c.mean(), "success_D": 100 * d.mean(),
    }


def bootstrap(t: np.ndarray, reps: int, seed: int, level: str) -> dict[str, np.ndarray]:
    """level='episode': task-stratified episode bootstrap; level='task': task-cluster bootstrap."""
    rng = np.random.default_rng(seed)
    n_tasks, n_eps, _ = t.shape
    out: dict[str, list[float]] = {}
    for _ in range(reps):
        if level == "episode":
            idx = rng.integers(0, n_eps, size=(n_tasks, n_eps))
            sample = np.take_along_axis(t, idx[:, :, None], axis=1)
        else:
            tasks = rng.integers(0, n_tasks, size=n_tasks)
            sample = t[tasks]
        for key, value in effects(sample).items():
            out.setdefault(key, []).append(value)
    return {k: np.asarray(v) for k, v in out.items()}


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def sign_flip_p(per_episode: np.ndarray, reps: int, seed: int) -> float:
    """Two-sided paired permutation test: under H0 the sign of each episode's effect is exchangeable."""
    rng = np.random.default_rng(seed)
    observed = abs(per_episode.mean())
    flips = rng.choice([-1.0, 1.0], size=(reps, per_episode.size))
    null = np.abs((flips * per_episode[None, :]).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (reps + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/pilot")
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--suffix", default="_matched")
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="factorial_20k")
    args = parser.parse_args()

    t, task_ids, keys = load_tuples(args.results_dir, args.step, args.episodes, args.suffix)
    n_tasks, n_eps, _ = t.shape
    point = effects(t)
    ep_boot = bootstrap(t, args.reps, args.seed, "episode")
    task_boot = bootstrap(t, args.reps, args.seed + 1, "task")
    flat = t.reshape(-1, 4).astype(float)
    a, b, c, d = flat.T
    per_episode = {
        "vlm_update_main": ((b - a) + (d - c)) / 2,
        "routing_main": ((c - a) + (d - b)) / 2,
        "interaction": (d - c) - (b - a),
        "B_minus_A": b - a, "D_minus_C": d - c, "C_minus_A": c - a, "D_minus_B": d - b,
    }
    paired = {}
    for name, (x, y) in {"B_minus_A": (a, b), "D_minus_C": (c, d), "C_minus_A": (a, c), "D_minus_B": (b, d)}.items():
        s = ep.paired_stats(x.astype(int), y.astype(int), n_boot=1000, seed=args.seed)  # wins/losses/ties + McNemar
        paired[name] = {"delta_points": point[name], "wins": s["wins"], "losses": s["losses"], "ties": s["ties"],
                        "mcnemar_p": s["mcnemar_p"], "episode_stratified_ci95": percentile_ci(ep_boot[name]),
                        "task_cluster_ci95": percentile_ci(task_boot[name])}
    # correlation between the two contrasts that were previously pooled as if independent
    corr_vlm = float(np.corrcoef(b - a, d - c)[0, 1])
    corr_routing = float(np.corrcoef(c - a, d - b)[0, 1])
    per_task = []
    for i, task in enumerate(task_ids):
        e = effects(t[i])
        per_task.append({"task_id": task, **{k: round(v, 1) for k, v in e.items()}})
    report = {
        "design": {"episodes_per_task": n_eps, "tasks": n_tasks, "matched_episodes": n_tasks * n_eps,
                   "bootstrap_replicates": args.reps, "seed": args.seed,
                   "primary_interval": "task-stratified episode bootstrap of complete A/B/C/D tuples",
                   "sensitivity_interval": "task-cluster bootstrap (10 clusters; robustness check only)"},
        "success_percent": {p: point[f"success_{p}"] for p in PRESETS},
        "factorial_effects": {
            name: {"points": point[name],
                   "episode_stratified_ci95": percentile_ci(ep_boot[name]),
                   "task_cluster_ci95": percentile_ci(task_boot[name]),
                   "sign_flip_permutation_p": sign_flip_p(per_episode[name], args.reps, args.seed + 2),
                   "bootstrap_fraction_below_zero": float((ep_boot[name] < 0).mean())}
            for name in ("vlm_update_main", "routing_main", "interaction")},
        "paired_contrasts": paired,
        "contrast_correlations": {"corr(B-A, D-C)": corr_vlm, "corr(C-A, D-B)": corr_routing},
        "per_task_effects": per_task,
    }
    out_json = args.results_dir / f"{args.out}.json"
    out_json.write_text(json.dumps(report, indent=1))

    def fmt(ci):
        return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
    lines = [f"# 2x2 factorial statistics, matched episodes @ step {args.step}", "",
             f"{n_tasks} tasks x {n_eps} matched episodes = {n_tasks * n_eps} A/B/C/D outcome tuples; "
             f"{args.reps} bootstrap replicates; primary CI = task-stratified episode bootstrap of complete tuples; "
             "task-cluster CI = 10-cluster robustness check only.", "",
             "| effect | points | episode-stratified 95% CI | task-cluster 95% CI (10 clusters) | sign-flip p |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in (("vlm_update_main", "VLM-update main effect ((B−A)+(D−C))/2"), ("routing_main", "routing main effect ((C−A)+(D−B))/2"), ("interaction", "interaction (D−C)−(B−A)")):
        f = report["factorial_effects"][name]
        lines.append(f"| {label} | {f['points']:+.2f} | {fmt(f['episode_stratified_ci95'])} | {fmt(f['task_cluster_ci95'])} | {f['sign_flip_permutation_p']:.3f} |")
    lines += ["", "| paired contrast | points | wins / losses / ties | McNemar p | episode-stratified 95% CI | task-cluster 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in (("B_minus_A", "B − A (VLM update, all-layer)"), ("D_minus_C", "D − C (VLM update, cross-only)"), ("C_minus_A", "C − A (routing, frozen)"), ("D_minus_B", "D − B (routing, trainable)")):
        p = paired[name]
        lines.append(f"| {label} | {p['delta_points']:+.1f} | {p['wins']} / {p['losses']} / {p['ties']} | {p['mcnemar_p']:.3f} | {fmt(p['episode_stratified_ci95'])} | {fmt(p['task_cluster_ci95'])} |")
    lines += ["", f"Within-episode correlation of the two VLM-update contrasts corr(B−A, D−C) = {corr_vlm:+.3f}; of the two routing contrasts corr(C−A, D−B) = {corr_routing:+.3f}.", "",
              "Per-task effects (points, 20 episodes each):", "", "| task | A | B | C | D | VLM main | routing main | interaction |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_task:
        lines.append(f"| {row['task_id']} | {row['success_A']:.0f} | {row['success_B']:.0f} | {row['success_C']:.0f} | {row['success_D']:.0f} | {row['vlm_update_main']:+.1f} | {row['routing_main']:+.1f} | {row['interaction']:+.1f} |")
    (args.results_dir / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
