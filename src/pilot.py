#!/usr/bin/env python3
"""Staged A/B/C/D pilot driver: train (fresh or resume) to a step, evaluate checkpoints, summarize.

    python src/pilot.py train --presets A B C D --stop-at 5000
    python src/pilot.py eval  --presets A B C D --step 5000 --episodes 2 --batch 2 [--parallel]
    python src/pilot.py summarize --step 5000 --episodes 2

Training uses LeRobot's official loop through ``train_semantic.py`` with a fixed 30k-step LR horizon
and stops after saving ``--stop-at``; resuming continues sample-exactly from ``checkpoints/last``.
Evaluation uses the pinned harness (``evaluate.py --checkpoint``), which restores and verifies the
semantic routing from the checkpoint before running any episode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import semantic_control as sc  # noqa: E402

TRAIN_ROOT = ROOT / "outputs/train"
LOG_ROOT = ROOT / "outputs/logs"
RESULTS_ROOT = ROOT / "results/pilot"
HORIZON_STEPS = 30000  # LR schedule horizon (== scheduler decay steps, so no auto-scaling)
SAVE_AT = "2000,5000,10000,15000,20000,25000,30000"
BASELINE = json.loads((ROOT / "configs/baseline.json").read_text())


def python() -> list[str]:
    return ["uv", "run", "--frozen", "python"]


def backbone_path() -> Path:
    checkpoint = sc.load_checkpoint_config()["checkpoint"]
    return sc.resolve_snapshot(checkpoint["backbone_repository"], checkpoint["backbone_revision"])


def run_dir(preset: str) -> Path:
    return TRAIN_ROOT / f"spatial_{preset}"


def checkpoint_dir(preset: str, step: int) -> Path:
    """LeRobot zero-pads checkpoint directory names; locate by numeric value."""
    checkpoints = run_dir(preset) / "checkpoints"
    for directory in sorted(checkpoints.glob("[0-9]*")) if checkpoints.exists() else []:
        if directory.name.isdigit() and int(directory.name) == step:
            return directory / "pretrained_model"
    return checkpoints / f"{step:06d}" / "pretrained_model"


def train_command(preset: str, stop_at: int) -> list[str]:
    out = run_dir(preset)
    last = out / "checkpoints" / "last"
    semantic = [
        f"--semantic.preset={preset}",
        "--semantic.trainable_fp32=true",
        "--semantic.vlm_lr=1e-5",
        f"--semantic.stop_at={stop_at}",
        f"--semantic.save_at={SAVE_AT}",
    ]
    if last.exists():
        return [*python(), "src/train_semantic.py", *semantic, "--resume=true",
                f"--config_path={last / 'pretrained_model' / 'train_config.json'}"]
    return [
        *python(), "src/train_semantic.py", *semantic, "--semantic.episodes=1261-1692",
        "--policy.type=smolvla", "--policy.load_vlm_weights=true", f"--policy.vlm_model_name={backbone_path()}",
        "--policy.num_vlm_layers=0", "--policy.expert_width_multiplier=0.5", "--policy.num_expert_layers=-1",
        "--policy.device=cuda", "--policy.push_to_hub=false",
        "--dataset.repo_id=HuggingFaceVLA/libero", "--dataset.revision=86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "--seed=1000", "--batch_size=32", f"--steps={HORIZON_STEPS}", "--save_freq=1000", "--eval_freq=0",
        "--log_freq=100", "--num_workers=32", f"--output_dir={out}", f"--job_name=spatial_{preset}", "--wandb.enable=false",
    ]


def train_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(ACCELERATE_MIXED_PRECISION="bf16", TOKENIZERS_PARALLELISM="false", MUJOCO_GL="osmesa")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")  # everything is cached; never let the loader re-fetch
    return env


def prune_training_state(preset: str) -> list[str]:
    """Drop optimizer/RNG state of every checkpoint except the one `last` points to."""
    checkpoints = run_dir(preset) / "checkpoints"
    last = (checkpoints / "last").resolve() if (checkpoints / "last").exists() else None
    pruned = []
    for directory in sorted(checkpoints.glob("[0-9]*")):
        state = directory / "training_state"
        if directory.resolve() != last and state.exists():
            shutil.rmtree(state)
            pruned.append(str(directory.name))
    return pruned


def train(preset: str, stop_at: int) -> dict:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log = LOG_ROOT / f"spatial_{preset}_to_{stop_at}.log"
    command = train_command(preset, stop_at)
    started = time.perf_counter()
    with log.open("w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=train_env(), stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    target = checkpoint_dir(preset, stop_at)
    ok = result.returncode == 0 and (target / sc.SEMANTIC_CONTROL_FILE).exists()
    pruned = prune_training_state(preset) if ok else []
    summary = {"preset": preset, "stop_at": stop_at, "returncode": result.returncode, "ok": ok,
               "wall_s": elapsed, "log": str(log), "checkpoint": str(target), "pruned_training_state": pruned}
    print(json.dumps(summary), flush=True)
    return summary


def result_stem(preset: str, step: int, episodes: int, deterministic: bool) -> str:
    return f"{preset}_{step:05d}_e{episodes}" + ("_matched" if deterministic else "")


def eval_command(preset: str, step: int, episodes: int, batch: int, deterministic: bool = False) -> tuple[list[str], Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = result_stem(preset, step, episodes, deterministic)
    output = RESULTS_ROOT / f"{stem}.json"
    command = [*python(), "src/evaluate.py", "--checkpoint", str(checkpoint_dir(preset, step)),
               "--semantic-control", preset, "--task-ids", *map(str, range(10)),
               "--episodes-per-task", str(episodes), "--batch-size", str(batch), "--render-episodes-per-task", "0",
               *(["--deterministic-noise"] if deterministic else []),
               "--output", str(output), "--report", str(RESULTS_ROOT / f"{stem}.md")]
    return command, output


def evaluate(presets: list[str], step: int, episodes: int, batch: int, parallel: bool, max_parallel: int = 2, deterministic: bool = False) -> list[dict]:
    """Each evaluation holds 10 tasks x batch MuJoCo subprocesses alive; cap concurrency to avoid OOM."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", LIBERO_CONFIG_PATH=str(ROOT / ".cache/libero"))
    procs, outputs = [], []
    for preset in presets:
        command, output = eval_command(preset, step, episodes, batch, deterministic)
        log = (LOG_ROOT / f"eval_{result_stem(preset, step, episodes, deterministic)}.log").open("w")
        log.write(" ".join(command) + "\n"); log.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((preset, proc, log, output))
        if not parallel:
            proc.wait()
        else:
            while sum(1 for _, p_, _, _ in procs if p_.poll() is None) >= max_parallel:
                time.sleep(15)
    results = []
    for preset, proc, log, output in procs:
        proc.wait(); log.close()
        entry = {"preset": preset, "step": step, "episodes": episodes, "returncode": proc.returncode, "output": str(output)}
        if output.exists():
            metrics = json.loads(output.read_text())
            entry.update(successes=metrics["result"]["successes"], total=metrics["result"]["episodes"],
                         success_percent=metrics["result"]["overall_success_percent"],
                         per_task=[t["successes"] for t in metrics["result"]["per_task"]],
                         routing_verified=bool(metrics.get("semantic_control", {}).get("routing", {}).get("verified")),
                         eval_s=metrics["timing"]["evaluation_seconds"])
        results.append(entry)
        print(json.dumps(entry), flush=True)
    return results


def load_curve(preset: str) -> list[dict]:
    path = run_dir(preset) / "semantic_train_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(step: int, episodes: int | None, presets: list[str]) -> dict:
    rows = {}
    for preset in presets:
        control_file = checkpoint_dir(preset, step) / sc.SEMANTIC_CONTROL_FILE
        entry = {"checkpoint_present": control_file.exists()}
        if control_file.exists():
            data = json.loads(control_file.read_text())
            training = data.get("training", {})
            entry.update(preset=data.get("preset"), init_fingerprint=data.get("init_fingerprint"),
                         routing_verified=data.get("routing", {}).get("verified"),
                         coupled_layers=len(data.get("routing", {}).get("coupled_layers", [])),
                         trainable_parameters=data.get("trainable_parameters"), lr=training.get("lr"),
                         loss_last=training.get("loss_last"), loss_mean_last_100=training.get("loss_mean_last_100"),
                         grad_norm_total_preclip=training.get("grad_norm_total_preclip_last"),
                         grad_norm_groups=training.get("grad_norm_groups_postclip_last"),
                         runtime=training.get("runtime"), peak_vram=training.get("peak_vram"), step=data.get("step"),
                         mixed_precision=data.get("mixed_precision"), resumed_from_step=data.get("resumed_from_step"))
        curve = load_curve(preset)
        if curve:
            def window(lo, hi):
                vals = [c["loss"] for c in curve if lo < c["step"] <= hi and c["loss"] is not None]
                return sum(vals) / len(vals) if vals else None
            entry["loss_curve_mean_by_1k"] = {f"{k}k": window((k - 1) * 1000, k * 1000) for k in range(1, step // 1000 + 1)}
            entry["max_step_logged"] = curve[-1]["step"]
            entry["nonfinite_losses"] = sum(1 for c in curve if c["loss"] is None or c["loss"] != c["loss"])
        if episodes is not None:
            result_file = RESULTS_ROOT / f"{preset}_{step:05d}_e{episodes}.json"
            if result_file.exists():
                metrics = json.loads(result_file.read_text())
                entry["eval"] = {"successes": metrics["result"]["successes"], "episodes": metrics["result"]["episodes"],
                                 "success_percent": metrics["result"]["overall_success_percent"],
                                 "per_task": [t["successes"] for t in metrics["result"]["per_task"]],
                                 "routing_verified": bool(metrics.get("semantic_control", {}).get("routing", {}).get("verified"))}
        rows[preset] = entry
    if "A" in rows and rows["A"].get("eval"):
        base = rows["A"]["eval"]["success_percent"]
        for preset, entry in rows.items():
            if entry.get("eval"):
                entry["eval"]["delta_vs_A_points"] = entry["eval"]["success_percent"] - base
    fingerprints = {e.get("init_fingerprint") for e in rows.values() if e.get("init_fingerprint")}
    summary = {"step": step, "episodes": episodes, "identical_init": len(fingerprints) == 1, "rows": rows}
    print(json.dumps(summary, indent=1, default=str))
    return summary


def compare(step: int, episodes: int, presets: list[str], deterministic: bool, baseline: str = "A") -> dict:
    """Paired comparison of every preset against the baseline preset on matched episodes."""
    import eval_protocol as ep

    loaded = {}
    for preset in presets:
        path = RESULTS_ROOT / f"{result_stem(preset, step, episodes, deterministic)}.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            if not metrics.get("semantic_control", {}).get("routing", {}).get("verified"):
                raise RuntimeError(f"{path}: routing not verified; result is invalid")
            loaded[preset] = metrics
    if baseline not in loaded:
        raise RuntimeError(f"baseline {baseline} result missing")
    keys_a, a = ep.episode_outcomes(loaded[baseline])
    report = {"step": step, "episodes_per_task": episodes, "baseline": baseline, "n_episodes": int(len(a)),
              "success": {p: float(100 * ep.episode_outcomes(m)[1].mean()) for p, m in loaded.items()}, "paired": {}, "per_task": {}}
    for preset, metrics in loaded.items():
        if preset == baseline:
            continue
        keys, other = ep.episode_outcomes(metrics)
        if keys != keys_a:
            raise RuntimeError(f"{preset}: episode keys differ from {baseline}; not matched")
        report["paired"][preset] = ep.paired_stats(a, other)
        report["per_task"][preset] = ep.per_task_deltas(keys, a, other)
    out = RESULTS_ROOT / f"paired_{step:05d}_e{episodes}{'_matched' if deterministic else ''}.json"
    out.write_text(json.dumps(report, indent=1))
    lines = [f"# Paired comparison vs {baseline} @ step {step} ({episodes} episodes/task, {'matched' if deterministic else 'unmatched'} protocol)", "",
             "| preset | success | Δ vs A (pts) | rel Δ | paired bootstrap 95% CI | wins / losses / ties | McNemar p |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for preset in presets:
        if preset not in loaded:
            continue
        s = report["success"][preset]
        if preset == baseline:
            lines.append(f"| {preset} | {s:.1f}% | — | — | — | — | — |"); continue
        p = report["paired"][preset]; ci = p["paired_bootstrap_ci95_points"]
        rel = f"{p['relative_delta_percent']:+.1f}%" if p["relative_delta_percent"] is not None else "n/a"
        lines.append(f"| {preset} | {s:.1f}% | {p['delta_points']:+.1f} | {rel} | [{ci[0]:+.1f}, {ci[1]:+.1f}] | {p['wins']} / {p['losses']} / {p['ties']} | {p['mcnemar_p']:.3f} |")
    lines += ["", "Per-task successes (delta vs A):", ""]
    header = "| task | A | " + " | ".join(f"{p} (Δ)" for p in presets if p in loaded and p != baseline) + " |"
    lines += [header, "| --- | ---: | " + " | ".join("---:" for p in presets if p in loaded and p != baseline) + " |"]
    a_tasks = ep.per_task_deltas(keys_a, a, a)
    for i, row in enumerate(a_tasks):
        cells = [f"{report['per_task'][p][i]['other']} ({report['per_task'][p][i]['delta']:+d})" for p in presets if p in loaded and p != baseline]
        lines.append(f"| {row['task_id']} | {row['a']} | " + " | ".join(cells) + " |")
    (out.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("--presets", nargs="+", default=list("ABCD")); t.add_argument("--stop-at", type=int, required=True)
    e = sub.add_parser("eval"); e.add_argument("--presets", nargs="+", default=list("ABCD")); e.add_argument("--step", type=int, required=True)
    e.add_argument("--episodes", type=int, required=True); e.add_argument("--batch", type=int, required=True); e.add_argument("--parallel", action="store_true"); e.add_argument("--max-parallel", type=int, default=2); e.add_argument("--deterministic", action="store_true")
    c = sub.add_parser("compare"); c.add_argument("--presets", nargs="+", default=list("ABCD")); c.add_argument("--step", type=int, required=True); c.add_argument("--episodes", type=int, required=True); c.add_argument("--deterministic", action="store_true")
    s = sub.add_parser("summarize"); s.add_argument("--presets", nargs="+", default=list("ABCD")); s.add_argument("--step", type=int, required=True); s.add_argument("--episodes", type=int)
    args = parser.parse_args()
    if args.command == "train":
        results = [train(p, args.stop_at) for p in args.presets]
        return 0 if all(r["ok"] for r in results) else 1
    if args.command == "eval":
        results = evaluate(args.presets, args.step, args.episodes, args.batch, args.parallel, args.max_parallel, args.deterministic)
        return 0 if all(r["returncode"] == 0 for r in results) else 1
    if args.command == "compare":
        compare(args.step, args.episodes, args.presets, args.deterministic)
        return 0
    summarize(args.step, args.episodes, args.presets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
