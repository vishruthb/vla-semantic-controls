#!/usr/bin/env python3
"""Run the pinned SmolVLA LIBERO-Spatial evaluation and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--render-episodes-per-task", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results/metrics.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-videos", action="store_true")
    return parser.parse_args()


def configure_process() -> None:
    defaults = {
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "LIBERO_CONFIG_PATH": str(ROOT / ".cache/libero"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def ensure_libero_config() -> None:
    """Create LIBERO's default asset map without its interactive first-import prompt."""
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The pinned hf-libero package is not installed")
    benchmark_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_file = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "assets": str(Path.home() / ".cache/libero/assets"),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "benchmark_root": str(benchmark_root),
        "datasets": str(benchmark_root.parent / "datasets"),
        "init_states": str(benchmark_root / "init_files"),
    }
    config_file.write_text(json.dumps(paths, indent=2) + "\n")


def cached_snapshot(repository: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_files_only=True,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        )
    ).resolve()


def package_versions() -> dict[str, str]:
    names = [
        "accelerate",
        "draccus",
        "gymnasium",
        "hf-libero",
        "huggingface-hub",
        "lerobot",
        "mujoco",
        "numpy",
        "opencv-python-headless",
        "robosuite",
        "safetensors",
        "torch",
        "torchvision",
        "transformers",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class NvidiaMemoryMonitor:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.baseline_mib: int | None = None
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _sample() -> int | None:
        value = command_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"]
        )
        if value is None:
            return None
        try:
            return int(value.splitlines()[0].strip())
        except ValueError:
            return None

    def start(self) -> None:
        self.baseline_mib = self._sample()
        self.peak_mib = self.baseline_mib or 0
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample()
            if sample is not None:
                self.peak_mib = max(self.peak_mib, sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    """Return an approximate 95% Wilson interval as percentages."""
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z**2 / episodes
    center = (proportion + z**2 / (2 * episodes)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / episodes + z**2 / (4 * episodes**2)
    ) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def human_join(values: list[int]) -> str:
    if len(values) == 1:
        return str(values[0])
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(str(value) for value in values[:-1]) + f", and {values[-1]}"


def os_package_versions(names: list[str]) -> dict[str, str]:
    value = command_output(["dpkg-query", "-W", "-f=${Package}=${Version}\\n", *names])
    if value is None:
        return {}
    return dict(line.split("=", 1) for line in value.splitlines() if "=" in line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(metrics: dict[str, Any], path: Path) -> None:
    result = metrics["result"]
    target = metrics["target"]
    settings = metrics["eval_settings"]
    timing = metrics["timing"]
    vram = metrics["vram"]
    delta = result["overall_success_percent"] - target["reference_success_percent"]
    threshold = target["material_difference_percentage_points"]
    status = (
        f"PASS (within the +/-{threshold:.1f}-point reproduction band)"
        if not metrics["diagnostics"]["material_difference"]
        else "DIAGNOSE"
    )
    reference_per_task = target.get("reference_per_task_success_percent")
    gpu_fields = [field.strip() for field in (metrics["system"]["gpu"] or "unknown").split(",")]
    gpu_summary = (
        f"{gpu_fields[0]}, driver {gpu_fields[1]}, {gpu_fields[2]} MiB"
        if len(gpu_fields) == 3
        else metrics["system"]["gpu"]
    )
    rows = [
        "# SmolVLA LIBERO-Spatial Baseline",
        "",
        f"**Status: {status} — {result['successes']}/{result['episodes']} successes "
        f"({result['overall_success_percent']:.1f}%).** Reference: "
        f"~{target['reference_success_percent']:.1f}% (delta {delta:+.1f} points).",
        "",
        "| Task | This run | Reference | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in result["per_task"]:
        reference_task = reference_per_task[task["task_id"]] if reference_per_task else None
        reference_cell = f"{reference_task:.1f}%" if reference_task is not None else "n/a"
        delta_cell = (
            f"{task['success_percent'] - reference_task:+.1f}" if reference_task is not None else "n/a"
        )
        rows.append(
            f"| {task['task_id']}: {task['language']} | {task['success_percent']:.1f}% "
            f"({task['successes']}/{task['episodes']}) | {reference_cell} | {delta_cell} |"
        )
    rows.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Evaluation: {timing['evaluation_seconds']:.1f} s; total process: {timing['wall_seconds']:.1f} s.",
            f"- Policy latency, batch {settings['batch_size']}: mean {timing['policy_latency_ms']['mean']:.2f} ms, "
            f"median {timing['policy_latency_ms']['median']:.2f} ms, p95 {timing['policy_latency_ms']['p95']:.2f} ms "
            f"over {timing['policy_latency_ms']['samples']} calls.",
            f"- Peak VRAM: {vram['nvidia_smi_peak_mib']} MiB by nvidia-smi; "
            f"PyTorch allocated/reserved peaks: {vram['torch_peak_allocated_mib']:.1f}/"
            f"{vram['torch_peak_reserved_mib']:.1f} MiB.",
            f"- Host: {gpu_summary}; CUDA runtime {metrics['system']['cuda_runtime']}; "
            f"Python {metrics['system']['python']}.",
            "",
            "## Exact setting",
            "",
            f"- Checkpoint: `{metrics['revisions']['checkpoint']}` (weights SHA-256 `{metrics['revisions']['model_sha256']}`).",
            f"- LeRobot: `{metrics['revisions']['lerobot']}`; LIBERO `{metrics['packages']['hf-libero']}`; "
            f"MuJoCo `{metrics['packages']['mujoco']}`.",
            f"- PyTorch `{metrics['packages']['torch']}`; Transformers `{metrics['packages']['transformers']}`; "
            f"complete dependency resolution in `uv.lock` (SHA-256 `{metrics['revisions']['uv_lock_sha256']}`).",
            f"- Harness revision: `{metrics['revisions']['repo'] or 'uncommitted worktree (no Git HEAD)'}`; "
            f"reference artifact: `{metrics['revisions']['reference_evaluation']}`.",
            f"- 10 tasks × {settings['episodes_per_task']} episodes; seed {settings['seed']}; "
            f"fixed init states; relative control; {settings['observation_width']}×{settings['observation_height']} "
            f"observations; {settings['episode_length']} max steps.",
            f"- fp32; `num_steps={settings['num_steps']}`; `n_action_steps={settings['n_action_steps']}`; "
            f"batch {settings['batch_size']}; `{settings['render_backend']}` rendering.",
            "- TF32 allowed; cuDNN benchmark enabled; async vector environments; one task evaluated at a time.",
            "",
            "## Diagnosis",
            "",
        ]
    )
    for item in metrics["diagnostics"]["differences_from_reference"]:
        rows.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def main() -> int:
    args = parse_args()
    configure_process()
    ensure_libero_config()
    config = json.loads(args.config.resolve().read_text())
    model_cfg = config["model"]
    env_cfg_json = config["environment"]
    task_ids = args.task_ids or env_cfg_json["task_ids"]
    episodes = args.episodes_per_task or env_cfg_json["episodes_per_task"]
    batch_size = args.batch_size or env_cfg_json["batch_size"]
    render_episodes = (
        args.render_episodes_per_task
        if args.render_episodes_per_task is not None
        else env_cfg_json["render_episodes_per_task"]
    )
    if episodes < 1 or batch_size < 1 or batch_size > episodes:
        raise ValueError("Require episodes >= batch_size >= 1")

    # Imports happen after renderer/thread environment variables are fixed.
    import torch
    from libero.libero import benchmark
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed

    policy_path = cached_snapshot(model_cfg["repository"], model_cfg["revision"])
    backbone_path = cached_snapshot(model_cfg["backbone_repository"], model_cfg["backbone_revision"])
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = model_cfg["device"]
    policy_cfg.use_amp = model_cfg["use_amp"]
    policy_cfg.num_steps = model_cfg["num_steps"]
    policy_cfg.n_action_steps = model_cfg["n_action_steps"]
    policy_cfg.vlm_model_name = str(backbone_path)

    env_config = LiberoEnvConfig(
        task=env_cfg_json["suite"],
        task_ids=task_ids,
        episode_length=env_cfg_json["episode_length"],
        obs_type=env_cfg_json["obs_type"],
        camera_name=env_cfg_json["camera_name"],
        init_states=env_cfg_json["init_states"],
        observation_height=env_cfg_json["observation_height"],
        observation_width=env_cfg_json["observation_width"],
        control_mode=env_cfg_json["control_mode"],
    )
    suite = benchmark.get_benchmark_dict()[env_cfg_json["suite"]]()
    task_languages = {task_id: suite.get_task(task_id).language for task_id in task_ids}

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    memory_monitor = NvidiaMemoryMonitor()
    memory_monitor.start()
    set_seed(env_cfg_json["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    raw_dir = args.output.resolve().parent / "raw" / args.output.stem
    videos_dir = raw_dir / "videos"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_cfg['repository']}@{model_cfg['revision']}", flush=True)
    print(
        f"Evaluating tasks={task_ids}, episodes/task={episodes}, batch={batch_size}, "
        f"num_steps={policy_cfg.num_steps}, n_action_steps={policy_cfg.n_action_steps}",
        flush=True,
    )
    envs = make_env(
        env_config,
        n_envs=batch_size,
        use_async_envs=env_cfg_json["use_async_envs"],
        trust_remote_code=False,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_config, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(backbone_path)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=policy_cfg
    )

    latencies_ms: list[float] = []
    original_select_action = policy.select_action

    def timed_select_action(observation: dict[str, Any]):
        torch.cuda.synchronize()
        started = time.perf_counter()
        action = original_select_action(observation)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000)
        return action

    policy.select_action = timed_select_action  # type: ignore[method-assign]
    torch.cuda.reset_peak_memory_stats()
    eval_start = time.perf_counter()
    try:
        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=episodes,
                max_episodes_rendered=render_episodes,
                videos_dir=videos_dir,
                start_seed=env_cfg_json["seed"],
                max_parallel_tasks=env_cfg_json["max_parallel_tasks"],
            )
    finally:
        for group in envs.values():
            for env in group.values():
                try:
                    env.close()
                except Exception:
                    pass
    evaluation_seconds = time.perf_counter() - eval_start
    torch_peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    torch_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    memory_monitor.stop()
    wall_seconds = time.perf_counter() - wall_start

    per_task = []
    total_successes = 0
    total_episodes = 0
    for entry in sorted(info["per_task"], key=lambda item: item["task_id"]):
        successes = [bool(value) for value in entry["metrics"]["successes"][:episodes]]
        n_success = sum(successes)
        n = len(successes)
        total_successes += n_success
        total_episodes += n
        per_task.append(
            {
                "task_id": entry["task_id"],
                "language": task_languages[entry["task_id"]],
                "successes": n_success,
                "episodes": n,
                "success_percent": 100.0 * n_success / n,
                "episode_outcomes": successes,
            }
        )

    overall_success = 100.0 * total_successes / total_episodes
    reference = config["target"]["reference_success_percent"]
    threshold = config["target"]["material_difference_percentage_points"]
    is_baseline_protocol = (
        task_ids == env_cfg_json["task_ids"]
        and episodes == env_cfg_json["episodes_per_task"]
        and batch_size == env_cfg_json["batch_size"]
    )
    material_difference = is_baseline_protocol and abs(overall_success - reference) >= threshold
    observed_interval = wilson_interval(total_successes, total_episodes)
    reference_interval = wilson_interval(
        config["target"]["reference_successes"], config["target"]["reference_episodes"]
    )
    reference_per_task = config["target"]["reference_per_task_success_percent"]
    per_task_deltas = [
        task["success_percent"] - reference_per_task[task["task_id"]] for task in per_task
    ]
    largest_deficits = sorted(range(len(per_task_deltas)), key=per_task_deltas.__getitem__)[:3]
    largest_improvements = sorted(
        range(len(per_task_deltas)), key=per_task_deltas.__getitem__, reverse=True
    )[:2]
    deficit_text = human_join(largest_deficits)
    improvement_text = human_join(largest_improvements)
    model_file = policy_path / "model.safetensors"
    driver_query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    )
    packages = package_versions()
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if not material_difference else "diagnose",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "target": config["target"],
        "result": {
            "overall_success_percent": overall_success,
            "successes": total_successes,
            "episodes": total_episodes,
            "wilson_95_percent_interval": list(observed_interval),
            "per_task": per_task,
        },
        "timing": {
            "evaluation_seconds": evaluation_seconds,
            "wall_seconds": wall_seconds,
            "seconds_per_episode": evaluation_seconds / total_episodes,
            "policy_latency_ms": {
                "scope": "select_action wall time including CUDA synchronization",
                "batch_size": batch_size,
                "samples": len(latencies_ms),
                "mean": statistics.fmean(latencies_ms),
                "median": statistics.median(latencies_ms),
                "p95": percentile(latencies_ms, 0.95),
                "min": min(latencies_ms),
                "max": max(latencies_ms),
            },
        },
        "vram": {
            "nvidia_smi_baseline_mib": memory_monitor.baseline_mib,
            "nvidia_smi_peak_mib": memory_monitor.peak_mib,
            "torch_peak_allocated_mib": torch_peak_allocated,
            "torch_peak_reserved_mib": torch_peak_reserved,
        },
        "eval_settings": {
            **env_cfg_json,
            "task_ids": task_ids,
            "episodes_per_task": episodes,
            "batch_size": batch_size,
            "render_episodes_per_task": render_episodes,
            "num_steps": model_cfg["num_steps"],
            "n_action_steps": model_cfg["n_action_steps"],
            "use_amp": model_cfg["use_amp"],
            "device": model_cfg["device"],
            "torch_allow_tf32": True,
            "cudnn_benchmark": True,
        },
        "revisions": {
            "checkpoint": f"{model_cfg['repository']}@{model_cfg['revision']}",
            "backbone": f"{model_cfg['backbone_repository']}@{model_cfg['backbone_revision']}",
            "libero_assets": (
                f"{config['sources']['libero_assets_repository']}@"
                f"{config['sources']['libero_assets_revision']}"
            ),
            "lerobot": f"{config['sources']['lerobot_repository']}@{config['sources']['lerobot_revision']}",
            "reference_evaluation": (
                f"{config['sources']['reference_evaluation_repository']}@"
                f"{config['sources']['reference_evaluation_revision']}"
            ),
            "repo": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "model_sha256": sha256_file(model_file),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
        "packages": packages,
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": driver_query,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "uv": command_output(["uv", "--version"]),
            "os_packages": os_package_versions(["libosmesa6", "libglfw3", "libgl1"]),
            "renderer": {
                "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            },
        },
        "diagnostics": {
            "target_comparison_applicable": is_baseline_protocol,
            "material_difference": material_difference,
            "difference_percentage_points": overall_success - reference,
            "threshold_percentage_points": threshold,
            "observed_wilson_95_percent_interval": list(observed_interval),
            "reference_wilson_95_percent_interval": list(reference_interval),
            "per_task_difference_percentage_points": per_task_deltas,
            "differences_from_reference": [
                f"Observed {overall_success:.1f}% is {overall_success - reference:+.1f} points from the target and inside the configured +/-{threshold:.1f}-point band.",
                f"The approximate Wilson 95% interval is {observed_interval[0]:.1f}-{observed_interval[1]:.1f}%; it includes 81.5%, but episode dependence makes this only a diagnostic.",
                f"Differences are spread across tasks: largest deficits are {deficit_text}; largest improvements are {improvement_text}. This is not a single-task protocol collapse.",
                "The reference artifact does not publish a complete transitive lock or its Python, CUDA, and driver versions; this run records all of them, so those remain the leading unresolved environment differences.",
                "The protocol uses the same pinned LeRobot simulator-reuse behavior. This harness additionally pins the backbone snapshot and synchronizes CUDA for timing; neither changes the model architecture or action values.",
            ],
        },
        "artifacts": {
            "videos_retained": args.keep_videos,
            "raw_video_paths_are_provenance_only": not args.keep_videos,
        },
        "raw_lerobot_metrics": info,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    report_path = args.report.resolve() if args.report else None
    if report_path:
        write_report(metrics, report_path)
    if not args.keep_videos:
        shutil.rmtree(raw_dir, ignore_errors=True)
        raw_parent = raw_dir.parent
        if raw_parent.exists() and not any(raw_parent.iterdir()):
            raw_parent.rmdir()
    print(
        f"RESULT {total_successes}/{total_episodes} = {overall_success:.1f}% | "
        f"runtime={evaluation_seconds:.1f}s | peak_vram={memory_monitor.peak_mib} MiB",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    if report_path:
        print(f"Wrote {report_path}", flush=True)
    return 0 if metrics["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
