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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        metavar="PRETRAINED_MODEL_DIR",
        help="Evaluate a local LeRobot checkpoint (…/checkpoints/<step>/pretrained_model) instead of the "
        "pinned Hub snapshot; semantic routing is restored from its semantic_control.json and verified",
    )
    parser.add_argument(
        "--semantic-control",
        choices=["A", "B", "C", "D"],
        help="Preset to enforce with --checkpoint (must match the checkpoint's own record if present)",
    )
    parser.add_argument(
        "--deterministic-noise",
        action="store_true",
        help="Matched protocol: per-(task, episode, step) seeded flow-matching noise via the policy's "
        "select_action(noise=...) argument, deterministic cuDNN kernels, TF32 off",
    )
    parser.add_argument(
        "--rerender",
        type=Path,
        metavar="METRICS_JSON",
        help="Recompute status and diagnostics from an existing metrics file and write --report "
        "without running the evaluation",
    )
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


def is_baseline_protocol(config: dict[str, Any], settings: dict[str, Any]) -> bool:
    env_cfg = config["environment"]
    return (
        list(settings["task_ids"]) == list(env_cfg["task_ids"])
        and settings["episodes_per_task"] == env_cfg["episodes_per_task"]
        and settings["batch_size"] == env_cfg["batch_size"]
    )


def describe_per_task_spread(
    per_task: list[dict[str, Any]], deltas: list[float], threshold: float
) -> str:
    """Summarize where the difference to the reference comes from, without presuming a direction."""
    if len(deltas) < 2:
        return (
            f"Only task {per_task[0]['task_id']} was evaluated ({deltas[0]:+.1f} points versus its "
            "reference); no spread analysis on a single task."
        )
    order = sorted(range(len(deltas)), key=deltas.__getitem__)
    deficits = [per_task[i]["task_id"] for i in order if deltas[i] < 0][:3]
    improvements = [per_task[i]["task_id"] for i in reversed(order) if deltas[i] > 0][:2]
    if not deficits and not improvements:
        return "Every task matches its per-task reference exactly."
    parts = []
    if deficits:
        parts.append(f"largest deficits are task(s) {human_join(deficits)}")
    if improvements:
        parts.append(f"largest improvements are task(s) {human_join(improvements)}")
    lost_episodes = [
        -deltas[i] * per_task[i]["episodes"] / 100 for i in range(len(deltas)) if deltas[i] < 0
    ]
    if lost_episodes:
        worst = order[0]
        share = (-deltas[worst] * per_task[worst]["episodes"] / 100) / sum(lost_episodes)
        if share >= 0.5 and -deltas[worst] >= 2 * threshold:
            verdict = (
                f"the deficit is concentrated in task {per_task[worst]['task_id']} "
                f"({share:.0%} of lost episodes), which points to a task-specific protocol problem"
            )
        else:
            verdict = "the deficit is spread across tasks rather than a single-task protocol collapse"
    else:
        verdict = "no task is below its reference"
    return f"Per-task deltas: {'; '.join(parts)}; {verdict}."


def build_diagnostics(
    config: dict[str, Any], result: dict[str, Any], comparable: bool
) -> dict[str, Any]:
    """Derive the status and the diagnosis bullets from the measured result only."""
    target = config["target"]
    reference = target["reference_success_percent"]
    threshold = target["material_difference_percentage_points"]
    overall = result["overall_success_percent"]
    delta = overall - reference
    material = comparable and abs(delta) >= threshold
    observed = wilson_interval(result["successes"], result["episodes"])
    reference_interval = wilson_interval(target["reference_successes"], target["reference_episodes"])
    per_task = result["per_task"]
    reference_per_task = target.get("reference_per_task_success_percent")

    bullets = []
    if comparable:
        band = "inside" if not material else "outside"
        bullets.append(
            f"Observed {overall:.1f}% is {delta:+.1f} points from the {reference:.1f}% target and "
            f"{band} the configured +/-{threshold:.1f}-point band."
        )
    else:
        bullets.append(
            f"This run used {len(per_task)} task(s) with {result['episodes']} episodes in total, "
            "which is not the baseline protocol; the reference comparison is not applicable."
        )
    includes = observed[0] <= reference <= observed[1]
    bullets.append(
        f"The approximate Wilson 95% interval is {observed[0]:.1f}-{observed[1]:.1f}%; it "
        f"{'includes' if includes else 'excludes'} the {reference:.1f}% reference (whose own interval is "
        f"{reference_interval[0]:.1f}-{reference_interval[1]:.1f}%). Episode dependence makes this only a diagnostic."
    )
    per_task_deltas: list[float] | None = None
    if reference_per_task is not None and all(
        task["task_id"] < len(reference_per_task) for task in per_task
    ):
        per_task_deltas = [
            task["success_percent"] - reference_per_task[task["task_id"]] for task in per_task
        ]
        bullets.append(describe_per_task_spread(per_task, per_task_deltas, threshold))
    else:
        bullets.append("No per-task reference covers these tasks; per-task deltas were not computed.")
    if comparable:
        bullets.extend(config.get("report", {}).get("notes", []))

    status = "not_comparable" if not comparable else ("diagnose" if material else "pass")
    return {
        "status": status,
        "target_comparison_applicable": comparable,
        "material_difference": material,
        "difference_percentage_points": delta,
        "threshold_percentage_points": threshold,
        "observed_wilson_95_percent_interval": list(observed),
        "reference_wilson_95_percent_interval": list(reference_interval),
        "per_task_difference_percentage_points": per_task_deltas,
        "differences_from_reference": bullets,
    }


class Tolerant(dict):
    """Dict view that renders missing keys as 'n/a' so older metrics files still report."""

    def __getitem__(self, key: str) -> Any:
        return super().get(key, "n/a")


def write_report(metrics: dict[str, Any], path: Path) -> None:
    result = metrics["result"]
    target = metrics["target"]
    settings = Tolerant(metrics["eval_settings"])
    timing = metrics["timing"]
    vram = Tolerant(metrics["vram"])
    revisions = Tolerant(metrics["revisions"])
    packages = Tolerant(metrics["packages"])
    system = Tolerant(metrics["system"])
    delta = result["overall_success_percent"] - target["reference_success_percent"]
    threshold = target["material_difference_percentage_points"]
    status_text = {
        "pass": f"PASS (within the +/-{threshold:.1f}-point reproduction band)",
        "diagnose": f"DIAGNOSE (outside the +/-{threshold:.1f}-point reproduction band)",
        "not_comparable": "NOT COMPARABLE (protocol differs from the baseline)",
    }[metrics["status"]]
    title = metrics.get("report", {}).get("title") or (
        f"{revisions['checkpoint'].split('@')[0]} on {settings['suite']}"
    )
    reference_per_task = target.get("reference_per_task_success_percent")
    gpu_fields = [field.strip() for field in str(system["gpu"] or "unknown").split(",")]
    gpu_summary = (
        f"{gpu_fields[0]}, driver {gpu_fields[1]}, {gpu_fields[2]} MiB"
        if len(gpu_fields) == 3
        else system["gpu"]
    )
    latency = timing["policy_latency_ms"]
    rows = [
        f"# {title}",
        "",
        f"**Status: {status_text} — {result['successes']}/{result['episodes']} successes "
        f"({result['overall_success_percent']:.1f}%).** Reference: "
        f"~{target['reference_success_percent']:.1f}% (delta {delta:+.1f} points).",
        "",
        "| Task | This run | Reference | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in result["per_task"]:
        reference_task = (
            reference_per_task[task["task_id"]]
            if reference_per_task is not None and task["task_id"] < len(reference_per_task)
            else None
        )
        reference_cell = f"{reference_task:.1f}%" if reference_task is not None else "n/a"
        delta_cell = (
            f"{task['success_percent'] - reference_task:+.1f}" if reference_task is not None else "n/a"
        )
        rows.append(
            f"| {task['task_id']}: {task['language']} | {task['success_percent']:.1f}% "
            f"({task['successes']}/{task['episodes']}) | {reference_cell} | {delta_cell} |"
        )
    def flag(key: str, on: str, off: str) -> str:
        value = settings[key]
        return "n/a" if value == "n/a" else (on if value else off)

    parallel_text = (
        "n/a"
        if settings["max_parallel_tasks"] == "n/a"
        else "one task evaluated at a time"
        if settings["max_parallel_tasks"] <= 1
        else f"up to {settings['max_parallel_tasks']} tasks evaluated in parallel"
    )
    rows.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Evaluation: {timing['evaluation_seconds']:.1f} s; total process: {timing['wall_seconds']:.1f} s.",
            f"- Policy latency, batch {settings['batch_size']}: mean {latency['mean']:.2f} ms, "
            f"median {latency['median']:.2f} ms, p95 {latency['p95']:.2f} ms over {latency['samples']} calls.",
            f"- Peak VRAM: {vram['nvidia_smi_peak_mib']} MiB by nvidia-smi; "
            f"PyTorch allocated/reserved peaks: {vram['torch_peak_allocated_mib']:.1f}/"
            f"{vram['torch_peak_reserved_mib']:.1f} MiB.",
            f"- Host: {gpu_summary}; CUDA runtime {system['cuda_runtime']}; Python {system['python']}.",
            "",
            "## Exact setting",
            "",
            f"- Checkpoint: `{revisions['checkpoint']}` (weights SHA-256 `{revisions['model_sha256']}`).",
            *([semantic_report_line(metrics["semantic_control"])] if metrics.get("semantic_control") else []),
            *(
                [f"- Policy kind: `{metrics['policy_kind']}`; semantic control active: {metrics['semantic_control_active']}; "
                 f"config SHA-256 `{metrics['fingerprints']['config_sha256'][:16]}`; protocol SHA-256 `{metrics['fingerprints']['protocol_sha256'][:16]}`."]
                if metrics.get("policy_kind") else []
            ),
            f"- LeRobot: `{revisions['lerobot']}`; LIBERO `{packages['hf-libero']}`; "
            f"MuJoCo `{packages['mujoco']}`.",
            f"- PyTorch `{packages['torch']}`; Transformers `{packages['transformers']}`; "
            f"complete dependency resolution in `uv.lock` (SHA-256 `{revisions['uv_lock_sha256']}`).",
            f"- Harness revision: `{revisions['repo'] or 'uncommitted worktree (no Git HEAD)'}`; "
            f"reference artifact: `{revisions['reference_evaluation']}`.",
            f"- {len(result['per_task'])} tasks × {settings['episodes_per_task']} episodes; seed {settings['seed']}; "
            f"{flag('init_states', 'fixed', 'random')} init states; {settings['control_mode']} control; "
            f"{settings['observation_width']}×{settings['observation_height']} observations; "
            f"{settings['episode_length']} max steps.",
            f"- {flag('use_amp', 'autocast (AMP)', 'fp32')}; `num_steps={settings['num_steps']}`; "
            f"`n_action_steps={settings['n_action_steps']}`; batch {settings['batch_size']}; "
            f"`{settings['render_backend']}` rendering.",
            f"- TF32 {flag('torch_allow_tf32', 'allowed', 'disabled')}; cuDNN benchmark "
            f"{flag('cudnn_benchmark', 'enabled', 'disabled')}; "
            f"{flag('use_async_envs', 'async', 'sync')} vector environments; {parallel_text}.",
            "",
            "## Diagnosis",
            "",
        ]
    )
    for item in metrics["diagnostics"]["differences_from_reference"]:
        rows.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def report_block(config: dict[str, Any], semantic_info: dict[str, Any] | None) -> dict[str, Any]:
    block = dict(config.get("report", {}))
    if semantic_info:
        preset = semantic_info["control"].get("preset") or semantic_info["control"]["semantic_layers"]
        step = semantic_info.get("checkpoint_metadata", {}).get("step")
        suffix = f"preset {preset}" + (f" @ step {step}" if step is not None else "")
        block["title"] = f"{block.get('title', 'SmolVLA evaluation')} — {suffix}"
    return block


def semantic_report_line(semantic_info: dict[str, Any]) -> str:
    control = semantic_info["control"]
    routing = semantic_info.get("routing", {})
    metadata = semantic_info.get("checkpoint_metadata", {})
    fingerprint = metadata.get("init_fingerprint") or semantic_info.get("init_fingerprint") or ""
    return (
        f"- Semantic control: preset {control.get('preset')} (semantic_layers={control['semantic_layers']}, "
        f"update_vlm={control['update_vlm']}; source {semantic_info.get('control_source')}); routing verified: "
        f"{routing.get('verified')} on {len(routing.get('coupled_layers', []))}/{routing.get('num_vlm_layers')} "
        f"coupled layers; training step {metadata.get('step', 'n/a')}; init fingerprint `{fingerprint[:16]}`; "
        f"parameter dtypes {semantic_info.get('parameter_dtypes')}."
    )


def rerender(args: argparse.Namespace) -> int:
    if args.report is None:
        raise SystemExit("--rerender requires --report")
    config = json.loads(args.config.resolve().read_text())
    metrics = json.loads(args.rerender.resolve().read_text())
    comparable = is_baseline_protocol(config, metrics["eval_settings"])
    metrics["target"] = config["target"]
    metrics["report"] = config.get("report", {})
    metrics["diagnostics"] = build_diagnostics(config, metrics["result"], comparable)
    metrics["status"] = metrics["diagnostics"].pop("status")
    report_path = args.report.resolve()
    write_report(metrics, report_path)
    print(f"Wrote {report_path} (status={metrics['status']})", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.rerender is not None:
        return rerender(args)
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

    policy = None
    semantic_info = None
    if args.checkpoint is not None:
        import semantic_control as sc

        policy_path = args.checkpoint.resolve()
        requested = (
            sc.SemanticControlConfig.from_preset(args.semantic_control) if args.semantic_control else None
        )
        policy, control, semantic_info = sc.load_policy_with_control(
            policy_path,
            requested,
            device=model_cfg["device"],
            policy_config_overrides={
                "device": model_cfg["device"],
                "use_amp": model_cfg["use_amp"],
                "num_steps": model_cfg["num_steps"],
                "n_action_steps": model_cfg["n_action_steps"],
            },
        )
        if not semantic_info.get("routing", {}).get("verified"):
            raise RuntimeError("Semantic routing verification failed; the evaluation would be invalid")
        policy_cfg = policy.config
        backbone_path = Path(policy_cfg.vlm_model_name)
        print(
            f"Checkpoint {policy_path}: preset {control.preset} "
            f"(source={semantic_info['control_source']}), routing verified on layers "
            f"{semantic_info['routing']['coupled_layers']}",
            flush=True,
        )
    else:
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
    torch.backends.cudnn.benchmark = not args.deterministic_noise
    torch.backends.cudnn.deterministic = bool(args.deterministic_noise)
    torch.backends.cuda.matmul.allow_tf32 = not args.deterministic_noise
    torch.backends.cudnn.allow_tf32 = not args.deterministic_noise

    raw_dir = args.output.resolve().parent / "raw" / args.output.stem
    videos_dir = raw_dir / "videos"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {policy_path}" if args.checkpoint is not None else f"Loading {model_cfg['repository']}@{model_cfg['revision']}",
        flush=True,
    )
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
    if policy is None:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_config, rename_map={})
    policy.eval()
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    semantic_control_active = (
        type(policy.model.vlm_with_expert) is not SmolVLMWithExpertModel or hasattr(policy, "semantic_control")
    )
    policy_kind = "semantic_control_checkpoint" if args.checkpoint is not None else "stock_released_smolvla"
    if args.checkpoint is None and semantic_control_active:
        raise RuntimeError("Stock policy evaluation must not have semantic control installed")
    print(f"Policy kind: {policy_kind} (semantic control active: {semantic_control_active})", flush=True)
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

    noise_protocol = None
    if args.deterministic_noise:
        import eval_protocol

        noise_protocol = eval_protocol.install_deterministic_noise(policy, envs, env_cfg_json["seed"])
        noise_protocol = {k: v for k, v in noise_protocol.items() if k != "state"}
        print(f"Deterministic matched noise enabled: {noise_protocol['scheme']}", flush=True)

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
    result = {
        "overall_success_percent": overall_success,
        "successes": total_successes,
        "episodes": total_episodes,
        "wilson_95_percent_interval": list(wilson_interval(total_successes, total_episodes)),
        "per_task": per_task,
    }
    comparable = (
        task_ids == env_cfg_json["task_ids"]
        and episodes == env_cfg_json["episodes_per_task"]
        and batch_size == env_cfg_json["batch_size"]
    )
    diagnostics = build_diagnostics(config, result, comparable)
    status = diagnostics.pop("status")
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
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "target": config["target"],
        "report": report_block(config, semantic_info),
        "result": result,
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
            "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_noise": noise_protocol or {"enabled": False},
        },
        "policy_kind": policy_kind,
        "semantic_control_active": semantic_control_active,
        "semantic_control": semantic_info,
        "fingerprints": {
            "model_sha256": sha256_file(model_file),
            "config_sha256": sha256_file(policy_path / "config.json"),
            "protocol_sha256": None,  # filled below from eval_settings
        },
        "revisions": {
            "checkpoint": (
                f"local:{policy_path}" if args.checkpoint is not None
                else f"{model_cfg['repository']}@{model_cfg['revision']}"
            ),
            "backbone": (
                str(backbone_path) if args.checkpoint is not None
                else f"{model_cfg['backbone_repository']}@{model_cfg['backbone_revision']}"
            ),
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
        "diagnostics": diagnostics,
        "artifacts": {
            "videos_retained": args.keep_videos,
            "raw_video_paths_are_provenance_only": not args.keep_videos,
        },
        "raw_lerobot_metrics": info,
    }
    metrics["fingerprints"]["protocol_sha256"] = hashlib.sha256(
        json.dumps(metrics["eval_settings"], sort_keys=True, default=str).encode()
    ).hexdigest()
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
    return 3 if metrics["status"] == "diagnose" else 0


if __name__ == "__main__":
    raise SystemExit(main())
