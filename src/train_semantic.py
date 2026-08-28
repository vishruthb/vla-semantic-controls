#!/usr/bin/env python3
"""Train SmolVLA with a semantic-control preset through LeRobot's official training loop.

    python src/train_semantic.py --semantic.preset C [any lerobot-train arguments ...]

The ``--semantic.*`` arguments are consumed here; everything else is handed verbatim to
``lerobot.scripts.lerobot_train.train``. Hooks installed on that module:

* ``make_policy`` — after LeRobot builds the policy, install the semantic control, optionally hold
  trainable blocks in fp32 (master weights), verify the routing by observation, record fingerprints,
  and give the optimizer two parameter groups (expert lr, VLM lr) when the VLM trains.
* ``make_optimizer_and_scheduler`` — register a step pre-hook that records per-group gradient norms.
* ``update_policy`` — record loss / lr / step time / peak memory for every step (training curve).
* ``save_checkpoint`` / ``update_last_checkpoint`` — keep only the steps in ``--semantic.save_at``,
  write ``semantic_control.json`` (control + fingerprints + training metrics) into every kept
  checkpoint, append the curve to ``semantic_train_log.jsonl``, and stop the process cleanly at
  ``--semantic.stop_at`` so a fixed LR horizon (``--steps``) can be trained in stages.

``--semantic.trainable_fp32 true`` keeps *trainable* parameters in float32 — pair it with
``ACCELERATE_MIXED_PRECISION=bf16`` for bf16 autocast. Required for small VLM learning rates: bf16
weights cannot represent 1e-5-relative updates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import semantic_control as sc  # noqa: E402

TRAIN_LOG_FILE = "semantic_train_log.jsonl"
STATE: dict[str, Any] = {}


def _bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def expand_episode_range(spec: str) -> list[int]:
    """'1261-1692' -> [1261, ..., 1692]; comma-separated ranges/ints are accepted."""
    episodes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if end < start:
                raise argparse.ArgumentTypeError(f"bad range {part!r}")
            episodes.extend(range(start, end + 1))
        elif part:
            episodes.append(int(part))
    return episodes


def parse_semantic_args(argv: list[str]) -> tuple[sc.SemanticControlConfig, argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--semantic.preset", dest="preset", choices=sorted(sc.PRESETS))
    parser.add_argument("--semantic.semantic_layers", dest="semantic_layers", choices=sc.SEMANTIC_LAYER_CHOICES)
    parser.add_argument("--semantic.update_vlm", dest="update_vlm", type=_bool)
    parser.add_argument("--semantic.trainable_fp32", dest="trainable_fp32", type=_bool, default=False)
    parser.add_argument("--semantic.vlm_lr", dest="vlm_lr", type=float, default=1e-5)
    parser.add_argument("--semantic.episodes", dest="episodes", help="episode range, e.g. 1261-1692")
    parser.add_argument("--semantic.stop_at", dest="stop_at", type=int, help="stop after saving this step")
    parser.add_argument("--semantic.save_at", dest="save_at", help="only keep checkpoints at these steps")
    args, rest = parser.parse_known_args(argv)
    if args.preset is None and args.semantic_layers is None and args.update_vlm is None:
        raise SystemExit("Give --semantic.preset {A,B,C,D} or --semantic.semantic_layers/--semantic.update_vlm")
    if args.preset is not None:
        control = sc.SemanticControlConfig.from_preset(args.preset)
        if args.semantic_layers is not None and args.semantic_layers != control.semantic_layers:
            raise SystemExit(f"--semantic.semantic_layers conflicts with preset {args.preset}")
        if args.update_vlm is not None and args.update_vlm != control.update_vlm:
            raise SystemExit(f"--semantic.update_vlm conflicts with preset {args.preset}")
    else:
        control = sc.SemanticControlConfig(
            semantic_layers=args.semantic_layers or "all", update_vlm=bool(args.update_vlm)
        )
    if args.episodes:
        rest = [*rest, "--dataset.episodes=" + json.dumps(expand_episode_range(args.episodes), separators=(",", ":"))]
    args.save_at = sorted({int(x) for x in args.save_at.split(",") if x.strip()}) if args.save_at else None
    if args.stop_at is not None and args.save_at is not None and args.stop_at not in args.save_at:
        args.save_at = sorted({*args.save_at, args.stop_at})
    return control, args, rest


# --------------------------------------------------------------------------------------------
# Policy preparation
# --------------------------------------------------------------------------------------------


def cast_trainable_to_fp32(policy) -> int:
    """Keep every trainable parameter in float32 (master weights); frozen modules stay as loaded.

    Whole blocks are cast (not individual parameters): upstream casts activations to the dtype of a
    sibling weight (e.g. `q_proj`) before applying `k_proj`, so a block must be dtype-homogeneous.
    """
    model = policy.model
    vwe = model.vlm_with_expert
    vlm_model = vwe.get_vlm_model()
    blocks = [vwe.lm_expert, model.state_proj, model.action_in_proj, model.action_out_proj,
              model.action_time_mlp_in, model.action_time_mlp_out]
    if any(p.requires_grad for p in vwe.vlm.parameters()):
        blocks += [vlm_model.text_model, vlm_model.connector]
    count = 0
    for block in blocks:
        for parameter in block.parameters():
            if parameter.dtype != sc.torch.float32:
                parameter.data = parameter.data.float()
                count += parameter.numel()
    stragglers = [n for n, p in policy.named_parameters() if p.requires_grad and p.dtype != sc.torch.float32]
    if stragglers:
        raise RuntimeError(f"Trainable parameters left in reduced precision: {stragglers[:5]}")
    sc.align_vision_connector_dtype(policy)
    return count


def parameter_groups(policy, vlm_lr: float) -> list[dict[str, Any]]:
    """Expert/projections at the optimizer's default lr; VLM parameters (when trainable) at ``vlm_lr``."""
    expert, vlm = [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        (vlm if ".vlm_with_expert.vlm." in name else expert).append(parameter)
    groups = [{"params": expert, "name": "expert"}]
    if vlm:
        groups.append({"params": vlm, "lr": vlm_lr, "name": "vlm"})
    return groups


def provenance() -> dict[str, Any]:
    source = Path(sc.__file__).read_bytes()
    try:
        from accelerate.state import AcceleratorState

        mixed_precision = AcceleratorState().mixed_precision if AcceleratorState._shared_state else None
    except Exception:  # noqa: BLE001
        mixed_precision = None
    return {
        "lerobot_version": importlib.metadata.version("lerobot"),
        "semantic_control_sha256": hashlib.sha256(source).hexdigest(),
        "mixed_precision": mixed_precision,
    }


def resumed_metadata(pretrained_path) -> dict[str, Any] | None:
    """When resuming, read the control file of the checkpoint being resumed."""
    if not pretrained_path:
        return None
    control_file = Path(pretrained_path) / sc.SEMANTIC_CONTROL_FILE
    if not control_file.exists():
        return None
    saved, metadata = sc.parse_semantic_control(json.loads(control_file.read_text()))
    return {"control": saved, "metadata": metadata}


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True
        )
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------------------------


def install_hooks(
    control: sc.SemanticControlConfig,
    trainable_fp32: bool = False,
    vlm_lr: float = 1e-5,
    stop_at: int | None = None,
    save_at: list[int] | None = None,
):
    """Patch LeRobot's training module. Returns the module and the original callables."""
    import lerobot.scripts.lerobot_train as train_module

    originals = {
        name: getattr(train_module, name)
        for name in ("make_policy", "save_checkpoint", "update_last_checkpoint", "update_policy", "make_optimizer_and_scheduler")
    }
    STATE.clear()
    STATE.update(
        control=control, trainable_fp32=trainable_fp32, vlm_lr=vlm_lr, stop_at=stop_at, save_at=save_at,
        step=0, start_step=0, t_start=time.perf_counter(), curve=[], unflushed=0, peak_mem_gib=0.0,
        group_grad_norms=None, skipped_save=False, steps_in_process=0, step_seconds=0.0,
    )

    def make_policy_with_control(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        policy = originals["make_policy"](cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)
        pretrained = getattr(cfg, "pretrained_path", None)
        resumed = resumed_metadata(pretrained)
        if resumed is not None and resumed["control"] != control:
            raise sc.RoutingError(f"Resuming {pretrained} recorded {resumed['control'].to_dict()}, requested {control.to_dict()}")
        sc.install_semantic_control(policy, control)
        cast = 0
        if trainable_fp32:
            cast = cast_trainable_to_fp32(policy)
            if pretrained:
                sc.restore_saved_dtypes(policy, pretrained)  # undo bf16 rounding of saved fp32 masters
        routing = sc.verify_routing(policy, control)
        report = sc.parameter_report(policy)
        groups = parameter_groups(policy, vlm_lr)
        policy.get_optim_params = lambda: groups
        metadata = {
            "init_fingerprint": sc.init_fingerprint(policy),
            "vlm_fingerprint": sc.vlm_fingerprint(policy),
            "trainable_fp32": trainable_fp32,
            "trainable_parameters": report["total"]["trainable"],
            "total_parameters": report["total"]["total"],
            "parameter_groups": {g["name"]: {"parameters": sum(p.numel() for p in g["params"]), "lr": g.get("lr", "default")} for g in groups},
            "routing": routing,
            **provenance(),
        }
        if resumed is not None:
            saved_meta = resumed["metadata"]
            metadata["resumed_from_step"] = saved_meta.get("step")
            metadata["resume_fingerprint"] = metadata["init_fingerprint"]
            metadata["init_fingerprint"] = saved_meta.get("init_fingerprint", metadata["init_fingerprint"])
            metadata["vlm_fingerprint"] = saved_meta.get("vlm_fingerprint", metadata["vlm_fingerprint"])
            if not control.update_vlm and saved_meta.get("vlm_fingerprint") not in (None, sc.vlm_fingerprint(policy)):
                raise sc.RoutingError("Frozen VLM weights changed between the original run and the resumed checkpoint")
            step_file = Path(pretrained).parent / "training_state" / "training_step.json"
            if step_file.exists():
                STATE["step"] = STATE["start_step"] = int(json.loads(step_file.read_text())["step"])
        policy.semantic_metadata = metadata
        logging.info(
            "Semantic control %s installed: coupled layers %s, trainable %s of %s, %s cast to fp32, groups %s, init %s",
            control.to_dict(), routing["coupled_layers"], f"{report['total']['trainable']:,}",
            f"{report['total']['total']:,}", f"{cast:,}", metadata["parameter_groups"], metadata["init_fingerprint"][:16],
        )
        return policy

    def make_optimizer_with_hook(cfg, policy):
        optimizer, scheduler = originals["make_optimizer_and_scheduler"](cfg, policy)

        def record_group_norms(opt, args, kwargs):
            norms = {}
            for group in opt.param_groups:
                grads = [p.grad.detach().float().norm() for p in group["params"] if p.grad is not None]
                norms[group.get("name", "group")] = float(sc.torch.stack(grads).norm()) if grads else 0.0
            STATE["group_grad_norms"] = norms

        optimizer.register_step_pre_hook(record_group_norms)
        STATE["optimizer"] = optimizer
        return optimizer, scheduler

    def update_policy_with_curve(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler=None, **kwargs):
        t0 = time.perf_counter()
        train_metrics, output_dict = originals["update_policy"](
            train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler=lr_scheduler, **kwargs
        )
        dt = time.perf_counter() - t0
        STATE["step"] += 1
        STATE["steps_in_process"] += 1
        STATE["step_seconds"] += dt
        peak = sc.torch.cuda.max_memory_allocated() / 2**30 if sc.torch.cuda.is_available() else 0.0
        STATE["peak_mem_gib"] = max(STATE["peak_mem_gib"], peak)
        total_norm = None
        try:
            meter = getattr(train_metrics, "grad_norm", None)
            total_norm = float(meter.val) if hasattr(meter, "val") else (float(meter) if meter is not None else None)
        except Exception:  # noqa: BLE001
            total_norm = None
        entry = {
            "step": STATE["step"],
            "loss": float(output_dict["loss"]) if output_dict and "loss" in output_dict else None,
            "lr": {g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)},
            "grad_norm_total_preclip": total_norm,
            "grad_norm_groups_postclip": STATE.get("group_grad_norms"),
            "step_s": dt,
            "peak_mem_gib": peak,
        }
        STATE["curve"].append(entry)
        STATE["unflushed"] += 1
        stop_at = STATE["stop_at"]
        if stop_at is not None and STATE["step"] > stop_at + 1000:
            raise RuntimeError(f"Ran past stop_at={stop_at} without a checkpoint; is stop_at a multiple of save_freq?")
        return train_metrics, output_dict

    def flush_curve(output_dir: Path) -> None:
        if not STATE["unflushed"]:
            return
        path = Path(output_dir) / TRAIN_LOG_FILE
        with path.open("a") as stream:
            for entry in STATE["curve"][-STATE["unflushed"] :]:
                stream.write(json.dumps(entry) + "\n")
        STATE["unflushed"] = 0

    def checkpoint_metrics(step: int, optimizer) -> dict[str, Any]:
        recent = [e["loss"] for e in STATE["curve"][-100:] if e["loss"] is not None]
        elapsed = time.perf_counter() - STATE["t_start"]
        return {
            "step": step,
            "loss_last": STATE["curve"][-1]["loss"] if STATE["curve"] else None,
            "loss_mean_last_100": sum(recent) / len(recent) if recent else None,
            "lr": {g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)} if optimizer is not None else None,
            "grad_norm_total_preclip_last": STATE["curve"][-1]["grad_norm_total_preclip"] if STATE["curve"] else None,
            "grad_norm_groups_postclip_last": STATE.get("group_grad_norms"),
            "runtime": {
                "process_elapsed_s": elapsed,
                "steps_in_process": STATE["steps_in_process"],
                "start_step_in_process": STATE["start_step"],
                "mean_step_s": STATE["step_seconds"] / max(1, STATE["steps_in_process"]),
            },
            "peak_vram": {"torch_max_allocated_gib_process": STATE["peak_mem_gib"], "nvidia_smi_used_mib_now": nvidia_smi_used_mib()},
        }

    def save_checkpoint_with_control(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None, **kwargs):
        if save_at is not None and step not in save_at:
            STATE["skipped_save"] = True
            logging.info("Skipping checkpoint at step %s (save_at=%s)", step, save_at)
            return
        STATE["skipped_save"] = False
        originals["save_checkpoint"](checkpoint_dir, step, cfg, policy, optimizer, scheduler, **kwargs)
        from lerobot.utils.constants import PRETRAINED_MODEL_DIR

        metadata = {**getattr(policy, "semantic_metadata", {}), "training": checkpoint_metrics(step, optimizer), "step": step}
        path = sc.save_semantic_control(control, Path(checkpoint_dir) / PRETRAINED_MODEL_DIR, metadata)
        # Resume only ever uses the newest checkpoint: drop earlier optimizer/RNG states to bound disk use.
        import shutil

        for earlier in sorted(Path(checkpoint_dir).parent.glob("[0-9]*")):
            state = earlier / "training_state"
            if earlier.name.isdigit() and int(earlier.name) < step and state.exists():
                shutil.rmtree(state)
                logging.info("Pruned %s", state)
        output_dir = getattr(cfg, "output_dir", None)
        flush_curve(Path(output_dir) if output_dir else Path(checkpoint_dir).parent.parent)
        logging.info("Wrote %s", path)

    def update_last_checkpoint_with_stop(checkpoint_dir):
        if STATE["skipped_save"]:
            return None
        result = originals["update_last_checkpoint"](checkpoint_dir)
        if stop_at is not None and STATE["step"] >= stop_at:
            logging.info("Reached stop_at=%s after saving %s; stopping this stage", stop_at, checkpoint_dir)
            raise SystemExit(0)
        return result

    train_module.make_policy = make_policy_with_control
    train_module.make_optimizer_and_scheduler = make_optimizer_with_hook
    train_module.update_policy = update_policy_with_curve
    train_module.save_checkpoint = save_checkpoint_with_control
    train_module.update_last_checkpoint = update_last_checkpoint_with_stop
    return train_module, originals


def remove_hooks(train_module, originals: dict[str, Any]) -> None:
    for name, value in originals.items():
        setattr(train_module, name, value)


def patch_subset_indexing() -> None:
    """LeRobot @8515d45: `EpisodeAwareSampler` yields absolute frame indices, but with an episode
    subset the reader's `get_item` expects indices relative to the filtered table (they coincide only
    for the full dataset). Apply the reader's own absolute->relative map in `__getitem__`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset, "_semantic_subset_patch", False):
        return
    original = LeRobotDataset.__getitem__

    def __getitem__(self, idx):
        reader = self._ensure_reader()
        mapping = getattr(reader, "_absolute_to_relative_idx", None)
        if mapping is not None and reader.hf_dataset is not None:
            idx = mapping.get(int(idx), idx)
        return original(self, idx)

    LeRobotDataset.__getitem__ = __getitem__
    LeRobotDataset._semantic_subset_patch = True


def main(argv: list[str] | None = None) -> int:
    control, args, rest = parse_semantic_args(sys.argv[1:] if argv is None else argv)
    patch_subset_indexing()
    train_module, _ = install_hooks(control, args.trainable_fp32, args.vlm_lr, args.stop_at, args.save_at)
    sys.argv = [sys.argv[0], *rest]
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    train_module.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
