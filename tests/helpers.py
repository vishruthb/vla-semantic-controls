"""Shared fixtures for the semantic-control tests (CPU tiny model and real-model GPU tests)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import semantic_control as sc  # noqa: E402


def find_backbone_snapshot() -> Path | None:
    """Locate a cached SmolVLM2 snapshot to borrow tokenizer/processor files from."""
    candidates = []
    semantic = sc.load_checkpoint_config()["checkpoint"]
    candidates.append((semantic["backbone_repository"], semantic["backbone_revision"]))
    baseline = json.loads((ROOT / "configs/baseline.json").read_text())["model"]
    candidates.append((baseline["backbone_repository"], baseline["backbone_revision"]))
    for repository, revision in candidates:
        try:
            return sc.resolve_snapshot(repository, revision, local_files_only=True)
        except Exception:  # noqa: BLE001 - any cache miss means "try the next candidate"
            continue
    return None


def build_tiny_vlm_dir(target: Path, backbone_snapshot: Path) -> Path:
    """Write a tiny SmolVLM config next to the real tokenizer/processor files."""
    target.mkdir(parents=True, exist_ok=True)
    for file in backbone_snapshot.iterdir():
        if file.is_file() and not file.name.endswith(".safetensors") and not file.name.startswith("."):
            shutil.copy(file, target / file.name)
    config = json.loads((backbone_snapshot / "config.json").read_text())
    config["text_config"].update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    config["vision_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=64,
        patch_size=16,
    )
    config["scale_factor"] = 4
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = "float32"
    (target / "config.json").write_text(json.dumps(config, indent=2))
    return target


def make_tiny_policy(tiny_dir: Path, num_steps: int = 2, seed: int = 0):
    """A randomly initialised SmolVLA with 4 VLM layers (self, cross, self, cross)."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(seed)
    config = SmolVLAConfig(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        vlm_model_name=str(tiny_dir),
        load_vlm_weights=False,
        num_vlm_layers=0,
        num_expert_layers=-1,
        self_attn_every_n_layers=2,
        expert_width_multiplier=0.5,
        attention_mode="cross_attn",
        resize_imgs_with_padding=(64, 64),
        chunk_size=8,
        n_action_steps=1,
        tokenizer_max_length=16,
        pad_language_to="max_length",
        num_steps=num_steps,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
    )
    return SmolVLAPolicy(config)


def make_batch(
    policy, text: str = "pick up the black bowl and place it on the plate", batch_size: int = 2, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Deterministic CPU batch shaped from the policy config (shared implementation in semantic_control)."""
    return sc.make_dummy_batch(policy, text=text, batch_size=batch_size, seed=seed, device="cpu")


def fixed_noise_and_time(policy, batch_size: int = 2, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        (batch_size, policy.config.chunk_size, policy.config.max_action_dim), generator=generator
    )
    time = torch.linspace(0.25, 0.75, batch_size)
    return noise, time


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


AttentionRecorder = sc.AttentionRecorder


def prefix_kv_cache(policy, batch: dict[str, torch.Tensor]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Run the VLM-only KV-cache fill pass (what inference does first) and return the cache on CPU."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, cache = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
    return {
        layer: (entry["key_states"].to("cpu").clone(), entry["value_states"].to("cpu").clone())
        for layer, entry in cache.items()
    }


def grad_norms(policy) -> dict[str, float | None]:
    return {
        name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm()))
        for name, parameter in policy.named_parameters()
    }


def expected_trainable(name: str, control: sc.SemanticControlConfig, last_idx: int, reads_last: bool) -> bool:
    """Independent specification of the trainable set, by parameter name."""
    if ".vlm.model.vision_model." in name:
        return False
    if ".vlm_with_expert.vlm." in name:
        if not control.update_vlm:
            return False
        if ".lm_head." in name or ".text_model.norm." in name:
            return False
        marker = f".text_model.layers.{last_idx}."
        if marker in name:
            sub = name.split(marker, 1)[1]
            if sub.startswith(("self_attn.q_proj", "self_attn.o_proj", "mlp.", "post_attention_layernorm")):
                return False
            if not reads_last and sub.startswith(("input_layernorm", "self_attn.k_proj", "self_attn.v_proj")):
                return False
        return True
    if ".lm_expert." in name and "lm_head" in name:
        return False
    return True
