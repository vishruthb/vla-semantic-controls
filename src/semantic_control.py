#!/usr/bin/env python3
"""Minimal semantic-control interface for SmolVLA.

Two knobs, one code path (see ``tmp/arch.md`` section 7 for the interface analysis):

* ``semantic_layers``: which transformer layers let the action expert read the VLM's key/value
  projections. ``"all"`` is the native SmolVLA wiring (joint self-attention layers *and*
  cross-attention layers). ``"cross_only"`` hides the VLM prefix from the expert in the joint
  self-attention layers, so VLM features reach the expert only through the cross-attention layers'
  dedicated ``k_proj``/``v_proj`` adapters. The VLM stream itself is never changed.
* ``update_vlm``: whether the action loss updates VLM parameters (text layers, token embeddings,
  connector). The vision encoder is frozen in every configuration; ``state_proj`` is trainable in
  every configuration.

Presets:  A = all/frozen   B = all/trainable   C = cross_only/frozen   D = cross_only/trainable
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/semantic_control.json"
SEMANTIC_LAYER_CHOICES = ("all", "cross_only")
PRESETS: dict[str, tuple[str, bool]] = {
    "A": ("all", False),
    "B": ("all", True),
    "C": ("cross_only", False),
    "D": ("cross_only", True),
}
SEMANTIC_CONTROL_FILE = "semantic_control.json"
SNAPSHOT_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticControlConfig:
    semantic_layers: str = "all"
    update_vlm: bool = False

    def __post_init__(self) -> None:
        if self.semantic_layers not in SEMANTIC_LAYER_CHOICES:
            raise ValueError(
                f"semantic_layers must be one of {SEMANTIC_LAYER_CHOICES}, got {self.semantic_layers!r}"
            )
        if not isinstance(self.update_vlm, bool):
            raise TypeError(f"update_vlm must be a bool, got {type(self.update_vlm).__name__}")

    @property
    def preset(self) -> str | None:
        for name, knobs in PRESETS.items():
            if knobs == (self.semantic_layers, self.update_vlm):
                return name
        return None

    @classmethod
    def from_preset(cls, name: str) -> SemanticControlConfig:
        try:
            semantic_layers, update_vlm = PRESETS[name.upper()]
        except KeyError as error:
            raise ValueError(f"Unknown preset {name!r}; expected one of {sorted(PRESETS)}") from error
        return cls(semantic_layers=semantic_layers, update_vlm=update_vlm)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticControlConfig:
        known = {"semantic_layers", "update_vlm", "preset"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown semantic-control keys: {sorted(unknown)}")
        config = cls(
            semantic_layers=data.get("semantic_layers", "all"),
            update_vlm=data.get("update_vlm", False),
        )
        if "preset" in data and data["preset"] is not None and config.preset != data["preset"].upper():
            raise ValueError(
                f"preset {data['preset']!r} does not match knobs "
                f"semantic_layers={config.semantic_layers!r}, update_vlm={config.update_vlm}"
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_layers": self.semantic_layers,
            "update_vlm": self.update_vlm,
            "preset": self.preset,
        }


CONTROL_KEYS = ("semantic_layers", "update_vlm", "preset")


def save_semantic_control(
    config: SemanticControlConfig, directory: Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Write ``semantic_control.json`` next to a checkpoint's ``config.json``/``model.safetensors``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SEMANTIC_CONTROL_FILE
    payload = {**config.to_dict(), **{k: v for k, v in (metadata or {}).items() if k not in CONTROL_KEYS}}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def parse_semantic_control(data: dict[str, Any]) -> tuple[SemanticControlConfig, dict[str, Any]]:
    """Split a control file into the knobs and the provenance metadata stored alongside them."""
    knobs = {k: data[k] for k in CONTROL_KEYS if k in data}
    metadata = {k: v for k, v in data.items() if k not in CONTROL_KEYS}
    return SemanticControlConfig.from_dict(knobs), metadata


def load_semantic_control(directory: Path) -> SemanticControlConfig:
    return parse_semantic_control(json.loads((Path(directory) / SEMANTIC_CONTROL_FILE).read_text()))[0]


# --------------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------------


def is_cross_attention_layer(layer_idx: int, attention_mode: str, self_attn_every_n_layers: int) -> bool:
    """Mirror of the layer dispatch in ``SmolVLMWithExpertModel.forward``.

    A layer is a cross-attention layer when the model runs in a ``cross`` attention mode and the
    layer index is not one of the interleaved joint self-attention layers.
    """
    if "cross" not in attention_mode:
        return False
    if self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers == 0:
        return False
    return True


def _import_upstream():
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    return SmolVLMWithExpertModel


class SemanticSmolVLMWithExpertModel(_import_upstream()):
    """``SmolVLMWithExpertModel`` whose joint self-attention layers can hide the VLM prefix from the
    action expert. Module structure and state-dict keys are identical to upstream."""

    semantic_layers: str = "all"

    @classmethod
    def install(cls, model: nn.Module, semantic_layers: str) -> SemanticSmolVLMWithExpertModel:
        upstream = _import_upstream()
        if type(model) is not upstream and not isinstance(model, cls):
            raise TypeError(f"Expected a SmolVLMWithExpertModel, got {type(model).__name__}")
        if semantic_layers not in SEMANTIC_LAYER_CHOICES:
            raise ValueError(f"semantic_layers must be one of {SEMANTIC_LAYER_CHOICES}")
        model.__class__ = cls
        model.semantic_layers = semantic_layers
        if semantic_layers == "cross_only" and not model.coupled_layers():
            raise ValueError(
                "semantic_layers='cross_only' requires cross-attention layers "
                f"(attention_mode={model.attention_mode!r}, "
                f"self_attn_every_n_layers={model.self_attn_every_n_layers})"
            )
        return model

    # -- routing predicates ------------------------------------------------------------------

    def is_cross_attn_layer(self, layer_idx: int) -> bool:
        return is_cross_attention_layer(layer_idx, self.attention_mode, self.self_attn_every_n_layers)

    def expert_reads_vlm_kv(self, layer_idx: int) -> bool:
        """Whether the action expert may attend to the VLM key/values of ``layer_idx``."""
        if self.semantic_layers == "all":
            return True
        return self.is_cross_attn_layer(layer_idx)

    def coupled_layers(self) -> list[int]:
        return [idx for idx in range(self.num_vlm_layers) if self.expert_reads_vlm_kv(idx)]

    # -- the single behavioural override -----------------------------------------------------

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        if not self.expert_reads_vlm_kv(layer_idx):
            attention_mask = self._hide_prefix_from_suffix(
                attention_mask, inputs_embeds, layer_idx, use_cache, fill_kv_cache, past_key_values
            )
        return super().forward_attn_layer(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            past_key_values=past_key_values,
        )

    @staticmethod
    def _hide_prefix_from_suffix(
        attention_mask, inputs_embeds, layer_idx, use_cache, fill_kv_cache, past_key_values
    ):
        """Return a copy of ``attention_mask`` in which suffix (action) queries cannot see prefix keys.

        Prefix rows are left untouched, so the VLM stream is unaffected. Masked logits receive the
        float32 minimum before the softmax, which yields exactly zero probability and exactly zero
        gradient for the hidden keys/values.
        """
        prefix = inputs_embeds[0] if len(inputs_embeds) > 0 else None
        suffix = inputs_embeds[1] if len(inputs_embeds) > 1 else None
        if suffix is None:
            return attention_mask  # prefix-only pass (e.g. KV-cache fill): nothing to hide
        if prefix is not None:
            prefix_len = prefix.shape[1]
        elif use_cache and not fill_kv_cache and past_key_values:
            prefix_len = past_key_values[layer_idx]["key_states"].shape[1]
        else:
            raise RuntimeError("Suffix tokens without a prefix or a filled KV cache")
        suffix_len = suffix.shape[1]
        mask = attention_mask.clone()
        mask[:, -suffix_len:, :prefix_len] = False
        return mask


# --------------------------------------------------------------------------------------------
# Trainability
# --------------------------------------------------------------------------------------------


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def apply_trainability(policy: nn.Module, control: SemanticControlConfig) -> None:
    """Set ``requires_grad`` for every parameter from scratch (upstream ``set_requires_grad`` can
    only freeze, never unfreeze).

    * vision encoder: always frozen
    * ``state_proj``: always trainable
    * ``update_vlm=False``: the whole VLM (text layers, token embeddings, connector, lm_head) frozen
    * ``update_vlm=True``: text layers, token embeddings and connector trainable; parameters with no
      path to the action loss stay frozen (``lm_head``, final text norm, and the last VLM layer's
      q/o projections, MLP and post-attention norm — plus its k/v projections and input norm when the
      expert does not read that layer's key/values)
    """
    vwe = policy.model.vlm_with_expert
    vlm_model = vwe.get_vlm_model()
    text_model = vlm_model.text_model

    _set_requires_grad(policy, True)
    _set_requires_grad(vlm_model.vision_model, False)
    if not control.update_vlm:
        _set_requires_grad(vwe.vlm, False)
    else:
        _set_requires_grad(vwe.vlm.lm_head, False)
        _set_requires_grad(text_model.norm, False)
        last_idx = vwe.num_vlm_layers - 1
        last = text_model.layers[last_idx]
        for dead in (last.self_attn.q_proj, last.self_attn.o_proj, last.mlp, last.post_attention_layernorm):
            _set_requires_grad(dead, False)
        reads_last = vwe.expert_reads_vlm_kv(last_idx) if hasattr(vwe, "expert_reads_vlm_kv") else True
        if not reads_last:
            for dead in (last.input_layernorm, last.self_attn.k_proj, last.self_attn.v_proj):
                _set_requires_grad(dead, False)
    for name, parameter in vwe.lm_expert.named_parameters():  # mirrors upstream
        if "lm_head" in name:
            parameter.requires_grad_(False)
    _set_requires_grad(policy.model.state_proj, True)

    # Keep upstream mode flags coherent so `train()` keeps the frozen VLM in eval mode as upstream does.
    vwe.freeze_vision_encoder = True
    vwe.train_expert_only = not control.update_vlm
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = not control.update_vlm
    policy.config.train_state_proj = True


def install_semantic_control(policy: nn.Module, control: SemanticControlConfig) -> nn.Module:
    """Apply routing and trainability to an already-constructed ``SmolVLAPolicy`` in place."""
    SemanticSmolVLMWithExpertModel.install(policy.model.vlm_with_expert, control.semantic_layers)
    apply_trainability(policy, control)
    policy.semantic_control = control
    return policy


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

PARAMETER_GROUPS: tuple[tuple[str, str], ...] = (
    ("vision_encoder", "model.vlm_with_expert.vlm.model.vision_model."),
    ("connector", "model.vlm_with_expert.vlm.model.connector."),
    ("vlm_embed_tokens", "model.vlm_with_expert.vlm.model.text_model.embed_tokens."),
    ("vlm_text_layers", "model.vlm_with_expert.vlm.model.text_model.layers."),
    ("vlm_text_norm", "model.vlm_with_expert.vlm.model.text_model.norm."),
    ("vlm_lm_head", "model.vlm_with_expert.vlm.lm_head."),
    ("action_expert", "model.vlm_with_expert.lm_expert."),
    ("state_proj", "model.state_proj."),
    ("action_in_proj", "model.action_in_proj."),
    ("action_out_proj", "model.action_out_proj."),
    ("action_time_mlp", "model.action_time_mlp_"),
)


def parameter_group(name: str) -> str:
    for group, prefix in PARAMETER_GROUPS:
        if name.startswith(prefix):
            return group
    return "other"


def parameter_report(policy: nn.Module) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {group: {"total": 0, "trainable": 0} for group, _ in PARAMETER_GROUPS}
    for name, parameter in policy.named_parameters():
        entry = report.setdefault(parameter_group(name), {"total": 0, "trainable": 0})
        entry["total"] += parameter.numel()
        if parameter.requires_grad:
            entry["trainable"] += parameter.numel()
    report["total"] = {
        "total": sum(v["total"] for k, v in report.items() if k != "total"),
        "trainable": sum(v["trainable"] for k, v in report.items() if k != "total"),
    }
    return report


def format_parameter_report(report: dict[str, dict[str, int]]) -> str:
    rows = [f"{'group':20} {'total':>14} {'trainable':>14}"]
    for group, entry in report.items():
        rows.append(f"{group:20} {entry['total']:>14,} {entry['trainable']:>14,}")
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------
# Checkpoint access and policy construction
# --------------------------------------------------------------------------------------------


def load_checkpoint_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_snapshot(repository: str, revision: str, local_files_only: bool = True) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=SNAPSHOT_PATTERNS,
        )
    ).resolve()


def cache_checkpoint(config: dict[str, Any]) -> tuple[Path, Path]:
    checkpoint = config["checkpoint"]
    policy_path = resolve_snapshot(checkpoint["repository"], checkpoint["revision"], local_files_only=False)
    backbone_path = resolve_snapshot(
        checkpoint["backbone_repository"], checkpoint["backbone_revision"], local_files_only=False
    )
    return policy_path, backbone_path


def align_vision_connector_dtype(policy: nn.Module) -> bool:
    """The frozen vision encoder feeds the connector directly; if their dtypes differ (fp32 master
    weights for the connector, bf16 vision) a forward hook casts the vision output. Returns whether
    a hook was installed."""
    vlm_model = policy.model.vlm_with_expert.get_vlm_model()
    connector_dtype = next(vlm_model.connector.parameters()).dtype
    if next(vlm_model.vision_model.parameters()).dtype == connector_dtype:
        return False
    existing = getattr(policy, "semantic_dtype_hook", None)
    if existing is not None:
        existing.remove()

    def align(module, inputs, output):
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = output.last_hidden_state.to(connector_dtype)
            return output
        return output.to(connector_dtype) if torch.is_tensor(output) else output

    policy.semantic_dtype_hook = vlm_model.vision_model.register_forward_hook(align)
    return True


def restore_saved_dtypes(policy: nn.Module, checkpoint: Path) -> int:
    """`from_pretrained` copies checkpoint tensors into parameters built in the backbone's dtype, which
    silently rounds fp32 master weights to bf16. Re-copy every tensor whose saved dtype differs from
    the constructed parameter, keeping the saved dtype. Returns the number of parameters restored."""
    from safetensors import safe_open

    path = Path(checkpoint) / "model.safetensors"
    parameters = dict(policy.named_parameters())
    restored = 0
    with safe_open(str(path), "pt") as handle:
        for name in handle.keys():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            tensor = handle.get_tensor(name)
            if tensor.dtype != parameter.dtype:
                parameter.data = tensor.to(device=parameter.device)
                restored += parameter.numel()
    if restored:
        align_vision_connector_dtype(policy)
    return restored


def build_policy(
    control: SemanticControlConfig,
    checkpoint: Path,
    backbone: Path | None = None,
    device: str = "cpu",
    policy_config_overrides: dict[str, Any] | None = None,
    keep_saved_dtypes: bool = True,
) -> nn.Module:
    """Load a SmolVLA checkpoint from a local snapshot directory and apply the semantic control."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint)
    if not (checkpoint / "config.json").exists() or not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(f"{checkpoint} is not a checkpoint directory (config.json + model.safetensors)")
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = device
    policy_cfg.freeze_vision_encoder = True
    policy_cfg.train_expert_only = not control.update_vlm
    policy_cfg.train_state_proj = True
    if backbone is not None:
        policy_cfg.vlm_model_name = str(backbone)
    for key, value in (policy_config_overrides or {}).items():
        setattr(policy_cfg, key, value)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_cfg)
    if keep_saved_dtypes:
        policy.restored_dtype_parameters = restore_saved_dtypes(policy, checkpoint)
    return install_semantic_control(policy, control)


def build_policy_from_config(
    control: SemanticControlConfig,
    config: dict[str, Any] | None = None,
    device: str = "cpu",
    policy_config_overrides: dict[str, Any] | None = None,
) -> nn.Module:
    config = config or load_checkpoint_config()
    checkpoint = config["checkpoint"]
    policy_path = resolve_snapshot(checkpoint["repository"], checkpoint["revision"])
    backbone_path = resolve_snapshot(checkpoint["backbone_repository"], checkpoint["backbone_revision"])
    return build_policy(control, policy_path, backbone_path, device, policy_config_overrides)


# --------------------------------------------------------------------------------------------
# Verification by observation, fingerprints, guarded loading
# --------------------------------------------------------------------------------------------


class RoutingError(RuntimeError):
    """Raised when a policy's effective routing does not match the requested semantic control."""


def make_dummy_batch(
    policy: nn.Module,
    text: str = "pick up the black bowl and place it on the plate",
    batch_size: int = 2,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic batch shaped from the policy config (images, state, action, language)."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    config = policy.config
    generator = torch.Generator().manual_seed(seed)
    batch: dict[str, torch.Tensor] = {}
    for key, feature in config.image_features.items():
        batch[key] = torch.rand((batch_size, *feature.shape), generator=generator)
    batch[OBS_STATE] = torch.randn((batch_size, *config.robot_state_feature.shape), generator=generator)
    batch[ACTION] = torch.randn((batch_size, config.chunk_size, *config.action_feature.shape), generator=generator)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(
        [text + "\n"] * batch_size,
        padding=config.pad_language_to,
        max_length=config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"]
    batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].bool()
    if device is None:
        device = next(policy.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


class AttentionRecorder:
    """Record the mask and Q/K/V shapes of every attention call, tagged with the dispatching layer."""

    def __init__(self, vlm_with_expert: nn.Module) -> None:
        self.vwe = vlm_with_expert
        self.records: list[dict[str, Any]] = []
        self._current: tuple[str, int] | None = None

    def __enter__(self) -> AttentionRecorder:
        vwe = self.vwe
        original_self = vwe.forward_attn_layer
        original_cross = vwe.forward_cross_attn_layer
        original_attention = vwe.eager_attention_forward

        def tag(kind, original):
            def wrapped(model_layers, inputs_embeds, layer_idx, *args, **kwargs):
                self._current = (kind, layer_idx)
                return original(model_layers, inputs_embeds, layer_idx, *args, **kwargs)

            return wrapped

        def attention(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            kind, layer_idx = self._current
            self.records.append(
                {
                    "dispatch": kind,
                    "layer": layer_idx,
                    "mask": attention_mask.detach().to("cpu").clone(),
                    "q_shape": tuple(query_states.shape),
                    "k_shape": tuple(key_states.shape),
                    "v_shape": tuple(value_states.shape),
                    "k_dtype": key_states.dtype,
                    "device": key_states.device.type,
                }
            )
            return original_attention(attention_mask, batch_size, head_dim, query_states, key_states, value_states)

        vwe.forward_attn_layer = tag("self", original_self)
        vwe.forward_cross_attn_layer = tag("cross", original_cross)
        vwe.eager_attention_forward = attention
        return self

    def __exit__(self, *exc) -> None:
        for name in ("forward_attn_layer", "forward_cross_attn_layer", "eager_attention_forward"):
            if name in self.vwe.__dict__:
                delattr(self.vwe, name)

    def by_layer(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for record in self.records:
            grouped.setdefault(record["layer"], []).append(record)
        return grouped


def observed_routing(policy: nn.Module, batch: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Run one training-style forward and report, per joint self-attention layer, whether the action
    rows can see any prefix key. This inspects the masks actually applied, not configuration."""
    vwe = policy.model.vlm_with_expert
    batch = batch or make_dummy_batch(policy)
    suffix_len = policy.config.chunk_size
    was_training = policy.training
    with torch.no_grad(), AttentionRecorder(vwe) as recorder:
        policy.eval()
        policy.forward(dict(batch))
    policy.train(was_training)
    prefix_visible: dict[int, bool] = {}
    prefix_matches_pad: dict[int, bool] = {}
    cross_layers: list[int] = []
    prefix_len = None
    for record in recorder.records:
        layer = record["layer"]
        if record["dispatch"] == "self" and record["q_shape"][1] > suffix_len:
            total = record["k_shape"][1]
            prefix_len = total - suffix_len
            block = record["mask"][:, prefix_len:, :prefix_len]
            pad = record["mask"][:, :prefix_len, :prefix_len].diagonal(dim1=1, dim2=2)
            prefix_visible[layer] = bool(block.any())
            prefix_matches_pad[layer] = bool(torch.equal(block, pad[:, None, :].expand(-1, suffix_len, -1)))
        elif record["dispatch"] == "cross" and record["q_shape"][1] == suffix_len:
            cross_layers.append(layer)
    return {
        "num_layers": vwe.num_vlm_layers,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "joint_layers": sorted(prefix_visible),
        "cross_layers": sorted(cross_layers),
        "joint_prefix_visible": prefix_visible,
        "joint_prefix_matches_pad_mask": prefix_matches_pad,
    }


def verify_routing(policy: nn.Module, control: SemanticControlConfig | None = None) -> dict[str, Any]:
    """Assert, by observing the applied attention masks, that ``policy`` routes as ``control`` says.

    Raises ``RoutingError`` on any mismatch, including a policy on which no semantic control was
    installed (upstream class) — the case that would otherwise silently evaluate C/D as native.
    """
    control = control or getattr(policy, "semantic_control", None)
    if control is None:
        raise RoutingError("No semantic control given and none installed on the policy")
    vwe = policy.model.vlm_with_expert
    if not isinstance(vwe, SemanticSmolVLMWithExpertModel):
        raise RoutingError(f"Policy uses upstream {type(vwe).__name__}; semantic control was never installed")
    if vwe.semantic_layers != control.semantic_layers:
        raise RoutingError(
            f"Installed semantic_layers={vwe.semantic_layers!r} but control requests {control.semantic_layers!r}"
        )
    observed = observed_routing(policy)
    expect_visible = control.semantic_layers == "all"
    bad = {
        layer: visible
        for layer, visible in observed["joint_prefix_visible"].items()
        if visible != expect_visible or (expect_visible and not observed["joint_prefix_matches_pad_mask"][layer])
    }
    if bad:
        raise RoutingError(
            f"semantic_layers={control.semantic_layers!r} but prefix visibility in joint layers is {bad}"
        )
    if not observed["joint_layers"]:
        raise RoutingError("No joint self-attention layer was observed; cannot verify routing")
    expected_cross = [idx for idx in range(vwe.num_vlm_layers) if vwe.is_cross_attn_layer(idx)]
    if observed["cross_layers"] != expected_cross:
        raise RoutingError(f"Cross-attention layers {observed['cross_layers']} != expected {expected_cross}")
    trainable_vlm = any(
        p.requires_grad for n, p in policy.named_parameters() if ".vlm_with_expert.vlm." in n
    )
    if trainable_vlm != control.update_vlm:
        raise RoutingError(f"update_vlm={control.update_vlm} but VLM parameters trainable={trainable_vlm}")
    return {
        "verified": True,
        "semantic_layers": control.semantic_layers,
        "update_vlm": control.update_vlm,
        "coupled_layers": vwe.coupled_layers(),
        "num_vlm_layers": vwe.num_vlm_layers,
        "joint_layers": observed["joint_layers"],
        "cross_layers": observed["cross_layers"],
        "prefix_len": observed["prefix_len"],
    }


def _fingerprint(policy: nn.Module, predicate) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        if not predicate(name):
            continue
        tensor = parameter.detach().to("cpu").contiguous().reshape(-1)
        digest.update(f"{name}|{tuple(parameter.shape)}|{tensor.dtype}".encode())
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())  # raw bytes, works for bf16 too
    return digest.hexdigest()


def init_fingerprint(policy: nn.Module) -> str:
    """SHA-256 of the non-VLM parameters (action expert, state/action projections)."""
    return _fingerprint(policy, lambda name: ".vlm_with_expert.vlm." not in name)


def vlm_fingerprint(policy: nn.Module) -> str:
    """SHA-256 of the VLM parameters (vision encoder, connector, text model)."""
    return _fingerprint(policy, lambda name: ".vlm_with_expert.vlm." in name)


def load_policy_with_control(
    checkpoint: Path,
    control: SemanticControlConfig | None = None,
    backbone: Path | None = None,
    device: str = "cuda",
    verify: bool = True,
    policy_config_overrides: dict[str, Any] | None = None,
) -> tuple[nn.Module, SemanticControlConfig, dict[str, Any]]:
    """Load a checkpoint directory for evaluation without any chance of silently reverting to native
    routing.

    Rules: a ``semantic_control.json`` in the checkpoint is authoritative — if ``control`` is also
    given it must match. Without the file, ``control`` must be given explicitly (native SmolVLA is
    preset A). After loading, the routing is verified by observing the applied attention masks.
    """
    checkpoint = Path(checkpoint)
    control_file = checkpoint / SEMANTIC_CONTROL_FILE
    saved, metadata = (None, {})
    if control_file.exists():
        saved, metadata = parse_semantic_control(json.loads(control_file.read_text()))
    if control is None and saved is None:
        raise RoutingError(
            f"{checkpoint} has no {SEMANTIC_CONTROL_FILE} and no semantic control was given; "
            "pass one explicitly (native SmolVLA routing is preset A)"
        )
    if control is not None and saved is not None and control != saved:
        raise RoutingError(
            f"Requested {control.to_dict()} but {control_file} records {saved.to_dict()}"
        )
    effective = control or saved
    policy = build_policy(effective, checkpoint, backbone, device, policy_config_overrides)
    info: dict[str, Any] = {
        "control": effective.to_dict(),
        "control_source": "checkpoint_file" if saved is not None else "argument",
        "checkpoint_metadata": metadata,
        "init_fingerprint": init_fingerprint(policy),
        "restored_dtype_parameters": getattr(policy, "restored_dtype_parameters", 0),
        "parameter_dtypes": sorted({str(p.dtype).replace("torch.", "") for p in policy.parameters()}),
    }
    if verify:
        info["routing"] = verify_routing(policy, effective)
    return policy, effective, info


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets", help="list the supported presets")
    sub.add_parser("cache", help="download the pinned checkpoint and backbone snapshots")
    report = sub.add_parser("report", help="print parameter counts for a preset")
    report.add_argument("--preset", required=True, choices=sorted(PRESETS))
    report.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_checkpoint_config(args.config)
    if args.command == "presets":
        for name, (layers, update) in PRESETS.items():
            print(f"{name}: semantic_layers={layers!r} update_vlm={update}")
        return 0
    if args.command == "cache":
        policy_path, backbone_path = cache_checkpoint(config)
        print(f"policy: {policy_path}\nbackbone: {backbone_path}")
        return 0
    control = SemanticControlConfig.from_preset(args.preset)
    policy = build_policy_from_config(control, config, device=args.device)
    vwe = policy.model.vlm_with_expert
    print(f"preset {args.preset}: {control.to_dict()}")
    print(f"coupled layers: {vwe.coupled_layers()} of {vwe.num_vlm_layers}")
    print(format_parameter_report(parameter_report(policy)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
