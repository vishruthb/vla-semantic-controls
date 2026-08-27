"""GPU integration tests for the semantic-control interface on the real pinned SmolVLA checkpoint.

Builds the native policy and presets A-D from ``lerobot/smolvla_base`` on CUDA, runs fixed-input /
fixed-noise forward passes plus one backward pass each, and checks the routing, gradient reach,
numerical equivalence of A with upstream, and well-formedness of all outputs. No LIBERO episodes,
no optimizer steps.
"""

from __future__ import annotations

import unittest

import torch
from safetensors import safe_open

from tests import helpers
from tests.helpers import sc

DEVICE = "cuda"


def _snapshots():
    checkpoint = sc.load_checkpoint_config()["checkpoint"]
    try:
        return (
            sc.resolve_snapshot(checkpoint["repository"], checkpoint["revision"]),
            sc.resolve_snapshot(checkpoint["backbone_repository"], checkpoint["backbone_revision"]),
        )
    except Exception:  # noqa: BLE001
        return None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class RealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshots = _snapshots()
        if snapshots is None:
            raise unittest.SkipTest("Pinned checkpoint not cached; run `python src/semantic_control.py cache`")
        cls.policy_path, cls.backbone_path = snapshots
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with safe_open(str(cls.policy_path / "model.safetensors"), "pt") as handle:
            cls.checkpoint_keys = set(handle.keys())

        native = cls._build_native()
        cls.batch = helpers.make_batch(native, text="pick up the cube and place it in the bin")
        cls.noise, cls.time = helpers.fixed_noise_and_time(native)
        cls.native = cls._run(native, keep_grad_tensors=True)
        cls.native["requires_grad"] = {n: p.requires_grad for n, p in native.named_parameters()}
        del native
        torch.cuda.empty_cache()

        cls.summaries = {}
        for preset in sc.PRESETS:
            control = sc.SemanticControlConfig.from_preset(preset)
            policy = sc.build_policy(control, cls.policy_path, cls.backbone_path, device=DEVICE)
            summary = cls._run(policy, keep_grad_tensors=(preset == "A"))
            vwe = policy.model.vlm_with_expert
            summary.update(
                control=control,
                report=sc.parameter_report(policy),
                coupled_layers=vwe.coupled_layers(),
                num_layers=vwe.num_vlm_layers,
                requires_grad={n: p.requires_grad for n, p in policy.named_parameters()},
                state_dict_keys=set(policy.state_dict().keys()),
                all_on_cuda=all(p.device.type == "cuda" for p in policy.parameters()),
            )
            cls.summaries[preset] = summary
            del policy
            torch.cuda.empty_cache()

    @classmethod
    def _build_native(cls):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(cls.policy_path)
        config.pretrained_path = cls.policy_path
        config.device = DEVICE
        config.vlm_model_name = str(cls.backbone_path)
        config.freeze_vision_encoder = True
        config.train_expert_only = True
        config.train_state_proj = True
        return SmolVLAPolicy.from_pretrained(cls.policy_path, config=config)

    @classmethod
    def _run(cls, policy, keep_grad_tensors: bool) -> dict:
        vwe = policy.model.vlm_with_expert
        batch = helpers.to_device(cls.batch, DEVICE)
        policy.train()
        modes = {
            "vlm_training": vwe.vlm.training,
            "vision_training": vwe.get_vlm_model().vision_model.training,
            "expert_training": vwe.lm_expert.training,
        }
        policy.zero_grad(set_to_none=True)
        with helpers.AttentionRecorder(vwe) as train_recorder:
            loss, _ = policy.forward(dict(batch), noise=cls.noise.to(DEVICE), time=cls.time.to(DEVICE))
        loss.backward()
        summary = {
            "modes": modes,
            "loss": loss.detach().float().cpu(),
            "loss_dtype": loss.dtype,
            "grad_norms": helpers.grad_norms(policy),
            "train_records": train_recorder.records,
        }
        if keep_grad_tensors:
            summary["grad_tensors"] = {
                n: p.grad.detach().clone() for n, p in policy.named_parameters() if p.grad is not None
            }
        with helpers.AttentionRecorder(vwe) as infer_recorder:
            actions = policy.predict_action_chunk(dict(batch), noise=cls.noise.to(DEVICE))
        summary.update(
            actions=actions.detach().float().cpu(),
            actions_device=actions.device.type,
            actions_dtype=actions.dtype,
            actions_shape=tuple(actions.shape),
            infer_records=infer_recorder.records,
            kv_cache=helpers.prefix_kv_cache(policy, batch),
        )
        return summary

    # -- structure ----------------------------------------------------------------------------

    def test_state_dict_matches_checkpoint(self):
        for preset, summary in self.summaries.items():
            self.assertEqual(summary["state_dict_keys"], self.checkpoint_keys, preset)
            self.assertTrue(summary["all_on_cuda"], preset)

    def test_routing_on_real_model(self):
        self.assertEqual(self.summaries["A"]["num_layers"], 16)
        self.assertEqual(self.summaries["A"]["coupled_layers"], list(range(16)))
        self.assertEqual(self.summaries["B"]["coupled_layers"], list(range(16)))
        self.assertEqual(self.summaries["C"]["coupled_layers"], list(range(1, 16, 2)))
        self.assertEqual(self.summaries["D"]["coupled_layers"], list(range(1, 16, 2)))

    # -- numerics -----------------------------------------------------------------------------

    def test_all_frozen_is_numerically_native(self):
        native, a = self.native, self.summaries["A"]
        self.assertTrue(torch.equal(native["loss"], a["loss"]), f"loss {native['loss']} vs {a['loss']}")
        self.assertTrue(torch.equal(native["actions"], a["actions"]))
        self.assertEqual(native["requires_grad"], a["requires_grad"])
        self.assertEqual(set(native["grad_tensors"]), set(a["grad_tensors"]))
        for name, grad in native["grad_tensors"].items():
            self.assertTrue(torch.equal(grad, a["grad_tensors"][name]), name)
        self.assertEqual(len(native["train_records"]), len(a["train_records"]))
        for x, y in zip(native["train_records"], a["train_records"], strict=True):
            self.assertTrue(torch.equal(x["mask"], y["mask"]))
        self.assertEqual(len(native["infer_records"]), len(a["infer_records"]))
        for x, y in zip(native["infer_records"], a["infer_records"], strict=True):
            self.assertTrue(torch.equal(x["mask"], y["mask"]))

    def test_all_configs_finite_and_well_formed(self):
        action_dim = 6
        for preset, summary in self.summaries.items():
            self.assertTrue(torch.isfinite(summary["loss"]).all(), preset)
            self.assertTrue(torch.isfinite(summary["actions"]).all(), preset)
            self.assertEqual(summary["actions_shape"], (2, 50, action_dim), preset)
            self.assertEqual(summary["actions_device"], "cuda", preset)
            self.assertEqual(summary["loss_dtype"], torch.float32, preset)
            for record in summary["train_records"] + summary["infer_records"]:
                self.assertEqual(record["device"], "cuda", preset)
            finite_grads = [v for v in summary["grad_norms"].values() if v is not None]
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in finite_grads), preset)

    def _joint_and_cross_masks(self, records, num_layers):
        joint, cross = {}, {}
        for record in records:
            target = cross if sc.is_cross_attention_layer(record["layer"], "cross_attn", 2) else joint
            target.setdefault(record["layer"], []).append(record)
        return joint, cross

    def test_cross_only_changes_only_expert_access_to_vlm_kv(self):
        suffix_len = 50
        for all_preset, cross_preset in (("A", "C"), ("B", "D")):
            a, c = self.summaries[all_preset], self.summaries[cross_preset]
            # VLM stream identical: the KV cache produced by the prefix-only pass is bitwise equal
            self.assertEqual(a["kv_cache"].keys(), c["kv_cache"].keys())
            for layer, (k_a, v_a) in a["kv_cache"].items():
                self.assertTrue(torch.equal(k_a, c["kv_cache"][layer][0]), (all_preset, layer))
                self.assertTrue(torch.equal(v_a, c["kv_cache"][layer][1]), (all_preset, layer))
            # same parameters, same trainable set, same state dict
            self.assertEqual(a["requires_grad"], c["requires_grad"])
            self.assertEqual(a["state_dict_keys"], c["state_dict_keys"])
            # training-pass masks: cross layers untouched, joint layers hide prefix from suffix rows only
            pad_mask = a["train_records"][0]["mask"]
            prefix_len = pad_mask.shape[1] - suffix_len
            pad_vector = pad_mask[:, :prefix_len, :prefix_len].diagonal(dim1=1, dim2=2)
            self.assertTrue(pad_vector.any() and not pad_vector.all())  # language is padded to 48 tokens
            joint_a, cross_a = self._joint_and_cross_masks(a["train_records"], a["num_layers"])
            joint_c, cross_c = self._joint_and_cross_masks(c["train_records"], c["num_layers"])
            self.assertEqual(sorted(joint_a), list(range(0, 16, 2)))
            self.assertEqual(sorted(cross_a), list(range(1, 16, 2)))
            for layer in cross_a:
                for x, y in zip(cross_a[layer], cross_c[layer], strict=True):
                    self.assertTrue(torch.equal(x["mask"], y["mask"]), layer)
                    self.assertEqual(x["k_shape"], y["k_shape"], layer)
            for layer in joint_a:
                (x,), (y,) = joint_a[layer], joint_c[layer]
                self.assertEqual(x["k_shape"], y["k_shape"])
                self.assertTrue(torch.equal(x["mask"][:, :prefix_len, :], y["mask"][:, :prefix_len, :]), layer)
                self.assertTrue(torch.equal(x["mask"][:, prefix_len:, prefix_len:], y["mask"][:, prefix_len:, prefix_len:]))
                self.assertTrue(torch.equal(x["mask"][:, prefix_len:, :prefix_len], pad_vector[:, None, :].expand(-1, suffix_len, -1)))
                self.assertFalse(y["mask"][:, prefix_len:, :prefix_len].any(), layer)
            # inference-pass masks: same story after the 16-layer cache fill
            fill = a["num_layers"]
            for x, y in zip(a["infer_records"][:fill], c["infer_records"][:fill], strict=True):
                self.assertTrue(torch.equal(x["mask"], y["mask"]))
            for x, y in zip(a["infer_records"][fill:], c["infer_records"][fill:], strict=True):
                self.assertEqual(x["layer"], y["layer"])
                if sc.is_cross_attention_layer(x["layer"], "cross_attn", 2):
                    self.assertTrue(torch.equal(x["mask"], y["mask"]))
                else:
                    self.assertTrue(torch.equal(x["mask"][:, :, :prefix_len], pad_vector[:, None, :].expand(-1, suffix_len, -1)))
                    self.assertFalse(y["mask"][:, :, :prefix_len].any())
                    self.assertTrue(torch.equal(x["mask"][:, :, prefix_len:], y["mask"][:, :, prefix_len:]))
            # and the expert's outputs do change
            self.assertFalse(torch.equal(a["loss"], c["loss"]), (all_preset, cross_preset))
            self.assertFalse(torch.equal(a["actions"], c["actions"]), (all_preset, cross_preset))

    # -- gradients ----------------------------------------------------------------------------

    def test_update_vlm_false_gives_no_vlm_gradients(self):
        for preset in ("A", "C"):
            summary = self.summaries[preset]
            vlm_names = [n for n in summary["grad_norms"] if ".vlm_with_expert.vlm." in n]
            self.assertTrue(vlm_names)
            self.assertTrue(all(summary["grad_norms"][n] is None for n in vlm_names), preset)
            self.assertTrue(all(not summary["requires_grad"][n] for n in vlm_names), preset)
            self.assertFalse(summary["modes"]["vlm_training"], preset)

    def test_update_vlm_true_gives_gradients_in_intended_vlm_parameters(self):
        for preset in ("B", "D"):
            summary = self.summaries[preset]
            control, last_idx = summary["control"], summary["num_layers"] - 1
            reads_last = last_idx in summary["coupled_layers"]
            self.assertTrue(summary["modes"]["vlm_training"], preset)
            for name, trainable in summary["requires_grad"].items():
                self.assertEqual(trainable, helpers.expected_trainable(name, control, last_idx, reads_last), (preset, name))
                norm = summary["grad_norms"][name]
                if trainable:
                    self.assertIsNotNone(norm, (preset, name))
                    self.assertGreater(norm, 0.0, (preset, name))
                else:
                    self.assertIsNone(norm, (preset, name))
            intended_prefixes = (
                "model.vlm_with_expert.vlm.model.connector.",
                "model.vlm_with_expert.vlm.model.text_model.embed_tokens.",
                *[f"model.vlm_with_expert.vlm.model.text_model.layers.{i}." for i in range(last_idx)],
                f"model.vlm_with_expert.vlm.model.text_model.layers.{last_idx}.self_attn.k_proj",
                f"model.vlm_with_expert.vlm.model.text_model.layers.{last_idx}.self_attn.v_proj",
                f"model.vlm_with_expert.vlm.model.text_model.layers.{last_idx}.input_layernorm",
            )
            for prefix in intended_prefixes:
                names = [n for n in summary["grad_norms"] if n.startswith(prefix)]
                self.assertTrue(names, prefix)
                self.assertTrue(all(summary["grad_norms"][n] not in (None, 0.0) for n in names), (preset, prefix))

    def test_state_proj_receives_gradients_in_every_config(self):
        for preset, summary in self.summaries.items():
            self.assertTrue(summary["requires_grad"]["model.state_proj.weight"], preset)
            self.assertGreater(summary["grad_norms"]["model.state_proj.weight"], 0.0, preset)
            self.assertGreater(summary["grad_norms"]["model.state_proj.bias"], 0.0, preset)

    def test_vision_encoder_frozen_in_every_config(self):
        for preset, summary in self.summaries.items():
            vision = [n for n in summary["grad_norms"] if ".vision_model." in n]
            self.assertTrue(vision)
            self.assertTrue(all(not summary["requires_grad"][n] for n in vision), preset)
            self.assertTrue(all(summary["grad_norms"][n] is None for n in vision), preset)
            self.assertFalse(summary["modes"]["vision_training"], preset)
            self.assertEqual(summary["report"]["vision_encoder"]["trainable"], 0, preset)

    def test_expert_and_projections_trainable_everywhere(self):
        for preset, summary in self.summaries.items():
            report = summary["report"]
            for group in ("action_expert", "state_proj", "action_in_proj", "action_out_proj", "action_time_mlp"):
                self.assertEqual(report[group]["trainable"], report[group]["total"], (preset, group))
            self.assertEqual(report["vlm_lm_head"]["trainable"], 0, preset)

    def test_parameter_counts(self):
        reports = {preset: s["report"] for preset, s in self.summaries.items()}
        totals = {preset: r["total"]["total"] for preset, r in reports.items()}
        self.assertEqual(len(set(totals.values())), 1)
        self.assertEqual(reports["A"]["total"]["trainable"], reports["C"]["total"]["trainable"])
        self.assertEqual(reports["B"]["total"]["trainable"], reports["D"]["total"]["trainable"])
        self.assertGreater(reports["B"]["total"]["trainable"], reports["A"]["total"]["trainable"])
        print("\n\nParameter counts (lerobot/smolvla_base):")
        for preset, report in reports.items():
            print(f"\n[{preset}] {self.summaries[preset]['control'].to_dict()}  coupled layers: {self.summaries[preset]['coupled_layers']}")
            print(sc.format_parameter_report(report))
        print("\nLosses / action checksums:")
        for preset, summary in self.summaries.items():
            print(f"  {preset}: loss={summary['loss'].item():.6f} actions.abs().sum()={summary['actions'].abs().sum().item():.4f}")
        print(f"  native: loss={self.native['loss'].item():.6f} actions.abs().sum()={self.native['actions'].abs().sum().item():.4f}")


if __name__ == "__main__":
    unittest.main()
