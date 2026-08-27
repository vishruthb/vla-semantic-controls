"""CPU unit tests for the semantic-control interface (tiny randomly initialised SmolVLA)."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests import helpers
from tests.helpers import sc


class ConfigTests(unittest.TestCase):
    def test_presets_map_to_expected_knobs(self):
        expected = {
            "A": ("all", False),
            "B": ("all", True),
            "C": ("cross_only", False),
            "D": ("cross_only", True),
        }
        for name, (layers, update) in expected.items():
            control = sc.SemanticControlConfig.from_preset(name)
            self.assertEqual((control.semantic_layers, control.update_vlm), (layers, update))
            self.assertEqual(control.preset, name)
        self.assertEqual(sc.SemanticControlConfig.from_preset("c").preset, "C")

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            sc.SemanticControlConfig(semantic_layers="self_only")
        with self.assertRaises(TypeError):
            sc.SemanticControlConfig(update_vlm="yes")
        with self.assertRaises(ValueError):
            sc.SemanticControlConfig.from_preset("E")
        with self.assertRaises(ValueError):
            sc.SemanticControlConfig.from_dict({"semantic_layers": "all", "gradient_scale": 0.5})
        with self.assertRaises(ValueError):
            sc.SemanticControlConfig.from_dict({"semantic_layers": "all", "update_vlm": True, "preset": "A"})

    def test_dict_and_file_roundtrip(self):
        for name in sc.PRESETS:
            control = sc.SemanticControlConfig.from_preset(name)
            self.assertEqual(sc.SemanticControlConfig.from_dict(control.to_dict()), control)
        with tempfile.TemporaryDirectory() as directory:
            path = sc.save_semantic_control(sc.SemanticControlConfig.from_preset("D"), Path(directory))
            self.assertEqual(json.loads(path.read_text())["preset"], "D")
            self.assertEqual(sc.load_semantic_control(Path(directory)).preset, "D")

    def test_presets_file_matches_code(self):
        config = sc.load_checkpoint_config()
        for name, knobs in config["presets"].items():
            self.assertEqual(sc.SemanticControlConfig.from_dict(knobs), sc.SemanticControlConfig.from_preset(name))
        self.assertTrue(config["invariants"]["vision_encoder_frozen"])
        self.assertTrue(config["invariants"]["state_proj_trainable"])


class RoutingLogicTests(unittest.TestCase):
    def test_cross_attention_layer_predicate(self):
        kinds = [sc.is_cross_attention_layer(i, "cross_attn", 2) for i in range(6)]
        self.assertEqual(kinds, [False, True, False, True, False, True])
        self.assertTrue(all(sc.is_cross_attention_layer(i, "cross_attn", -1) for i in range(6)))
        self.assertFalse(any(sc.is_cross_attention_layer(i, "self_attn", 2) for i in range(6)))
        self.assertEqual([sc.is_cross_attention_layer(i, "cross_attn", 3) for i in range(6)], [False, True, True, False, True, True])


class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backbone = helpers.find_backbone_snapshot()
        if backbone is None:
            raise unittest.SkipTest("No cached SmolVLM2 snapshot to borrow tokenizer files from")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tiny_dir = helpers.build_tiny_vlm_dir(Path(cls._tmp.name) / "tiny_vlm", backbone)
        cls.base_policy = helpers.make_tiny_policy(cls.tiny_dir)
        cls.batch = helpers.make_batch(cls.base_policy)
        cls.noise, cls.time = helpers.fixed_noise_and_time(cls.base_policy)
        cls.num_layers = cls.base_policy.model.vlm_with_expert.num_vlm_layers
        assert cls.num_layers == 4

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def policy_for(self, preset: str):
        policy = copy.deepcopy(self.base_policy)
        return sc.install_semantic_control(policy, sc.SemanticControlConfig.from_preset(preset))

    # -- structure -------------------------------------------------------------------------

    def test_install_keeps_state_dict_keys(self):
        upstream_keys = list(self.base_policy.state_dict().keys())
        for preset in sc.PRESETS:
            policy = self.policy_for(preset)
            self.assertEqual(list(policy.state_dict().keys()), upstream_keys)
            self.assertIsInstance(policy.model.vlm_with_expert, sc.SemanticSmolVLMWithExpertModel)

    def test_install_rejects_cross_only_without_cross_layers(self):
        policy = copy.deepcopy(self.base_policy)
        policy.model.vlm_with_expert.attention_mode = "self_attn"
        with self.assertRaises(ValueError):
            sc.install_semantic_control(policy, sc.SemanticControlConfig.from_preset("C"))

    def test_coupled_layers(self):
        self.assertEqual(self.policy_for("A").model.vlm_with_expert.coupled_layers(), [0, 1, 2, 3])
        self.assertEqual(self.policy_for("B").model.vlm_with_expert.coupled_layers(), [0, 1, 2, 3])
        self.assertEqual(self.policy_for("C").model.vlm_with_expert.coupled_layers(), [1, 3])
        self.assertEqual(self.policy_for("D").model.vlm_with_expert.coupled_layers(), [1, 3])

    # -- trainability ------------------------------------------------------------------------

    def test_trainable_sets_match_specification(self):
        for preset in sc.PRESETS:
            policy = self.policy_for(preset)
            control = policy.semantic_control
            vwe = policy.model.vlm_with_expert
            last_idx = vwe.num_vlm_layers - 1
            reads_last = vwe.expert_reads_vlm_kv(last_idx)
            mismatches = [
                name
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad != helpers.expected_trainable(name, control, last_idx, reads_last)
            ]
            self.assertEqual(mismatches, [], f"preset {preset}")

    def test_vision_always_frozen_state_proj_always_trainable(self):
        for preset in sc.PRESETS:
            policy = self.policy_for(preset)
            vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
            self.assertFalse(any(p.requires_grad for p in vision.parameters()), preset)
            self.assertTrue(all(p.requires_grad for p in policy.model.state_proj.parameters()), preset)
            self.assertTrue(policy.config.freeze_vision_encoder)
            self.assertTrue(policy.config.train_state_proj)
            self.assertEqual(policy.config.train_expert_only, not policy.semantic_control.update_vlm)

    def test_train_mode_keeps_frozen_vlm_in_eval(self):
        for preset in sc.PRESETS:
            policy = self.policy_for(preset)
            policy.train()
            vwe = policy.model.vlm_with_expert
            self.assertEqual(vwe.vlm.training, policy.semantic_control.update_vlm, preset)
            self.assertFalse(vwe.get_vlm_model().vision_model.training, preset)
            self.assertTrue(vwe.lm_expert.training, preset)

    def test_parameter_counts(self):
        reports = {preset: sc.parameter_report(self.policy_for(preset)) for preset in sc.PRESETS}
        total = sum(p.numel() for p in self.base_policy.parameters())
        for preset, report in reports.items():
            self.assertEqual(report["total"]["total"], total, preset)
            self.assertEqual(report["vision_encoder"]["trainable"], 0, preset)
            self.assertEqual(report["state_proj"]["trainable"], report["state_proj"]["total"], preset)
            self.assertEqual(report["action_expert"]["trainable"], report["action_expert"]["total"], preset)
            self.assertEqual(report["vlm_lm_head"]["trainable"], 0, preset)
            self.assertEqual(report["vlm_text_norm"]["trainable"], 0, preset)
        frozen_vlm = ("vision_encoder", "connector", "vlm_embed_tokens", "vlm_text_layers")
        for preset in ("A", "C"):
            for group in frozen_vlm:
                self.assertEqual(reports[preset][group]["trainable"], 0, (preset, group))
        # routing does not change the trainable set (the last tiny layer is a cross layer in both)
        self.assertEqual(reports["A"]["total"]["trainable"], reports["C"]["total"]["trainable"])
        self.assertEqual(reports["B"]["total"]["trainable"], reports["D"]["total"]["trainable"])
        # update_vlm adds exactly: connector + embed_tokens + live text-layer parameters
        policy_b = self.policy_for("B")
        text_layers = policy_b.model.vlm_with_expert.get_vlm_model().text_model.layers
        last = text_layers[-1]
        dead = sum(
            p.numel()
            for m in (last.self_attn.q_proj, last.self_attn.o_proj, last.mlp, last.post_attention_layernorm)
            for p in m.parameters()
        )
        live_text = reports["B"]["vlm_text_layers"]["total"] - dead
        self.assertEqual(reports["B"]["vlm_text_layers"]["trainable"], live_text)
        self.assertEqual(
            reports["B"]["total"]["trainable"] - reports["A"]["total"]["trainable"],
            reports["B"]["connector"]["total"] + reports["B"]["vlm_embed_tokens"]["total"] + live_text,
        )
        # hand-computed sizes for the tiny model
        self.assertEqual(reports["A"]["state_proj"]["total"], 32 * 64 + 64)
        self.assertEqual(reports["A"]["action_in_proj"]["total"], 32 * 32 + 32)
        self.assertEqual(reports["A"]["action_out_proj"]["total"], 32 * 32 + 32)

    # -- routing and numerics ------------------------------------------------------------------

    def _train_forward(self, policy):
        policy.train()
        policy.zero_grad(set_to_none=True)
        loss, _ = policy.forward(dict(self.batch), noise=self.noise.clone(), time=self.time.clone())
        return loss

    def test_all_is_identical_to_upstream(self):
        native = copy.deepcopy(self.base_policy)
        policy = self.policy_for("A")
        with helpers.AttentionRecorder(native.model.vlm_with_expert) as native_rec:
            native_loss = self._train_forward(native)
        with helpers.AttentionRecorder(policy.model.vlm_with_expert) as rec:
            loss = self._train_forward(policy)
        self.assertTrue(torch.equal(native_loss, loss))
        self.assertEqual(len(native_rec.records), len(rec.records))
        for a, b in zip(native_rec.records, rec.records, strict=True):
            self.assertTrue(torch.equal(a["mask"], b["mask"]))
            self.assertEqual(a["k_shape"], b["k_shape"])
        native_loss.backward()
        loss.backward()
        for (name, p_native), (_, p) in zip(native.named_parameters(), policy.named_parameters(), strict=True):
            self.assertEqual(p_native.grad is None, p.grad is None, name)
            if p.grad is not None:
                self.assertTrue(torch.equal(p_native.grad, p.grad), name)
        native_actions = native.predict_action_chunk(dict(self.batch), noise=self.noise.clone())
        actions = policy.predict_action_chunk(dict(self.batch), noise=self.noise.clone())
        self.assertTrue(torch.equal(native_actions, actions))

    def test_cross_only_masks_prefix_in_joint_layers_only_training(self):
        policy_all = self.policy_for("A")
        policy_cross = self.policy_for("C")
        with helpers.AttentionRecorder(policy_all.model.vlm_with_expert) as rec_all:
            loss_all = self._train_forward(policy_all)
        with helpers.AttentionRecorder(policy_cross.model.vlm_with_expert) as rec_cross:
            loss_cross = self._train_forward(policy_cross)
        self.assertFalse(torch.equal(loss_all, loss_cross))
        by_all, by_cross = rec_all.by_layer(), rec_cross.by_layer()
        suffix_len = policy_all.config.chunk_size
        for layer in range(self.num_layers):
            records_all, records_cross = by_all[layer], by_cross[layer]
            self.assertEqual(len(records_all), len(records_cross))
            if sc.is_cross_attention_layer(layer, "cross_attn", 2):
                # prefix branch + expert branch, both untouched by the routing knob
                self.assertEqual([r["q_shape"][1] for r in records_cross], [r["q_shape"][1] for r in records_all])
                for a, c in zip(records_all, records_cross, strict=True):
                    self.assertTrue(torch.equal(a["mask"], c["mask"]))
                expert = records_cross[-1]
                prefix_len = expert["k_shape"][1]
                self.assertEqual(expert["q_shape"][1], suffix_len)
                self.assertEqual(expert["mask"].shape[1:], (suffix_len, prefix_len))
            else:
                (a,), (c,) = records_all, records_cross
                total = a["k_shape"][1]
                prefix_len = total - suffix_len
                self.assertEqual(c["k_shape"], a["k_shape"])  # keys still concatenated; access is masked
                self.assertEqual(c["mask"].shape, a["mask"].shape)
                self.assertTrue(torch.equal(c["mask"][:, :prefix_len, :], a["mask"][:, :prefix_len, :]))
                # native: every suffix row sees exactly the valid (non-padded) prefix tokens
                pad_mask = a["mask"][:, :prefix_len, :prefix_len].diagonal(dim1=1, dim2=2)
                self.assertTrue(pad_mask.any() and not pad_mask.all())  # the fixture has padded language tokens
                self.assertTrue(torch.equal(a["mask"][:, prefix_len:, :prefix_len], pad_mask[:, None, :].expand(-1, suffix_len, -1)))
                self.assertFalse(c["mask"][:, prefix_len:, :prefix_len].any())
                causal = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool))
                self.assertTrue(torch.equal(c["mask"][0, prefix_len:, prefix_len:], causal))
                self.assertTrue(torch.equal(a["mask"][0, prefix_len:, prefix_len:], causal))

    def test_cross_only_masks_prefix_in_joint_layers_only_inference(self):
        policy_all = self.policy_for("A")
        policy_cross = self.policy_for("C")
        with helpers.AttentionRecorder(policy_all.model.vlm_with_expert) as rec_all:
            actions_all = policy_all.predict_action_chunk(dict(self.batch), noise=self.noise.clone())
        with helpers.AttentionRecorder(policy_cross.model.vlm_with_expert) as rec_cross:
            actions_cross = policy_cross.predict_action_chunk(dict(self.batch), noise=self.noise.clone())
        self.assertEqual(actions_all.shape, actions_cross.shape)
        self.assertFalse(torch.equal(actions_all, actions_cross))
        self.assertEqual(len(rec_all.records), len(rec_cross.records))
        suffix_len = policy_all.config.chunk_size
        fill_calls = self.num_layers
        # KV-cache fill pass: prefix only, identical
        for a, c in zip(rec_all.records[:fill_calls], rec_cross.records[:fill_calls], strict=True):
            self.assertTrue(torch.equal(a["mask"], c["mask"]))
            self.assertEqual(a["q_shape"][1], a["k_shape"][1])  # prefix-only: query and key lengths match
        pad_mask = rec_all.records[0]["mask"].diagonal(dim1=1, dim2=2)
        # denoising steps: joint layers hide the prefix, cross layers unchanged
        for a, c in zip(rec_all.records[fill_calls:], rec_cross.records[fill_calls:], strict=True):
            self.assertEqual(a["layer"], c["layer"])
            self.assertEqual(a["q_shape"][1], suffix_len)
            if sc.is_cross_attention_layer(a["layer"], "cross_attn", 2):
                self.assertTrue(torch.equal(a["mask"], c["mask"]))
                self.assertEqual(c["k_shape"][1], a["k_shape"][1])
            else:
                prefix_len = a["k_shape"][1] - suffix_len
                self.assertTrue(torch.equal(a["mask"][:, :, :prefix_len], pad_mask[:, None, :].expand(-1, suffix_len, -1)))
                self.assertFalse(c["mask"][:, :, :prefix_len].any())
                self.assertTrue(torch.equal(a["mask"][:, :, prefix_len:], c["mask"][:, :, prefix_len:]))

    def test_cross_only_leaves_vlm_stream_unchanged(self):
        cache_all = helpers.prefix_kv_cache(self.policy_for("A"), dict(self.batch))
        cache_cross = helpers.prefix_kv_cache(self.policy_for("C"), dict(self.batch))
        self.assertEqual(cache_all.keys(), cache_cross.keys())
        for layer in cache_all:
            self.assertTrue(torch.equal(cache_all[layer][0], cache_cross[layer][0]), layer)
            self.assertTrue(torch.equal(cache_all[layer][1], cache_cross[layer][1]), layer)

    def test_backward_reaches_exactly_the_trainable_parameters(self):
        for preset in sc.PRESETS:
            policy = self.policy_for(preset)
            loss = self._train_forward(policy)
            self.assertTrue(torch.isfinite(loss), preset)
            loss.backward()
            norms = helpers.grad_norms(policy)
            for name, parameter in policy.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(norms[name], (preset, name))
                    self.assertGreater(norms[name], 0.0, (preset, name))
                else:
                    self.assertIsNone(norms[name], (preset, name))
            self.assertGreater(norms["model.state_proj.weight"], 0.0, preset)
            vision_grads = [norms[n] for n in norms if ".vision_model." in n]
            self.assertTrue(all(g is None for g in vision_grads), preset)


if __name__ == "__main__":
    unittest.main()
