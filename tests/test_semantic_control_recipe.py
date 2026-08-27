"""CPU tests for the save/restore contract: routing verification by observation, fingerprints,
the guarded loader, and the training-wrapper hooks (tiny random SmolVLA, no dataset)."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests import helpers
from tests.helpers import sc

sys.path.insert(0, str(helpers.SRC))
import train_semantic  # noqa: E402


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backbone = helpers.find_backbone_snapshot()
        if backbone is None:
            raise unittest.SkipTest("No cached SmolVLM2 snapshot to borrow tokenizer files from")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tiny_dir = helpers.build_tiny_vlm_dir(Path(cls._tmp.name) / "tiny_vlm", backbone)
        cls.base_policy = helpers.make_tiny_policy(cls.tiny_dir, seed=0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def policy_for(self, preset: str):
        return sc.install_semantic_control(copy.deepcopy(self.base_policy), sc.SemanticControlConfig.from_preset(preset))

    # -- verification by observation -------------------------------------------------------------

    def test_verify_routing_accepts_every_preset(self):
        for preset in sc.PRESETS:
            info = sc.verify_routing(self.policy_for(preset))
            self.assertTrue(info["verified"], preset)
            self.assertEqual(info["joint_layers"], [0, 2], preset)
            self.assertEqual(info["cross_layers"], [1, 3], preset)
            self.assertEqual(info["coupled_layers"], [0, 1, 2, 3] if preset in "AB" else [1, 3], preset)

    def test_verify_routing_rejects_uninstalled_policy(self):
        with self.assertRaises(sc.RoutingError):
            sc.verify_routing(copy.deepcopy(self.base_policy), sc.SemanticControlConfig.from_preset("C"))
        with self.assertRaises(sc.RoutingError):
            sc.verify_routing(copy.deepcopy(self.base_policy))  # nothing installed, nothing requested

    def test_verify_routing_rejects_mismatch_and_tampering(self):
        policy = self.policy_for("A")
        with self.assertRaises(sc.RoutingError):
            sc.verify_routing(policy, sc.SemanticControlConfig.from_preset("C"))
        tampered = self.policy_for("C")
        tampered.model.vlm_with_expert.semantic_layers = "all"  # config says C, routing is native
        with self.assertRaises(sc.RoutingError):
            sc.verify_routing(tampered, sc.SemanticControlConfig.from_preset("C"))
        frozen_wrong = self.policy_for("D")
        for p in frozen_wrong.model.vlm_with_expert.vlm.parameters():
            p.requires_grad_(False)
        with self.assertRaises(sc.RoutingError):
            sc.verify_routing(frozen_wrong)

    # -- fingerprints ----------------------------------------------------------------------------

    def test_fingerprints_identify_identical_initialization(self):
        same = helpers.make_tiny_policy(self.tiny_dir, seed=0)
        other = helpers.make_tiny_policy(self.tiny_dir, seed=1)
        self.assertEqual(sc.init_fingerprint(self.base_policy), sc.init_fingerprint(same))
        self.assertNotEqual(sc.init_fingerprint(self.base_policy), sc.init_fingerprint(other))
        fingerprints = {preset: sc.init_fingerprint(self.policy_for(preset)) for preset in sc.PRESETS}
        self.assertEqual(len(set(fingerprints.values())), 1)  # routing/trainability never touch weights
        self.assertEqual(sc.vlm_fingerprint(self.base_policy), sc.vlm_fingerprint(self.policy_for("D")))

    # -- guarded loading -------------------------------------------------------------------------

    def _save(self, policy, directory: Path):
        policy.save_pretrained(directory)
        return directory

    def test_load_requires_explicit_or_saved_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._save(self.base_policy, Path(tmp) / "pretrained_model")
            with self.assertRaises(sc.RoutingError):
                sc.load_policy_with_control(directory, device="cpu")
            policy, control, info = sc.load_policy_with_control(
                directory, sc.SemanticControlConfig.from_preset("A"), device="cpu"
            )
            self.assertEqual(control.preset, "A")
            self.assertEqual(info["control_source"], "argument")
            self.assertTrue(info["routing"]["verified"])
            self.assertEqual(sc.init_fingerprint(policy), sc.init_fingerprint(self.base_policy))

    def test_saved_control_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._save(self.base_policy, Path(tmp) / "pretrained_model")
            sc.save_semantic_control(sc.SemanticControlConfig.from_preset("C"), directory, {"step": 7})
            policy, control, info = sc.load_policy_with_control(directory, device="cpu")
            self.assertEqual(control.preset, "C")
            self.assertEqual(info["control_source"], "checkpoint_file")
            self.assertEqual(info["checkpoint_metadata"]["step"], 7)
            self.assertEqual(info["routing"]["coupled_layers"], [1, 3])
            self.assertEqual(policy.model.vlm_with_expert.coupled_layers(), [1, 3])
            with self.assertRaises(sc.RoutingError):
                sc.load_policy_with_control(directory, sc.SemanticControlConfig.from_preset("A"), device="cpu")
            same = sc.load_policy_with_control(directory, sc.SemanticControlConfig.from_preset("C"), device="cpu")
            self.assertEqual(same[1].preset, "C")

    # -- training wrapper ------------------------------------------------------------------------

    def test_parse_semantic_args(self):
        control, args, rest = train_semantic.parse_semantic_args(
            ["--semantic.preset", "D", "--steps=10", "--semantic.trainable_fp32", "true", "--batch_size=4"]
        )
        self.assertEqual(control.preset, "D")
        self.assertTrue(args.trainable_fp32)
        self.assertEqual(rest, ["--steps=10", "--batch_size=4"])
        control, _, _ = train_semantic.parse_semantic_args(["--semantic.semantic_layers", "cross_only"])
        self.assertEqual(control.preset, "C")
        with self.assertRaises(SystemExit):
            train_semantic.parse_semantic_args(["--semantic.preset", "A", "--semantic.update_vlm", "true"])
        with self.assertRaises(SystemExit):
            train_semantic.parse_semantic_args(["--steps=10"])

    def test_hooks_install_control_and_write_control_file(self):
        import importlib

        train_module = importlib.import_module("lerobot.scripts.lerobot_train")
        saved = (train_module.make_policy, train_module.save_checkpoint)
        base = self.base_policy
        calls = []

        def stub_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
            return copy.deepcopy(base)

        def stub_save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None, **kwargs):
            calls.append((step, sorted(kwargs)))
            policy.save_pretrained(Path(checkpoint_dir) / "pretrained_model")

        control = sc.SemanticControlConfig.from_preset("D")
        try:
            train_module.make_policy = stub_make_policy
            train_module.save_checkpoint = stub_save_checkpoint
            module, originals = train_semantic.install_hooks(control, trainable_fp32=True)
            self.assertIs(originals["make_policy"], stub_make_policy)
            policy = module.make_policy(cfg=None, ds_meta=object())
            self.assertEqual(policy.semantic_control, control)
            self.assertTrue(policy.semantic_metadata["routing"]["verified"])
            self.assertEqual(policy.semantic_metadata["routing"]["coupled_layers"], [1, 3])
            self.assertTrue(all(p.dtype == torch.float32 for p in policy.parameters() if p.requires_grad))
            self.assertEqual(policy.semantic_metadata["init_fingerprint"], sc.init_fingerprint(policy))
            loss, _ = policy.forward(helpers.make_batch(policy))  # fp32 trainable + frozen modules run together
            self.assertTrue(torch.isfinite(loss))
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint_dir = Path(tmp) / "000010"
                module.save_checkpoint(checkpoint_dir, 10, None, policy, None, None, preprocessor=None)
                self.assertEqual(calls, [(10, ["preprocessor"])])
                saved_file = json.loads((checkpoint_dir / "pretrained_model" / sc.SEMANTIC_CONTROL_FILE).read_text())
                self.assertEqual(saved_file["preset"], "D")
                self.assertEqual(saved_file["step"], 10)
                self.assertEqual(saved_file["init_fingerprint"], policy.semantic_metadata["init_fingerprint"])
                self.assertIn("semantic_control_sha256", saved_file)
                restored, restored_control, info = sc.load_policy_with_control(
                    checkpoint_dir / "pretrained_model", device="cpu"
                )
                self.assertEqual(restored_control.preset, "D")
                self.assertEqual(info["routing"]["coupled_layers"], [1, 3])
                self.assertEqual(info["checkpoint_metadata"]["init_fingerprint"], saved_file["init_fingerprint"])
        finally:
            train_semantic.remove_hooks(train_module, originals)
            train_module.make_policy, train_module.save_checkpoint = saved

    def test_episode_range_expansion(self):
        self.assertEqual(train_semantic.expand_episode_range("3-5,9"), [3, 4, 5, 9])
        _, _, rest = train_semantic.parse_semantic_args(["--semantic.preset", "A", "--semantic.episodes", "1261-1263"])
        self.assertEqual(rest, ["--dataset.episodes=[1261,1262,1263]"])


if __name__ == "__main__":
    unittest.main()


class PilotHookTests(unittest.TestCase):
    """Two-LR parameter groups, save_at/stop_at control, and fp32-dtype round trips (tiny model)."""

    @classmethod
    def setUpClass(cls):
        backbone = helpers.find_backbone_snapshot()
        if backbone is None:
            raise unittest.SkipTest("No cached SmolVLM2 snapshot to borrow tokenizer files from")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tiny_dir = helpers.build_tiny_vlm_dir(Path(cls._tmp.name) / "tiny_vlm", backbone)
        cls.base_policy = helpers.make_tiny_policy(cls.tiny_dir, seed=0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_parameter_groups(self):
        policy_b = sc.install_semantic_control(copy.deepcopy(self.base_policy), sc.SemanticControlConfig.from_preset("B"))
        groups = train_semantic.parameter_groups(policy_b, vlm_lr=1e-5)
        self.assertEqual([g["name"] for g in groups], ["expert", "vlm"])
        self.assertEqual(groups[1]["lr"], 1e-5)
        self.assertNotIn("lr", groups[0])
        trainable = sum(p.numel() for p in policy_b.parameters() if p.requires_grad)
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]), trainable)
        self.assertTrue(all(p.requires_grad for g in groups for p in g["params"]))
        policy_a = sc.install_semantic_control(copy.deepcopy(self.base_policy), sc.SemanticControlConfig.from_preset("A"))
        self.assertEqual([g["name"] for g in train_semantic.parameter_groups(policy_a, 1e-5)], ["expert"])
        opt = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
        self.assertEqual([g["lr"] for g in opt.param_groups], [1e-4, 1e-5])

    def test_parse_pilot_args(self):
        control, args, rest = train_semantic.parse_semantic_args(
            ["--semantic.preset=B", "--semantic.vlm_lr=1e-5", "--semantic.stop_at=5000", "--semantic.save_at=2000,10000", "--steps=30000"]
        )
        self.assertEqual(control.preset, "B")
        self.assertEqual(args.vlm_lr, 1e-5)
        self.assertEqual(args.stop_at, 5000)
        self.assertEqual(args.save_at, [2000, 5000, 10000])  # stop_at is always saved
        self.assertEqual(rest, ["--steps=30000"])

    def test_fp32_masters_survive_save_and_reload(self):
        policy = sc.install_semantic_control(copy.deepcopy(self.base_policy), sc.SemanticControlConfig.from_preset("D"))
        train_semantic.cast_trainable_to_fp32(policy)
        with torch.no_grad():
            for p in policy.parameters():
                if p.requires_grad:
                    p.add_(1e-5)  # a perturbation below bf16 resolution for O(1) weights
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "pretrained_model"
            policy.save_pretrained(directory)
            sc.save_semantic_control(policy.semantic_control, directory)
            restored, _, info = sc.load_policy_with_control(directory, device="cpu")
            self.assertGreater(info["restored_dtype_parameters"], 0)
            for (name, p), (_, q) in zip(policy.named_parameters(), restored.named_parameters(), strict=True):
                self.assertEqual(p.dtype, q.dtype, name)
                self.assertTrue(torch.equal(p.detach(), q.detach()), name)
            loss, _ = restored.forward(helpers.make_batch(restored))
            self.assertTrue(torch.isfinite(loss))

    def test_save_at_filters_and_stop_at_exits(self):
        import importlib

        train_module = importlib.import_module("lerobot.scripts.lerobot_train")
        saved = {n: getattr(train_module, n) for n in ("make_policy", "save_checkpoint", "update_last_checkpoint", "update_policy", "make_optimizer_and_scheduler")}
        base = self.base_policy
        calls, links = [], []
        try:
            train_module.make_policy = lambda cfg, ds_meta=None, env_cfg=None, rename_map=None: copy.deepcopy(base)
            train_module.save_checkpoint = lambda checkpoint_dir, step, cfg, policy, optimizer, scheduler=None, **kw: (calls.append(step), policy.save_pretrained(Path(checkpoint_dir) / "pretrained_model"))
            train_module.update_last_checkpoint = lambda checkpoint_dir: links.append(checkpoint_dir)
            control = sc.SemanticControlConfig.from_preset("C")
            module, originals = train_semantic.install_hooks(control, trainable_fp32=False, stop_at=2, save_at=[2])
            policy = module.make_policy(cfg=None, ds_meta=object())
            optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
            with tempfile.TemporaryDirectory() as tmp:
                cfg = type("Cfg", (), {"output_dir": Path(tmp), "save_freq": 1})()
                train_semantic.STATE["step"] = 1
                module.save_checkpoint(Path(tmp) / "000001", 1, cfg, policy, optimizer, None)
                module.update_last_checkpoint(Path(tmp) / "000001")
                self.assertEqual(calls, [])  # step 1 not in save_at -> skipped, link not updated
                self.assertEqual(links, [])
                train_semantic.STATE["step"] = 2
                module.save_checkpoint(Path(tmp) / "000002", 2, cfg, policy, optimizer, None)
                self.assertEqual(calls, [2])
                record = json.loads((Path(tmp) / "000002" / "pretrained_model" / sc.SEMANTIC_CONTROL_FILE).read_text())
                self.assertEqual(record["step"], 2)
                self.assertIn("training", record)
                self.assertEqual(record["parameter_groups"]["expert"]["lr"], "default")
                with self.assertRaises(SystemExit):
                    module.update_last_checkpoint(Path(tmp) / "000002")
                self.assertEqual(links, [Path(tmp) / "000002"])
        finally:
            train_semantic.remove_hooks(train_module, originals)
            for n, v in saved.items():
                setattr(train_module, n, v)
