# A/B/C/D training recipe — LIBERO-Spatial, apples-to-apples

Status: defined and dry-tested (loader, hooks, fingerprints, routing verification); **no training run**.
Sources: pinned LeRobot `8515d45` (`lerobot/scripts/lerobot_train.py`, `configs/train.py`,
`policies/factory.py`, `datasets/`), the official LIBERO doc page at that commit, the
`HuggingFaceVLA/libero` dataset metadata, and a step-time benchmark on this RTX 5090.

## 0. Commands

```bash
export ACCELERATE_MIXED_PRECISION=bf16 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BACKBONE=/workspace/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467
for P in A B C D; do
  uv run --frozen python src/train_semantic.py \
    --semantic.preset=$P --semantic.trainable_fp32=true --semantic.episodes=1261-1692 \
    --policy.type=smolvla --policy.load_vlm_weights=true --policy.vlm_model_name=$BACKBONE \
    --policy.num_vlm_layers=0 --policy.expert_width_multiplier=0.5 --policy.num_expert_layers=-1 \
    --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=HuggingFaceVLA/libero --dataset.revision=86958911c0f959db2bbbdb107eb3e17c5f9c798e \
    --seed=1000 --batch_size=32 --steps=20000 --save_freq=5000 --eval_freq=0 --log_freq=100 --num_workers=8 \
    --output_dir=outputs/train/spatial_$P --job_name=spatial_$P --wandb.enable=false
done
```

Only `--semantic.*` differs between the four runs. Everything else is LeRobot's official `lerobot-train`
path; `train_semantic.py` hooks it at exactly two points (§8).

## 1. What the official LIBERO training path does (pinned code)

Official doc command (`docs/source/libero.mdx` @ `8515d45`):
`lerobot-train --policy.type=smolvla --policy.load_vlm_weights=true --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_10 --steps=100000 --batch_size=4 …`

- `make_policy(cfg.policy, ds_meta=dataset.meta)` (`lerobot_train.py:281`): a **fresh `SmolVLAConfig`**
  whose `input_features`/`output_features` come from the dataset (`dataset_to_policy_features`), and
  whose normalization stats come from `meta/stats.json` via `make_pre_post_processors`.
- `load_vlm_weights=true` → `AutoModelForImageTextToText.from_pretrained(vlm_model_name, bf16)`;
  the action expert is built by `AutoModel.from_config` from the *global torch RNG* — i.e. random,
  fully determined by `--seed` (`set_seed` seeds `random`, `numpy`, `torch`, CUDA at
  `lerobot_train.py:231`, before the policy is built).
- Sampler: `EpisodeAwareSampler(episode_indices_to_use=dataset.episodes, drop_n_last_frames=0,
  shuffle=True, seed=cfg.seed)` — deterministic order given the seed; `dataset.episodes` restricts to
  the subset and `LeRobotDataset` downloads only the parquet files those episodes need.
- Optimizer/scheduler: `use_policy_training_preset=True` → SmolVLA's presets (§5).
- Precision: the `Accelerator` is created **without** `mixed_precision`, so `policy.use_amp` has no
  effect; `accelerator.autocast()` is a no-op unless `ACCELERATE_MIXED_PRECISION` is set. Default
  parameters: VLM bf16, expert bf16 (config dtype inherited from the backbone), projections fp32.
- Checkpoints: `outputs/train/<job>/checkpoints/<step:06d>/{pretrained_model/{config.json,
  model.safetensors, policy_preprocessor*, policy_postprocessor*}, training_state/…}` + `last` link.
- `HuggingFaceVLA/smolvla_libero` (the pinned baseline checkpoint) matches this path with
  `num_vlm_layers=0` (all 32 layers), `expert_width_multiplier=0.5`, `num_expert_layers=-1`,
  `pad_language_to=longest` — its model card declares `base_model: lerobot/smolvla_base`, but its
  architecture cannot load `smolvla_base` weights (16 vs 32 VLM layers, 0.75 vs 0.5 width), so its
  expert was trained from scratch on top of the pretrained VLM. `SmolVLM2-500M-Instruct` and
  `SmolVLM2-500M-Video-Instruct` are the same Hub repo (redirect, sha `7b375e1`).

## 2. Exact input/output schema

Dataset `HuggingFaceVLA/libero@8695891` (LeRobot v3, panda, 10 fps, 1693 episodes, 273,465 frames, 40 tasks):

| Key | Dataset | Policy feature | Processing |
| --- | --- | --- | --- |
| `observation.images.image` (agentview) | image 256×256×3 uint8 | VISUAL (3,256,256) float [0,1] | `resize_with_pad` → 512×512 → SigLIP → 64 tokens; `[-1,1]` |
| `observation.images.image2` (wrist) | image 256×256×3 | VISUAL (3,256,256) | same |
| `observation.state` | float32[8] | STATE (8,) → padded to 32 | MEAN_STD (dataset stats) → `state_proj` |
| `task` | string (per episode) | language | `NewLineTaskProcessorStep` + tokenizer, `pad_language_to=longest`, max 48 |
| `action` | float32[7] | ACTION (7,) → padded to 32 | MEAN_STD; chunk of 50 future actions (`action_delta_indices` 0..49 = 5 s at 10 fps); frames past the episode end are masked via `action_is_pad` |

Prefix = 64 + 64 + L(≈20) + 1 ≈ 149 tokens; suffix = 50 action tokens. Output at eval: 7-d action,
unnormalized by the saved postprocessor. The LIBERO env (`envs/libero.py`) emits the same keys at
360×360 (eval) — the policy resizes either way.

## 3. Architecture and initialization

Recommended architecture = the pinned baseline's (`smolvla_libero`): backbone
`SmolVLM2-500M-Video-Instruct@7b375e1`, `num_vlm_layers=0` (32 layers → 16 joint + 16 cross),
`expert_width_multiplier=0.5` (expert hidden 480), `num_expert_layers=-1`, `self_attn_every_n_layers=2`,
`chunk_size=50`. Reason: like-for-like with the only LIBERO-Spatial reference numbers we have
(81.5% reference, 77.0% reproduced) and with `configs/baseline.json`. Cheaper alternative: the
`smolvla_base` shape (`num_vlm_layers=16`, width 0.75) at ~0.65× the step time (§7).

Initialization, identical across A–D by construction:
- VLM (vision encoder, connector, text layers, embeddings): pretrained backbone weights, bf16.
  Verified by `vlm_fingerprint` (SHA-256 of all VLM tensors) recorded in `semantic_control.json`.
- Action expert + `state_proj` + `action_in/out_proj` + `action_time_mlp_*`: random init from
  `--seed=1000`; the semantic control is installed *after* construction and never touches weights, so
  the `init_fingerprint` is identical for A–D (unit-tested on the tiny model; the GPU test showed the
  installed policy's weights equal the checkpoint's). Any pilot must assert all four
  `init_fingerprint`s are equal before proceeding.
- Not used: `lerobot/smolvla_base`'s pretrained expert. It was trained under native ("all") routing,
  so it would hand A/B an in-distribution expert and C/D an out-of-distribution one (on identical
  inputs the pretrained expert's output changes by ~80% in |sum| under `cross_only`). A seeded
  random expert removes that confound; the cost is lower absolute success than a pretrained-expert
  fine-tune at equal budget.

## 4. The four variants

| | `semantic_layers` | `update_vlm` | Expert reads VLM K/V in | Trainable parameters (32-layer arch) |
| --- | --- | --- | --- | --- |
| A | all | false | all 32 layers | **97,462,960** = expert 96,700,800 (incl. 16×2 cross `k/v_proj`) + `state_proj` 31,680 + `action_in_proj` 15,840 + `action_out_proj` 15,392 + `action_time_mlp` 692,160 |
| B | all | true | all 32 layers | **461,985,520** = A + connector 11,796,480 + `embed_tokens` 47,308,800 + text layers 305,417,280 (314,634,240 minus layer 31's dead q/o/MLP/post-norm 9,216,960) |
| C | cross_only | false | odd layers 1,3,…,31 (16) | = A |
| D | cross_only | true | odd layers (16) | = B |

Always frozen: vision encoder 86,433,024; `lm_head` 47,308,800; final text norm 960. Total 604,934,176.
`cross_only` masks the prefix columns for the 50 action rows in the 16 joint layers (exactly zero
probability and gradient); the VLM stream is untouched (KV cache bitwise identical, GPU-tested).

## 5. Optimizer, schedule, precision (identical for A–D)

- AdamW, lr 1e-4, betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, grad-clip 10 (SmolVLA preset).
- Cosine decay with 1,000 warm-up steps, 30,000 decay steps to 2.5e-6 (preset). At `--steps=20000`
  the run stops mid-decay (lr ≈ 2.7e-5); acceptable because identical across variants — set
  `--steps=30000` if the full schedule is wanted.
- One parameter group, one LR for VLM and expert in B/D (upstream behaviour). Known risk: 1e-4 is
  high for a pretrained VLM; see §9.
- Precision (recommended): `--semantic.trainable_fp32=true` + `ACCELERATE_MIXED_PRECISION=bf16`:
  every trainable block (expert, projections, and for B/D the text model + connector) is held in
  fp32 master weights with fp32 AdamW states; matmuls run under bf16 autocast; frozen modules stay
  bf16 (a forward hook aligns the vision→connector dtype boundary). Cost +9% (A) / +19% (B) step
  time, +3.4 GiB (B). Alternative = upstream default (bf16 weights *and* bf16 Adam states, no
  autocast); with bf16's 8-bit mantissa, lr·update below ~0.4% of a weight's magnitude is lost.
- Determinism: `--seed=1000` fixes init and sampling order; GPU kernels are not bitwise
  deterministic across runs (leave `cudnn_deterministic` off; it does not cover attention matmuls).

## 6. Dataset subset

LIBERO-Spatial = task indices 30–39 of `HuggingFaceVLA/libero@8695891` = episodes **1261–1692**
(432 episodes, contiguous, 52,970 frames, 35–47 demos per task, mean 122.6 frames/episode).
`--semantic.episodes=1261-1692` expands to `--dataset.episodes=[…]`. Batch 32 → 1,655 steps/epoch →
20k steps ≈ 12 epochs (≈ 640k samples; the official example is 100k × 4 = 400k).
Normalization stats are the dataset-wide `meta/stats.json` (all 1693 episodes; e.g. state std
[0.105, 0.152, 0.379, …] vs. the spatial-only values) — identical for all variants; recomputing
subset stats is optional and must then be applied to all four.

## 7. Pilot cost (measured: fwd+bwd+clip+AdamW on dummy LIBERO-shaped batches, RTX 5090)

| Arch | Variant | batch 16 | batch 32 | batch 64 | fp32-master + bf16 autocast, batch 32 |
| --- | --- | --- | --- | --- | --- |
| 32 layers / 0.5 (recommended) | A (=C) | 273 ms · 7.6 GiB | 321 ms · 13.5 GiB | 545 ms · 25.5 GiB | 349 ms · 13.3 GiB |
| 32 layers / 0.5 | B (=D) | 272 ms · 10.3 GiB | 329 ms · 17.5 GiB | OOM (32 GiB) | 390 ms · 20.9 GiB |
| 16 layers / 0.75 | A | 164 ms · 4.5 GiB | 212 ms · 7.7 GiB | 381 ms · 14.1 GiB | — |
| 16 layers / 0.75 | B | 173 ms · 6.0 GiB | 223 ms · 9.8 GiB | 409 ms · 17.4 GiB | — |

A and B cost the same per step because the VLM backward already runs in A (it feeds `state_proj`);
B only adds weight-gradient accumulation. Pilot estimate, recommended settings (batch 32, 20k steps,
fp32-master + autocast): A/C ≈ 1.9 h, B/D ≈ 2.2 h → **≈ 8.3 GPU-hours for the four runs**, plus
dataloader overhead (PNG decode of 64 images/step; 8 workers expected to keep up) and offline
evaluation: ≈ 100 min per 200-episode LIBERO-Spatial evaluation with the pinned harness protocol
(`num_steps=1`, `n_action_steps=1`) → ≈ 6.7 h for four final checkpoints, more if intermediate
checkpoints (5k/10k/15k) are also evaluated. Peak memory ≤ 21 GiB, so one run at a time.
Dataset download for the subset ≈ 8 GB (the repo is ~37 GB; only the subset's files are fetched).

## 8. Saving and restoring the semantic control (so C/D cannot revert to native routing)

Two hooks in `src/train_semantic.py` on `lerobot.scripts.lerobot_train`:

1. `make_policy` → after LeRobot builds the policy: `install_semantic_control` (routing + trainable
   set), optional fp32 cast, `verify_routing` **by observation** (records the attention masks of a
   dummy forward and checks that action rows see the prefix in joint layers iff `semantic_layers=all`,
   that cross layers are where the dispatch says, and that VLM trainability matches `update_vlm`),
   fingerprints and parameter counts stored on `policy.semantic_metadata`.
2. `save_checkpoint` → after every LeRobot checkpoint, writes
   `checkpoints/<step>/pretrained_model/semantic_control.json`:
   `{semantic_layers, update_vlm, preset, init_fingerprint, vlm_fingerprint, trainable_fp32,
   trainable_parameters, total_parameters, routing{verified, coupled_layers, joint_layers,
   cross_layers, …}, lerobot_version, semantic_control_sha256, step}`.
   Module structure and `model.safetensors` keys are unchanged, so the checkpoint still loads in
   upstream tools — which is exactly why the file plus the loader rules below are required.

Restoring (`semantic_control.load_policy_with_control(checkpoint_dir, control=None, …)`):
- `semantic_control.json` present → it is authoritative; a conflicting explicit `control` raises
  `RoutingError`. Absent and no explicit control → `RoutingError` (native SmolVLA = preset A must be
  stated, never assumed).
- After loading, `verify_routing` runs again on the loaded model; the returned `info`
  (`control`, `control_source`, `checkpoint_metadata`, `init_fingerprint`, `routing.verified`, …)
  must be written into the evaluation metrics. An evaluation result without
  `routing.verified == true` and a matching `preset` is invalid by definition.
- Eval wiring still to do (harness left untouched so far): give `src/evaluate.py` a
  `--checkpoint DIR` mode that calls `load_policy_with_control` (with `--semantic-control` optional),
  stores `info` under `metrics["semantic_control"]`, and puts the preset in the report title; the
  smoke/baseline configs keep working unchanged because they do not use `--checkpoint`.
- Eval backbone: the training config records `vlm_model_name` as the local snapshot path; pass the
  same pinned snapshot (`build_policy(backbone=…)`) so tokenizer and VLM weights are identical.

Tests covering this contract (`tests/test_semantic_control_recipe.py`, CPU): verification accepts
A–D and rejects an uninstalled policy, a mismatched control, a tampered `semantic_layers`, and a
wrong trainable set; fingerprints are equal for equal seeds and across A–D; the loader enforces the
rules above; the hooks install the control, cast to fp32, run a finite forward, and write a file that
round-trips through the loader.

## 9. Confounds and open decisions before the pilot

1. **B/D have 4.7× the trainable parameters and use lr 1e-4 on a pretrained VLM.** Expect drift;
   the clean comparison is "same recipe, different knob", but a VLM-LR sensitivity arm (e.g. 1e-5
   for VLM parameters via a second param group) is the first follow-up if B/D underperform A/C.
2. **Random expert from scratch** lowers absolute success vs. the 81.5%/77.0% reference numbers,
   which came from an unknown budget. Compare variants to each other, not to the reference.
3. **Eval noise**: 200 episodes gives a ±5–6 pt Wilson interval. Differences inside that band need
   more episodes (400) or a second training seed per variant (doubles the ≈8 h).
4. `cross_only` still computes (masked) attention over the prefix in joint layers — no cost saving,
   no correctness issue.
5. `update_vlm=true` trains `embed_tokens` (47 M) and the connector; the vision encoder is frozen,
   so images enter the VLM through a fixed SigLIP + trainable connector. Decide whether embeddings
   should train.
6. Global vs. subset normalization stats (§6). 7. `pad_language_to=longest` (baseline config) vs
   `max_length` (`smolvla_base`); keep `longest` for parity with the baseline harness.
8. Eval protocol for trained models = the harness protocol (`num_steps=1`, `n_action_steps=1`,
   seed 1000, fixed init states), applied identically to A–D; the harness `--checkpoint` mode (§8)
   must exist before the first evaluation.

## 10. Pilot execution notes (added when the staged pilot was set up)

- **Dataset file map is wrong on the Hub** (`HuggingFaceVLA/libero@8695891`): `meta/episodes` claims
  episodes 1261-1692 live in `data/chunk-000/file-055..068`, but those files hold episodes 137-174 and
  the map's indices stop at 68 while the repo has 377 data files. The data files themselves are
  consistent (global `episode_index`, `index`, `task_index`). The true map was rebuilt from each
  parquet footer's column statistics (`scratchpad/fetch_spatial_subset.py`), the 73 files that hold the
  spatial episodes were downloaded into the LeRobot hub cache (7.4 GB), and LeRobot's own
  `isin(episodes)` filter then loads exactly 432 episodes / 52,970 frames.
- **Episode subsets are broken in LeRobot @8515d45**: `EpisodeAwareSampler` yields absolute frame
  indices while the filtered reader expects relative ones. `train_semantic.patch_subset_indexing`
  applies the reader's own absolute→relative map in `LeRobotDataset.__getitem__`.
- **Fixed LR horizon**: `CosineDecayWithWarmupSchedulerConfig.build` rescales warm-up/decay to
  `--steps` when it is below 30k, so staged runs use `--steps=30000` (1k warm-up, 30k decay) and stop
  via `--semantic.stop_at`; `--semantic.save_at` keeps only 2k/5k/10k/…; resume is sample-exact.
- **Two LR groups**: expert/projections 1e-4, VLM (text layers + embeddings + connector) 1e-5, both
  scaled by the same cosine factor. fp32 master weights (`--semantic.trainable_fp32=true`) with
  `ACCELERATE_MIXED_PRECISION=bf16`; saved fp32 tensors are restored as fp32 on resume and at eval.
- **Checkpoint record** (`semantic_control.json`): preset, fingerprints, routing verification, trainable
  count, parameter groups, loss (last / mean of last 100), lr per group, gradient norms (pre-clip total,
  post-clip per group), runtime, peak VRAM, step; the full curve is in `semantic_train_log.jsonl`.
- **Evaluation**: `evaluate.py --checkpoint DIR --semantic-control P` restores the record, re-verifies
  routing on the loaded model, and aborts otherwise; `results/pilot/*.json` carry `semantic_control`.
- Driver: `src/pilot.py train|eval|summarize`. Trainable parameters on the fresh 32-layer model:
  A/C 97,451,872, B/D 461,974,432 (11,088 fewer than the released-checkpoint accounting in §4; same
  across variants, verified by identical init fingerprints).
