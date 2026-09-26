# Semantic controls for VLA action experts

In SmolVLA the action expert sees the vision-language model through exactly one channel: at every layer, its
attention reads the VLM's key/value projections of the image, language and state tokens. There is no residual mixing,
no FiLM, and no use of the VLM's final hidden state ([`docs/architecture.md`](docs/architecture.md)). That makes the
coupling between "semantics" and "control" something we can switch on and off precisely, in two independent ways:

- **Forward (`semantic_layers`)**: which layers let the action expert read VLM keys/values. `all` is native SmolVLA
  (32 layers). `cross_only` keeps only the 16 cross-attention layers and masks the VLM prefix out of the 16 joint
  self-attention layers for the action tokens (exactly zero attention probability and zero gradient).
- **Backward (`update_vlm`)**: whether the action loss may update the VLM (text layers, token embeddings, connector).
  The vision encoder is always frozen and `state_proj` is always trained.

We train the resulting 2x2 grid on LIBERO-Spatial with everything else held fixed, and compare the four policies on
the same 200 matched simulator episodes.

| preset | `semantic_layers` | `update_vlm` | expert reads VLM K/V in | trainable parameters |
| --- | --- | --- | --- | ---: |
| A | all | false | 32 layers | 97.5 M |
| B | all | true | 32 layers | 462.0 M |
| C | cross_only | false | 16 cross-attention layers | 97.5 M |
| D | cross_only | true | 16 cross-attention layers | 462.0 M |

## Results

LIBERO-Spatial success rate (10 tasks, 20 matched episodes each; Wilson 95% intervals at 30k):

| preset | 10k steps\* | 20k steps | 30k steps |
| --- | ---: | ---: | ---: |
| A: all, frozen VLM | 56% | 67.5% | 70.5% [63.8, 76.4] |
| B: all, trained VLM | 52% | 76.0% | **78.5%** [72.3, 83.6] |
| C: cross-only, frozen VLM | 62% | 67.5% | 66.0% [59.2, 72.2] |
| D: cross-only, trained VLM | 64% | 73.5% | 73.0% [66.5, 78.7] |
| released `smolvla_libero` | | | 82.0% [76.1, 86.7] |

\* 10k screening: 10 episodes per task, before the matched protocol existed; all four are within noise.

2x2 factorial effects in percentage points, with 95% intervals from a task-stratified bootstrap over complete
(A, B, C, D) episode tuples and p-values from a paired sign-flip test:

| effect | 20k steps | 30k steps |
| --- | ---: | ---: |
| update VLM: ((B-A) + (D-C)) / 2 | +7.25 [+1.2, +13.0], p = 0.026 | **+7.5 [+2.0, +13.0], p = 0.017** |
| cross-only routing: ((C-A) + (D-B)) / 2 | -1.25 [-7.0, +4.5], p = 0.75 | -5.0 [-11.0, +1.0], p = 0.14 |
| interaction: (D-C) - (B-A) | -2.5 [-14.0, +9.5], p = 0.75 | -1.0 [-11.0, +9.0], p = 0.93 |

What this shows so far:

1. **Letting the action loss update the VLM helps.** The effect is +7.5 points at 30k, the same size as at 20k
   (+7.25), and it appears under both routings (B-A = +8.0, D-C = +7.0 at 30k).
2. **Restricting the expert to the cross-attention layers does not help, and may hurt.** The routing effect grew from
   -1.25 to -5.0 points between 20k and 30k, but its interval still includes zero. Cross-only runs also end with higher
   training loss (C 0.075 vs A 0.057, D 0.063 vs B 0.050; mean over the last 1k steps).
3. **The two knobs do not interact** measurably (-1.0 points).
4. **The best configuration, B, is statistically indistinguishable from the released checkpoint** on the same 200
   episodes: -3.5 points [-11.0, +4.0], 27 wins / 34 losses / 139 ties, exact McNemar p = 0.44. B trains its expert
   from scratch for 30k steps on LIBERO-Spatial demonstrations only; the released checkpoint's training budget is not
   published.

Caveats: one training seed per configuration; 200 episodes per configuration; one LIBERO suite. Per-task effects are
heterogeneous (on task 3 cross-only routing costs 27.5 points; on task 5 updating the VLM costs 27.5 points), and with
only 10 tasks the task-clustered intervals include zero for every effect. Full tables:
[`factorial_30k.md`](results/pilot/factorial_30k.md), [`paired_30000_e20_matched.md`](results/pilot/paired_30000_e20_matched.md),
[`released_vs_B30k_matched.md`](results/pilot/released_vs_B30k_matched.md).

## Setup

**Model.** SmolVLA from LeRobot `8515d45`, in the shape of `HuggingFaceVLA/smolvla_libero`: the pretrained
`SmolVLM2-500M-Video-Instruct` backbone with all 32 text layers, and a 32-layer action expert of width 0.5
(hidden size 480). Even layers are joint self-attention over VLM and action tokens; odd layers are cross-attention
into VLM keys/values through a learned 320 -> 320 projection. Preset A is bitwise identical to upstream SmolVLA
(loss, actions, every gradient and attention mask; tested on GPU).

**Training** is identical for A-D apart from the two knobs ([`docs/recipe.md`](docs/recipe.md)):

- The action expert is initialised from scratch with seed 1000; all four runs start from the same weights (checked by
  an initialisation fingerprint). The pretrained `smolvla_base` expert is not used because it was trained with native
  routing, which would favour A and B.
- Data: LIBERO-Spatial demonstrations from `HuggingFaceVLA/libero` (episodes 1261-1692: 432 episodes, 52,970 frames).
- AdamW, batch 32, 30k steps (about 18 epochs), 1k warm-up and cosine decay to 2.5e-6. Learning rate 1e-4 for the
  expert and projections, 1e-5 for the VLM. fp32 master weights with bf16 autocast.
- One RTX 5090: about 0.43 s/step for A/C (11.5 GiB peak) and 0.48-0.50 s/step for B/D (18.6 GiB), so 3.6-4.2 hours
  per 30k-step run.

**Evaluation.** 10 LIBERO-Spatial tasks x 20 episodes, 280 steps max, batch 5, one flow-matching step
(`num_steps=1`) and one action per inference (`n_action_steps=1`). Initial states and environment seeds are bound to
the episode index, flow-matching noise is seeded per (task, episode, step), and cuDNN runs deterministically with TF32
off, so every policy faces exactly the same 200 episodes and repeated runs reproduce outcomes. Comparisons are
therefore paired: bootstrap intervals, exact McNemar tests, and the episode-clustered factorial analysis above.

**Baseline reproduction.** On the original reference protocol, the released checkpoint scores 154/200 (77.0%,
[70.7, 82.3]) against a reference of 163/200 (81.5%), inside the 5-point reproduction band
([`results/REPORT.md`](results/REPORT.md)).

Two data-loading issues had to be fixed along the way. The `meta/episodes` file map of `HuggingFaceVLA/libero` at
revision `8695891` points the LIBERO-Spatial episodes at the wrong parquet files, so
[`fetch_libero_subset.py`](src/fetch_libero_subset.py) rebuilds it from the parquet footers. LeRobot's episode-subset
sampler also yields absolute frame indices where the filtered dataset expects relative ones; `train_semantic.py`
patches this at runtime.

## Reproduce

Requires Linux, an NVIDIA GPU (the setup check expects an RTX 5090), uv 0.9.0 and OSMesa. All Python dependencies,
including the LeRobot commit, LIBERO, MuJoCo and PyTorch, are locked in `uv.lock`.

```bash
sudo apt-get install -y libosmesa6 libglfw3 libgl1
./scripts/setup.sh                                   # uv sync + cache the pinned checkpoints
./scripts/smoke.sh                                   # task 0, 2 episodes
./scripts/evaluate.sh                                # released checkpoint, reference protocol -> results/REPORT.md
```

Semantic-control interface and tests:

```bash
uv run --frozen python src/semantic_control.py cache                    # smolvla_base + backbone snapshots
uv run --frozen python src/fetch_libero_subset.py                       # LIBERO-Spatial training data
uv run --frozen python src/semantic_control.py report --preset C        # parameter counts per module group
uv run --frozen python -m pytest tests/test_semantic_control_unit.py    # CPU, tiny random model
uv run --frozen python -m pytest tests/test_semantic_control_gpu.py     # GPU, real checkpoint
uv run --frozen python -m pytest tests/test_semantic_control_recipe.py  # checkpoint save/restore contract
uv run --frozen python -m pytest tests/test_eval_protocol.py            # matched-protocol statistics
```

A-D pilot (training resumes from the last checkpoint at each stage; the learning-rate schedule always spans 30k steps):

```bash
for step in 5000 10000 20000 30000; do uv run --frozen python src/pilot.py train --stop-at $step; done
uv run --frozen python src/pilot.py eval --step 30000 --episodes 20 --batch 5 --deterministic
uv run --frozen python src/pilot.py compare --step 30000 --episodes 20 --deterministic
uv run --frozen python src/factorial_stats.py --step 30000 --out factorial_30k
uv run --frozen python src/compare_results.py --reference results/pilot/released_smolvla_matched_e20.json \
  --candidate results/pilot/B_30000_e20_matched.json --context A,B,C,D:30000 \
  --out results/pilot/released_vs_B30k_matched
```

Every checkpoint carries a `semantic_control.json` record (preset, fingerprints, verified routing, loss, learning
rates, gradient norms, runtime, peak memory). `evaluate.py --checkpoint DIR` restores the routing from it, verifies it
on the loaded model, and refuses to run otherwise. Stored results can be re-rendered without a GPU with
`python src/evaluate.py --rerender RESULT.json --report RESULT.md`. `src/make_gifs.py` turns rendered rollouts of
matched episodes into side-by-side GIFs.
