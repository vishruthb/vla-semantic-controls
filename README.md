# Semantic controls for the SmolVLA action expert

Does it matter where a VLA's action expert reads the vision-language model, and whether the action loss may update
that model? We answer this for [SmolVLA](https://arxiv.org/abs/2506.01844) on
[LIBERO-Spatial](https://arxiv.org/abs/2306.03310) with a 2x2 experiment in which only those two things change.

SmolVLA's action expert is a flow-matching head: starting from Gaussian noise, it produces a chunk of 50 future
actions. It sees the VLM through exactly one channel. At every layer, its attention reads the VLM's key/value
projections of the image, language and state tokens. There is no residual mixing, no FiLM, and no use of the VLM's
final hidden state ([`docs/architecture.md`](docs/architecture.md)). That makes the coupling precise to control in
two independent ways:

- **Forward (`semantic_layers`)**: which layers let the action expert read VLM keys/values. `all` is native SmolVLA
  (32 layers). `cross_only` keeps only the 16 cross-attention layers and masks the VLM tokens out of the 16 joint
  self-attention layers for the action tokens (exactly zero attention probability and zero gradient).
- **Backward (`update_vlm`)**: whether the action loss updates VLM weights (text layers, token embeddings, and the
  connector that maps image features into the text model). The vision encoder is always frozen. Even with
  `update_vlm=false`, gradients still flow through the frozen VLM, because `state_proj` (the linear layer that embeds
  the robot state as a VLM token) is always trained.

| preset | `semantic_layers` | `update_vlm` | expert reads VLM K/V in | trainable parameters |
| --- | --- | --- | --- | ---: |
| A | all | false | 32 layers | 97.5 M |
| B | all | true | 32 layers | 462.0 M |
| C | cross_only | false | 16 cross-attention layers | 97.5 M |
| D | cross_only | true | 16 cross-attention layers | 462.0 M |

All four policies are evaluated on the same 200 matched episodes: identical initial states, environment seeds and
flow-matching noise for every policy (see [Evaluation](#evaluation)), so every comparison is paired.

## Results

LIBERO-Spatial success rate over 10 tasks, with Wilson 95% intervals at 30k steps:

| preset | 10k steps\* | 20k steps | 30k steps |
| --- | ---: | ---: | ---: |
| A: all layers, frozen VLM | 56% | 67.5% | 70.5% [63.8, 76.4] |
| B: all layers, trained VLM | 52% | 76.0% | **78.5%** [72.3, 83.6] |
| C: cross-only, frozen VLM | 62% | 67.5% | 66.0% [59.2, 72.2] |
| D: cross-only, trained VLM | 64% | 73.5% | 73.0% [66.5, 78.7] |

\* Screening run: 10 episodes per task, before the matched protocol existed; the four are within noise.

For reference, the released [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
checkpoint scores 82.0% [76.1, 86.7] on the same 200 matched episodes. Its training budget is not published.

2x2 factorial effects in percentage points. Intervals come from an episode bootstrap: episodes are resampled within
each task, keeping each episode's four outcomes (A, B, C, D) together. p-values come from a paired sign-flip test.

| effect | 20k steps | 30k steps |
| --- | ---: | ---: |
| update VLM: ((B-A) + (D-C)) / 2 | +7.25 [+1.25, +13.0], p = 0.026 | **+7.5 [+2.0, +13.0], p = 0.017** |
| cross-only routing: ((C-A) + (D-B)) / 2 | -1.25 [-7.0, +4.5], p = 0.74 | -5.0 [-11.0, +1.0], p = 0.14 |
| interaction: (D-C) - (B-A) | -2.5 [-14.0, +9.5], p = 0.75 | -1.0 [-11.0, +9.0], p = 0.93 |

What this shows so far:

1. **Letting the action loss update the VLM improved success by about 7.5 points** at 30k. The estimate was similar at
   20k (+7.25), on the same runs. The point estimate is positive under both routings: B-A = +8.0 (McNemar p = 0.06)
   and D-C = +7.0 (p = 0.11). Neither is significant on its own.
2. **Restricting the expert to the cross-attention layers did not help.** The routing effect was -1.25 points at 20k
   and -5.0 at 30k. Neither estimate, nor the change between them (-3.75 [-11.25, +3.75]), is distinguishable from
   zero. Cross-only runs do end with higher training loss: C 0.075 vs A 0.057, and D 0.063 vs B 0.050 (mean over
   steps 29k-30k).
3. **No interaction between the two knobs was detected** (-1.0 points). With 200 episodes, interactions of up to about
   10 points cannot be ruled out.
4. **The best configuration, B, is not significantly different from the released checkpoint**: -3.5 points
   [-11.0, +4.0], 27 wins / 34 losses / 139 ties, exact McNemar p = 0.44. The interval still allows a deficit of up to
   11 points, and B was picked after the fact as the best of four.

Limitations:

- There is one training seed per configuration, 200 episodes per configuration, and one LIBERO suite.
- B and D train 4.7x more parameters than A and C, so the update-VLM effect includes added capacity. The VLM learning
  rate (1e-5) was not tuned.
- Per-task effects vary a lot. On task 3 ("the black bowl on the cookie box"), cross-only routing costs 27.5 points.
  On task 5 ("the black bowl on the ramekin"), updating the VLM costs 27.5 points.
- With only 10 tasks, a secondary task bootstrap (resampling whole tasks) gives intervals that include zero for every
  effect.

The natural next steps are a second seed, 400 episodes per configuration, the other LIBERO suites, and a sweep over
the VLM learning rate. Full tables: [`factorial_30k.md`](results/pilot/factorial_30k.md),
[`paired_30000_e20_matched.md`](results/pilot/paired_30000_e20_matched.md),
[`released_vs_B30k_matched.md`](results/pilot/released_vs_B30k_matched.md).

## Method

**Model.** SmolVLA from LeRobot `8515d45`, in the shape of `smolvla_libero`:

- The pretrained `SmolVLM2-500M-Video-Instruct` backbone with all 32 text layers.
- A 32-layer action expert of width 0.5 (hidden size 480).
- Even layers are joint self-attention over VLM and action tokens. Odd layers are cross-attention into VLM keys/values
  through a learned 320 -> 320 projection.

With preset A, the semantic-control code path reproduces upstream SmolVLA bit for bit: loss, actions, every gradient
and every attention mask. This was tested on GPU with the 16-layer `lerobot/smolvla_base` checkpoint; the 32-layer
shape trained here runs the same code path.

**Training** is identical for A-D apart from the two knobs ([`docs/recipe.md`](docs/recipe.md)):

- **Initialization.** The action expert and projections are initialized from scratch with seed 1000. All four runs
  start from the same weights (identical `init_fingerprint`) on the same pretrained VLM snapshot, which B and D hold as
  fp32 master copies. The pretrained `smolvla_base` expert is not used, for two reasons: it has a different shape (16
  VLM layers, width 0.75), and it was trained with native routing, which would favor A and B.
- **Data.** LIBERO-Spatial demonstrations from `HuggingFaceVLA/libero`: episodes 1261-1692, which is 432 episodes and
  52,970 frames. The dataset's `meta/episodes` file map (revision `8695891`) points these episodes at the wrong parquet
  files. [`fetch_libero_subset.py`](src/fetch_libero_subset.py) rebuilds the true file-to-episode map from the parquet
  footers and downloads the files that hold them. LeRobot's episode-subset sampler also yields absolute frame indices
  where the filtered dataset expects relative ones; [`train_semantic.py`](src/train_semantic.py) patches this at
  runtime.
- **Optimization.** AdamW, batch 32, 30k steps (about 18 epochs), 1k warm-up steps, then a single cosine factor
  applied to both learning-rate groups:
  - the expert and projections go from 1e-4 to 2.5e-6;
  - the VLM goes from 1e-5 to 2.5e-7.

  Weights are held as fp32 master copies, with bf16 autocast.
- **Compute.** One RTX 5090. A and C take about 0.43 s per step (11.5 GiB peak); B and D take 0.48-0.50 s per step
  (18.6 GiB). One 30k-step run takes 3.7-4.3 hours of wall-clock time.

### Evaluation

The protocol:

- 10 LIBERO-Spatial tasks x 20 episodes, 280 steps max, batch 5.
- One flow-matching integration step (`num_steps=1`; SmolVLA's default is 10).
- The first action of each chunk is executed before re-planning (`n_action_steps=1`).
- Initial states and environment seeds are bound to the episode index. Flow-matching noise is seeded per
  (task, episode, step). cuDNN runs deterministically with TF32 off.

Every policy therefore faces exactly the same 200 episodes. In one check (not committed), re-running A at 10k steps
at the same batch size reproduced every outcome and reward trace. Batch size still changes GPU numerics, so it is held
fixed. Comparisons use paired bootstrap intervals, exact McNemar tests, and the factorial analysis above.

**Baseline reproduction.** The reference protocol comes from
[zuoxingdong/smolvla-libero-eval](https://github.com/zuoxingdong/smolvla-libero-eval) (`114d19c`). On it, the
released checkpoint scores 154/200 (77.0%, [70.7, 82.3]) against the published 163/200 (81.5%), within the 5-point
band we set for reproduction ([`results/REPORT.md`](results/REPORT.md)). On the matched protocol, the same checkpoint
scores 164/200 (82.0%). The two runs differ only in how the flow-matching noise is drawn and in cuDNN/TF32 settings,
yet they disagree on 32 of 200 episodes. That swing, from evaluation noise alone, is why all comparisons above are
paired.

## Reproduce

Requires:

- Linux.
- An NVIDIA GPU with compute capability 12.0. `setup.sh` checks for it, and all results were produced on an RTX 5090.
  B/D training needs about 20 GB.
- uv 0.9.0 and OSMesa.

All Python dependencies, including the LeRobot commit, LIBERO, MuJoCo and PyTorch, are locked in `uv.lock`. The
trained A-D checkpoints are not included (about 16 GPU-hours to retrain). All statistics can be recomputed from the
committed `results/pilot/*.json` without a GPU.

```bash
sudo apt-get install -y libosmesa6 libglfw3 libgl1
./scripts/setup.sh                                   # uv sync + cache the released checkpoint and backbone
./scripts/smoke.sh                                   # task 0, 2 episodes
./scripts/evaluate.sh                                # released checkpoint, reference protocol -> results/REPORT.md
```

Semantic-control interface and tests. The interface tests and `report` use `lerobot/smolvla_base`
(`configs/semantic_control.json`), so their parameter counts are for the 16-layer shape, not for the 32-layer
training model in the table above.

```bash
uv run --frozen python src/semantic_control.py cache                    # smolvla_base + backbone (needed by pilot.py)
uv run --frozen python src/fetch_libero_subset.py                       # LIBERO-Spatial training data
uv run --frozen python src/semantic_control.py report --preset C        # parameter counts, smolvla_base shape
uv run --frozen python -m pytest tests/test_semantic_control_unit.py    # CPU, tiny random model
uv run --frozen python -m pytest tests/test_semantic_control_gpu.py     # GPU, smolvla_base checkpoint
uv run --frozen python -m pytest tests/test_semantic_control_recipe.py  # checkpoint save/restore contract
uv run --frozen python -m pytest tests/test_eval_protocol.py            # matched-protocol noise and statistics
```

A-D pilot. Training resumes from the last checkpoint at each stage, and the learning-rate schedule always spans 30k
steps.

```bash
for step in 5000 10000 20000 30000; do uv run --frozen python src/pilot.py train --stop-at $step; done
uv run --frozen python src/pilot.py eval --step 30000 --episodes 20 --batch 5 --deterministic
uv run --frozen python src/pilot.py compare --step 30000 --episodes 20 --deterministic
uv run --frozen python src/factorial_stats.py --step 30000 --out factorial_30k
uv run --frozen python src/evaluate.py --deterministic-noise --render-episodes-per-task 0 \
  --output results/pilot/released_smolvla_matched_e20.json --report results/pilot/released_smolvla_matched_e20.md
uv run --frozen python src/compare_results.py --reference results/pilot/released_smolvla_matched_e20.json \
  --candidate results/pilot/B_30000_e20_matched.json --context A,B,C,D:30000 \
  --out results/pilot/released_vs_B30k_matched
```

Every checkpoint carries a `semantic_control.json` record: preset, fingerprints, verified routing, loss, learning
rates, gradient norms, runtime and peak memory. `evaluate.py --checkpoint DIR` restores the routing from that record,
verifies it on the loaded model, and refuses to run otherwise. To rebuild a stored report without a GPU, run
`uv run --frozen python src/evaluate.py --rerender RESULT.json --report RESULT.md`.

The committed evaluations kept no videos. To make rollout GIFs, re-run the tasks you want with
`--render-episodes-per-task N --keep-videos`, check that the re-run's `episode_outcomes` match the stored result, and
tile matched episodes side by side with [`src/make_gifs.py`](src/make_gifs.py) (see its docstring).
