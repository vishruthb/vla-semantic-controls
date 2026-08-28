# SmolVLA LIBERO-Spatial Baseline — preset D @ step 30000

**Status: DIAGNOSE (outside the +/-5.0-point reproduction band) — 146/200 successes (73.0%).** Reference: ~81.5% (delta -8.5 points).

| Task | This run | Reference | Delta |
| --- | ---: | ---: | ---: |
| 0: pick up the black bowl between the plate and the ramekin and place it on the plate | 55.0% (11/20) | 75.0% | -20.0 |
| 1: pick up the black bowl next to the ramekin and place it on the plate | 95.0% (19/20) | 85.0% | +10.0 |
| 2: pick up the black bowl from table center and place it on the plate | 85.0% (17/20) | 95.0% | -10.0 |
| 3: pick up the black bowl on the cookie box and place it on the plate | 75.0% (15/20) | 80.0% | -5.0 |
| 4: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 65.0% (13/20) | 85.0% | -20.0 |
| 5: pick up the black bowl on the ramekin and place it on the plate | 55.0% (11/20) | 100.0% | -45.0 |
| 6: pick up the black bowl next to the cookie box and place it on the plate | 90.0% (18/20) | 65.0% | +25.0 |
| 7: pick up the black bowl on the stove and place it on the plate | 80.0% (16/20) | 70.0% | +10.0 |
| 8: pick up the black bowl next to the plate and place it on the plate | 50.0% (10/20) | 85.0% | -35.0 |
| 9: pick up the black bowl on the wooden cabinet and place it on the plate | 80.0% (16/20) | 75.0% | +5.0 |

## Runtime

- Evaluation: 5823.1 s; total process: 5823.9 s.
- Policy latency, batch 5: mean 115.47 ms, median 113.72 ms, p95 132.26 ms over 9607 calls.
- Peak VRAM: 5800 MiB by nvidia-smi; PyTorch allocated/reserved peaks: 2242.0/2412.0 MiB.
- Host: NVIDIA GeForce RTX 5090, driver 580.159.04, 32607 MiB; CUDA runtime 12.8; Python 3.12.3.

## Exact setting

- Checkpoint: `local:/workspace/vla-semantic-controls/outputs/train/spatial_D/checkpoints/030000/pretrained_model` (weights SHA-256 `7c09687da053a85ecd57b605d7259573320d8d39f768d1e95095315dc8fb3dff`).
- Semantic control: preset D (semantic_layers=cross_only, update_vlm=True; source checkpoint_file); routing verified: True on 16/32 coupled layers; training step 30000; init fingerprint `77cc336fa8786389`; parameter dtypes ['bfloat16', 'float32'].
- LeRobot: `https://github.com/huggingface/lerobot.git@8515d456be1dbef8c133f07188c785e683eca899`; LIBERO `0.1.4`; MuJoCo `3.3.2`.
- PyTorch `2.11.0+cu128`; Transformers `5.5.4`; complete dependency resolution in `uv.lock` (SHA-256 `a0bda71d5fc3e46a4383669b46a1166c1b47eac1e7f6f2b3c75dd648f193714e`).
- Harness revision: `6a9caf8e26534f5fe388ba2b69db4af31429acdc`; reference artifact: `https://github.com/zuoxingdong/smolvla-libero-eval.git@114d19c51bf7655e61fb994f3b344a0257ceb20b`.
- 10 tasks × 20 episodes; seed 1000; fixed init states; relative control; 360×360 observations; 280 max steps.
- fp32; `num_steps=1`; `n_action_steps=1`; batch 5; `osmesa` rendering.
- TF32 disabled; cuDNN benchmark disabled; async vector environments; one task evaluated at a time.

## Diagnosis

- Observed 73.0% is -8.5 points from the 81.5% target and outside the configured +/-5.0-point band.
- The approximate Wilson 95% interval is 66.5-78.7%; it excludes the 81.5% reference (whose own interval is 75.5-86.3%). Episode dependence makes this only a diagnostic.
- Per-task deltas: largest deficits are task(s) 5, 8, and 0; largest improvements are task(s) 6 and 7; the deficit is spread across tasks rather than a single-task protocol collapse.
- The reference artifact does not publish a complete transitive lock or its Python, CUDA, and driver versions; this run records all of them, so those remain the leading unresolved environment differences.
- The protocol uses the same pinned LeRobot simulator-reuse behavior. This harness additionally pins the backbone snapshot and synchronizes CUDA for timing; neither changes the model architecture or action values.
