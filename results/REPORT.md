# SmolVLA LIBERO-Spatial Baseline

**Status: PASS (within the +/-5.0-point reproduction band) — 154/200 successes (77.0%).** Reference: ~81.5% (delta -4.5 points).

| Task | This run | Reference | Delta |
| --- | ---: | ---: | ---: |
| 0: pick up the black bowl between the plate and the ramekin and place it on the plate | 60.0% (12/20) | 75.0% | -15.0 |
| 1: pick up the black bowl next to the ramekin and place it on the plate | 95.0% (19/20) | 85.0% | +10.0 |
| 2: pick up the black bowl from table center and place it on the plate | 90.0% (18/20) | 95.0% | -5.0 |
| 3: pick up the black bowl on the cookie box and place it on the plate | 75.0% (15/20) | 80.0% | -5.0 |
| 4: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 75.0% (15/20) | 85.0% | -10.0 |
| 5: pick up the black bowl on the ramekin and place it on the plate | 85.0% (17/20) | 100.0% | -15.0 |
| 6: pick up the black bowl next to the cookie box and place it on the plate | 50.0% (10/20) | 65.0% | -15.0 |
| 7: pick up the black bowl on the stove and place it on the plate | 70.0% (14/20) | 70.0% | +0.0 |
| 8: pick up the black bowl next to the plate and place it on the plate | 85.0% (17/20) | 85.0% | +0.0 |
| 9: pick up the black bowl on the wooden cabinet and place it on the plate | 85.0% (17/20) | 75.0% | +10.0 |

## Runtime

- Evaluation: 5940.9 s; total process: 5948.5 s.
- Policy latency, batch 5: mean 114.89 ms, median 112.73 ms, p95 131.26 ms over 9287 calls.
- Peak VRAM: 2577 MiB by nvidia-smi; PyTorch allocated/reserved peaks: 1340.3/1458.0 MiB.
- Host: NVIDIA GeForce RTX 5090, driver 580.159.04, 32607 MiB; CUDA runtime 12.8; Python 3.12.3.

## Exact setting

- Checkpoint: `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102` (weights SHA-256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`).
- LeRobot: `https://github.com/huggingface/lerobot.git@8515d456be1dbef8c133f07188c785e683eca899`; LIBERO `0.1.4`; MuJoCo `3.3.2`.
- PyTorch `2.11.0+cu128`; Transformers `5.5.4`; complete dependency resolution in `uv.lock` (SHA-256 `a0bda71d5fc3e46a4383669b46a1166c1b47eac1e7f6f2b3c75dd648f193714e`).
- Harness revision: `uncommitted worktree (no Git HEAD)`; reference artifact: `https://github.com/zuoxingdong/smolvla-libero-eval.git@114d19c51bf7655e61fb994f3b344a0257ceb20b`.
- 10 tasks × 20 episodes; seed 1000; fixed init states; relative control; 360×360 observations; 280 max steps.
- fp32; `num_steps=1`; `n_action_steps=1`; batch 5; `osmesa` rendering.
- TF32 allowed; cuDNN benchmark enabled; async vector environments; one task evaluated at a time.

## Diagnosis

- Observed 77.0% is -4.5 points from the 81.5% target and inside the configured +/-5.0-point band.
- The approximate Wilson 95% interval is 70.7-82.3%; it includes the 81.5% reference (whose own interval is 75.5-86.3%). Episode dependence makes this only a diagnostic.
- Per-task deltas: largest deficits are task(s) 0, 5, and 6; largest improvements are task(s) 9 and 1; the deficit is spread across tasks rather than a single-task protocol collapse.
- The reference artifact does not publish a complete transitive lock or its Python, CUDA, and driver versions; this run records all of them, so those remain the leading unresolved environment differences.
- The protocol uses the same pinned LeRobot simulator-reuse behavior. This harness additionally pins the backbone snapshot and synchronizes CUDA for timing; neither changes the model architecture or action values.
