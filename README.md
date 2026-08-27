# SmolVLA LIBERO-Spatial baseline

Minimal, pinned harness for the unmodified `HuggingFaceVLA/smolvla_libero` checkpoint.

Baseline result: **154/200 (77.0%)**, within the configured 5-point reproduction band around 81.5%.
See [`results/REPORT.md`](results/REPORT.md) for per-task results and diagnosis.

## Reproduce

```bash
# Requires uv 0.9.0.
sudo apt-get update
sudo apt-get install -y libosmesa6 libglfw3 libgl1
./scripts/setup.sh
./scripts/smoke.sh
./scripts/evaluate.sh
```

The smoke test runs task 0 for two episodes. The baseline runs 10 tasks × 20 episodes and writes
`results/metrics.json` plus `results/REPORT.md`. Model snapshots stay in the Hugging Face cache and
checkpoint extensions are ignored by git.

All Python dependencies, including the LeRobot git commit, LIBERO, MuJoCo, PyTorch, and transitive
packages, are locked in `uv.lock`. Exact evaluation settings are in `configs/baseline.json`.
No SmolVLA architecture or checkpoint files are present in this repository.
