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
`results/metrics.json` plus `results/REPORT.md`. Status and diagnosis are derived from the measured
result (`pass` / `diagnose` inside or outside the band; `not_comparable` when the protocol differs);
`uv run --frozen python src/evaluate.py --rerender results/metrics.json --report results/REPORT.md`
regenerates the report from a stored metrics file without running the evaluation. Model snapshots stay in the Hugging Face cache and
checkpoint extensions are ignored by git.

All Python dependencies, including the LeRobot git commit, LIBERO, MuJoCo, PyTorch, and transitive
packages, are locked in `uv.lock`. Exact evaluation settings are in `configs/baseline.json`.
No SmolVLA architecture or checkpoint files are present in this repository.

## Semantic-control interface

`src/semantic_control.py` adds two knobs on top of the unmodified SmolVLA code (`tmp/arch.md`
explains the interface): `semantic_layers = "all" | "cross_only"` selects the layers in which the
action expert may read the VLM key/values, and `update_vlm = false | true` selects whether the action
loss updates the VLM. The vision encoder is always frozen and `state_proj` is always trainable.
Presets A–D live in `configs/semantic_control.json`, pinned to `lerobot/smolvla_base`.

```bash
uv run --frozen python src/semantic_control.py cache               # pinned checkpoint + backbone
uv run --frozen python src/semantic_control.py report --preset C   # parameter counts
uv run --frozen python -m pytest tests/test_semantic_control_unit.py   # CPU, tiny random model
uv run --frozen python -m pytest tests/test_semantic_control_gpu.py    # RTX 5090, real checkpoint
uv run --frozen python -m pytest tests/test_semantic_control_recipe.py # save/restore contract
```

`src/train_semantic.py` runs LeRobot's official `lerobot-train` loop with a preset installed
(`--semantic.preset=C …`) and writes `semantic_control.json` into every checkpoint;
`semantic_control.load_policy_with_control` restores it for evaluation and refuses to guess.
The full apples-to-apples recipe (schema, initialization, optimizer, LIBERO-Spatial subset, cost) is in
`tmp/recipe.md`; `src/pilot.py train|eval|summarize` runs the staged A–D pilot and
`src/evaluate.py --checkpoint DIR` evaluates a checkpoint with its routing restored and verified.
