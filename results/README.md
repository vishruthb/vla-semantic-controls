# Results

Every number in the top-level README comes from the files here. The JSON files are the raw records; the markdown
tables are rendered from them and can be rebuilt without a GPU (see [Regenerating](#regenerating)).

## Files

| path | what it is |
| --- | --- |
| `baseline/metrics.json`, `baseline/report.md` | released `smolvla_libero` on the reference protocol (154/200) |
| `pilot/{P}_{step}_e{n}[_matched].json` | preset `P` at training step `step`, `n` episodes per task; `_matched` = matched protocol |
| `pilot/paired_{step}_e20_matched.{json,md}` | B, C and D against A on matched episodes (`pilot.py compare`) |
| `pilot/factorial_{20k,30k}.{json,md}` | 2x2 factorial effects (`factorial_stats.py`) |
| `pilot/released_smolvla_matched_e20.json` | released checkpoint on the matched protocol (164/200) |
| `pilot/released_vs_B30k_matched.{json,md}` | B at 30k against the released checkpoint (`compare_results.py`) |
| `pilot/summary_{10k,20k,30k}.json` | per-stage digest of training records and statistics, assembled by hand (no script) |
| `pilot/records/spatial_{P}_{step}_semantic_control.json` | copy of `outputs/train/spatial_{P}/checkpoints/{step}/pretrained_model/semantic_control.json` |
| `pilot/records/spatial_{P}_train_log.jsonl` | copy of `outputs/train/spatial_{P}/semantic_train_log.jsonl` (per-step loss, lr, gradient norms, step time) |
| `pilot/records/spatial_{P}_train_config.json` | copy of the run's final `train_config.json` |

Step numbers have five digits in evaluation files (`05000`), six in training records (`005000`, the LeRobot
checkpoint name), and `k` in the hand-named summaries. The scripts depend on the first two patterns.

## Reading the results

Per-run reports are not committed; render one with
`uv run --frozen python src/evaluate.py --rerender results/pilot/B_30000_e20_matched.json --report B_30000.md`.

- **Status.** Every evaluation result compares its success rate with the released checkpoint's published 81.5%.
  `pass` means within 5 points, `diagnose` means outside, and `not_comparable` means a different protocol (the 5k and
  10k screening runs). For the released checkpoint this is the reproduction check. For A-D it is only a distance from
  that reference, so `diagnose` on A, C and D at 30k is expected and is not an error. `evaluate.py` exits with code 3
  on `diagnose`, which is why `pilot.py eval` returns 1 when any preset lands outside the band.
- **Absolute paths.** Paths under `/workspace/...` in the JSON and in the report headers are from the machine that
  produced the results. They are provenance only.

## Provenance

The results record the harness commit they were produced with (`revisions.repo` in the JSON, shown as the
"Harness revision" line in a rendered report). These hashes come from the repository's original history, which was
later rewritten to reword commit messages, with every tree left unchanged:

| recorded revision | commit in this history | used for |
| --- | --- | --- |
| `04a653e` | `12470de` | A-D at 5k and 10k |
| `6bea8ba` | `e656d6f` | A-D at 20k (matched) |
| `6a9caf8` | `65a7f0d` | A-D at 30k (matched) |
| `a2729b7` | `b12642e` | released checkpoint (matched) |

After the results were produced, the code was reformatted, its comments rewritten, and a few scripts restructured
(imports, a shared helper, output paths). None of this changes an output: every committed table regenerates
byte-identically from the JSON. Two consequences:

- `semantic_control_sha256` in the training records (`7992b8ec...`) is the hash of `src/semantic_control.py` at
  `12470de`-`b12642e`, not of the current file. Check it with
  `git show b12642e:src/semantic_control.py | sha256sum`.
- `uv.lock` is unchanged, so the `uv_lock_sha256` recorded in every result still matches.

## Regenerating

With the locked environment (`uv sync --frozen`), everything below runs on CPU and rewrites the committed files:

```bash
uv run --frozen python src/evaluate.py --rerender results/baseline/metrics.json --report results/baseline/report.md
for step in 20000 30000; do uv run --frozen python src/pilot.py compare --step $step --episodes 20 --deterministic; done
uv run --frozen python src/factorial_stats.py --step 20000 --out factorial_20k
uv run --frozen python src/factorial_stats.py --step 30000 --out factorial_30k
uv run --frozen python src/compare_results.py --reference results/pilot/released_smolvla_matched_e20.json \
  --candidate results/pilot/B_30000_e20_matched.json --context A,B,C,D:30000 \
  --out results/pilot/released_vs_B30k_matched
git diff --stat results/   # expected: empty
```
