#!/usr/bin/env python3
"""Matched episode-level comparison of two evaluation results (same protocol), with context rows.

    python src/compare_results.py --reference results/pilot/released_smolvla_matched_e20.json \
        --candidate results/pilot/B_30000_e20_matched.json --context A,B,C,D:30000 \
        --out results/pilot/released_vs_B30k_matched
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eval_protocol as ep  # noqa: E402


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(100 * (c - h)), float(100 * (c + h))]


def load(path: Path) -> dict:
    metrics = json.loads(path.read_text())
    if metrics.get("policy_kind") == "semantic_control_checkpoint" or metrics.get("semantic_control"):
        if not metrics.get("semantic_control", {}).get("routing", {}).get("verified"):
            raise RuntimeError(f"{path}: routing not verified; invalid")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="released SmolVLA")
    parser.add_argument("--candidate-label", default="B@30k")
    parser.add_argument("--context", default="", help="e.g. A,B,C,D:30000 -> results/pilot/<P>_<step>_e20_matched.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20000)
    args = parser.parse_args()

    ref, cand = load(args.reference), load(args.candidate)
    keys_r, r = ep.episode_outcomes(ref)
    keys_c, c = ep.episode_outcomes(cand)
    if keys_r != keys_c:
        raise RuntimeError("episode keys differ; results are not matched")
    for name, m in (("reference", ref), ("candidate", cand)):
        e = m["eval_settings"]
        if not e.get("deterministic_noise", {}).get("enabled"):
            raise RuntimeError(f"{name}: not evaluated with the deterministic matched protocol")
    # Protocol fingerprint computed uniformly from the recorded eval_settings (older result files
    # predate the stored `fingerprints` field); identical settings -> identical digest.
    import hashlib

    protocol = {name: hashlib.sha256(json.dumps(m["eval_settings"], sort_keys=True, default=str).encode()).hexdigest()
                for name, m in (("reference", ref), ("candidate", cand))}
    stats = ep.paired_stats(r, c, n_boot=args.reps, seed=0)  # candidate minus reference
    per_task = ep.per_task_deltas(keys_r, r, c)
    n = int(len(r))
    rows = [{"label": args.reference_label, "kind": ref.get("policy_kind"), "successes": int(r.sum()), "episodes": n,
             "percent": float(100 * r.mean()), "wilson95": wilson(int(r.sum()), n), "per_task": [t["a"] for t in per_task],
             "checkpoint": ref["revisions"]["checkpoint"]}]
    context = []
    if args.context:
        presets, step = args.context.split(":")
        for preset in presets.split(","):
            m = load(args.results_dir / f"{preset}_{int(step):05d}_e20_matched.json")
            k, v = ep.episode_outcomes(m)
            if k != keys_r:
                raise RuntimeError(f"{preset}: keys differ")
            context.append({"label": f"{preset}@{int(step)//1000}k", "kind": m.get("policy_kind", "semantic_control_checkpoint"),
                            "successes": int(v.sum()), "episodes": n, "percent": float(100 * v.mean()), "wilson95": wilson(int(v.sum()), n),
                            "per_task": [int(x) for x in v.reshape(10, -1).sum(axis=1)], "checkpoint": m["revisions"]["checkpoint"]})
    verdict = ("NO RELIABLE DIFFERENCE" if stats["paired_bootstrap_ci95_points"][0] <= 0 <= stats["paired_bootstrap_ci95_points"][1]
               else (f"{args.candidate_label.upper()} OUTPERFORMS {args.reference_label.upper()}" if stats["delta_points"] > 0
                     else f"{args.reference_label.upper()} OUTPERFORMS {args.candidate_label.upper()}"))
    report = {"reference": rows[0], "candidate": {"label": args.candidate_label, "successes": int(c.sum()), "episodes": n, "percent": float(100 * c.mean()),
                                                  "wilson95": wilson(int(c.sum()), n), "checkpoint": cand["revisions"]["checkpoint"]},
              "paired_candidate_minus_reference": stats, "per_task": per_task, "context": context, "protocol_sha256": protocol, "verdict": verdict}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    ci = stats["paired_bootstrap_ci95_points"]
    lines = [f"# {args.candidate_label} vs {args.reference_label} — matched {n}-episode LIBERO-Spatial protocol", "",
             f"Protocol fingerprints: reference `{str(protocol['reference'])[:16]}`, candidate `{str(protocol['candidate'])[:16]}` "
             f"({'identical' if protocol['reference'] == protocol['candidate'] else 'DIFFERENT'}).", "",
             "| policy | successes | success | Wilson 95% CI | per-task (of 20) |", "| --- | ---: | ---: | ---: | --- |"]
    for row in rows + context:
        lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['percent']:.1f}% | [{row['wilson95'][0]:.1f}, {row['wilson95'][1]:.1f}] | {row['per_task']} |")
    rel = f"{stats['relative_delta_percent']:+.1f}%" if stats["relative_delta_percent"] is not None else "n/a"
    lines += ["", f"## Paired: {args.candidate_label} − {args.reference_label}", "",
              f"- absolute delta: **{stats['delta_points']:+.1f} pts** (relative {rel})",
              f"- paired bootstrap 95% CI: [{ci[0]:+.1f}, {ci[1]:+.1f}]",
              f"- episode-level wins / losses / ties: {stats['wins']} / {stats['losses']} / {stats['ties']}",
              f"- exact McNemar p = {stats['mcnemar_p']:.3f}", "",
              "| task | reference | candidate | Δ |", "| --- | ---: | ---: | ---: |"]
    for t in per_task:
        lines.append(f"| {t['task_id']} | {t['a']} | {t['other']} | {t['delta']:+d} |")
    lines += ["", f"**Verdict: {verdict}**"]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
