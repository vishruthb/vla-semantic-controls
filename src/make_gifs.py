#!/usr/bin/env python3
"""tile matched rollout videos from several evaluation results into one labelled gif.

videos exist only for evaluations run with ``--render-episodes-per-task N --keep-videos``. the pinned lerobot eval
writes episode ``k`` of task ``t`` to ``<output dir>/raw/<output stem>/videos/libero_spatial_<t>/eval_episode_<k>.mp4``,
and ``k`` indexes the same matched episode as ``result.per_task[t].episode_outcomes[k]``. the matched protocol binds
init states, env seeds and flow-matching noise to (task, episode) and is deterministic at a fixed batch size, so a
re-run with rendering on should replay the stored episodes; compare its ``episode_outcomes`` with the stored result.

    # re-render task 3 for presets A and C at 30k steps under the matched protocol
    for p in A C; do
      uv run --frozen python src/evaluate.py --deterministic-noise --semantic-control $p \\
        --checkpoint outputs/train/spatial_$p/checkpoints/030000/pretrained_model \\
        --task-ids 3 --episodes-per-task 20 --batch-size 5 --render-episodes-per-task 20 --keep-videos \\
        --output results/render/${p}_30000_task3.json
    done
    # tile the first episode where A and C disagree
    uv run --frozen python src/make_gifs.py results/render/A_30000_task3.json results/render/C_30000_task3.json \\
      --task 3 --discordant --out docs/media/task3_A_vs_C.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUITE = "libero_spatial"
SUCCESS_COLOR = (46, 160, 67)
FAILURE_COLOR = (207, 34, 46)


def load_result(path: Path) -> dict:
    metrics = json.loads(path.read_text())
    per_task = {row["task_id"]: row for row in metrics["result"]["per_task"]}
    control = (metrics.get("semantic_control") or {}).get("control") or {}
    return {"path": path, "metrics": metrics, "per_task": per_task, "preset": control.get("preset")}


def video_path(result: dict, task: int, episode: int) -> Path:
    """prefer the paths lerobot recorded; fall back to its directory layout next to the metrics file."""
    for row in result["metrics"].get("raw_lerobot_metrics", {}).get("per_task", []):
        if row["task_id"] == task:
            for recorded in row["metrics"].get("video_paths", []):
                if Path(recorded).name == f"eval_episode_{episode}.mp4":
                    return Path(recorded)
    path = result["path"].resolve()
    return path.parent / "raw" / path.stem / "videos" / f"{SUITE}_{task}" / f"eval_episode_{episode}.mp4"


def read_frames(path: Path, stride: int, height: int) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index % stride:
                continue
            image = frame.to_image()
            width = round(image.width * height / image.height)
            frames.append(np.asarray(image.resize((width, height), Image.Resampling.BILINEAR)))
    if not frames:
        raise RuntimeError(f"{path}: no frames")
    return frames


def label_bar(width: int, text: str, color: tuple[int, int, int], height: int) -> np.ndarray:
    bar = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(bar)
    font = ImageFont.load_default(size=max(10, int(height * 0.6)))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left, (height - (bottom - top)) / 2 - top), text, fill="white", font=font)
    return np.asarray(bar)


def pick_episode(results: list[dict], task: int, episode: int | None, discordant: bool) -> int:
    outcomes = [result["per_task"][task]["episode_outcomes"] for result in results]
    if episode is not None:
        return episode
    candidates = range(min(len(o) for o in outcomes))
    if discordant:
        candidates = [k for k in candidates if len({o[k] for o in outcomes}) > 1]
        if not candidates:
            raise SystemExit(f"task {task}: no episode where the results disagree")
    return next(iter(candidates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="Metrics JSON files evaluated with --keep-videos")
    parser.add_argument("--labels", nargs="+", help="Tile labels (default: the preset recorded in each result)")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--episode", type=int, help="Matched episode index within the task")
    parser.add_argument("--discordant", action="store_true", help="Pick the first episode where outcomes differ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224, help="Tile height in pixels")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th frame")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    results = [load_result(path) for path in args.results]
    labels = args.labels or [result["preset"] or result["path"].stem for result in results]
    if len(labels) != len(results):
        raise SystemExit("--labels needs one label per result")
    episode = pick_episode(results, args.task, args.episode, args.discordant)

    tiles = []
    for result, label in zip(results, labels, strict=True):
        success = result["per_task"][args.task]["episode_outcomes"][episode]
        frames = read_frames(video_path(result, args.task, episode), args.stride, args.height)
        text = f"{label}: {'success' if success else 'failure'}"
        bar = label_bar(frames[0].shape[1], text, SUCCESS_COLOR if success else FAILURE_COLOR, args.height // 8)
        tiles.append([np.concatenate([bar, frame], axis=0) for frame in frames])

    # hold each tile's last frame so rollouts of different lengths stay aligned
    length = max(len(tile) for tile in tiles)
    tiles = [tile + [tile[-1]] * (length - len(tile)) for tile in tiles]
    frames = [Image.fromarray(np.concatenate([tile[i] for tile in tiles], axis=1)) for i in range(length)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"Wrote {args.out} (task {args.task}, episode {episode}, {length} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
