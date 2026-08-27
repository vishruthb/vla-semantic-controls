#!/usr/bin/env python3
"""Download the data files of an episode range of HuggingFaceVLA/libero into the LeRobot hub cache.

The Hub revision's ``meta/episodes`` file map is wrong (it points episodes 1261-1692 at files that
hold episodes 137-174 and stops at file 68 of 377), so the true file -> episode map is rebuilt from
each parquet footer's column statistics (footer-only reads), the right files are fetched, and the
LeRobot loader is verified to yield exactly the requested episodes.

    python src/fetch_libero_subset.py            # LIBERO-Spatial: episodes 1261-1692
    python src/fetch_libero_subset.py --lo 0 --hi 431
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, snapshot_download

REPO = "HuggingFaceVLA/libero"
REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"


def footer_stats(fs: HfFileSystem, path: str, revision: str) -> tuple[str, int, int, int]:
    with fs.open(path, "rb") as handle:
        metadata = pq.read_metadata(handle)
    column = metadata.schema.names.index("episode_index")
    lo = min(metadata.row_group(i).column(column).statistics.min for i in range(metadata.num_row_groups))
    hi = max(metadata.row_group(i).column(column).statistics.max for i in range(metadata.num_row_groups))
    return path.split(f"@{revision}/")[1], int(lo), int(hi), metadata.num_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--lo", type=int, default=1261)
    parser.add_argument("--hi", type=int, default=1692)
    parser.add_argument("--map-out", type=Path, default=Path("outputs/libero_file_map.json"))
    args = parser.parse_args()

    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    started = time.time()
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{args.repo}@{args.revision}/data/*/*.parquet"))
    print(f"data files: {len(files)}", flush=True)
    with cf.ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(lambda p: footer_stats(fs, p, args.revision), files))
    wanted = [r for r in rows if r[2] >= args.lo and r[1] <= args.hi]
    print(f"files covering episodes {args.lo}-{args.hi}: {len(wanted)} ({sum(r[3] for r in wanted)} rows)", flush=True)
    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    args.map_out.write_text(json.dumps({"repo": args.repo, "revision": args.revision, "map": rows, "wanted": [r[0] for r in wanted]}, indent=1))
    snapshot_download(args.repo, repo_type="dataset", revision=args.revision, cache_dir=HF_LEROBOT_HUB_CACHE, allow_patterns=[r[0] for r in wanted])

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo, episodes=list(range(args.lo, args.hi + 1)), revision=args.revision)
    first, last = dataset[0], dataset[dataset.num_frames - 1]
    print(f"LeRobotDataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames; first episode {int(first['episode_index'])} "
          f"({first['task']}), last {int(last['episode_index'])} ({last['task']}); {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
