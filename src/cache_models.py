#!/usr/bin/env python3
"""Cache the exact policy and backbone revisions used by the baseline."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/baseline.json").read_text())


def main() -> None:
    model = CONFIG["model"]
    for label, repository, revision in (
        ("policy", model["repository"], model["revision"]),
        ("backbone", model["backbone_repository"], model["backbone_revision"]),
    ):
        path = snapshot_download(
            repo_id=repository,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        )
        print(f"{label}: {repository}@{revision} -> {path}")

    sources = CONFIG["sources"]
    asset_dir = Path.home() / ".cache/libero/assets"
    path = snapshot_download(
        repo_id=sources["libero_assets_repository"],
        repo_type="dataset",
        revision=sources["libero_assets_revision"],
        local_dir=asset_dir,
    )
    print(
        f"assets: {sources['libero_assets_repository']}@"
        f"{sources['libero_assets_revision']} -> {path}"
    )


if __name__ == "__main__":
    main()
