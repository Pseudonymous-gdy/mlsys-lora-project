#!/usr/bin/env python3
"""Convenience wrapper for generating all experiment YAML files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from analysis.generate_configs import generate_config_files

    paths = generate_config_files(
        ROOT / "configs/base.yaml",
        ROOT / "configs/sweeps.yaml",
        ROOT / "configs/generated",
    )
    print(f"Generated {len(paths)} configs in {ROOT / 'configs/generated'}")


if __name__ == "__main__":
    main()
