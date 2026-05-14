#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch Cosmos Predict training; cwd must be repo root for Hydra/paths."""

from __future__ import annotations

import os

_COSMOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cosmos-predict2.5-1.5.0"))


def main() -> None:
    if not os.path.isdir(_COSMOS_ROOT):
        raise SystemExit(f"Cosmos repo not found: {_COSMOS_ROOT}")
    os.chdir(_COSMOS_ROOT)
    from cosmos_oss.scripts.train import main as _train_main

    _train_main()


if __name__ == "__main__":
    main()
