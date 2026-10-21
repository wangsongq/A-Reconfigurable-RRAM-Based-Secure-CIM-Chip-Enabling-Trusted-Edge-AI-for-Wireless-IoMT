#!/usr/bin/env python3
"""Wrapper for the fixed-range QAT training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scim_iomt.train_compact_qat_patient import main


if __name__ == "__main__":
    main()
