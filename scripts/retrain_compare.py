#!/usr/bin/env python3
"""Wrapper for retraining and comparing float and effective-INT4 views."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scim_iomt.retrain_compare_float_int4 import main


if __name__ == "__main__":
    main()
