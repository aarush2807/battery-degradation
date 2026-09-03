#!/usr/bin/env python3
"""Train wrapper script."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main

if __name__ == "__main__":
    sys.argv = ["main.py", "train"] + sys.argv[1:]
    main()
