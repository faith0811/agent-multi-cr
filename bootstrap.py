"""
Helper to ensure the local ``src/`` directory is on ``sys.path``.

Both the CLI entrypoint (``main.py``) and the test suite use this so we keep
the bootstrap logic in one place instead of duplicating it.
"""

import os
import sys


def ensure_src_on_path() -> None:
    root = os.path.dirname(__file__)
    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

