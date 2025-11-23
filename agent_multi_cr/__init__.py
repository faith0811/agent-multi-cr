"""
Compatibility package so that tooling can import `agent_multi_cr` even though
the code currently lives under the `src` package.

This avoids breaking existing imports while we keep the CLI entry point
unchanged (`src.cli:main`).
"""

from src import *  # type: ignore  # re-export for convenience

