"""
Main package for agent-multi-cr.

The public API is intentionally minimal; most functionality is exposed via the
``agent-multi-cr`` console script. Library users should prefer the CLI or
call ``run_pipeline`` directly.
"""

from .pipeline import run_pipeline

__all__ = [
    "run_pipeline",
]
