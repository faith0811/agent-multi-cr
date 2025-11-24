"""
Main package for agent-multi-cr.

The public API is intentionally minimal; most functionality is exposed via the
``agent-multi-cr`` console script. Library users should prefer the CLI or
call ``run_pipeline`` directly.
"""


def run_pipeline(*args, **kwargs):
    """
    Lazily import and invoke the main review pipeline.

    This avoids importing the heavy pipeline module on package import,
    which keeps ``import agent_multi_cr`` lightweight.
    """
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = [
    "run_pipeline",
]
