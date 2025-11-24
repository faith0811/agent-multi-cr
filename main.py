#!/usr/bin/env python3
"""
Developer convenience entrypoint for agent-multi-cr.

Production usage should prefer the ``agent-multi-cr`` console script that is
installed via ``pip``. This file simply bootstraps the local ``src/`` layout
and forwards to ``agent_multi_cr.cli.main``.
"""

from bootstrap import ensure_src_on_path

ensure_src_on_path()

from agent_multi_cr.cli import main


if __name__ == "__main__":
    main()
