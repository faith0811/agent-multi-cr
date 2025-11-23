"""
Thin wrapper CLI module that delegates to the existing implementation in
`src.cli`. This allows `agent_multi_cr.cli:main` to be used as an entry
point for tools that expect a non-`src` package name.
"""

from src.cli import main  # type: ignore

