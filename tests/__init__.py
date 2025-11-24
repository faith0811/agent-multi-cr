"""
Test package for agent-multi-cr.

We reuse the bootstrap helper from the project root to ensure the local
``src/`` directory is importable when running tests without installing the
package.
"""

from bootstrap import ensure_src_on_path

ensure_src_on_path()
