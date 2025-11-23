import os
import sys
from typing import List

from .shell_utils import run_shell


def get_git_diff(use_cached: bool) -> str:
    """Get git diff text for the current repo."""
    diff_cmd = ["git", "diff", "--cached"] if use_cached else ["git", "diff"]
    diff = run_shell(diff_cmd)
    if not diff.strip():
        raise SystemExit("No git diff found (nothing to review).")
    return diff


def collect_repo_context(
    root: str,
    max_files: int = 40,
    max_bytes_per_file: int = 4000,
) -> str:
    """
    Build a textual snapshot of the repository for review.

    We attempt to list tracked files using `git ls-files`. If that fails, we
    fall back to walking the directory tree, ignoring some common directories.

    For each file, we include up to `max_bytes_per_file` bytes of content.
    """
    root_abs = os.path.abspath(root)

    files: List[str] = []
    try:
        out = run_shell(["git", "ls-files"], cwd=root_abs)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            files.append(line)
    except Exception:
        # Fallback: walk filesystem
        for dirpath, dirnames, filenames in os.walk(root_abs):
            # Skip typical non-source dirs
            rel = os.path.relpath(dirpath, root_abs)
            if rel.startswith(".git") or rel.startswith(".multi_cr_auditors"):
                dirnames[:] = []  # don't descend further
                continue
            for fname in filenames:
                relpath = os.path.relpath(os.path.join(dirpath, fname), root_abs)
                files.append(relpath)

    # Limit number of files
    files = files[:max_files]

    parts: List[str] = []
    parts.append(
        f"Repository snapshot from {root_abs}\n"
        f"(showing up to {max_files} files, {max_bytes_per_file} bytes per file)\n"
    )

    for path in files:
        full_path = os.path.join(root_abs, path)
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "rb") as f:
                data = f.read(max_bytes_per_file + 1)
        except Exception:
            continue
        truncated = len(data) > max_bytes_per_file
        content = data[:max_bytes_per_file].decode("utf-8", errors="replace")
        parts.append(f"\n\n===== FILE: {path} =====\n")
        parts.append(content)
        if truncated:
            parts.append("\n[...TRUNCATED...]\n")

    return "".join(parts)


def get_stdin_context() -> str:
    """Read all of stdin as the context."""
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No data read from stdin for context.")
    return data


def resolve_context_text(
    context_mode: str,
    use_cached: bool,
    repo_root: str,
    max_context_files: int,
    max_context_bytes_per_file: int,
) -> str:
    """
    Produce the text context based on the chosen mode:

    - diff: git diff / git diff --cached
    - repo: repository snapshot (limited by file count and bytes per file)
    - stdin: read from stdin
    """
    if context_mode == "diff":
        return get_git_diff(use_cached)
    if context_mode == "repo":
        return collect_repo_context(
            root=repo_root,
            max_files=max_context_files,
            max_bytes_per_file=max_context_bytes_per_file,
        )
    if context_mode == "stdin":
        return get_stdin_context()
    raise SystemExit(f"Unknown context mode: {context_mode}")

