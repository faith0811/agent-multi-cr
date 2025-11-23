import fnmatch
import os
import subprocess
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

    # Default patterns to avoid obviously sensitive / noisy files.
    default_exclude_patterns = [
        ".env*",
        "*.log",
        "*.sqlite",
        "*.sqlite3",
        "*.db",
    ]
    extra_excludes_env = os.environ.get("AGENT_MULTI_CR_SNAPSHOT_EXCLUDES", "")
    extra_exclude_patterns = [
        p.strip()
        for p in extra_excludes_env.split(",")
        if p.strip()
    ]
    all_exclude_patterns = default_exclude_patterns + extra_exclude_patterns

    def _is_excluded(path: str) -> bool:
        base = os.path.basename(path)
        for pattern in all_exclude_patterns:
            if fnmatch.fnmatch(base, pattern) or fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _is_git_ignored_path(rel_path: str) -> bool:
        """Check .gitignore using `git check-ignore`, if available."""
        git_dir = os.path.join(root_abs, ".git")
        if not os.path.isdir(git_dir):
            return False
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", rel_path],
                cwd=root_abs,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception:
            return False

    files: List[str] = []
    try:
        out = run_shell(["git", "ls-files"], cwd=root_abs)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if _is_excluded(line):
                continue
            # git ls-files returns tracked files, so we don't need to check _is_git_ignored_path
            files.append(line)
    except Exception:
        # Fallback: walk filesystem with an expanded ignore list to avoid
        # common virtualenvs, caches, and large dependency trees.
        ignore_dirs = {
            ".git",
            ".multi_cr_auditors",
            ".venv",
            "venv",
            "env",
            "ENV",
            "node_modules",
            "dist",
            "build",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".tox",
            ".idea",
            ".vscode",
        }
        for dirpath, dirnames, filenames in os.walk(root_abs):
            # Filter out ignored directories in-place so os.walk does not descend
            # into them.
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

            for fname in filenames:
                relpath = os.path.relpath(os.path.join(dirpath, fname), root_abs)
                if _is_excluded(relpath):
                    continue
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
        # Skip symlinks for security (avoid escaping repo root)
        if os.path.islink(full_path):
            continue
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "rb") as f:
                data = f.read(max_bytes_per_file + 1)
        except Exception:
            continue

        # Simple binary heuristic: look for null bytes in the first chunk
        if b"\0" in data[:8000]:
            parts.append(f"\n\n===== FILE: {path} =====\n")
            parts.append("[Binary file omitted]\n")
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

    - diff: current git diff / git diff --cached (LLMs are expected to inspect
      the repo and diff themselves; we only provide a short textual hint).
    - repo: instruct LLMs to inspect the repository in the working directory
      (no full snapshot is inlined into the prompt).
    - stdin: read additional context from stdin and pass it through as text.
    """
    if context_mode == "diff":
        # Still call git diff so we fail fast when there's nothing to review,
        # but we no longer inline the entire diff into the LLM prompt. Codex
        # and Gemini CLIs are expected to inspect the repo/diff using their
        # own tools in the working directory.
        _ = get_git_diff(use_cached)
        return (
            "CONTEXT_MODE: diff\n"
            "The code under review is the current git diff in this repository. "
            "Use your available tools (e.g., git diff, search, file inspection) "
            "to examine the changed code directly from the working directory."
        )
    if context_mode == "repo":
        # Do not inline a full repository snapshot; Codex/Gemini CLIs can read
        # files directly from the working directory. We only provide a brief
        # textual description so the models know how to interpret their tools.
        root_abs = os.path.abspath(repo_root)
        return (
            f"CONTEXT_MODE: repo\n"
            f"The code under review is the repository at {root_abs}. "
            "You have access to these files via your tools; inspect whatever "
            "you need directly from disk instead of relying on inlined code."
        )
    if context_mode == "stdin":
        # For stdin, we *do* pass through the text because it may be an
        # external diff, PR, or other context that does not live in the repo.
        data = get_stdin_context()
        return (
            "CONTEXT_MODE: stdin\n"
            "Additional text context from stdin follows below.\n\n"
            f"{data}"
        )
    raise SystemExit(f"Unknown context mode: {context_mode}")
