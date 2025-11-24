import os
import subprocess
import sys


def _ensure_git_diff_exists(use_cached: bool, repo_root: str) -> None:
    """Fail fast when there is no diff without materializing the full patch."""
    diff_cmd = ["git", "diff", "--cached", "--quiet"] if use_cached else [
        "git",
        "diff",
        "--quiet",
    ]
    try:
        # git diff --quiet exit codes:
        #   0 -> no differences
        #   1 -> differences found
        #  >1 -> error
        result = subprocess.run(
            diff_cmd,
            cwd=repo_root or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # pragma: no cover - very unlikely in normal use
        raise SystemExit(f"Failed to run git diff to check for changes: {exc}")

    if result.returncode == 0:
        raise SystemExit("No git diff found (nothing to review).")
    if result.returncode != 1:
        raise SystemExit(
            f"git diff exited with status {result.returncode}; "
            "are you running inside a git repository?",
        )


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
        _ensure_git_diff_exists(use_cached, repo_root)
        return (
            "CONTEXT_MODE: diff\n"
            "The code under review is the current git diff in this repository. "
            "Use your available tools (e.g., git diff, search, file inspection) "
            "to examine the changed code directly from the working directory."
        )
    if context_mode == "repo":
        # Do not inline a full repository snapshot; Codex/Gemini CLIs can read
        # files directly from their working directory. We only provide a brief
        # textual description so the models know how to interpret their tools,
        # without exposing the user's real filesystem paths.
        return (
            "CONTEXT_MODE: repo\n"
            "The code under review is the repository available in your current "
            "working directory. You have access to these files via your tools; "
            "inspect whatever you need directly from disk instead of relying "
            "on inlined code."
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
