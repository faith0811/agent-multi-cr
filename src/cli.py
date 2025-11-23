import argparse
from typing import List, Tuple

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-model code review using Codex (multiple models) + Gemini, "
            "with an arbiter that can iteratively query individual reviewers. "
            "You specify the review task in free-form English, and choose the context "
            "(git diff, repo snapshot, or stdin). Each auditor gets its own private "
            "working directory and memo."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=["diff", "repo", "stdin"],
        default="diff",
        help=(
            "What to use as the shared context for the review:\n"
            "  diff  - use git diff / git diff --cached (default)\n"
            "  repo  - use a repository snapshot (limited by file count and size)\n"
            "  stdin - read context from stdin (e.g. `gh pr diff 123 | ...`)"
        ),
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="When context-mode=diff, use staged changes (git diff --cached) instead of working tree.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Please review this code context for correctness, readability, and potential issues.",
        help=(
            "Free-form description of what you want the reviewers to do, in English.\n"
            "Example: \"Review the entire repository with a focus on security and concurrency issues.\""
        ),
    )
    parser.add_argument(
        "--codex-model",
        action="append",
        dest="codex_models",
        default=[],
        help=(
            "Codex model name (can be used multiple times). "
            "If omitted, defaults to two auditors: "
            "gpt-5.1 (high) and gpt-5.1-codex-max (xhigh)."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model id (default: gemini-3-pro-preview).",
    )
    parser.add_argument(
        "--arbiter-family",
        choices=["codex", "gemini"],
        default="codex",
        help=(
            "Which family acts as arbiter for the final decision: "
            "'codex' (default, uses the first Codex model) or 'gemini'."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=4,
        help="Maximum number of clarification questions the arbiter may ask (default: 4).",
    )
    parser.add_argument(
        "--auditors-workdir",
        type=str,
        default=".multi_cr_auditors",
        help=(
            "Base directory under which each auditor gets its own working directory "
            "(default: .multi_cr_auditors under the current repo)."
        ),
    )
    parser.add_argument(
        "--max-context-files",
        type=int,
        default=40,
        help="When context-mode=repo, maximum number of files to include in the snapshot (default: 40).",
    )
    parser.add_argument(
        "--max-context-bytes-per-file",
        type=int,
        default=4000,
        help="When context-mode=repo, maximum bytes per file to include (default: 4000).",
    )

    args = parser.parse_args()

    if args.codex_models:
        codex_configs: List[Tuple[str, str]] = [(name, "high") for name in args.codex_models]
    else:
        codex_configs = [
            ("gpt-5.1", "high"),
            ("gpt-5.1-codex-max", "xhigh"),
        ]

    result = run_pipeline(
        context_mode=args.context_mode,
        use_cached=args.cached,
        task_description=args.task,
        codex_configs=codex_configs,
        gemini_model=args.gemini_model,
        arbiter_family=args.arbiter_family,
        max_queries=args.max_queries,
        base_workdir=args.auditors_workdir,
        max_context_files=args.max_context_files,
        max_context_bytes_per_file=args.max_context_bytes_per_file,
    )

    print(result)

