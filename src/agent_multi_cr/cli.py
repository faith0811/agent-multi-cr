import argparse
from typing import List, Tuple

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-model code review using Codex (multiple models) + Gemini, "
            "with an arbiter that can iteratively query individual reviewers. "
            "You specify the review task in free-form English, and choose the context "
            "(git diff, repo, or stdin). Each auditor gets its own private "
            "working directory and memo."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=["diff", "repo", "stdin"],
        default="repo",
        help=(
            "What to use as the shared context for the review:\n"
            "  diff  - use git diff / git diff --cached\n"
            "  repo  - use the repository in the current working directory (default)\n"
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
            "'codex' (default, uses a dedicated gpt-5.1-codex|low arbiter "
            "separate from the reviewer models) or 'gemini' (uses the "
            "configured Gemini model as arbiter)."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum number of clarification questions the arbiter may ask (default: 50).",
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
        "--include-p2-p3",
        action="store_true",
        help=(
            "If set, include P2/P3 issues in the final output. "
            "By default, only P0/P1 issues are shown in detail."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print every prompt sent to Codex/Gemini to stderr.",
    )
    parser.add_argument(
        "--output-lang",
        choices=["en", "zh"],
        default="zh",
        help="Language of the final report: 'zh' (Chinese, default) or 'en'.",
    )
    parser.add_argument(
        "--arbiter-round-mode",
        choices=["single", "multi"],
        default="single",
        help=(
            "Arbiter behavior: 'single' (default) produces a one-shot summary "
            "with no follow-up questions; 'multi' allows multiple rounds of "
            "clarification questions."
        ),
    )

    args = parser.parse_args()

    if args.max_queries < 0:
        parser.error("--max-queries must be non-negative.")

    if args.codex_models:
        codex_configs: List[Tuple[str, str]] = []
        for val in args.codex_models:
            if ":" in val:
                m, e = val.split(":", 1)
                codex_configs.append((m, e))
            else:
                codex_configs.append((val, "high"))
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
        verbose=args.verbose,
        output_lang=args.output_lang,
        include_p2_p3=args.include_p2_p3,
        arbiter_round_mode=args.arbiter_round_mode,
    )

    print(result)
