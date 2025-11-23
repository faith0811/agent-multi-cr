import os
import shutil
from typing import Dict, List, Optional, Tuple

from .auditors import Auditor, slugify
from .context import resolve_context_text
from .llm_runners import (
    run_auditor_followup,
    run_auditor_initial_review,
    run_arbiter_step,
)


def build_qa_snippet_for_reviewer(
    qa_history: List[Dict[str, str]],
    reviewer_name: str,
) -> str:
    """Build a small text snippet of previous Q&A for the given reviewer."""
    parts: List[str] = []
    for item in qa_history:
        if item["reviewer"] != reviewer_name:
            continue
        parts.append(
            f"Question:\n{item['question']}\n\nAnswer:\n{item['answer']}\n"
        )
    return "\n---\n".join(parts) if parts else "(no previous Q&A with you)"


def _copy_repo_into_workdir(repo_root: str, workdir: str) -> None:
    """
    Copy the current repo contents into the auditor's working directory.

    We deliberately avoid copying VCS / cache directories so we do not create
    recursive .multi_cr_auditors trees or large .git clones.
    """
    shutil.copytree(
        repo_root,
        workdir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".multi_cr_auditors", "__pycache__", "*.pyc"),
    )


def run_pipeline(
    context_mode: str,
    use_cached: bool,
    task_description: str,
    codex_configs: List[Tuple[str, str]],
    gemini_model: str,
    arbiter_family: str,
    max_queries: int,
    base_workdir: str,
    max_context_files: int,
    max_context_bytes_per_file: int,
) -> str:
    """
    Main entry point for the multi-model review pipeline.
    """
    repo_root = os.getcwd()
    context_text = resolve_context_text(
        context_mode=context_mode,
        use_cached=use_cached,
        repo_root=repo_root,
        max_context_files=max_context_files,
        max_context_bytes_per_file=max_context_bytes_per_file,
    )

    base_workdir = os.path.abspath(base_workdir)
    os.makedirs(base_workdir, exist_ok=True)

    auditors: List[Auditor] = []

    for model_name, effort in codex_configs:
        name = f"Codex[{model_name}|{effort}]"
        workdir = os.path.join(base_workdir, slugify(name))
        _copy_repo_into_workdir(repo_root, workdir)
        auditors.append(
            Auditor(
                name=name,
                kind="codex",
                model_name=model_name,
                workdir=workdir,
                reasoning_effort=effort,
            )
        )

    gemini_name = f"Gemini[{gemini_model}]"
    gemini_workdir = os.path.join(base_workdir, slugify(gemini_name))
    _copy_repo_into_workdir(repo_root, gemini_workdir)
    gemini_auditor = Auditor(
        name=gemini_name,
        kind="gemini",
        model_name=gemini_model,
        workdir=gemini_workdir,
        reasoning_effort=None,
    )
    auditors.append(gemini_auditor)

    if not auditors:
        raise SystemExit("You must have at least one auditor configured.")

    # Choose arbiter
    if arbiter_family == "codex":
        arbiter = auditors[0]
    elif arbiter_family == "gemini":
        arbiter = gemini_auditor
    else:
        raise ValueError(f"Unknown arbiter_family: {arbiter_family}")

    print("▶ Round 1: independent reviews from each auditor...\n")

    initial_reviews: Dict[str, str] = {}
    for auditor in auditors:
        print(f"  - {auditor.name} reviewing...")
        review, _memo = run_auditor_initial_review(
            auditor=auditor,
            task_description=task_description,
            context_text=context_text,
        )
        initial_reviews[auditor.name] = review

    qa_history: List[Dict[str, str]] = []
    query_count = 0
    final_markdown: Optional[str] = None

    print("\n▶ Arbiter loop: asking clarifications when needed...\n")

    while query_count < max_queries:
        control = run_arbiter_step(
            arbiter=arbiter,
            task_description=task_description,
            context_text=context_text,
            auditors=auditors,
            initial_reviews=initial_reviews,
            qa_history=qa_history,
            max_queries=max_queries,
            query_count=query_count,
        )

        state = control.get("state")
        if state == "final":
            final_markdown = control.get("final_markdown", "").strip()
            break

        if state == "query":
            target = control.get("target_reviewer")
            question = control.get("question", "").strip()
            if not target or not question:
                final_markdown = control.get("final_markdown", "") or str(control)
                break

            print(f"  - Arbiter asks follow-up question to {target}...")

            target_auditor = next((a for a in auditors if a.name == target), None)
            if target_auditor is None:
                final_markdown = f"Arbiter referred to unknown reviewer {target}. Raw control: {control}"
                break

            qa_snippet = build_qa_snippet_for_reviewer(qa_history, target)
            initial_review_text = initial_reviews[target]

            answer, _memo = run_auditor_followup(
                auditor=target_auditor,
                task_description=task_description,
                context_text=context_text,
                initial_review=initial_review_text,
                question=question,
                qa_snippet=qa_snippet,
            )

            qa_history.append(
                {
                    "reviewer": target,
                    "question": question,
                    "answer": answer,
                }
            )
            query_count += 1
            continue

        final_markdown = control.get("final_markdown", "") or str(control)
        break

    if final_markdown is None:
        final_control = run_arbiter_step(
            arbiter=arbiter,
            task_description=task_description,
            context_text=context_text,
            auditors=auditors,
            initial_reviews=initial_reviews,
            qa_history=qa_history,
            max_queries=max_queries,
            query_count=max_queries,
        )
        final_markdown = final_control.get("final_markdown", "") or str(final_control)

    parts: List[str] = []
    parts.append(f"# Joint Code Review (arbiter: {arbiter.name})\n")
    parts.append("\n## 1. Task\n\n")
    parts.append(task_description.strip() or "(no task description provided)")
    parts.append("\n\n---\n\n")
    parts.append("## 2. Unified review\n\n")
    parts.append(final_markdown)
    parts.append("\n\n---\n\n")
    parts.append("## 3. Initial reviews from each auditor\n")
    for name, review in initial_reviews.items():
        parts.append(f"\n### {name}\n\n{review}\n")

    if qa_history:
        parts.append("\n---\n\n## 4. Arbiter follow-up Q&A log\n")
        for i, item in enumerate(qa_history, start=1):
            parts.append(
                f"\n### Q&A #{i} with {item['reviewer']}\n\n"
                f"**Question:**\n\n{item['question']}\n\n"
                f"**Answer:**\n\n{item['answer']}\n"
            )

    return "".join(parts).strip()
