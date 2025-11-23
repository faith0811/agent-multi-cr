import os

from .auditors import Auditor, extract_and_update_memo, load_memo
from .prompts import build_arbiter_prompt, build_followup_prompt, build_initial_review_prompt
from .shell_utils import run_shell


def run_codex(auditor: Auditor, prompt: str) -> str:
    """
    Call Codex CLI in non-interactive mode for a given auditor.

    We invoke Codex in this auditor's private working directory and rely on the
    prompt (and the fact that this is a copy of the repo) to enforce
    "do not modify the code".
    """
    if not auditor.reasoning_effort:
        raise ValueError(f"Codex auditor {auditor.name} is missing reasoning_effort.")

    cmd = [
        "codex",
        "--yolo",
        "--search",
        "--config",
        f"model_reasoning_effort={auditor.reasoning_effort}",
        "exec",
        "--model",
        auditor.model_name,
        "-",
    ]
    return run_shell(cmd, input_text=prompt, cwd=auditor.workdir)


def run_gemini(auditor: Auditor, prompt: str) -> str:
    """
    Call Gemini CLI in non-interactive mode.

    We run Gemini in the auditor's private working directory, which does not contain
    the real project files (only any scratch files or memo files we may create).
    """
    cmd = ["gemini", "--yolo", "--model", auditor.model_name]
    return run_shell(cmd, input_text=prompt, cwd=auditor.workdir)


def run_auditor_initial_review(
    auditor: Auditor,
    task_description: str,
    context_text: str,
):
    """
    Run the initial, independent review for one auditor.

    Returns (cleaned_output_without_memo_json, updated_memo_text).
    """
    os.makedirs(auditor.workdir, exist_ok=True)
    memo_before = load_memo(auditor)
    prompt = build_initial_review_prompt(
        reviewer_name=auditor.name,
        task_description=task_description,
        context_text=context_text,
        memo_text=memo_before,
    )

    if auditor.kind == "codex":
        raw = run_codex(auditor, prompt)
    elif auditor.kind == "gemini":
        raw = run_gemini(auditor, prompt)
    else:
        raise ValueError(f"Unknown auditor kind: {auditor.kind}")

    cleaned, memo_after = extract_and_update_memo(auditor, raw, memo_before)
    return cleaned, memo_after


def run_auditor_followup(
    auditor: Auditor,
    task_description: str,
    context_text: str,
    initial_review: str,
    question: str,
    qa_snippet: str,
):
    """
    Run a follow-up Q&A round with one auditor.

    Returns (cleaned_output_without_memo_json, updated_memo_text).
    """
    memo_before = load_memo(auditor)
    prompt = build_followup_prompt(
        reviewer_name=auditor.name,
        task_description=task_description,
        context_text=context_text,
        initial_review=initial_review,
        question=question,
        qa_snippet=qa_snippet,
        memo_text=memo_before,
    )

    if auditor.kind == "codex":
        raw = run_codex(auditor, prompt)
    elif auditor.kind == "gemini":
        raw = run_gemini(auditor, prompt)
    else:
        raise ValueError(f"Unknown auditor kind: {auditor.kind}")

    cleaned, memo_after = extract_and_update_memo(auditor, raw, memo_before)
    return cleaned, memo_after


def parse_arbiter_json(raw: str):
    """Parse arbiter response as JSON (best effort)."""
    import json

    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {"state": "final", "final_markdown": raw}


def run_arbiter_step(
    arbiter: Auditor,
    task_description: str,
    context_text: str,
    auditors,
    initial_reviews,
    qa_history,
    max_queries: int,
    query_count: int,
):
    """Run one step of the arbiter loop and return the parsed JSON control object."""
    prompt = build_arbiter_prompt(
        arbiter_name=arbiter.name,
        task_description=task_description,
        context_text=context_text,
        auditors=auditors,
        initial_reviews=initial_reviews,
        qa_history=qa_history,
        max_queries=max_queries,
        query_count=query_count,
    )
    if arbiter.kind == "codex":
        raw = run_codex(arbiter, prompt)
    elif arbiter.kind == "gemini":
        raw = run_gemini(arbiter, prompt)
    else:
        raise ValueError(f"Unknown arbiter kind: {arbiter.kind}")

    return parse_arbiter_json(raw)
