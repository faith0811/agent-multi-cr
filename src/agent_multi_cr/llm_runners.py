import os
import sys
from typing import List, Optional

from .auditors import Auditor, extract_and_update_memo, load_memo
from .prompts import (
    build_arbiter_prompt,
    build_followup_prompt,
    build_initial_review_prompt,
    build_peer_review_prompt,
)
from .shell_utils import run_shell


def _run_auditor_llm_with_memo(
    *,
    auditor: Auditor,
    label: str,
    prompt: str,
    memo_before: str,
    verbose: bool,
    error_heading: str,
    error_body_prefix: str,
):
    """Shared helper to run an auditor LLM call and handle memo updates."""
    if verbose:
        sys.stderr.write(
            f"\n==== LLM PROMPT BEGIN [{label}] ====\n{prompt}\n"
            f"==== LLM PROMPT END   [{label}] ====\n"
        )
        sys.stderr.flush()
    try:
        if auditor.kind == "codex":
            raw = run_codex(auditor, prompt)
        elif auditor.kind == "gemini":
            raw = run_gemini(auditor, prompt)
        else:
            raise ValueError(f"Unknown auditor kind: {auditor.kind}")
    except Exception as e:
        sys.stderr.write(f"Error running {label}: {e}\n")
        return f"{error_heading}\n\n{error_body_prefix}{e}", memo_before

    cleaned, memo_after = extract_and_update_memo(auditor, raw, memo_before)
    return cleaned, memo_after


def _build_codex_cmd(
    model_name: str,
    reasoning_effort: Optional[str] = None,
    search: bool = True,
) -> List[str]:
    cmd: List[str] = ["codex", "--yolo"]
    if search:
        cmd.append("--search")
    if reasoning_effort:
        cmd.extend(["--config", f"model_reasoning_effort={reasoning_effort}"])
    cmd.extend(
        [
            "exec",
            "--model",
            model_name,
            "-",
        ]
    )
    return cmd


def run_codex(auditor: Auditor, prompt: str) -> str:
    """
    Call Codex CLI in non-interactive mode for a given auditor.

    We invoke Codex in this auditor's private working directory and rely on the
    prompt (and the fact that this is a copy of the repo) to enforce
    "do not modify the code".
    """
    if not auditor.reasoning_effort:
        raise ValueError(f"Codex auditor {auditor.name} is missing reasoning_effort.")

    cmd = _build_codex_cmd(
        model_name=auditor.model_name,
        reasoning_effort=auditor.reasoning_effort,
        search=True,
    )
    # Use the default shell timeout, which can be overridden via
    # AGENT_MULTI_CR_SHELL_TIMEOUT_SEC.
    return run_shell(cmd, input_text=prompt, cwd=auditor.workdir)


def run_gemini(auditor: Auditor, prompt: str) -> str:
    """
    Call Gemini CLI in non-interactive mode.

    We run Gemini in the auditor's private working directory, which does not contain
    the real project files (only any scratch files or memo files we may create).
    """
    cmd = ["gemini", "--yolo", "--model", auditor.model_name]
    # Use the default shell timeout, which can be overridden via
    # AGENT_MULTI_CR_SHELL_TIMEOUT_SEC.
    return run_shell(cmd, input_text=prompt, cwd=auditor.workdir)


def run_auditor_initial_review(
    auditor: Auditor,
    task_description: str,
    context_text: str,
    verbose: bool = False,
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

    label = f"{auditor.name} initial review"
    return _run_auditor_llm_with_memo(
        auditor=auditor,
        label=label,
        prompt=prompt,
        memo_before=memo_before,
        verbose=verbose,
        error_heading="## Review Failed",
        error_body_prefix="Auditor encountered an error: ",
    )


def run_auditor_followup(
    auditor: Auditor,
    task_description: str,
    context_text: str,
    initial_review: str,
    question: str,
    qa_snippet: str,
    verbose: bool = False,
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

    label = f"{auditor.name} follow-up"
    return _run_auditor_llm_with_memo(
        auditor=auditor,
        label=label,
        prompt=prompt,
        memo_before=memo_before,
        verbose=verbose,
        error_heading="## Reply Failed",
        error_body_prefix="Error: ",
    )


def run_reviewer_peer_round(
    auditor: Auditor,
    task_description: str,
    context_text: str,
    own_review: str,
    other_reviews_block: str,
    round_index: int,
    max_rounds: int,
    verbose: bool = False,
):
    """
    Run a cross-check round where one reviewer compares their review with others.

    Returns (cleaned_output_without_memo_json, updated_memo_text).
    """
    memo_before = load_memo(auditor)
    prompt = build_peer_review_prompt(
        reviewer_name=auditor.name,
        task_description=task_description,
        context_text=context_text,
        own_review=own_review,
        other_reviews_block=other_reviews_block,
        memo_text=memo_before,
        round_index=round_index,
        max_rounds=max_rounds,
    )

    label = f"{auditor.name} peer round {round_index}"
    return _run_auditor_llm_with_memo(
        auditor=auditor,
        label=label,
        prompt=prompt,
        memo_before=memo_before,
        verbose=verbose,
        error_heading="## Cross-check Failed",
        error_body_prefix="Error: ",
    )


def parse_arbiter_json(raw: str):
    """Parse arbiter response as JSON (best effort)."""
    import json

    raw = raw.strip()
    parsed = None

    # 1. Try parsing the whole string
    try:
        parsed = json.loads(raw)
    except Exception:
        pass

    # 2. If that failed or yielded non-dict, try finding a JSON object block
    if not isinstance(parsed, dict):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                pass

    # 3. Validate result. If we have a dict, return it.
    if isinstance(parsed, dict):
        return parsed

    # 4. Fallback: treat the entire raw text as the final markdown.
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
    include_p2_p3: bool = False,
    allow_queries: bool = True,
    verbose: bool = False,
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
        include_p2_p3=include_p2_p3,
        allow_queries=allow_queries,
    )
    # Arbiter failures probably should be fatal or return a special "stop" state?
    # For now, we let them bubble up or the caller handles them?
    # The caller `run_pipeline` catches exceptions in the thread pool for initial reviews,
    # but `run_arbiter_step` is run in the main thread.
    # If arbiter fails, we probably want to stop.
    label = f"Arbiter {arbiter.name} step {query_count + 1}"
    if verbose:
        import sys

        sys.stderr.write(
            f"\n==== LLM PROMPT BEGIN [{label}] ====\n{prompt}\n"
            f"==== LLM PROMPT END   [{label}] ====\n"
        )
        sys.stderr.flush()

    if arbiter.kind == "codex":
        raw = run_codex(arbiter, prompt)
    elif arbiter.kind == "gemini":
        raw = run_gemini(arbiter, prompt)
    else:
        raise ValueError(f"Unknown arbiter kind: {arbiter.kind}")

    return parse_arbiter_json(raw)


def translate_markdown_to_zh(markdown: str) -> str:
    """
    Translate a Markdown code review into Simplified Chinese using Codex CLI.

    This is only used on the final unified report. We keep all Markdown
    structure and code blocks, and only translate natural language.
    """
    from textwrap import dedent

    prompt = dedent(
        f"""
        You are a professional technical translator.

        Task:
        - Translate the following Markdown software code review into Simplified Chinese.
        - Keep all Markdown structure, headings, bullet lists, code blocks, and inline
          code (text inside backticks) exactly as they are.
        - Do not add any new commentary or explanations.
        - Translate only natural language; keep code identifiers, file paths, and
          programming language keywords in their original form.

        <INPUT_MARKDOWN>
        {markdown}
        </INPUT_MARKDOWN>
        """
    ).strip()

    # For translation we do not need repo search, so we disable it.
    cmd = _build_codex_cmd(model_name="gpt-5.1", reasoning_effort=None, search=False)
    # Use current working directory; translation does not depend on repo files.
    # Timeout is controlled by AGENT_MULTI_CR_SHELL_TIMEOUT_SEC or the default.
    return run_shell(cmd, input_text=prompt, cwd=os.getcwd())
