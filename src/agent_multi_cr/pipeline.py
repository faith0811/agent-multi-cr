import os
import shutil
import subprocess
import sys
import threading
import time
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .auditors import Auditor, slugify
from .context import resolve_context_text
from .llm_runners import (
    run_auditor_followup,
    run_auditor_initial_review,
    run_arbiter_step,
    run_reviewer_peer_round,
    translate_markdown_to_zh,
)


WORKDIR_MARKER = ".agent_multi_cr_workspace"

_CLEANUP_RUN_WORKDIRS: List[str] = []
_ATEEXIT_REGISTERED = False


def _atexit_cleanup_run_workdirs() -> None:
    """Best-effort cleanup for any leftover per-run workdirs on process exit."""
    for path in list(_CLEANUP_RUN_WORKDIRS):
        if os.path.isdir(path):
            marker = os.path.join(path, WORKDIR_MARKER)
            if os.path.exists(marker):
                _cleanup_workdir(path)
        try:
            _CLEANUP_RUN_WORKDIRS.remove(path)
        except ValueError:
            pass


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


def _copy_repo_into_workdir(repo_root: str, workdir: str, include_git: bool) -> None:
    """
    Copy the current repo contents into the auditor's working directory.

    We deliberately avoid copying VCS / cache directories and obviously noisy
    or sensitive files so we do not create recursive .multi_cr_auditors trees,
    large dependency directories, or duplicate secrets.
    """
    extra_excludes_env = os.environ.get("AGENT_MULTI_CR_COPY_EXCLUDES", "")
    extra_exclude_patterns = [
        p.strip()
        for p in extra_excludes_env.split(",")
        if p.strip()
    ]
    ignore_patterns = [
        ".multi_cr_auditors",
        "__pycache__",
        "*.pyc",
        ".venv",
        "venv",
        "env",
        "ENV",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".idea",
        ".vscode",
        ".env*",
        "*.log",
        "*.sqlite",
        "*.sqlite3",
        "*.db",
    ]
    if not include_git:
        ignore_patterns.insert(0, ".git")
    ignore_patterns.extend(extra_exclude_patterns)

    pattern_ignore = shutil.ignore_patterns(*ignore_patterns)
    git_dir = os.path.join(repo_root, ".git")
    has_git = os.path.isdir(git_dir)

    # Bound the number of git check-ignore subprocesses so large repositories do
    # not spawn an excessive number of short-lived processes. The limit can be
    # tuned via AGENT_MULTI_CR_GIT_CHECK_IGNORE_LIMIT (default: 200).
    limit_env = os.environ.get("AGENT_MULTI_CR_GIT_CHECK_IGNORE_LIMIT")
    try:
        git_check_ignore_limit = int(limit_env) if limit_env is not None else 200
    except ValueError:
        git_check_ignore_limit = 200
    git_check_ignore_used = 0

    def ignore(dirpath: str, names: List[str]) -> List[str]:
        nonlocal git_check_ignore_used
        # Start with pattern-based ignores.
        ignored = set(pattern_ignore(dirpath, names))

        # If not a git repo or the limit for git check-ignore calls is exhausted,
        # we are done. The pattern-based ignores still apply.
        if not has_git or git_check_ignore_limit <= 0 or git_check_ignore_used >= git_check_ignore_limit:
            return list(ignored)

        # Identify candidates that are NOT yet ignored by patterns or already skipped
        candidates = [n for n in names if n not in ignored]
        if not candidates:
            return list(ignored)

        # Build relative paths for git check-ignore
        rel_dir = os.path.relpath(dirpath, repo_root)
        check_map = {}
        for name in candidates:
            full_path = os.path.join(dirpath, name)
            # Never follow symlinks; skip them entirely to avoid copying
            # arbitrary host files or creating recursive trees.
            if os.path.islink(full_path):
                ignored.add(name)
                continue
            path = os.path.join(rel_dir, name) if rel_dir != "." else name
            check_map[path] = name

        if not check_map:
            return list(ignored)

        # Batch check using git check-ignore --stdin -z
        # -z uses NUL as terminator for input and output
        try:
            git_check_ignore_used += 1
            proc = subprocess.Popen(
                ["git", "check-ignore", "--stdin", "-z"],
                cwd=repo_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            input_data = "\0".join(check_map.keys()) + "\0"
            stdout, _ = proc.communicate(input_data.encode("utf-8"))

            if proc.returncode in (0, 1):
                # 0 = some ignored, 1 = none ignored (but command ran ok)
                # stdout contains the paths that ARE ignored, NUL-separated
                ignored_paths = stdout.split(b"\0")
                for ip in ignored_paths:
                    if not ip:
                        continue
                    ip_str = ip.decode("utf-8")
                    # We sent relative paths, we get back relative paths (or absolute if we sent abs?)
                    # git check-ignore usually echoes the input format.
                    if ip_str in check_map:
                        ignored.add(check_map[ip_str])
        except Exception:
            # If git fails, fall back to copying (safe default)
            pass

        return list(ignored)

    shutil.copytree(
        repo_root,
        workdir,
        symlinks=False,  # Changed to False for isolation security
        dirs_exist_ok=True,
        ignore=ignore,
    )


def _cleanup_workdir(path: str) -> None:
    """Remove an existing auditors workdir tree, if present."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def run_pipeline(
    context_mode: str,
    use_cached: bool,
    task_description: str,
    codex_configs: List[Tuple[str, str]],
    gemini_model: str,
    arbiter_family: str,
    max_queries: int,
    base_workdir: str,
    verbose: bool = False,
    output_lang: str = "zh",
    include_p2_p3: bool = False,
    arbiter_round_mode: str = "single",
) -> str:
    """
    Main entry point for the multi-model review pipeline.
    """
    repo_root = os.getcwd()

    # Resolve and validate the auditors workdir root. We require it to live under
    # the current repo root (but not equal to it) to avoid accidentally
    # deleting unrelated directories when we clean up.
    if os.path.isabs(base_workdir):
        candidate = os.path.realpath(base_workdir)
    else:
        candidate = os.path.realpath(os.path.join(repo_root, base_workdir))

    repo_root_real = os.path.realpath(repo_root)
    if candidate == repo_root_real or not candidate.startswith(repo_root_real + os.sep):
        raise SystemExit(
            f"--auditors-workdir must point to a directory inside the repo root "
            f"({repo_root_real}), not '{base_workdir}'. Refusing to delete it."
        )

    base_workdir_abs = candidate

    # Use a per-run subdirectory under the auditors root to avoid concurrency
    # issues when multiple runs share the same --auditors-workdir.
    run_id = f"run-{int(time.time())}-{os.getpid()}"
    run_workdir_abs = os.path.join(base_workdir_abs, run_id)

    # Ensure the root exists but do not delete it; only the per-run directory
    # will be cleaned up at the end of this run.
    os.makedirs(base_workdir_abs, exist_ok=True)

    # Track this run's workdir for best-effort cleanup on process exit.
    global _ATEEXIT_REGISTERED
    _CLEANUP_RUN_WORKDIRS.append(run_workdir_abs)
    if not _ATEEXIT_REGISTERED:
        atexit.register(_atexit_cleanup_run_workdirs)
        _ATEEXIT_REGISTERED = True

    print(f"▶ Preparing auditors workdir for this run at {run_workdir_abs}...", flush=True)
    if os.path.isdir(run_workdir_abs):
        marker_path = os.path.join(run_workdir_abs, WORKDIR_MARKER)
        if not os.path.exists(marker_path):
            raise SystemExit(
                f"Refusing to delete existing auditors workdir '{run_workdir_abs}' "
                f"because it does not contain the expected marker file {WORKDIR_MARKER!r}. "
                "Please choose a different path or remove it manually if you are sure "
                "it is safe."
            )
        _cleanup_workdir(run_workdir_abs)
    print("  ✓ Run workdir ready.\n", flush=True)

    print(f"▶ Preparing context (mode={context_mode}, repo_root={repo_root})...", flush=True)
    context_text = resolve_context_text(
        context_mode=context_mode,
        use_cached=use_cached,
        repo_root=repo_root,
    )
    print(f"  ✓ Context ready ({len(context_text):,} characters).\n", flush=True)

    os.makedirs(run_workdir_abs, exist_ok=True)
    marker_path = os.path.join(run_workdir_abs, WORKDIR_MARKER)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("agent-multi-cr auditors workspace\n")

    # Shared progress state for interactive summaries.
    start_time = time.time()
    progress_lock = threading.Lock()
    progress: Dict[str, object] = {
        "phase": "setup",
        # Filled in after we determine which Codex configs are used as reviewers.
        "auditors_total": 0,
        "auditors_done": 0,
        "arbiter_step": 0,
        "arbiter_max": max_queries,
    }
    stop_progress = threading.Event()

    def _progress_reporter() -> None:
        last_len = 0
        # Emit a progress snapshot every 5 seconds until the run finishes.
        if stop_progress.wait(5.0):
            return
        while not stop_progress.is_set():
            with progress_lock:
                phase = progress.get("phase")
                auditors_total = int(progress.get("auditors_total", 0))
                auditors_done = int(progress.get("auditors_done", 0))
                arbiter_step = int(progress.get("arbiter_step", 0))
                arbiter_max_local = int(progress.get("arbiter_max", 0))
            elapsed = int(time.time() - start_time)
            line = (
                f"▶ Status[{elapsed}s]: phase={phase}, "
                f"auditors={auditors_done}/{auditors_total}, "
                f"arbiter_steps={arbiter_step}/{arbiter_max_local}"
            )
            padded = line.ljust(last_len)
            sys.stderr.write("\r" + padded)
            sys.stderr.flush()
            last_len = len(line)
            if stop_progress.wait(5.0):
                break

    reporter_thread = threading.Thread(target=_progress_reporter, daemon=True)
    reporter_thread.start()

    auditors: List[Auditor] = []

    # Determine Codex configs used for reviewers vs arbiter.
    arbiter: Optional[Auditor] = None
    # Never use gpt-5.1-codex|low as a reviewer when Codex acts as arbiter;
    # in that mode it is reserved for arbiter only.
    if arbiter_family == "codex":
        codex_auditor_configs: List[Tuple[str, str]] = []
        for model_name, effort in codex_configs:
            if model_name == "gpt-5.1-codex" and effort == "low":
                sys.stderr.write(
                    "Ignoring Codex[gpt-5.1-codex|low] as a reviewer because it "
                    "is reserved for use as the Codex arbiter.\n"
                )
                sys.stderr.flush()
                continue
            codex_auditor_configs.append((model_name, effort))
    else:
        codex_auditor_configs = list(codex_configs)

    if arbiter_family == "codex":
        # Use a dedicated low-effort Codex model as arbiter, separate from reviewers.
        arb_model = "gpt-5.1-codex"
        arb_effort = "low"
        arbiter_name = f"Codex[{arb_model}|{arb_effort}]"
        arbiter_workdir = os.path.join(run_workdir_abs, slugify(f"Arbiter-{arbiter_name}"))
        # Arbiter workspace intentionally does NOT contain the repo copy; it should
        # base decisions only on reviewers' messages and Q&A, not on direct code access.
        os.makedirs(arbiter_workdir, exist_ok=True)
        arbiter = Auditor(
            name=arbiter_name,
            kind="codex",
            model_name=arb_model,
            workdir=arbiter_workdir,
            reasoning_effort=arb_effort,
        )

    # Total planned auditors are Codex reviewers (if any) plus Gemini.
    total_planned_auditors = len(codex_auditor_configs) + 1  # +1 for Gemini
    with progress_lock:
        progress["auditors_total"] = total_planned_auditors

    # Set up Codex auditors (excluding arbiter-only config when arbiter_family=codex)
    for idx, (model_name, effort) in enumerate(codex_auditor_configs, start=1):
        name = f"Codex[{model_name}|{effort}]"
        slug = slugify(name)
        workdir = os.path.join(run_workdir_abs, slug)
        memo_root = os.path.join(base_workdir_abs, "memos", slug)
        print(f"▶ Setting up auditor {idx}/{total_planned_auditors}: {name}", flush=True)
        _copy_repo_into_workdir(
            repo_root=repo_root,
            workdir=workdir,
            include_git=context_mode == "diff",
        )
        auditors.append(
            Auditor(
                name=name,
                kind="codex",
                model_name=model_name,
                workdir=workdir,
                reasoning_effort=effort,
                memo_root=memo_root,
            )
        )

    # Set up Gemini auditor reviewer.
    gemini_index = total_planned_auditors
    gemini_name = f"Gemini[{gemini_model}]"
    gemini_slug = slugify(gemini_name)
    gemini_workdir = os.path.join(run_workdir_abs, gemini_slug)
    gemini_memo_root = os.path.join(base_workdir_abs, "memos", gemini_slug)
    print(f"▶ Setting up auditor {gemini_index}/{total_planned_auditors}: {gemini_name}", flush=True)
    _copy_repo_into_workdir(
        repo_root=repo_root,
        workdir=gemini_workdir,
        include_git=context_mode == "diff",
    )
    gemini_auditor = Auditor(
        name=gemini_name,
        kind="gemini",
        model_name=gemini_model,
        workdir=gemini_workdir,
        reasoning_effort=None,
        memo_root=gemini_memo_root,
    )
    auditors.append(gemini_auditor)

    # If Gemini acts as arbiter, give it its own isolated workspace without a repo copy.
    if arbiter_family == "gemini":
        arbiter_name = f"Gemini-Arbiter[{gemini_model}]"
        arbiter_workdir = os.path.join(run_workdir_abs, slugify(arbiter_name))
        os.makedirs(arbiter_workdir, exist_ok=True)
        arbiter = Auditor(
            name=arbiter_name,
            kind="gemini",
            model_name=gemini_model,
            workdir=arbiter_workdir,
            reasoning_effort=None,
        )

    # Choose arbiter
    if arbiter_family in ("codex", "gemini"):
        if arbiter is None:
            raise SystemExit(
                "Internal error: arbiter was not initialized."
            )
    else:
        raise ValueError(f"Unknown arbiter_family: {arbiter_family}")

    print("\n▶ Round 1: independent reviews from each auditor...\n", flush=True)
    with progress_lock:
        progress["phase"] = "initial_reviews"

    initial_reviews: Dict[str, str] = {}
    num_auditors = len(auditors)

    # Run all auditors in parallel so we don't block on each model sequentially.
    with ThreadPoolExecutor(max_workers=num_auditors or 1) as executor:
        future_to_auditor = {}
        for idx, auditor in enumerate(auditors, start=1):
            print(f"  - [{idx}/{num_auditors}] {auditor.name} queued...", flush=True)
            future = executor.submit(
                run_auditor_initial_review,
                auditor=auditor,
                task_description=task_description,
                context_text=context_text,
                verbose=verbose,
            )
            future_to_auditor[future] = auditor

        completed = 0
        for future in as_completed(future_to_auditor):
            auditor = future_to_auditor[future]
            try:
                review, _memo = future.result()
            except Exception as exc:
                print(f"    ✗ {auditor.name} review failed: {exc}", flush=True)
                raise

            completed += 1
            initial_reviews[auditor.name] = review
            with progress_lock:
                progress["auditors_done"] = completed
            print(
                f"    ✓ [{completed}/{num_auditors}] {auditor.name} review finished.",
                flush=True,
            )

    # Preserve the original initial reviews for the arbiter so it can
    # distinguish between issues that were first proposed in an initial
    # review vs. those adopted later during peer reconciliation.
    raw_initial_reviews: Dict[str, str] = dict(initial_reviews)

    # Reviewer cross-check round before arbiter aggregation: each reviewer sees
    # others' initial reviews once and updates their own view.
    print("\n▶ Round 2: reviewers cross-check each other...\n", flush=True)
    with progress_lock:
        progress["phase"] = "peer_round"

    def _build_other_reviews_block(target_name: str) -> str:
        parts: List[str] = []
        for name, review in initial_reviews.items():
            if name == target_name:
                continue
            parts.append(f'<REVIEW name="{name}">\n{review}\n</REVIEW>')
        return "\n\n".join(parts)

    updated_reviews: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=num_auditors or 1) as executor:
        future_to_auditor = {}
        for auditor in auditors:
            own_review = initial_reviews.get(auditor.name, "")
            other_block = _build_other_reviews_block(auditor.name)
            future = executor.submit(
                run_reviewer_peer_round,
                auditor=auditor,
                task_description=task_description,
                context_text=context_text,
                own_review=own_review,
                other_reviews_block=other_block,
                verbose=verbose,
            )
            future_to_auditor[future] = auditor

        for future in as_completed(future_to_auditor):
            auditor = future_to_auditor[future]
            try:
                review, _memo = future.result()
            except Exception as exc:
                print(f"    ✗ {auditor.name} peer round failed: {exc}", flush=True)
                raise
            updated_reviews[auditor.name] = review
            print(f"    ✓ Peer cross-check finished for {auditor.name}.", flush=True)

    # The arbiter will see both the original initial reviews and the latest
    # cross-checked reviews as its primary input.
    latest_reviews: Dict[str, str] = updated_reviews

    qa_history: List[Dict[str, str]] = []
    query_count = 0
    final_markdown: Optional[str] = None

    def _call_arbiter_step(
        max_queries_for_call: int,
        query_index: int,
        allow_queries_flag: bool,
    ):
        return run_arbiter_step(
            arbiter=arbiter,
            task_description=task_description,
            context_text=context_text,
            auditors=auditors,
            initial_reviews=raw_initial_reviews,
            latest_reviews=latest_reviews,
            qa_history=qa_history,
            max_queries=max_queries_for_call,
            query_count=query_index,
            include_p2_p3=include_p2_p3,
            allow_queries=allow_queries_flag,
            verbose=verbose,
        )

    if arbiter_round_mode == "single":
        print("\n▶ Arbiter: summarizing cross-checked reviews (single-shot)...\n", flush=True)
    else:
        print("\n▶ Arbiter loop: asking clarifications when needed...\n", flush=True)
    with progress_lock:
        progress["phase"] = "arbiter"

    if arbiter_round_mode == "single":
        # One-shot arbiter: no clarification questions allowed.
        print(f"  - Arbiter single-shot decision (no follow-up questions)...", flush=True)
        control = _call_arbiter_step(
            max_queries_for_call=0,
            query_index=0,
            allow_queries_flag=False,
        )

        state = control.get("state")
        if state == "final":
            with progress_lock:
                progress["arbiter_step"] = 0
            final_markdown = control.get("final_markdown", "").strip()
        else:
            # Arbiter incorrectly tried to ask a question in single-shot mode.
            final_markdown = (
                "## Arbiter Error\n\n"
                "Arbiter attempted to ask follow-up questions in single-shot mode.\n\n"
                f"Raw control object:\n\n{control}"
            )
    else:
        while query_count < max_queries:
            print(f"  - Arbiter step {query_count + 1}/{max_queries}...", flush=True)
            control = _call_arbiter_step(
                max_queries_for_call=max_queries,
                query_index=query_count,
                allow_queries_flag=True,
            )

            state = control.get("state")
            if state == "final":
                # No new clarification question was asked in this step.
                # Keep arbiter_step equal to the number of questions asked so far.
                with progress_lock:
                    progress["arbiter_step"] = query_count
                final_markdown = control.get("final_markdown", "").strip()
                break

            if state == "query":
                target = control.get("target_reviewer")
                question = control.get("question", "").strip()
                if not target or not question:
                    final_markdown = control.get("final_markdown", "") or str(control)
                    break

                print(f"    ▹ Arbiter asks follow-up question to {target}...", flush=True)

                target_auditor = next((a for a in auditors if a.name == target), None)
                if target_auditor is None:
                    final_markdown = f"Arbiter referred to unknown reviewer {target}. Raw control: {control}"
                    break

                qa_snippet = build_qa_snippet_for_reviewer(qa_history, target)
                initial_review_text = latest_reviews[target]

                answer, _memo = run_auditor_followup(
                    auditor=target_auditor,
                    task_description=task_description,
                    context_text=context_text,
                    initial_review=initial_review_text,
                    question=question,
                    qa_snippet=qa_snippet,
                    verbose=verbose,
                )

                qa_history.append(
                    {
                        "reviewer": target,
                        "question": question,
                        "answer": answer,
                    }
                )
                query_count += 1
                # We have successfully used one more clarification question.
                with progress_lock:
                    progress["arbiter_step"] = query_count
                print(f"    ✓ Received answer from {target}.\n", flush=True)
                continue

            final_markdown = control.get("final_markdown", "") or str(control)
            break

        if final_markdown is None:
            # All allowed clarification questions have been used; record this in progress.
            with progress_lock:
                progress["arbiter_step"] = query_count
            final_control = _call_arbiter_step(
                max_queries_for_call=max_queries,
                query_index=max_queries,
                allow_queries_flag=False,
            )
            if final_control.get("state") == "final":
                final_markdown = final_control.get("final_markdown", "") or str(final_control)
            else:
                final_markdown = (
                    "## Arbiter Error\n\n"
                    "Arbiter attempted to ask additional questions after reaching the "
                    "configured max_queries limit.\n\n"
                    f"Raw control object:\n\n{final_control}"
                )

    print("\n▶ Arbiter finished; assembling final joint review...\n", flush=True)
    with progress_lock:
        progress["phase"] = "assembling"

    # Optionally translate the final unified review to Chinese.
    if output_lang == "zh":
        try:
            print("\n▶ Translating final report to Chinese...\n", flush=True)
            final_markdown = translate_markdown_to_zh(final_markdown)
        except Exception as exc:
            sys.stderr.write(f"Translation to Chinese failed: {exc}\n")
            sys.stderr.flush()

    # Clean up this run's auditors workdir after the run to avoid accumulation.
    print(f"\n▶ Cleaning auditors workdir for this run at {run_workdir_abs}...\n", flush=True)
    if os.path.isdir(run_workdir_abs):
        marker_path = os.path.join(run_workdir_abs, WORKDIR_MARKER)
        if os.path.exists(marker_path):
            _cleanup_workdir(run_workdir_abs)
            print("  ✓ Run auditors workdir removed.\n", flush=True)
        else:
            print(
                "  ⚠ Run auditors workdir is missing the safety marker; "
                "leaving it in place. Please inspect and remove manually if desired.\n",
                flush=True,
            )
    # This run's workdir has been cleaned up (or left in place intentionally);
    # drop it from the global cleanup list to avoid unbounded growth.
    try:
        _CLEANUP_RUN_WORKDIRS.remove(run_workdir_abs)
    except ValueError:
        pass

    # Stop the progress reporter.
    stop_progress.set()

    # Move the status line to the next line so it does not interfere
    # with subsequent terminal output.
    sys.stderr.write("\n")
    sys.stderr.flush()

    return (final_markdown or "").strip()
