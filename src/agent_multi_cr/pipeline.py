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

# Track per-run workdirs plus their associated repo roots so that both normal
# cleanup and atexit handlers can safely remove any git worktrees and
# temporary directories without affecting unrelated paths.
_CLEANUP_RUN_WORKDIRS: List[Tuple[str, str]] = []
_ATEEXIT_REGISTERED = False


def _is_git_repo(path: str) -> bool:
    """Best-effort check for whether path looks like a git repository root."""
    git_dir = os.path.join(path, ".git")
    return os.path.isdir(git_dir)


def _unique_slug(name: str, used: Dict[str, str]) -> str:
    """
    Generate a filesystem-safe slug that is unique for this run.

    If different auditor names would map to the same slug (after slugify),
    append a numeric suffix (-2, -3, ...) to avoid collisions and keep their
    workdirs/memos separate.
    """
    base = slugify(name)
    if not base:
        base = "auditor"
    slug = base
    counter = 2
    while slug in used.values():
        slug = f"{base}-{counter}"
        counter += 1
    used[name] = slug
    return slug


def _run_git(
    args: List[str],
    *,
    cwd: str,
    check: bool = True,
    capture_output: bool = False,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Small wrapper around git subprocess invocation."""
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=capture_output,
        )
    except Exception as exc:  # pragma: no cover - very unlikely in normal use
        raise SystemExit(f"Failed to run {' '.join(cmd)}: {exc}")

    if check and result.returncode != 0:
        raise SystemExit(
            f"git command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def _git_diff_patch(repo_root: str, use_cached: bool) -> str:
    """
    Return a unified diff patch for the current repo state.

    The patch is produced via `git diff` (or `git diff --cached`) and may be
    empty if there are no differences. We rely on the context preparation
    step to enforce "diff must exist" when context_mode == diff.
    """
    args = ["diff", "--cached"] if use_cached else ["diff"]
    result = _run_git(args, cwd=repo_root, check=False, capture_output=True)

    # git diff exit codes:
    #   0 -> no differences
    #   1 -> differences found
    #  >1 -> error
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"git diff exited with status {result.returncode}; "
            "are you running inside a git repository?"
        )
    return result.stdout or ""


def _create_worktree(
    repo_root: str,
    worktree_path: str,
    base_ref: str,
) -> None:
    """Create a detached git worktree at the given path."""
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
    # Use --detach so the worktree is not tied to a branch.
    _run_git(
        ["worktree", "add", "--detach", worktree_path, base_ref],
        cwd=repo_root,
        check=True,
        capture_output=False,
    )


def _apply_patch_to_worktree(
    worktree_path: str,
    patch_text: str,
    *,
    apply_to_index: bool,
) -> None:
    """
    Apply a diff patch to a worktree.

    When apply_to_index is True, use `git apply --index` so that staged diffs
    (`git diff --cached`) inside the worktree reflect the original repo's
    staged changes. Otherwise we only modify the working tree (`git diff`).
    """
    if not patch_text.strip():
        return

    args = ["apply", "--index"] if apply_to_index else ["apply"]
    result = _run_git(
        args,
        cwd=worktree_path,
        check=False,
        capture_output=True,
        input_text=patch_text,
    )
    if result.returncode != 0:
        # If patch application fails, we still want the run to continue, but
        # we surface a warning so the user can investigate.
        sys.stderr.write(
            "Warning: failed to apply git diff patch in auditor worktree.\n"
            f"cwd: {worktree_path}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
        sys.stderr.flush()


def _remove_worktrees_under(repo_root: str, parent_path: str) -> None:
    """
    Best-effort removal of any git worktrees whose paths live under parent_path.
    """
    parent_real = os.path.realpath(parent_path)
    try:
        result = _run_git(
            ["worktree", "list", "--porcelain"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
    except SystemExit:
        # Not a git repo or git missing; nothing to do.
        return

    current_worktree: Optional[str] = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("worktree "):
            current_worktree = line.split(" ", 1)[1]
            wt_real = os.path.realpath(current_worktree)
            if wt_real == parent_real or wt_real.startswith(parent_real + os.sep):
                # Best-effort removal; ignore failures.
                try:
                    _run_git(
                        ["worktree", "remove", "--force", current_worktree],
                        cwd=repo_root,
                        check=False,
                        capture_output=True,
                    )
                except SystemExit:
                    pass
            current_worktree = None


def _cleanup_run_workdir(path: str, repo_root: Optional[str]) -> None:
    """
    Remove an auditor run workdir tree and any associated git worktrees.

    Errors are logged but do not abort the main process so that a single
    failure does not prevent overall cleanup.
    """
    if repo_root:
        try:
            _remove_worktrees_under(repo_root, path)
        except Exception as exc:  # pragma: no cover - very unlikely
            sys.stderr.write(
                f"Warning: failed to remove git worktrees under {path}: {exc}\n"
            )
            sys.stderr.flush()

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        sys.stderr.write(
            f"Warning: failed to remove auditors workdir '{path}': {exc}\n"
        )
        sys.stderr.flush()


def _atexit_cleanup_run_workdirs() -> None:
    """Best-effort cleanup for any leftover per-run workdirs on process exit."""
    for path, repo_root in list(_CLEANUP_RUN_WORKDIRS):
        if os.path.isdir(path):
            marker = os.path.join(path, WORKDIR_MARKER)
            if os.path.exists(marker):
                _cleanup_run_workdir(path, repo_root)
        try:
            _CLEANUP_RUN_WORKDIRS.remove((path, repo_root))
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

    def ignore(dirpath: str, names: List[str]) -> List[str]:
        # Start with pattern-based ignores only. For non-git directories we do
        # not attempt to apply .gitignore semantics here.
        ignored = set(pattern_ignore(dirpath, names))

        # Never follow symlinks; skip them entirely to avoid copying arbitrary
        # host files or creating recursive trees.
        for name in list(names):
            if name in ignored:
                continue
            full_path = os.path.join(dirpath, name)
            if os.path.islink(full_path):
                ignored.add(name)

        return list(ignored)

    shutil.copytree(
        repo_root,
        workdir,
        symlinks=False,  # Changed to False for isolation security
        dirs_exist_ok=True,
        ignore=ignore,
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
    _CLEANUP_RUN_WORKDIRS.append((run_workdir_abs, repo_root))
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
        _cleanup_run_workdir(run_workdir_abs, repo_root)
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
    # Map from human-readable auditor/arbiter names to their unique slugs
    # for this run, so that collisions are avoided.
    slug_map: Dict[str, str] = {}

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
        arbiter_slug = _unique_slug(f"Arbiter-{arbiter_name}", slug_map)
        arbiter_workdir = os.path.join(run_workdir_abs, arbiter_slug)
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

    # Prepare git-based view of the repository, if available. For non-git
    # directories we fall back to a plain copy.
    has_git = _is_git_repo(repo_root)
    # For repo/stdin modes, always reflect the current working tree diff
    # (tracked files) in the auditors' views. For diff mode we also compute
    # the appropriate patch so that tools like `git diff`/`git diff --cached`
    # inside the worktree see the expected changes.
    repo_diff_patch = ""
    diff_mode_patch = ""
    if has_git:
        try:
            repo_diff_patch = _git_diff_patch(repo_root, use_cached=False)
        except SystemExit:
            repo_diff_patch = ""
        if context_mode == "diff":
            try:
                diff_mode_patch = _git_diff_patch(repo_root, use_cached=use_cached)
            except SystemExit:
                diff_mode_patch = ""

    # Set up Codex auditors (excluding arbiter-only config when arbiter_family=codex)
    for idx, (model_name, effort) in enumerate(codex_auditor_configs, start=1):
        name = f"Codex[{model_name}|{effort}]"
        slug = _unique_slug(name, slug_map)
        workdir = os.path.join(run_workdir_abs, slug)
        memo_root = os.path.join(base_workdir_abs, "memos", slug)
        print(f"▶ Setting up auditor {idx}/{total_planned_auditors}: {name}", flush=True)
        if has_git:
            worktree_repo = os.path.join(workdir, "repo")
            _create_worktree(repo_root=repo_root, worktree_path=worktree_repo, base_ref="HEAD")
            # For repo and stdin modes, reflect the current working tree diff
            # (tracked files) so auditors see the same code as in the user's
            # working directory. For diff mode, use a patch that matches the
            # requested diff semantics.
            if context_mode in ("repo", "stdin"):
                _apply_patch_to_worktree(
                    worktree_path=worktree_repo,
                    patch_text=repo_diff_patch,
                    apply_to_index=False,
                )
            elif context_mode == "diff":
                _apply_patch_to_worktree(
                    worktree_path=worktree_repo,
                    patch_text=diff_mode_patch,
                    apply_to_index=use_cached,
                )
            # The auditor's workdir is the worktree root so that tools run
            # directly inside the repo view.
            auditor_workdir = worktree_repo
        else:
            _copy_repo_into_workdir(
                repo_root=repo_root,
                workdir=workdir,
                include_git=context_mode == "diff",
            )
            auditor_workdir = workdir
        auditors.append(
            Auditor(
                name=name,
                kind="codex",
                model_name=model_name,
                workdir=auditor_workdir,
                reasoning_effort=effort,
                memo_root=memo_root,
            )
        )

    # Set up Gemini auditor reviewer.
    gemini_index = total_planned_auditors
    gemini_name = f"Gemini[{gemini_model}]"
    gemini_slug = _unique_slug(gemini_name, slug_map)
    gemini_workdir = os.path.join(run_workdir_abs, gemini_slug)
    gemini_memo_root = os.path.join(base_workdir_abs, "memos", gemini_slug)
    print(f"▶ Setting up auditor {gemini_index}/{total_planned_auditors}: {gemini_name}", flush=True)
    if has_git:
        gemini_repo = os.path.join(gemini_workdir, "repo")
        _create_worktree(repo_root=repo_root, worktree_path=gemini_repo, base_ref="HEAD")
        if context_mode in ("repo", "stdin"):
            _apply_patch_to_worktree(
                worktree_path=gemini_repo,
                patch_text=repo_diff_patch,
                apply_to_index=False,
            )
        elif context_mode == "diff":
            _apply_patch_to_worktree(
                worktree_path=gemini_repo,
                patch_text=diff_mode_patch,
                apply_to_index=use_cached,
            )
        gemini_workdir_final = gemini_repo
    else:
        _copy_repo_into_workdir(
            repo_root=repo_root,
            workdir=gemini_workdir,
            include_git=context_mode == "diff",
        )
        gemini_workdir_final = gemini_workdir
    gemini_auditor = Auditor(
        name=gemini_name,
        kind="gemini",
        model_name=gemini_model,
        workdir=gemini_workdir_final,
        reasoning_effort=None,
        memo_root=gemini_memo_root,
    )
    auditors.append(gemini_auditor)

    # If Gemini acts as arbiter, give it its own isolated workspace without a repo copy.
    if arbiter_family == "gemini":
        arbiter_name = f"Gemini-Arbiter[{gemini_model}]"
        arbiter_slug = _unique_slug(arbiter_name, slug_map)
        arbiter_workdir = os.path.join(run_workdir_abs, arbiter_slug)
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

    latest_reviews: Dict[str, str] = {}
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
            latest_reviews[auditor.name] = review
            print(f"    ✓ Peer cross-check finished for {auditor.name}.", flush=True)

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
            initial_reviews=initial_reviews,
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
            final_markdown = translate_markdown_to_zh(
                final_markdown,
                cwd=run_workdir_abs,
            )
        except Exception as exc:
            sys.stderr.write(f"Translation to Chinese failed: {exc}\n")
            sys.stderr.flush()

    # Clean up this run's auditors workdir after the run to avoid accumulation.
    print(f"\n▶ Cleaning auditors workdir for this run at {run_workdir_abs}...\n", flush=True)
    if os.path.isdir(run_workdir_abs):
        marker_path = os.path.join(run_workdir_abs, WORKDIR_MARKER)
        if os.path.exists(marker_path):
            _cleanup_run_workdir(run_workdir_abs, repo_root)
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
        _CLEANUP_RUN_WORKDIRS.remove((run_workdir_abs, repo_root))
    except ValueError:
        pass

    # Stop the progress reporter.
    stop_progress.set()

    # Move the status line to the next line so it does not interfere
    # with subsequent terminal output.
    sys.stderr.write("\n")
    sys.stderr.flush()

    return (final_markdown or "").strip()
