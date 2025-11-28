import os
import subprocess
from typing import List, Optional


def _resolve_timeout(explicit_timeout: Optional[float]) -> Optional[float]:
    """
    Determine the timeout for shell commands.

    Priority:
    1. Explicit timeout parameter (if provided).
    2. Environment variable AGENT_MULTI_CR_SHELL_TIMEOUT_SEC (float seconds).
    3. Default of 3600 seconds (1 hour) for long-running review tasks.
    """
    if explicit_timeout is not None:
        return explicit_timeout

    env_value = os.environ.get("AGENT_MULTI_CR_SHELL_TIMEOUT_SEC")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            # Fall through to default if the env var is invalid.
            pass

    # Default to 1 hour for long-running LLM and review tasks.
    return 3600.0


def run_shell(
    cmd: List[str],
    input_text: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    Run an external command and return stdout as text, raise on error.

    A timeout is applied to avoid indefinite hangs from external CLIs. By default
    this is 1 hour, and can be overridden via the AGENT_MULTI_CR_SHELL_TIMEOUT_SEC
    environment variable or the timeout argument.
    """
    resolved_timeout = _resolve_timeout(timeout)

    # Retry on non-zero exit codes to handle transient CLI failures.
    # We perform one initial attempt plus up to three retries (four attempts total).
    max_retries = 3
    last_result: Optional[subprocess.CompletedProcess] = None

    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                capture_output=True,
                cwd=cwd,
                timeout=resolved_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Timeouts are treated as fatal so we do not repeatedly wait for
            # long-running commands that are unlikely to succeed on retry.
            raise RuntimeError(
                f"Command timed out after {resolved_timeout} seconds: {' '.join(cmd)}\n"
                f"cwd: {cwd}\n"
                f"partial stdout:\n{exc.stdout or ''}\n"
                f"partial stderr:\n{exc.stderr or ''}"
            ) from exc

        if result.returncode == 0:
            return result.stdout.strip()

        last_result = result
        # If there are retries left, loop again; otherwise fall through and
        # raise using the last non-zero result.
        if attempt < max_retries:
            continue

    assert last_result is not None
    raise RuntimeError(
        f"Command failed with exit code {last_result.returncode} "
        f"after {max_retries + 1} attempts: {' '.join(cmd)}\n"
        f"cwd: {cwd}\n"
        f"stdout:\n{last_result.stdout}\n"
        f"stderr:\n{last_result.stderr}"
    )

