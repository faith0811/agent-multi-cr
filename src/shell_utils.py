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
        raise RuntimeError(
            f"Command timed out after {resolved_timeout} seconds: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"partial stdout:\n{exc.stdout or ''}\n"
            f"partial stderr:\n{exc.stderr or ''}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()
