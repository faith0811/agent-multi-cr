import subprocess
from typing import List, Optional


def run_shell(cmd: List[str], input_text: Optional[str] = None, cwd: Optional[str] = None) -> str:
    """Run an external command and return stdout as text, raise on error."""
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()

