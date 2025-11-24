from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def read_requirements() -> List[str]:
    """Read requirements.txt, ignoring comments and blank lines."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return []

    requirements: List[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="agent-multi-cr",
    version="0.1.0",
    description="Multi-model code review using Codex and Gemini with an arbiter.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "agent-multi-cr=agent_multi_cr.cli:main",
        ]
    },
)
