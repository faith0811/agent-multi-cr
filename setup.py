from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list:
    """Read requirements.txt, ignoring comments and blank lines."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return []

    requirements: list[str] = []
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
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "agent-multi-cr=src.cli:main",
        ]
    },
)

