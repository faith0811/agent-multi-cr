import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


MEMO_JSON_PREFIX = "MEMO_JSON:"


@dataclass
class Auditor:
    name: str           # e.g. "Codex[gpt-5.1|high]" or "Gemini[gemini-3-pro-preview]"
    kind: str           # "codex" or "gemini"
    model_name: str     # model id for CLI
    workdir: str        # private working directory path (per-run workspace)
    reasoning_effort: Optional[str] = None  # only for codex; e.g. "high", "xhigh"
    # Optional persistent root for memo files. If not set, memos live in workdir.
    memo_root: Optional[str] = None


def slugify(name: str) -> str:
    """Create a filesystem-safe slug from an auditor name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return slug or "auditor"


def memo_path(auditor: Auditor) -> str:
    """Return the memo file path for an auditor."""
    root = auditor.memo_root or auditor.workdir
    return os.path.join(root, "memo.txt")


def load_memo(auditor: Auditor) -> str:
    """Load memo text for an auditor (private)."""
    path = memo_path(auditor)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def save_memo(auditor: Auditor, memo_text: str) -> None:
    """Persist memo text for an auditor."""
    path = memo_path(auditor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(memo_text)


def extract_and_update_memo(
    auditor: Auditor,
    raw_output: str,
    current_memo: str,
) -> Tuple[str, str]:
    """
    Look for a line starting with MEMO_JSON_PREFIX in the model's output.

    If present and parses as JSON, update the memo accordingly and return:
        (clean_output_without_memo_line, new_memo_text)
    """
    lines = raw_output.splitlines()
    new_memo = current_memo
    cleaned_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(MEMO_JSON_PREFIX):
            json_part = stripped[len(MEMO_JSON_PREFIX):].strip()
            try:
                obj = json.loads(json_part)
                append_text = obj.get("append", "")
                overwrite = bool(obj.get("overwrite", False))
                if append_text:
                    if overwrite or not new_memo:
                        new_memo = append_text
                    else:
                        if not new_memo.endswith("\n"):
                            new_memo += "\n"
                        new_memo += append_text
            except Exception as exc:
                sys.stderr.write(f"Warning: Failed to parse MEMO_JSON for {auditor.name}: {exc}\n")
        else:
            cleaned_lines.append(line)

    cleaned_output = "\n".join(cleaned_lines).strip()
    if new_memo != current_memo:
        save_memo(auditor, new_memo)
    return cleaned_output, new_memo
