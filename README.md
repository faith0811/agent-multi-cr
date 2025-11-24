# agent-multi-cr

Multi‑model code review using Codex and Gemini, coordinated by an arbiter model.

This tool spins up several independent “auditors” (multiple Codex models plus one
Gemini model), lets them review the same codebase, cross‑check each other, and
finally asks a dedicated arbiter model to synthesize a single joint review.

Reviews are structured by priority (`P0`–`P3`) and can optionally be translated
to Simplified Chinese.

## Features

- Multiple Codex reviewers (configurable models and reasoning effort).
- Gemini reviewer plus optional Gemini arbiter.
- Arbiter loop that can ask targeted follow‑up questions to individual reviewers.
- File‑based memo system per reviewer, persisted across runs under a dedicated
  auditors workdir.
- Three context modes:
  - `diff` – review the current Git diff (or staged diff with `--cached`).
  - `repo` – review the repository in the current working directory.
  - `stdin` – review arbitrary text piped in via stdin (e.g. `git diff` or `gh pr diff`).

## Installation

From a clone of this repository:

```bash
python3 -m pip install -e .
```

This will expose a console script named `agent-multi-cr`.

You can verify that it is installed with:

```bash
agent-multi-cr --help
```

> Tip: on systems with an externally‑managed Python (e.g. Homebrew Python on
> macOS), you may prefer installing into a virtualenv or using the `--user`
> flag.

## Basic usage

Run a review over the current Git diff:

```bash
agent-multi-cr \
  --context-mode diff \
  --task "Please review this change for correctness, readability, and potential issues." \
  --arbiter-family codex \
  --arbiter-round-mode single \
  --output-lang zh
```

Key options:

- `--context-mode`:
  - `diff` – use `git diff` / `git diff --cached`.
  - `repo` – let auditors inspect the repository on disk.
  - `stdin` – read context from stdin.
- `--task` – free‑form description of what you want the reviewers to focus on.
- `--codex-model` – may be passed multiple times; each value is either
  `model` or `model:effort` (e.g. `gpt-5.1:xhigh`).
- `--gemini-model` – Gemini model id (default: `gemini-3-pro-preview`).
- `--arbiter-family` – `codex` (default, uses a dedicated `gpt-5.1-codex|low`
  arbiter) or `gemini` (uses the configured Gemini model as arbiter).
- `--max-queries` – maximum number of clarification questions the arbiter may ask.
- `--auditors-workdir` – base directory under which each auditor gets its own
  working directory (default: `.multi_cr_auditors` under the repo root).
- `--output-lang` – `zh` (Chinese, default) or `en`.
- `--arbiter-round-mode` – `single` (one‑shot arbiter, no questions) or
  `multi` (allow iterative clarification questions).

## Memos

Each reviewer maintains a private memo for the repository, updated via a
`MEMO_JSON:` control line appended to its outputs. Memos are stored under:

```text
<auditors-workdir>/memos/<slugified-auditor-name>/memo.txt
```

They are per‑reviewer and persist across runs, so auditors can accumulate
long‑lived notes about a codebase.

## Development

Run the test suite with:

```bash
python3 -m unittest
```

This project is intended to be used in editable/development mode during active
work on the agent. The launcher script `main.py` and the test package both
reuse `bootstrap.ensure_src_on_path()` so that the `src/` layout works without
requiring a full install, but normal usage should rely on the installed
`agent-multi-cr` console script.

