#!/usr/bin/env python3
"""Fail if a real secret is about to be committed.

Run by the test suite and by CI, and usable by hand:

    python scripts/check_secrets.py

Two checks:

1. **Exact.** Anything set in the local environment or `.env` (MONGODB, API_KEY,
   TUNNEL_TOKEN, HF_TOKEN) must not appear, verbatim, in a tracked file. This is
   the check that actually catches a leaked credential, and it works because the
   secret lives in `.env` on the machine doing the committing.
2. **Backstop.** Any `mongodb://user:pass@` URI in a tracked file must use
   obviously fake credentials. This is what protects a CI run, where there is no
   `.env` to compare against.

Exits non-zero with a list of the offending file and line.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SECRET_ENV_KEYS = ("MONGODB", "API_KEY", "TUNNEL_TOKEN", "HF_TOKEN")

URI_WITH_CREDENTIALS = re.compile(
    r"mongodb(?:\+srv)?://(?P<creds>[^/\s:@]+:[^/\s@]+)@", re.IGNORECASE
)
# Credentials made of these are obviously placeholders, not a leaked secret.
PLACEHOLDER_CREDENTIALS = {
    "u:p", "user:pass", "user:password", "username:password", "user:secret",
    "user:example", "user:todo", "user:placeholder", "user:xxxx", "user:<password>",
    "<user>:<password>", "user:", ":pass", "todo:todo", "xxx:xxx",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def local_secrets() -> dict[str, str]:
    """Secrets from the process environment, then from a local .env."""
    import os

    found = {key: os.environ[key].strip() for key in SECRET_ENV_KEYS if os.environ.get(key)}
    env_file = ROOT / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key in SECRET_ENV_KEYS and value and key not in found:
                found[key] = value
    return {k: v for k, v in found.items() if len(v) >= 8}


def scan(text: str, path: str, secrets: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for key, value in secrets.items():
        if value in text:
            problems.append(f"{path}: contains the real value of ${key}")
    for match in URI_WITH_CREDENTIALS.finditer(text):
        creds = match.group("creds").lower()
        if creds in PLACEHOLDER_CREDENTIALS:
            continue
        if any(marker in creds for marker in ("todo", "example", "placeholder", "xxx")):
            continue
        line = text.count("\n", 0, match.start()) + 1
        problems.append(f"{path}:{line}: mongodb URI with non-placeholder credentials")
    return problems


def main() -> int:
    secrets = local_secrets()
    problems: list[str] = []
    for path in tracked_files():
        full = ROOT / path
        try:
            text = full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not ours
        problems.extend(scan(text, path, secrets))
    if problems:
        print("secret check FAILED:")
        for problem in sorted(set(problems)):
            print(f"  {problem}")
        print(
            "\nIf a real secret reached git, rotate it first (a leaked credential is\n"
            "compromised), then replace it in the file with a placeholder."
        )
        return 1
    checked = len(tracked_files())
    scope = f", {len(secrets)} local secret(s) compared" if secrets else ""
    print(f"secret check ok: {checked} tracked files{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
