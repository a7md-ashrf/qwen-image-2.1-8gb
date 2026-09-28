"""`.env` loading plus a registry of the keys the user has to paste in.

Nothing here ever writes a real secret. `qwen21 keygen` prints a fresh local API
key; the Cloudflare token is copied from the Cloudflare Zero Trust dashboard.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER_MARKERS = ("change-me", "paste", "your-", "xxxx", "todo", "<", ">")


@dataclass(frozen=True)
class KeySpec:
    name: str
    required: bool
    why: str
    where: str


KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        "API_KEY",
        required=True,
        why="Bearer token every caller must send: Authorization: Bearer $API_KEY",
        where="run `qwen21 keygen` - it generates a random one for you",
    ),
    KeySpec(
        "TUNNEL_TOKEN",
        required=False,
        why="Stable public hostname. Only needed for TUNNEL_MODE=named.",
        where=(
            "Cloudflare Zero Trust -> Networks -> Tunnels -> Create a tunnel -> "
            "copy the token from the install command"
        ),
    ),
    KeySpec(
        "TUNNEL_HOSTNAME",
        required=False,
        why="Optional convenience: the public URL to print on start.",
        where="any hostname on a zone that is already on Cloudflare DNS, e.g. img.example.com",
    ),
    KeySpec(
        "HF_TOKEN",
        required=False,
        why="Raises Hugging Face download limits; required only for gated repositories.",
        where="huggingface.co/settings/tokens -> Create new token (read is enough)",
    ),
)


def load_env(path: Path, override: bool = False) -> dict[str, str]:
    """Parse a dotenv file into os.environ. Returns the parsed mapping."""
    parsed: dict[str, str] = {}
    if not path.is_file():
        return parsed
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def unfilled_keys(env: dict[str, str] | None = None) -> list[KeySpec]:
    env = env if env is not None else dict(os.environ)
    return [k for k in KEYS if (env.get(k.name) is None and k.required) or is_placeholder(env.get(k.name))]


def describe_unfilled(env: dict[str, str] | None = None) -> list[str]:
    source = env if env is not None else dict(os.environ)
    lines = []
    for spec in unfilled_keys(source):
        value = source.get(spec.name)
        state = "missing" if not value else "still a placeholder"
        lines.append(f"  {spec.name} ({state}): {spec.why}\n      get it: {spec.where}")
    return lines


def generate_api_key() -> str:
    return "qk-" + secrets.token_urlsafe(32)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def env_path() -> Path:
    override = os.environ.get("QWEN_ENV_FILE")
    return Path(override).expanduser() if override else repo_root() / ".env"


def runtime_dir() -> Path:
    return repo_root() / "runtime"
