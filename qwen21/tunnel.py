"""cloudflared acquisition and tunnel modes.

Two modes:

* ``quick``  - `cloudflared tunnel --url http://127.0.0.1:8000`
  Zero configuration, no Cloudflare account, random ``*.trycloudflare.com``
  hostname that changes on every restart. Good enough to try the API from
  anywhere in the world in ten seconds.
* ``named``  - `cloudflared tunnel run --token $TUNNEL_TOKEN`
  Stable hostname, survives restarts, supports SSE/WebSocket. Needs a
  Cloudflare account and a zone; the token is pasted into ``.env``.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download"
BINARY = "cloudflared"

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TOKEN_RE = re.compile(r"eyJhIjoi[A-Za-z0-9_.=-]+")


def tool_path(tools_dir: Path) -> Path:
    if os.name == "nt":
        return tools_dir / f"{BINARY}.exe"
    return tools_dir / BINARY


def asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return f"{BINARY}-darwin-arm64.tgz" if machine in {"arm64", "aarch64"} else f"{BINARY}-darwin-amd64.tgz"
    if system == "Windows":
        return f"{BINARY}-windows-amd64.exe"
    if machine in {"aarch64", "arm64"}:
        return f"{BINARY}-linux-arm64"
    return f"{BINARY}-linux-amd64"


def find_existing(tools_dir: Path) -> Path | None:
    local = tool_path(tools_dir)
    if local.is_file():
        return local
    return shutil.which(BINARY)


def ensure_cloudflared(tools_dir: Path, quiet: bool = False) -> Path:
    """Return a usable cloudflared binary, downloading it if necessary."""
    found = find_existing(tools_dir)
    if found:
        return found
    tools_dir.mkdir(parents=True, exist_ok=True)
    asset = asset_name()
    url = f"{RELEASES}/{asset}"
    target = tools_dir / asset
    if not quiet:
        print("cloudflared")
        print(f"  ↓ {asset}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    if asset.endswith(".tgz"):
        with tarfile.open(target) as archive:
            member = archive.getmember(BINARY)
            member.name = BINARY
            archive.extract(member, path=str(tools_dir))
        target.unlink()
    binary = tool_path(tools_dir)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    if not quiet:
        version = binary_version(binary)
        print(f"  ✓ {binary} {version}")
    return binary


def binary_version(binary: Path) -> str:
    try:
        out = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=20, check=False
        )
        return (out.stdout or out.stderr or "").strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "(version unknown)"


def quick_tunnel_url(log_file: Path, timeout: float = 60.0) -> str | None:
    """Scrape the ephemeral hostname out of cloudflared's log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_file.is_file():
            match = URL_RE.search(log_file.read_text(encoding="utf-8", errors="replace"))
            if match:
                return match.group(0)
        time.sleep(1.0)
    return None


def build_cmd(binary: Path, mode: str, origin: str, token: str | None) -> list[str]:
    base = [str(binary), "tunnel", "--no-autoupdate"]
    if mode == "quick":
        return [*base, "--url", origin]
    if not token:
        raise SystemExit(
            "TUNNEL_MODE=named needs TUNNEL_TOKEN in .env\n"
            "  Cloudflare Zero Trust -> Networks -> Tunnels -> Create a tunnel ->\n"
            "  copy the token out of the install command for your OS."
        )
    return [*base, "run", "--token", token]


def public_url(mode: str, hostname: str | None, quick: str | None = None) -> str | None:
    if mode == "named":
        if not hostname:
            return None
        return hostname if hostname.startswith("http") else f"https://{hostname}"
    return quick


def load_token(env: dict[str, str] | None = None) -> str | None:
    source = env if env is not None else dict(os.environ)
    value = source.get("TUNNEL_TOKEN", "")
    if not TOKEN_RE.search(value or ""):
        return None
    return value


def describe() -> str:
    exe = shutil.which(BINARY)
    return exe or "not installed (the installer drops a private copy into runtime/tools)"
