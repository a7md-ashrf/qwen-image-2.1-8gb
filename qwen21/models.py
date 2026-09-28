"""Resumable model downloads with an optional Hugging Face token."""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .profiles import ModelFile

USER_AGENT = "qwen-image-2.1-8gb/2.0"
CHUNK = 4 * 1024 * 1024
RETRIES = 4


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def free_disk_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return os.statvfs(probe).f_bavail * os.statvfs(probe).f_frsize / (1024**3)
    except (OSError, AttributeError, ValueError):
        return float("inf")


def _headers(offset: int) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return headers


def _fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers=_headers(offset))
        suffix = f" (resume at {human_bytes(offset)})" if offset else ""
        print(f"  ↓ {target.name}{suffix}", flush=True)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and partial.exists():
                # Already complete: the server refused the range request.
                partial.replace(target)
                return
            if exc.code in (401, 403):
                raise SystemExit(
                    f"{url} returned {exc.code}. Set HF_TOKEN in .env if the repository is gated."
                ) from None
            last_error = exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
        else:
            with response:
                status = getattr(response, "status", None)
                mode = "ab" if offset and status == 206 else "wb"
                done = offset if mode == "ab" else 0
                length = response.headers.get("Content-Length")
                total = done + int(length) if length and length.isdigit() else None
                # Carriage-return progress is unreadable in a log file or CI.
                live = sys.stdout.isatty()
                with partial.open(mode) as fh:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total and live:
                            pct = min(100.0, done * 100.0 / total)
                            print(
                                f"    {human_bytes(done)} / {human_bytes(total)} ({pct:5.1f}%)",
                                end="\r",
                                flush=True,
                            )
                if total and live:
                    print()
            partial.replace(target)
            return
        if attempt < RETRIES:
            wait = 2**attempt
            print(f"  ! {last_error}; retry {attempt}/{RETRIES - 1} in {wait}s", flush=True)
            time.sleep(wait)

    raise SystemExit(f"Failed to download {url}: {last_error}")


def sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if limit is not None and read >= limit:
                break
    return digest.hexdigest()


def verify(path: Path, model: ModelFile) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if model.size and abs(size - model.size) > max(50_000_000, model.size * 0.02):
        return False, f"size {human_bytes(size)} != expected {human_bytes(model.size)}"
    if model.sha256:
        actual = sha256(path)
        if actual != model.sha256:
            return False, f"sha256 mismatch ({actual[:12]}…)"
    return True, human_bytes(size)


def download_model(comfy_models: Path, model: ModelFile, force: bool = False) -> Path:
    target = comfy_models / model.folder / model.name
    ok, detail = verify(target, model)
    if ok and not force:
        print(f"  ✓ {model.rel} ({detail})")
        return target
    if target.exists() and force:
        target.unlink()
    free = free_disk_gib(target.parent)
    if free < 4:
        raise SystemExit(
            f"Only {free:.1f} GiB free near {target.parent}, need ~4 GiB of headroom."
        )
    _fetch(model.url, target)
    ok, detail = verify(target, model)
    if not ok:
        raise SystemExit(f"{model.rel} failed verification: {detail}")
    print(f"  ✓ {model.rel} ({detail})")
    return target


def install_models(comfy: Path, models: tuple[ModelFile, ...], force: bool = False) -> None:
    print("Models")
    for model in models:
        download_model(comfy / "models", model, force=force)
