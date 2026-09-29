"""Where ComfyUI puts images, and how they are removed again.

ComfyUI 0.37.0 is file-based for images and offers no way to delete them:
`POST /history {"delete": [...]}` only pops the in-memory record, there is no
route that unlinks anything, and `/system_stats` does not report the configured
directories. So this service does two things:

* `qwen21` launches ComfyUI with `--input-directory`, `--output-directory` and
  `--temp-directory` pointing at `runtime/comfy/*`, which makes the paths known
  by construction and keeps them away from anything a user owns.
* after reading a rendered image (and after a job ends, whatever the outcome),
  the file is unlinked immediately. What is left over from a crash is swept by
  age, never by age-less blanket delete, so a sweep cannot race a render.

Every deletion goes through `unlink_quietly`, which refuses to touch anything
outside the three roots.
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path


def _flag_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ComfyDirs:
    input: Path
    output: Path
    temp: Path
    keep_files: bool = False

    @classmethod
    def from_env(cls) -> ComfyDirs:
        comfy = Path(os.environ.get("QWEN_COMFY_DIR", "ComfyUI"))

        def pick(variable: str, fallback: Path) -> Path:
            value = os.environ.get(variable, "").strip()
            return Path(value).expanduser() if value else fallback

        return cls(
            input=pick("QWEN_COMFY_INPUT_DIR", comfy / "input"),
            output=pick("QWEN_COMFY_OUTPUT_DIR", comfy / "output"),
            temp=pick("QWEN_COMFY_TEMP_DIR", comfy / "temp"),
            keep_files=_flag_bool("KEEP_IMAGE_FILES", False),
        )

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.input, self.output, self.temp)

    def ensure(self) -> None:
        for root in self.roots:
            root.mkdir(parents=True, exist_ok=True)

    def root_for(self, type_: str) -> Path:
        return {"input": self.input, "temp": self.temp}.get(type_, self.output)

    def resolve(self, filename: str, subfolder: str = "", type_: str = "output") -> Path | None:
        """Absolute path of a file ComfyUI reported, or None if it is not ours.

        A traversal in `subfolder` or an absolute `filename` resolves outside the
        root and is refused rather than deleted.
        """
        if not filename or self.keep_files:
            return None
        root = self.root_for(type_).resolve()
        candidate = (root / subfolder / filename).resolve()
        if not self._inside(candidate, root):
            return None
        return candidate

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            return path == root or root in path.parents
        except OSError:
            return False

    def unlink_quietly(self, path: Path | None) -> bool:
        """Delete one file. Never raises; a missing file is success."""
        if path is None:
            return False
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def sweep(self, max_age_seconds: float) -> list[str]:
        """Delete files older than `max_age_seconds`, then prune empty dirs.

        Age-based on purpose: a render that is in flight right now has a fresh
        mtime, so a periodic sweep cannot pull a file out from under a job.
        """
        removed: list[str] = []
        if self.keep_files:
            return removed
        now = time.time()
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                if not self._inside(path.resolve(), root.resolve()):
                    continue
                try:
                    if now - path.stat().st_mtime < max_age_seconds:
                        continue
                    path.unlink()
                    removed.append(str(path))
                except OSError:
                    continue
        self.purge_empty()
        return removed

    def purge_empty(self) -> None:
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    with contextlib.suppress(OSError):
                        path.rmdir()

    def files(self) -> list[Path]:
        found: list[Path] = []
        for root in self.roots:
            if root.is_dir():
                found += [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
        return sorted(found)
