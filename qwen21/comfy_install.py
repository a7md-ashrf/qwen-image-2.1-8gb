"""ComfyUI acquisition and Python environment setup.

ComfyUI lives in this repo as a git submodule pinned to a known-good tag. The
alternative (`--comfy-path`) attaches to an install that already exists, which
also covers the Windows portable layout with its embedded interpreter.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from .profiles import GGUF_NODE_REPO, MIN_COMFY_VERSION, RECOMMENDED_COMFY_REF, Profile

GGUF_NODE_DIR = "ComfyUI-GGUF"


class InstallError(RuntimeError):
    pass


def git() -> str:
    exe = shutil.which("git")
    if not exe:
        raise InstallError("git is required (submodule + custom node installation).")
    return exe


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise InstallError(f"{' '.join(cmd[:3])}… failed:\n{detail}")
    if not quiet:
        for line in (result.stdout or "").splitlines():
            if line.strip():
                print(f"    {line}")
    return result.stdout


# --------------------------------------------------------------------------- #
# ComfyUI checkout
# --------------------------------------------------------------------------- #


def ensure_submodule(root: Path, ref: str | None = None) -> Path:
    """Initialise (and optionally move) the ComfyUI submodule."""
    submodule = root / "ComfyUI"
    if not (submodule / "main.py").is_file():
        print("ComfyUI")
        run([git(), "submodule", "update", "--init", "--recursive", "ComfyUI"], cwd=root)
    if ref and ref != RECOMMENDED_COMFY_REF:
        run([git(), "-C", str(submodule), "fetch", "--depth", "1", "origin", ref])
        run([git(), "-C", str(submodule), "checkout", "FETCH_HEAD"])
    if not (submodule / "main.py").is_file():
        raise InstallError(
            "ComfyUI submodule is empty. Run: git submodule update --init --recursive ComfyUI"
        )
    return submodule


def normalize_comfy(path: Path) -> Path:
    for candidate in (path, path / "ComfyUI"):
        if (candidate / "main.py").is_file():
            return candidate.resolve()
    raise InstallError(f"ComfyUI not found at {path} (no main.py)")


def comfy_version(comfy: Path) -> tuple[int, ...] | None:
    version_file = comfy / "comfyui_version.py"
    if not version_file.is_file():
        return None
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_file.read_text(encoding="utf-8"))
    if not match:
        return None
    parts: list[int] = []
    for chunk in match.group(1).split("."):
        digits = re.match(r"\d+", chunk)
        parts.append(int(digits.group()) if digits else 0)
    return tuple(parts)


def version_ok(comfy: Path) -> bool:
    version = comfy_version(comfy)
    if version is None:
        return False
    return version >= MIN_COMFY_VERSION


# --------------------------------------------------------------------------- #
# Python environment
# --------------------------------------------------------------------------- #


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def python_executable(python: Path | None = None) -> Path:
    """The interpreter ComfyUI should run with (embedded python wins)."""
    if python:
        return python
    portable = Path(sys.executable).parent.parent / "python_embeded" / "python.exe"
    if portable.is_file():
        return portable
    return Path(sys.executable)


def create_venv(venv_dir: Path, force: bool = False) -> Path:
    python = venv_python(venv_dir)
    if python.is_file() and not force:
        return python
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)
    print(f"Python env -> {venv_dir}")
    venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt", clear=force).create(str(venv_dir))
    if not python.is_file():
        raise InstallError(f"venv creation did not produce {python}")
    return python


def pip_install(python: Path, *args: str, quiet: bool = False) -> None:
    cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check", *args]
    print(f"  $ {' '.join(args[:6])}" + (" …" if len(args) > 6 else ""))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-15:])
        raise InstallError(f"pip install failed:\n{tail}")
    if not quiet:
        for line in (result.stdout or "").splitlines()[-3:]:
            if line.strip():
                print(f"    {line.strip()}")


def install_torch(python: Path, profile: Profile) -> None:
    print("Torch")
    if profile.torch_index_url:
        pip_install(
            python,
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            profile.torch_index_url,
        )
    else:
        pip_install(python, "torch", "torchvision", "torchaudio")


def install_comfy_requirements(comfy: Path, python: Path) -> None:
    print("ComfyUI requirements")
    pip_install(python, "-r", str(comfy / "requirements.txt"))


# --------------------------------------------------------------------------- #
# custom nodes and workflows
# --------------------------------------------------------------------------- #


def install_gguf_node(comfy: Path, python: Path) -> None:
    target = comfy / "custom_nodes" / GGUF_NODE_DIR
    target.parent.mkdir(parents=True, exist_ok=True)
    print("ComfyUI-GGUF")
    if (target / ".git").is_dir():
        run([git(), "-C", str(target), "pull", "--ff-only", "--depth", "1"], quiet=True)
    elif target.exists():
        raise InstallError(
            f"{target} exists but is not a git checkout. Move it aside and re-run the install."
        )
    else:
        run([git(), "clone", "--depth", "1", GGUF_NODE_REPO, str(target)], quiet=True)
    requirements = target / "requirements.txt"
    if requirements.is_file():
        # The loader needs the `gguf` package inside *ComfyUI's* interpreter.
        pip_install(python, "-r", str(requirements), quiet=True)


def gguf_node_ready(comfy: Path) -> bool:
    return (comfy / "custom_nodes" / GGUF_NODE_DIR / "nodes.py").is_file()


def install_workflows(comfy: Path, root: Path) -> list[Path]:
    source = root / "workflows"
    target = comfy / "user" / "default" / "workflows"
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in sorted(source.glob("*.json")):
        dst = target / src.name
        shutil.copy2(src, dst)
        written.append(dst)
        print(f"  ✓ workflow {src.name}")
    return written
