"""Resolved runtime configuration: repo paths, ports, profile, tunnel mode."""
from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from . import comfy_install, profiles
from .env import env_path, load_env, repo_root, runtime_dir

DEFAULT_COMFY_PORT = 8188
DEFAULT_API_PORT = 8000
DEFAULT_COMFY_HOST = "127.0.0.1"
DEFAULT_API_HOST = "127.0.0.1"


def _int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value else fallback
    except ValueError:
        return fallback


@dataclass
class Settings:
    root: Path
    profile: profiles.Profile
    comfy: Path
    comfy_python: Path
    api_python: Path
    comfy_host: str
    comfy_port: int
    api_host: str
    api_port: int
    tunnel_mode: str

    @property
    def comfy_url(self) -> str:
        return f"http://{self.comfy_host}:{self.comfy_port}"

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def api_origin(self) -> str:
        return self.api_url

    @property
    def host_name(self) -> str:
        """Stable per-machine key. The OS hostname unless .env overrides it."""
        return os.environ.get("HOST_NAME", "").strip() or socket.gethostname()

    @property
    def mongodb_uri(self) -> str:
        return os.environ.get("MONGODB", "").strip()

    @property
    def mongodb_collection(self) -> str:
        return os.environ.get("MONGODB_COLLECTION", "devices").strip() or "devices"

    @property
    def venv_comfy(self) -> Path:
        return runtime_dir() / "venvs" / "comfy"

    @property
    def venv_api(self) -> Path:
        return runtime_dir() / "venvs" / "api"

    @property
    def tools_dir(self) -> Path:
        return runtime_dir() / "tools"

    def api_env(self) -> dict[str, str]:
        """Environment handed to the API process."""
        return {
            "COMFY_URL": self.comfy_url,
            "API_HOST": self.api_host,
            "API_PORT": str(self.api_port),
            "HOST_NAME": self.host_name,
            "MONGODB": self.mongodb_uri,
            "MONGODB_COLLECTION": self.mongodb_collection,
            "QWEN_PROFILE": self.profile.name,
            "QWEN_UNET": self.profile.model("diffusion_models").name
            if self.profile.model("diffusion_models")
            else "",
            "QWEN_UNET_GGUF": "1" if self.profile.loader == "gguf" else "0",
            "QWEN_CLIP": self.profile.model("text_encoders").name
            if self.profile.model("text_encoders")
            else "",
            "QWEN_VAE": self.profile.model("vae").name if self.profile.model("vae") else "",
            "QWEN_ENCODER_DEVICE": self.profile.encoder_device,
            "QWEN_CACHE_DEVICE": self.profile.cache_device,
            "QWEN_CACHE_DTYPE": self.profile.cache_dtype,
            "QWEN_DEFAULT_RESOLUTION": str(self.profile.resolution),
            "QWEN_DEFAULT_STEPS": str(self.profile.steps),
            "QWEN_DEFAULT_CFG": str(self.profile.cfg),
            "QWEN_DEFAULT_SAMPLER": self.profile.sampler,
            "QWEN_DEFAULT_SCHEDULER": self.profile.scheduler,
        }


def resolve(
    root: Path | None = None,
    profile_name: str | None = None,
    comfy_path: str | Path | None = None,
    require_comfy: bool = True,
) -> Settings:
    """Resolve settings for a command.

    `require_comfy=False` is for the commands that only need the profile (the
    model names, the flags) and must work in a clone whose ComfyUI submodule has
    not been initialised yet - otherwise `qwen21 export` fails for anyone who
    cloned without --recurse-submodules.
    """
    root = root or repo_root()
    load_env(env_path())

    requested = (profile_name or os.environ.get("QWEN_PROFILE") or "auto").strip().lower()
    if requested in {"", "auto"}:
        detected, reasons = profiles.detect_profile()
        profile = profiles.get_profile(detected)
        if os.environ.get("QWEN_VERBOSE_PROFILE"):
            for reason in reasons:
                print(f"  · {reason}")
    else:
        profile = profiles.get_profile(requested)

    if comfy_path:
        comfy = comfy_install.normalize_comfy(Path(comfy_path).expanduser())
    else:
        comfy = root / "ComfyUI"
        if require_comfy or (comfy / "main.py").is_file():
            comfy = comfy_install.normalize_comfy(comfy)

    comfy_python = comfy_install.venv_python(runtime_dir() / "venvs" / "comfy")
    if not comfy_python.is_file():
        portable = comfy.parent / "python_embeded" / "python.exe"
        comfy_python = portable if portable.is_file() else Path(sys.executable)

    return Settings(
        root=root,
        profile=profile,
        comfy=comfy,
        comfy_python=comfy_python,
        api_python=comfy_install.venv_python(runtime_dir() / "venvs" / "api"),
        comfy_host=os.environ.get("COMFY_HOST", DEFAULT_COMFY_HOST),
        comfy_port=_int(os.environ.get("COMFY_PORT"), DEFAULT_COMFY_PORT),
        api_host=os.environ.get("API_HOST", DEFAULT_API_HOST),
        api_port=_int(os.environ.get("API_PORT"), DEFAULT_API_PORT),
        tunnel_mode=os.environ.get("TUNNEL_MODE", "quick").lower(),
    )
