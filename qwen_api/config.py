"""Environment-driven configuration for the API service.

The service is a plain FastAPI app: it can be started with `python -m qwen_api`
inside the venv created by `qwen21 install`, or by any process manager that can
set these variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    comfy_url: str
    host: str
    port: int
    api_key: str
    allow_anonymous: bool
    public_base_url: str

    unet: str
    unet_gguf: bool
    clip: str
    vae: str
    encoder_device: str
    cache_device: str
    cache_dtype: str

    default_resolution: int
    default_steps: int
    default_cfg: float
    default_sampler: str
    default_scheduler: str

    sync_max_wait: float
    max_concurrent: int
    max_queue: int
    job_timeout: float
    job_ttl_hours: float
    rate_limit_per_min: float
    max_upload_mb: float
    max_images: int
    poll_interval: float
    output_dir: Path

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def output_dir_default() -> Path:
    override = os.environ.get("QWEN_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    here = Path(__file__).resolve().parent.parent
    return here / "runtime" / "outputs"


def load() -> Settings:
    return Settings(
        comfy_url=os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/"),
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=_int("API_PORT", 8000),
        api_key=os.environ.get("API_KEY", "").strip(),
        allow_anonymous=_bool("ALLOW_ANONYMOUS", False),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/"),
        unet=os.environ.get("QWEN_UNET", "").strip(),
        unet_gguf=_bool("QWEN_UNET_GGUF", False),
        clip=os.environ.get("QWEN_CLIP", "").strip(),
        vae=os.environ.get("QWEN_VAE", "").strip(),
        encoder_device=os.environ.get("QWEN_ENCODER_DEVICE", "cpu"),
        cache_device=os.environ.get("QWEN_CACHE_DEVICE", "auto"),
        cache_dtype=os.environ.get("QWEN_CACHE_DTYPE", "default"),
        default_resolution=_int("QWEN_DEFAULT_RESOLUTION", 1024),
        default_steps=_int("QWEN_DEFAULT_STEPS", 25),
        default_cfg=_float("QWEN_DEFAULT_CFG", 1.0),
        default_sampler=os.environ.get("QWEN_DEFAULT_SAMPLER", "euler"),
        default_scheduler=os.environ.get("QWEN_DEFAULT_SCHEDULER", "simple"),
        # Cloudflare's proxy drops origin responses after ~100s of silence, so a
        # blocking request must give up before that and hand back a job id.
        sync_max_wait=_float("SYNC_MAX_WAIT", 90.0),
        max_concurrent=_int("MAX_CONCURRENT", 1),
        max_queue=_int("MAX_QUEUE", 8),
        job_timeout=_float("JOB_TIMEOUT", 3600.0),
        job_ttl_hours=_float("JOB_TTL_HOURS", 24.0),
        rate_limit_per_min=_float("RATE_LIMIT_PER_MIN", 10.0),
        max_upload_mb=_float("MAX_UPLOAD_MB", 25.0),
        max_images=_int("MAX_IMAGES", 4),
        poll_interval=_float("POLL_INTERVAL", 1.5),
        output_dir=output_dir_default(),
    )


settings = load()
