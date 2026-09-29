"""Request/response models for the public API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_PROMPT_CHARS = 8000


class SampleParams(BaseModel):
    """Knobs shared by /v1/generate and /v1/edit."""

    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    negative_prompt: str = Field("", max_length=MAX_PROMPT_CHARS)
    resolution: int | None = Field(None, ge=64, le=4096, description="longest edge in pixels")
    width: int | None = Field(None, ge=64, le=4096)
    height: int | None = Field(None, ge=64, le=4096)
    steps: int | None = Field(None, ge=1, le=100)
    cfg: float | None = Field(None, ge=0.0, le=20.0)
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    sampler: str | None = None
    scheduler: str | None = None
    wait: bool | None = Field(
        None,
        description="block until the job finishes; falls back to 202 + job id on timeout",
    )
    n: int = Field(1, ge=1, le=4)
    response_format: str | None = Field(
        None,
        pattern="^(b64|url)$",
        description=(
            "b64 embeds the image in the JSON (default); url returns only the "
            "download URL, which avoids the 33%% base64 inflation on large results"
        ),
    )

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value


class GenerateRequest(SampleParams):
    pass


class EditRequest(SampleParams):
    pass


class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    created_at: float
    updated_at: float
    elapsed: float
    queue_position: int | None = None
    comfy_prompt_id: str | None = None
    seed: int | None = None
    duration: float | None = None
    images: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    status_url: str = ""


class JobList(BaseModel):
    jobs: list[JobStatus]


class Health(BaseModel):
    status: str
    comfy: dict[str, Any] | None = None
    queue: dict[str, Any] | None = None
    running: int = 0
    queued: int = 0
    models: dict[str, Any] = Field(default_factory=dict)


class TunnelPublish(BaseModel):
    """Body of POST /v1/internal/tunnel.

    The device is deliberately *not* accepted here: it comes from the server's
    own HOST_NAME, so a caller with a valid key cannot write rows for other
    machines.
    """

    link: str = Field(
        ...,
        max_length=512,
        description="the device's public edit URL, e.g. https://x.trycloudflare.com/v1/edit",
    )


class RegistryEntry(BaseModel):
    published: bool
    device: str | None = None
    link: str | None = None
    updated_at: str | None = None
    target: str = ""


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    profile: str
    loader: str
    edit: bool = True
    unet: str
    clip: str
    vae: str
    defaults: dict[str, Any]
