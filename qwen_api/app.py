"""FastAPI service: image + prompt in, edited image out.

Design notes worth knowing before changing anything here:

* Cloudflare's proxy gives up on an origin response after roughly 100 seconds
  of silence, and a 25-step edit takes minutes. So a request never commits to
  blocking: it waits up to ``SYNC_MAX_WAIT`` and otherwise returns ``202`` with a
  job id the caller can poll.
* ComfyUI already serialises execution, so this service never runs two jobs at
  once unless ``MAX_CONCURRENT`` is raised. Its job table is bookkeeping, not a
  scheduler.
* Every route except ``/healthz`` requires ``Authorization: Bearer $API_KEY``.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import mimetypes
import os
import re
import time
from collections import defaultdict, deque
from fnmatch import fnmatch
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from . import workflows
from .comfy_client import ComfyClient, ComfyError, history_error
from .config import settings
from .jobs import FAILED, SUCCEEDED, Job, JobStore
from .registry import Registry, RegistryError, validate_link
from .schemas import (
    EditRequest,
    GenerateRequest,
    Health,
    JobList,
    JobStatus,
    ModelCard,
    RegistryEntry,
    SampleParams,
    TunnelPublish,
)
from .storage import ComfyDirs
from .workflows import ModelSpec

API = "/v1"
IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
CANDIDATES = {
    "diffusion_models": [
        "qwen_image_2.1-Q6_K.gguf",
        "qwen_image_2.1-Q4_K.gguf",
        "qwen_image_2.1-Q5_0.gguf",
        "qwen_image_2.1-Q3_K.gguf",
        "qwen_image_2.1-Q8_0.gguf",
        "qwen_image_2.1-Q4_0.gguf",
        "qwen_image_2.1-Q2_K.gguf",
        "qwen_image_2.1_int8_convrot.safetensors",
        "qwen_image_2.1_bf16.safetensors",
    ],
    "text_encoders": [
        "qwen3vl_8b_w4a8.safetensors",
        "qwen3vl_8b_int8_convrot.safetensors",
        "qwen3vl_8b_bf16.safetensors",
    ],
    "vae": [
        "qwen_image_2.1_vae_bf16.safetensors",
        "qwen_image_2.1_vae_*.safetensors",
    ],
}

store = JobStore(settings.db_path, settings.job_ttl_hours, settings.image_cache_mb)
comfy: ComfyClient | None = None
registry = Registry(settings.mongodb_uri, settings.mongodb_collection, settings.host_name)
storage = ComfyDirs.from_env()
_spec_cache: dict[str, Any] = {"spec": None, "at": 0.0, "warnings": []}
_hits: dict[str, deque[float]] = defaultdict(deque)
_capacity_lock = asyncio.Lock()


def _client() -> ComfyClient:
    """The ComfyUI client, created on first use and closed by the lifespan."""
    global comfy
    if comfy is None:
        comfy = ComfyClient(settings.comfy_url)
    return comfy


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global comfy
    storage.ensure()
    # Anything a previous crash left behind in ComfyUI's image directories.
    removed = storage.sweep(settings.orphan_max_age_min * 60)
    if removed:
        print(f"[startup] removed {len(removed)} leftover image file(s)")
    _client()
    sweeper = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        if comfy is not None:
            await comfy.aclose()
        comfy = None
        await registry.close()


app = FastAPI(
    title="Qwen-Image-2.1 edit API",
    version="2.0.0",
    description=(
        "Local Qwen-Image-2.1 over HTTP. POST /v1/edit with an image plus an instruction, "
        "or POST /v1/generate with a prompt only. Every route needs "
        "`Authorization: Bearer $API_KEY`."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# auth, rate limiting, capacity
# --------------------------------------------------------------------------- #


def require_key(request: Request) -> str:
    if settings.allow_anonymous:
        return "anonymous"
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="API_KEY is not set. Copy .env.example to .env and run `qwen21 keygen`.",
        )
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    token = token or request.headers.get("x-api-key", "").strip()
    if not token or not hmac.compare_digest(token, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _rate_limited(token)


def _rate_limited(token: str) -> str:
    limit = settings.rate_limit_per_min
    if limit <= 0:
        return token
    now = time.time()
    window = _hits[token]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit: {limit:.0f} requests per minute",
            headers={"Retry-After": "30"},
        )
    window.append(now)
    return token


# A name is only ever a bare filename inside ComfyUI's input directory.
INPUT_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def check_capacity() -> None:
    if store.queue_length() >= settings.max_queue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"queue is full ({settings.max_queue} jobs); retry in a few seconds",
            headers={"Retry-After": "10"},
        )


def check_request_size(request: Request) -> None:
    """Reject an oversized body from Content-Length, before reading any of it.

    Cloudflare answers an over-cap request with its own 413 page; catching it
    here means the caller gets a message that says which limit to lower.
    """
    declared = request.headers.get("content-length")
    if not declared or not declared.isdigit():
        return
    # + slack for the multipart boundaries and headers around the file parts.
    if int(declared) > settings.max_request_bytes + 64 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=too_large(int(declared)),
        )


def too_large(size: int) -> str:
    plan = "Cloudflare caps uploads at 100 MB on Free/Pro, 200 MB on Business"
    return (
        f"request body is {size / 1024 / 1024:.1f} MB; this service accepts "
        f"{settings.max_request_mb:.0f} MB ({plan}). Send fewer/smaller images, "
        f"use response_format=url, or reference a file already in "
        f"ComfyUI/input/ by name instead of uploading it."
    )


async def read_capped(upload: UploadFile, budget: int) -> bytes:
    """Read a part in chunks, giving up the moment the shared budget is gone."""
    chunks: list[bytes] = []
    used = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        used += len(chunk)
        if used > budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=too_large(used)
            )
        chunks.append(chunk)
    return b"".join(chunks)


def reference_existing_input(name: str) -> str:
    """Use a file already in ComfyUI/input/ instead of uploading one.

    This is the way past the tunnel's upload cap: put the file there yourself
    (scp, a sync folder, anything) and send only its name. The name is
    attacker-controlled text pointing at a server-side file, so it is matched
    against a strict pattern and must exist.
    """
    candidate = name.strip()
    if not INPUT_FILENAME.match(candidate) or ".." in candidate:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{name!r} is not a plain filename. Reference mode takes a bare "
                "file name that already exists in ComfyUI/input/ "
                "(letters, digits, dot, dash, underscore)."
            ),
        )
    path = storage.resolve(candidate, "", "input")
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"{candidate!r} is not in ComfyUI's input directory ({storage.input})",
        )
    return candidate


# --------------------------------------------------------------------------- #
# model resolution
# --------------------------------------------------------------------------- #


def _pick(available: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        for name in available:
            if fnmatch(name, candidate):
                return name
    return None


async def resolve_spec() -> tuple[ModelSpec, list[str]]:
    """Find the model files on disk, preferring the ones the profile declared."""
    now = time.time()
    if _spec_cache["spec"] and now - _spec_cache["at"] < 60:
        return _spec_cache["spec"], _spec_cache["warnings"]
    try:
        listing = {
            folder: await _client().models(folder)
            for folder in ("diffusion_models", "text_encoders", "vae")
        }
    except ComfyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"cannot reach ComfyUI at {settings.comfy_url}: {exc}",
        ) from exc

    preferred = {
        "diffusion_models": [settings.unet] if settings.unet else [],
        "text_encoders": [settings.clip] if settings.clip else [],
        "vae": [settings.vae] if settings.vae else [],
    }
    chosen: dict[str, str | None] = {}
    warnings: list[str] = []
    for folder, names in listing.items():
        pick = _pick(names, preferred[folder] + CANDIDATES[folder])
        chosen[folder] = pick
        if preferred[folder] and names and pick and pick not in preferred[folder]:
            warnings.append(f"profile asked for {preferred[folder][0]}, using {pick}")

    missing = [name for name, value in chosen.items() if not value]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"no model file for: {', '.join(missing)}. Run `qwen21 models` to download them. "
                f"ComfyUI sees: {listing}"
            ),
        )

    spec = ModelSpec(
        unet=chosen["diffusion_models"],
        clip=chosen["text_encoders"],
        vae=chosen["vae"],
        unet_gguf=chosen["diffusion_models"].endswith(".gguf"),
        encoder_device=settings.encoder_device,
        cache_device=settings.cache_device,
        cache_dtype=settings.cache_dtype,
    )
    _spec_cache.update({"spec": spec, "at": now, "warnings": warnings})
    return spec, warnings


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def _params(req: SampleParams) -> dict[str, Any]:
    resolution = req.resolution or settings.default_resolution
    return {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "resolution": resolution,
        "width": req.width or resolution,
        "height": req.height or resolution,
        "steps": req.steps or settings.default_steps,
        "cfg": settings.default_cfg if req.cfg is None else req.cfg,
        "sampler": req.sampler or settings.default_sampler,
        "scheduler": req.scheduler or settings.default_scheduler,
        "seed": req.seed,
        "n": req.n,
    }


def _graph(kind: str, spec: ModelSpec, params: dict[str, Any], names: list[str], job_id: str,
           seed: int | None) -> dict[str, Any]:
    prefix = f"qwen21/{job_id}"
    if kind == "edit":
        return workflows.build_edit_prompt(
            spec,
            params["prompt"],
            names,
            negative_prompt=params["negative_prompt"],
            resolution=params["resolution"],
            seed=seed,
            steps=params["steps"],
            cfg=params["cfg"],
            sampler=params["sampler"],
            scheduler=params["scheduler"],
            prefix=prefix,
        )
    return workflows.build_generate_prompt(
        spec,
        params["prompt"],
        negative_prompt=params["negative_prompt"],
        width=params["width"],
        height=params["height"],
        seed=seed,
        steps=params["steps"],
        cfg=params["cfg"],
        sampler=params["sampler"],
        scheduler=params["scheduler"],
        prefix=prefix,
    )


async def _execute(kind: str, spec: ModelSpec, params: dict[str, Any], names: list[str],
                   wait: bool, request: Request, response_format: str = "b64",
                   uploaded: list[str] | None = None) -> Response:
    """Queue a job and return its result.

    `names` go into the graph. `uploaded` is the subset this service created and
    must clean up - in reference mode `names` are files the caller put in
    ComfyUI/input/ themselves, and those are never ours to delete.
    """
    async with _capacity_lock:
        check_capacity()
        job = store.create(
            kind,
            {
                "prompt": params["prompt"],
                "images": names,
                "params": {k: v for k, v in params.items() if k != "prompt"},
            },
        )

    async def work(job: Job) -> None:
        client = _client()
        saved: list[dict[str, Any]] = []
        blobs: dict[int, bytes] = {}
        try:
            for run in range(max(1, params["n"])):
                seed = None if params["seed"] is None else params["seed"] + run
                graph = _graph(
                    kind, spec, params, names,
                    f"{job.id}-{run}" if params["n"] > 1 else job.id, seed,
                )
                prompt_id = await client.queue_prompt(graph, client_id=job.id)
                if run == 0:
                    store.update(
                        job, comfy_prompt_id=prompt_id,
                        seed=graph[workflows.SAMPLER]["inputs"]["seed"],
                    )
                record = await client.wait_for_result(
                    prompt_id, settings.job_timeout, settings.poll_interval
                )
                failure = history_error(record)
                if failure:
                    raise RuntimeError(failure)
                for meta in await client.fetch_outputs(record):
                    index = len(saved)
                    try:
                        data = await client.view(
                            meta["filename"], meta.get("subfolder", ""),
                            meta.get("type", "output"),
                        )
                    finally:
                        # The bytes are in RAM (or they never arrived); ComfyUI's
                        # copy goes now either way, so a failed or crashed job
                        # cannot leave a render behind.
                        storage.unlink_quietly(
                            storage.resolve(
                                meta["filename"], meta.get("subfolder", ""),
                                meta.get("type", "output"),
                            )
                        )
                    saved.append(
                        {
                            "index": index,
                            "filename": meta["filename"],
                            "bytes": len(data),
                            "media_type": mimetypes.guess_type(meta["filename"])[0]
                            or "image/png",
                            "retained": False,
                        }
                    )
                    blobs[index] = data
            if not saved:
                raise RuntimeError("ComfyUI finished but produced no image")
            store.update(
                job, status=SUCCEEDED, images=saved,
                duration=round(time.time() - job.created_at, 2),
            )
            # After the update, because it replaces `images` and therefore the
            # `retained` flags that caching decides.
            store.cache_images(job, blobs)
        finally:
            # Inputs *we* uploaded go when the job ends: succeeded, failed,
            # cancelled or interrupted. A file the caller referenced by name is
            # theirs and stays exactly where it is.
            for name in uploaded or []:
                storage.unlink_quietly(storage.resolve(name, "", "input"))

    await store.run(job, work)
    await asyncio.sleep(0)  # let the worker leave `queued`

    if wait:
        if await store.wait(job, settings.sync_max_wait):
            if job.status == SUCCEEDED:
                return _completed(job, request, response_format)
            raise HTTPException(status_code=502, detail=job.error or "job failed")
    return _accepted(job, request)


def _base_url(request: Request) -> str:
    """Public base URL: the configured one, else whatever the caller used."""
    return settings.public_base_url or str(request.base_url).rstrip("/")


def _image_payload(job: Job, request: Request, response_format: str = "b64") -> list[dict[str, Any]]:
    """Per-image response entries, read from RAM.

    `retained` says whether the bytes are still in the cache, i.e. whether
    `url` will still resolve. With response_format=b64 the bytes are inline, so
    a cache miss only matters for a later fetch of the url.
    """
    base = _base_url(request)
    out: list[dict[str, Any]] = []
    for entry in job.images:
        index = entry.get("index", 0)
        payload: dict[str, Any] = {
            "index": index,
            "media_type": entry.get("media_type", "image/png"),
            "bytes": entry.get("bytes"),
            "url": f"{base}{API}/jobs/{job.id}/image/{index}",
            "retained": bool(entry.get("retained")),
        }
        if response_format != "url":
            blob = job.blob(index)
            payload["b64_json"] = base64.b64encode(blob).decode() if blob else None
        out.append(payload)
    return out


def _completed(job: Job, request: Request, response_format: str = "b64") -> Response:
    return JSONResponse(
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "seed": job.seed,
            "duration_s": job.duration,
            "images": _image_payload(job, request, response_format),
        }
    )


def _accepted(job: Job, request: Request) -> Response:
    base = _base_url(request)
    return JSONResponse(
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "status_url": f"{base}{API}/jobs/{job.id}",
            "image_url_pattern": f"{base}{API}/jobs/{job.id}/image/{{index}}",
            "retry_after": 2,
        },
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": f"{API}/jobs/{job.id}", "Retry-After": "2"},
    )


def _job_status(job: Job) -> JobStatus:
    return JobStatus(
        id=job.id,
        kind=job.kind,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        elapsed=round(job.elapsed, 2),
        comfy_prompt_id=job.comfy_prompt_id,
        seed=job.seed,
        duration=job.duration,
        images=[
            {
                "index": entry.get("index", 0),
                "media_type": entry.get("media_type", "image/png"),
                "bytes": entry.get("bytes"),
                "retained": bool(entry.get("retained")),
                "url": f"{API}/jobs/{job.id}/image/{entry.get('index', 0)}",
            }
            for entry in job.images
        ],
        error=job.error,
        status_url=f"{API}/jobs/{job.id}",
    )


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def index() -> dict[str, Any]:
    return {
        "service": "qwen-image-2.1-edit-api",
        "auth": "Authorization: Bearer $API_KEY",
        "endpoints": [
            f"POST {API}/edit",
            f"POST {API}/generate",
            f"GET  {API}/models",
            f"GET  {API}/jobs",
            f"GET  {API}/jobs/{{id}}",
            f"GET  {API}/jobs/{{id}}/image/{{index}}",
            f"POST {API}/jobs/{{id}}/cancel",
            "GET  /healthz",
            "GET  /docs",
        ],
    }


@app.get("/healthz", response_model=Health)
async def healthz() -> Health:
    running, queued = store.counts()
    payload = Health(status="ok", running=running, queued=queued)
    try:
        payload.comfy = await _client().system_stats()
        payload.queue = await _client().queue_state()
    except ComfyError as exc:
        payload.status = "degraded"
        payload.comfy = {"error": str(exc)}
    return payload


@app.get(f"{API}/models", response_model=ModelCard, dependencies=[Depends(require_key)])
async def model_card() -> ModelCard:
    spec, warnings = await resolve_spec()
    return ModelCard(
        id="qwen-image-2.1",
        profile=os.environ.get("QWEN_PROFILE", "custom"),
        loader="gguf" if spec.unet_gguf else "safetensors",
        unet=spec.unet,
        clip=spec.clip,
        vae=spec.vae,
        defaults={
            "resolution": settings.default_resolution,
            "steps": settings.default_steps,
            "cfg": settings.default_cfg,
            "sampler": settings.default_sampler,
            "scheduler": settings.default_scheduler,
            "max_images": settings.max_images,
            "encoder_device": spec.encoder_device,
            "kv_cache": {"device": spec.cache_device, "dtype": spec.cache_dtype},
            "warnings": warnings,
        },
    )


@app.post(f"{API}/generate", dependencies=[Depends(require_key)])
async def generate(request: Request, body: GenerateRequest) -> Response:
    check_request_size(request)
    spec, _ = await resolve_spec()
    return await _execute(
        "generate", spec, _params(body), [], body.wait is not False, request,
        response_format=body.response_format or settings.response_format,
    )


@app.post(f"{API}/edit", dependencies=[Depends(require_key)])
async def edit(
    request: Request,
    image: Annotated[
        list[UploadFile | str],
        File(description="image(s) to edit: a file, or the name of a file already in ComfyUI/input/"),
    ],
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    resolution: Annotated[int | None, Form()] = None,
    width: Annotated[int | None, Form()] = None,
    height: Annotated[int | None, Form()] = None,
    steps: Annotated[int | None, Form()] = None,
    cfg: Annotated[float | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
    sampler: Annotated[str | None, Form()] = None,
    scheduler: Annotated[str | None, Form()] = None,
    wait: Annotated[bool | None, Form()] = None,
    n: Annotated[int, Form()] = 1,
    response_format: Annotated[str | None, Form()] = None,
) -> Response:
    check_request_size(request)
    if len(image) > settings.max_images:
        raise HTTPException(
            status_code=400, detail=f"at most {settings.max_images} images per request"
        )
    # Classify by what the part actually is. Note that FastAPI hands back
    # starlette's UploadFile, which is NOT an instance of fastapi.UploadFile, so
    # `isinstance(part, UploadFile)` is false for every real upload.
    references = [item for item in image if isinstance(item, str)]
    if references and len(references) != len(image):
        raise HTTPException(
            status_code=400,
            detail="mixing uploads and filenames in one request is not supported",
        )
    try:
        body = EditRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler=sampler,
            scheduler=scheduler,
            wait=wait,
            n=n,
            response_format=response_format,
        )
    except ValidationError as exc:
        # The form is assembled by hand, so pydantic errors would otherwise
        # surface as a 500 instead of the 422 the JSON route returns.
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        raise HTTPException(
            status_code=422, detail=f"{field}: {first.get('msg')}" if field else first.get("msg", "")
        ) from exc
    check_capacity()  # reject before spending bandwidth on uploads
    # Resolve the model files *before* uploading: a 503 here would otherwise
    # leave the uploaded files on disk with no job to clean them up.
    spec, _ = await resolve_spec()

    names: list[str] = []
    uploaded: list[str] = []
    if references:
        # Reference mode: nothing crosses the tunnel, so the upload cap is moot.
        names = [reference_existing_input(item) for item in references]
    else:
        budget = settings.max_request_bytes
        try:
            for index, upload in enumerate(image):
                data = await read_capped(upload, min(budget, settings.max_upload_bytes))
                budget -= len(data)
                suffix = next((s for magic, s in IMAGE_MAGIC if data.startswith(magic)), None)
                if suffix is None:
                    raise HTTPException(
                        status_code=415,
                        detail=f"{getattr(upload, 'filename', 'upload')}: expected PNG or JPEG",
                    )
                uploaded.append(await _client().upload_image(data, f"qwen21_{index}{suffix}"))
                names.extend(uploaded[-1:])
        except Exception:
            # A part refused halfway through must not leave its predecessors.
            for name in uploaded:
                storage.unlink_quietly(storage.resolve(name, "", "input"))
            raise
    return await _execute(
        "edit", spec, _params(body), names, body.wait is not False, request,
        response_format=body.response_format or settings.response_format,
        uploaded=uploaded,
    )


@app.get(f"{API}/jobs", response_model=JobList, dependencies=[Depends(require_key)])
async def list_jobs(limit: int = 50) -> JobList:
    return JobList(jobs=[_job_status(job) for job in store.list(limit)])


@app.get(f"{API}/jobs/{{job_id}}", response_model=JobStatus, dependencies=[Depends(require_key)])
async def job_detail(job_id: str) -> JobStatus:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return _job_status(job)


@app.get(f"{API}/jobs/{{job_id}}/image/{{index}}", dependencies=[Depends(require_key)])
async def job_image(job_id: str, index: int = 0) -> Response:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    entry = job.image(index)
    if entry is None:
        raise HTTPException(status_code=404, detail="no such image on this job")
    blob = job.blob(index)
    if blob is None:
        # Images are held in RAM and nothing is written to disk, so once the
        # cache has moved on (or the service restarted) there is nothing to give.
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=(
                "image no longer retained: rendered images are kept in memory only "
                "(IMAGE_CACHE_MB). Re-run with wait=true, or poll sooner."
            ),
        )
    return Response(
        content=blob,
        media_type=entry.get("media_type", "image/png"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post(f"{API}/jobs/{{job_id}}/cancel", dependencies=[Depends(require_key)])
async def job_cancel(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status in {SUCCEEDED, FAILED}:
        return {"id": job.id, "status": job.status, "cancelled": False}
    await store.cancel(job)
    if job.comfy_prompt_id:
        await _client().interrupt()
    return {"id": job.id, "status": job.status, "cancelled": True}


@app.post(
    f"{API}/internal/tunnel",
    response_model=RegistryEntry,
    dependencies=[Depends(require_key)],
)
async def publish_tunnel(body: TunnelPublish) -> RegistryEntry:
    """Record this device's public link, overwriting whatever was there.

    Called by `qwen21 start` once the tunnel is live. Idempotent: a device that
    restarts publishes again and the same row is updated, which is what makes
    the collection a "where is each machine right now" list rather than a log.
    """
    try:
        validate_link(body.link)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        record = await registry.publish(body.link)
    except RegistryError as exc:
        # The endpoint is healthy; only the directory is not. Say so, do not
        # make the caller think the tunnel is down.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"device registry unavailable: {exc}",
            headers={"Retry-After": "30"},
        ) from exc
    except Exception as exc:  # nothing the registry does may become a 500
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"device registry failed: {type(exc).__name__}",
            headers={"Retry-After": "30"},
        ) from exc
    record["published"] = True
    record["target"] = registry.target
    return RegistryEntry(**record)


@app.get(f"{API}/registry", response_model=RegistryEntry, dependencies=[Depends(require_key)])
async def registry_entry() -> RegistryEntry:
    """What is currently stored for this device."""
    record = await registry.current()
    if record is None:
        return RegistryEntry(published=False, device=settings.host_name,
                             target=registry.target)
    record["published"] = True
    record["target"] = registry.target
    return RegistryEntry(**record)


async def _sweep_loop() -> None:
    while True:
        await asyncio.sleep(600)
        try:
            removed = await asyncio.to_thread(store.sweep)
            orphans = await asyncio.to_thread(
                storage.sweep, settings.orphan_max_age_min * 60
            )
            if removed:
                print(f"[sweeper] removed {removed} expired job(s)")
            if orphans:
                print(f"[sweeper] removed {len(orphans)} leftover image file(s)")
        except Exception as exc:  # housekeeping must never kill the service
            print(f"[sweeper] {exc}")
