"""`qwen21` - install, check, run, and publish the local Qwen-Image-2.1 stack."""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import comfy_install, config, profiles, tunnel
from . import models as models_mod
from .comfy_install import InstallError
from .env import describe_unfilled, env_path, generate_api_key, load_env, unfilled_keys
from .runner import Runner, Service

BANNER = "Qwen-Image-2.1 on 8GB - installer / API toolkit"
OK = "OK"
MISS = "MISS"
BAD = "BAD"


def log(message: str = "") -> None:
    print(message, flush=True)


def head(title: str) -> None:
    log()
    log(title)
    log("-" * len(title))


def die(message: str) -> int:
    print(f"\nerror: {message}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib so the CLI works before any venv exists)
# --------------------------------------------------------------------------- #


def _decode(body: str) -> object:
    """ComfyUI answers 400s with a JSON *string* holding JSON; unwrap that."""
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except ValueError:
            return payload
    return payload


def http_json(
    url: str, payload: dict | None = None, headers: dict[str, str] | None = None, timeout: int = 30
) -> tuple[int, dict | list | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers or {})
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decode(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


def http_multipart(
    url: str, files: list[tuple[str, str, bytes, str]], fields: dict[str, str], headers: dict[str, str]
) -> tuple[int, dict | str]:
    boundary = "----qwen21" + secrets.token_hex(12)
    body = bytearray()
    for name, filename, payload, content_type in files:
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        body += payload + b"\r\n"
    for name, value in fields.items():
        if value is None:
            continue
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=bytes(body), headers=headers)
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #


def cmd_install(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        settings = config.resolve(root, args.profile, args.comfy_path)
    except InstallError as exc:
        return die(str(exc))

    profile = settings.profile
    log(BANNER)
    log(f"profile: {profile.name} - {profile.summary}")
    detected, reasons = profiles.detect_profile()
    for reason in reasons:
        log(f"  detected: {reason} -> {detected}")
    for warning in profile.warnings:
        log(f"  ! {warning}")

    free = models_mod.free_disk_gib(settings.comfy)
    needed = profile.total_bytes / (1024**3)
    log(f"  disk: {free:.0f} GiB free, this profile downloads ~{needed:.0f} GiB")
    if free < needed + 3:
        return die(f"need about {needed + 3:.0f} GiB free, only {free:.0f} GiB available")

    head("ComfyUI")
    if args.comfy_path:
        log(f"  using {settings.comfy}")
    else:
        comfy_install.ensure_submodule(root, args.ref)

    head("Python environment")
    comfy_python = comfy_install.venv_python(settings.venv_comfy)
    if args.recreate_env:
        comfy_install.create_venv(settings.venv_comfy, force=True)
    else:
        comfy_install.create_venv(settings.venv_comfy)
    comfy_install.install_torch(comfy_python, profile)
    comfy_install.install_comfy_requirements(settings.comfy, comfy_python)
    if profile.needs_gguf_node:
        comfy_install.install_gguf_node(settings.comfy, comfy_python)
    else:
        log("ComfyUI-GGUF not needed for this profile (core UNETLoader is used)")

    if not args.skip_models:
        models_mod.install_models(settings.comfy, profile.models, force=args.force)

    comfy_install.install_workflows(settings.comfy, root)

    head("API service")
    api_python = comfy_install.venv_python(settings.venv_api)
    if args.recreate_env:
        comfy_install.create_venv(settings.venv_api, force=True)
    else:
        comfy_install.create_venv(settings.venv_api)
    comfy_install.pip_install(
        api_python, "-r", str(root / "api" / "requirements.txt"), quiet=True
    )

    head("Configuration")
    target = env_path()
    if not target.is_file():
        shutil.copy2(root / ".env.example", target)
        log(f"  ✓ wrote {target.name} (fill in the keys listed by `qwen21 doctor`)")
    else:
        log(f"  · {target.name} already exists, left untouched")

    return cmd_doctor(args)


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        settings = config.resolve(root, args.profile, args.comfy_path)
    except InstallError as exc:
        return die(str(exc))
    load_env(env_path())
    profile = settings.profile
    failures: list[str] = []
    warnings: list[str] = []

    log(BANNER)
    log(f"profile {profile.name} | comfy {settings.comfy}")

    head("Machine")
    log(f"  python     {sys.version.split()[0]} ({sys.executable})")
    log(f"  git        {shutil.which('git') or 'MISSING'}")
    if not shutil.which("git"):
        failures.append("git is required for the submodule and ComfyUI-GGUF")
    info = profiles.gpu_info()
    if info:
        name, mib = info
        log(f"  GPU        {name} - {mib / 1024:.1f} GiB VRAM")
    elif profiles.apple_silicon():
        log(f"  GPU        Apple Silicon - {profiles.mac_total_memory_gib():.0f} GiB unified memory")
    else:
        warnings.append("no NVIDIA or Apple GPU detected; the nvidia-8gb profile assumes one")
    log(f"  memory     {profiles.system_memory_gib():.0f} GiB system")
    if profiles.system_memory_gib() < 12:
        warnings.append("below 12GB system RAM: expect heavy paging during encoding and sampling")
    free = models_mod.free_disk_gib(settings.comfy)
    log(f"  disk free  {free:.0f} GiB")

    head("ComfyUI")
    version = comfy_install.comfy_version(settings.comfy)
    if not (settings.comfy / "main.py").is_file():
        failures.append(f"no ComfyUI at {settings.comfy} (run `qwen21 install`)")
        log(f"  {MISS}   ComfyUI not found")
    else:
        shown = ".".join(map(str, version)) if version else "unknown version"
        log(f"  {OK if version else '?'}   checkout {shown}")
        if not comfy_install.version_ok(settings.comfy):
            minimum = ".".join(map(str, profiles.MIN_COMFY_VERSION))
            failures.append(
                f"ComfyUI {shown} is older than {minimum}; the Qwen-Image-2.1 nodes are missing"
            )
    comfy_python = settings.comfy_python
    log(f"  {OK if comfy_python.is_file() else BAD}   interpreter {comfy_python}")
    if profile.needs_gguf_node:
        ready = comfy_install.gguf_node_ready(settings.comfy)
        log(f"  {OK if ready else BAD}   ComfyUI-GGUF")
        if not ready:
            failures.append("ComfyUI-GGUF is not installed (run `qwen21 install`)")
    api_python = settings.api_python
    log(f"  {OK if api_python.is_file() else BAD}   api interpreter {api_python}")
    if not api_python.is_file():
        failures.append("API venv missing (run `qwen21 install`)")

    head("Models")
    for model in profile.models:
        path = settings.comfy / "models" / model.rel
        ok, detail = models_mod.verify(path, model)
        log(f"  {OK if ok else BAD}   {model.name} ({detail})")
        if not ok:
            failures.append(f"{model.rel} {detail} (run `qwen21 models`)")

    head("Configuration")
    parsed = load_env(env_path())
    for spec in unfilled_keys(parsed):
        state = "set" if parsed.get(spec.name) else "missing"
        log(f"  {BAD if spec.required else '?'}   {spec.name} ({state}) - {spec.where}")
    if parsed.get("API_KEY"):
        log("  OK   API_KEY present")
    else:
        failures.append("API_KEY is not set (run `qwen21 keygen --write`)")

    head("Services")
    runner = Runner(root)
    running = runner.running()
    log(f"  processes  {', '.join(f'{k}={v}' for k, v in running.items()) or 'none'}")
    if "comfy" in running:
        status, payload = http_json(f"{settings.comfy_url}/system_stats")
        if status == 200 and isinstance(payload, dict):
            device = (payload.get("devices") or [{}])[0]
            log(
                f"  {OK}   ComfyUI up: {device.get('name', '?')}, "
                f"vram_total={device.get('vram_total', 0) / 2**30:.1f} GiB"
            )
            for node in ("TextEncodeQwenImage21", "QwenImage21Cache", "VAEDecode", "SaveImage"):
                code, _ = http_json(f"{settings.comfy_url}/object_info/{node}")
                if code != 200:
                    failures.append(f"ComfyUI does not know node {node}")
    else:
        log("  ·   ComfyUI is not running (`qwen21 start`)")
    if "api" in running:
        code, payload = http_json(f"{settings.api_url}/healthz")
        log(f"  {OK if code == 200 else BAD}   API {payload if code != 200 else 'ok'}")
    else:
        log("  ·   API is not running (`qwen21 start`)")
    binary = tunnel.find_existing(settings.tools_dir)
    log(f"  {'OK' if binary else '?'}   cloudflared {binary or 'not downloaded yet'}")
    log(f"  tunnel mode {settings.tunnel_mode}")

    head("Verdict")
    for warning in warnings:
        log(f"  warn  {warning}")
    for failure in failures:
        log(f"  FAIL  {failure}")
    if failures:
        log(f"\n{len(failures)} problem(s) to fix. Start with: qwen21 install")
        return 1
    log("\nReady. `qwen21 start` then `qwen21 status`")
    return 0


# --------------------------------------------------------------------------- #
# process control
# --------------------------------------------------------------------------- #


def _comfy_service(settings: config.Settings) -> Service:
    cmd = [
        str(settings.comfy_python),
        "-s",
        str(settings.comfy / "main.py"),
        "--listen",
        settings.comfy_host,
        "--port",
        str(settings.comfy_port),
        *settings.profile.comfy_args,
    ]
    return Service(
        name="comfy",
        cmd=cmd,
        cwd=settings.comfy,
        env=dict(settings.profile.comfy_env),
        url=f"{settings.comfy_url}/system_stats",
    )


def _api_service(settings: config.Settings) -> Service:
    return Service(
        name="api",
        cmd=[str(settings.api_python), "-m", "qwen_api"],
        cwd=settings.root,
        env=settings.api_env(),
        url=f"{settings.api_url}/healthz",
    )


def _tunnel_service(settings: config.Settings, mode: str) -> Service | None:
    if mode == "off":
        return None
    binary = tunnel.ensure_cloudflared(settings.tools_dir)
    token = tunnel.load_token()
    if mode == "named" and not token:
        return None
    origin = settings.api_origin
    return Service(
        name="tunnel",
        cmd=tunnel.build_cmd(binary, mode, origin, token),
        cwd=settings.root,
        env={},
    )


def cmd_start(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    load_env(env_path())
    try:
        settings = config.resolve(root, args.profile, args.comfy_path)
    except InstallError as exc:
        return die(str(exc))
    runner = Runner(root)

    if not (settings.comfy / "main.py").is_file():
        return die(f"ComfyUI is not installed at {settings.comfy}. Run `qwen21 install` first.")
    if not settings.api_python.is_file():
        return die("the API virtualenv is missing. Run `qwen21 install` first.")
    if not load_env(env_path()).get("API_KEY"):
        log("! API_KEY is not set. Generate one now with `qwen21 keygen --write`,")
        log("  or the service will answer 500 until you do.")

    mode = args.tunnel or settings.tunnel_mode
    head("Starting")
    if not runner.start(_comfy_service(settings)):
        return die(
            f"ComfyUI did not come up on {settings.comfy_url}. "
            f"See {runner.log_file('comfy')} (or run `qwen21 comfy` in the foreground)."
        )
    if not runner.start(_api_service(settings)):
        return die(
            f"the API service did not come up on {settings.api_url}. "
            f"See {runner.log_file('api')}; is API_KEY set in .env?"
        )

    public = None
    if mode != "off":
        service = _tunnel_service(settings, mode)
        if service is None:
            log("  ! named tunnel requested but TUNNEL_TOKEN is missing; staying local")
        else:
            runner.start(service)
            if mode == "quick":
                log("  … waiting for the cloudflared hostname")
                public = tunnel.quick_tunnel_url(runner.log_file("tunnel"), timeout=60)
            else:
                public = tunnel.public_url("named", os.environ.get("TUNNEL_HOSTNAME"))

    head("Ready")
    log(f"  ComfyUI   {settings.comfy_url}")
    log(f"  API       {settings.api_url}   (docs at /docs)")
    if public:
        log(f"  Public    {public}/v1/edit")
    else:
        log("  Public    not exposed (TUNNEL_MODE=off)")
    log(f"  Logs      {runner.logs}")
    log("  Stop with qwen21 stop")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    runner = Runner(Path(__file__).resolve().parent.parent)
    head("Stopping")
    runner.stop_all()
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cmd_stop(args)
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    settings = config.resolve(root, args.profile, args.comfy_path)
    runner = Runner(root)
    head("Status")
    running = runner.running()
    for name in ("comfy", "api", "tunnel"):
        pid = running.get(name)
        log(f"  {'running' if pid else 'stopped'}  {name}" + (f" (pid {pid})" if pid else ""))
    code, payload = http_json(f"{settings.api_url}/healthz")
    if code == 200 and isinstance(payload, dict):
        log(f"  api health: {payload.get('status')}  running={payload.get('running')} "
            f"queued={payload.get('queued')}")
    else:
        log(f"  api health: unreachable ({payload if code == 0 else code})")
    quick = tunnel.quick_tunnel_url(runner.log_file("tunnel"), timeout=1)
    if quick:
        log(f"  public url: {quick}/v1/edit")
    elif os.environ.get("TUNNEL_HOSTNAME"):
        log(f"  public url: https://{os.environ['TUNNEL_HOSTNAME']}/v1/edit")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    runner = Runner(Path(__file__).resolve().parent.parent)
    names = [args.name] if args.name else ["api", "comfy", "tunnel"]
    for name in names:
        head(f"logs: {name}")
        log(runner.tail(name, args.lines).rstrip())
    return 0


def cmd_tunnel(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    load_env(env_path())
    settings = config.resolve(root, args.profile, args.comfy_path)
    runner = Runner(root)
    mode = args.mode or settings.tunnel_mode
    head(f"Tunnel ({mode})")
    if mode == "off":
        if runner.stop("tunnel"):
            log("  ✓ stopped the running tunnel")
        log("  the API stays on " + settings.api_url)
        return 0
    if mode == "named" and not tunnel.load_token():
        log("TUNNEL_TOKEN looks empty. Create a tunnel at")
        log("  Cloudflare Zero Trust -> Networks -> Tunnels -> Create a tunnel")
        log("  then paste that base64 blob into .env as TUNNEL_TOKEN=<the blob>")
        return 2
    service = _tunnel_service(settings, mode)
    assert service
    runner.start(service)
    if mode == "quick":
        url = tunnel.quick_tunnel_url(runner.log_file("tunnel"), timeout=60)
        log(f"  public: {url}/v1/edit" if url else "  ! no hostname yet, check `qwen21 logs tunnel`")
    else:
        hostname = os.environ.get("TUNNEL_HOSTNAME")
        log(f"  public: {tunnel.public_url('named', hostname)}/v1/edit")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    load_env(env_path())
    settings = config.resolve(root, args.profile, args.comfy_path)
    environment = {**os.environ, **settings.api_env()}
    import subprocess

    return subprocess.call(
        [str(settings.api_python), "-m", "qwen_api"], cwd=str(root), env=environment
    )


def cmd_comfy(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    settings = config.resolve(root, args.profile, args.comfy_path)
    extra = args.extra or []
    environment = {**os.environ, **settings.profile.comfy_env}
    import subprocess

    return subprocess.call(
        [str(_comfy_service(settings).cmd[0]), "-s", str(settings.comfy / "main.py"), *extra],
        cwd=str(settings.comfy),
        env=environment,
    )


# --------------------------------------------------------------------------- #
# smoke tests
# --------------------------------------------------------------------------- #


def _sample_png() -> bytes:
    """A 256x256 PNG made without Pillow: enough to exercise the edit path."""
    import struct
    import zlib

    width = height = 256
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw += bytes(((x * 255) // width, (y * 255) // height, 128))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    load_env(env_path())
    settings = config.resolve(root, args.profile, args.comfy_path)
    if args.direct:
        return _smoke_direct(settings, args)
    return _smoke_api(settings, args)


def _smoke_api(settings: config.Settings, args: argparse.Namespace) -> int:
    key = load_env(env_path()).get("API_KEY") or os.environ.get("API_KEY", "")
    if not key:
        return die("API_KEY is not set (qwen21 keygen --write)")
    headers = {"Authorization": f"Bearer {key}"}
    head("Smoke test through the API")

    code, payload = http_json(f"{settings.api_url}/healthz", headers=headers)
    if code != 200:
        return die(f"API is not answering on {settings.api_url} ({payload})")
    log(f"  {OK}   healthz")

    code, payload = http_json(f"{settings.api_url}/v1/models", headers=headers)
    if code != 200:
        return die(f"/v1/models failed: {code} {payload}")
    log(f"  {OK}   models: {payload.get('unet')} + {payload.get('clip')}")

    fields = {
        "prompt": "make the gradient warmer, keep everything else identical",
        "resolution": str(args.resolution),
        "steps": str(args.steps),
        "wait": "true",
    }
    log(f"  …   editing a synthetic {args.resolution}px png, {args.steps} steps")
    started = time.time()
    code, payload = http_multipart(
        f"{settings.api_url}/v1/edit",
        [("image", "smoke.png", _sample_png(), "image/png")],
        fields,
        headers,
    )
    if code not in (200, 202):
        return die(f"/v1/edit failed: {code} {payload}")
    if code == 202:
        log(f"  …   queued as job {payload.get('id')}; polling (this runs a real generation)")
        job = _poll_job(settings, payload["id"], headers, args.timeout)
        if job is None:
            return die("job did not finish; check `qwen21 logs api`")
        blob = _fetch_image(settings, job["id"], 0, headers)
    else:
        blob = base64.b64decode(payload["images"][0]["b64_json"])
        job = payload
    if not blob:
        return die("no image bytes came back")
    log(f"  {OK}   {len(blob) / 1024:.0f} KiB PNG in {time.time() - started:.0f}s")
    log(f"  {OK}   seed={job.get('seed')} job={job.get('id')}")
    return 0


def _poll_job(settings: config.Settings, job_id: str, headers: dict[str, str], timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, payload = http_json(f"{settings.api_url}/v1/jobs/{job_id}", headers=headers)
        if code != 200:
            return None
        status = payload.get("status")
        if status == "succeeded":
            return payload
        if status in {"failed", "cancelled"}:
            log(f"  !   job {status}: {payload.get('error')}")
            return None
        time.sleep(3)
    return None


def _fetch_image(settings: config.Settings, job_id: str, index: int, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(
        f"{settings.api_url}/v1/jobs/{job_id}/image/{index}", headers=headers
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _classify(code: int, payload: object) -> tuple[str, list[str]]:
    """Split a /prompt response into a verdict and the node errors.

    `value_not_in_list` on a loader node only means the model file is not
    installed yet, which is a normal state before `qwen21 models`. Anything
    else is a real problem with the graph we generated.
    """
    if code == 200 and isinstance(payload, dict) and payload.get("prompt_id"):
        return "accepted", []
    errors = payload.get("node_errors", {}) if isinstance(payload, dict) else {}
    messages: list[str] = []
    blocking: list[str] = []
    for node_id, detail in errors.items():
        for entry in (detail or {}).get("errors", []) or []:
            text = f"node {node_id}: {entry.get('type')} - {entry.get('message', '')}".strip()
            messages.append(text)
            if entry.get("type") != "value_not_in_list":
                blocking.append(text)
    if not messages:
        messages.append(f"ComfyUI rejected the graph with HTTP {code} and no node detail: {payload}")
        blocking.append(messages[-1])
    return ("blocked" if not blocking else "invalid"), messages


def _smoke_direct(settings: config.Settings, args: argparse.Namespace) -> int:
    """Validate the generated graphs against ComfyUI's own /prompt validator.

    This is the pre-flight check that needs no model files: node wiring, input
    names, the autogrow image slots and the sampler options are all validated
    before anything is loaded onto the GPU.
    """
    head("Smoke test: workflow graph vs ComfyUI")
    code, payload = http_json(f"{settings.comfy_url}/system_stats")
    if code != 200:
        return die(f"ComfyUI is not answering on {settings.comfy_url} ({payload})")
    version = (payload.get("system") or {}).get("comfyui_version", "?")
    log(f"  {OK}   ComfyUI {version} up")

    sys.path.insert(0, str(settings.root))
    from qwen_api.workflows import ModelSpec, build_edit_prompt, build_generate_prompt

    profile = settings.profile
    listing = {}
    for folder, fallback in (
        ("diffusion_models", profile.model("diffusion_models").name),
        ("text_encoders", profile.model("text_encoders").name),
        ("vae", profile.model("vae").name),
    ):
        code, found = http_json(f"{settings.comfy_url}/models/{folder}")
        found = [f for f in (found or []) if isinstance(found, list) and f] if code == 200 else []
        listing[folder] = found[0] if found else fallback
        if not found:
            log(f"  ·   no {folder} installed yet (will validate as '{fallback}')")
    spec = ModelSpec(
        unet=listing["diffusion_models"],
        clip=listing["text_encoders"],
        vae=listing["vae"],
        unet_gguf=str(listing["diffusion_models"]).endswith(".gguf"),
        encoder_device=profile.encoder_device,
        cache_device=profile.cache_device,
        cache_dtype=profile.cache_dtype,
    )
    log(f"  ·   loader: {spec.unet} + {spec.clip} + {spec.vae}")

    failures = 0
    for name, graph in (
        ("generate", build_generate_prompt(spec, "test", width=256, height=256, steps=1,
                                          prefix="qwen21/smoke")),
        ("edit", build_edit_prompt(spec, "test", ["qwen21_smoke_input.png"], resolution=256,
                                   steps=1, prefix="qwen21/smoke")),
    ):
        code, payload = http_json(
            f"{settings.comfy_url}/prompt", {"prompt": graph, "client_id": "smoke"}
        )
        verdict, messages = _classify(code, payload)
        if verdict == "accepted":
            log(f"  {OK}   {name} graph accepted - ComfyUI will run it")
        elif verdict == "blocked":
            log(f"  {OK}   {name} graph is valid; only the model files are missing")
            for message in messages:
                log(f"        {message}")
        else:
            failures += 1
            log(f"  {BAD}  {name} graph is invalid:")
            for message in messages:
                log(f"        {message}")
    if failures:
        log(f"\n{failures} graph(s) rejected. This is a bug in qwen_api/workflows.py.")
        return 1
    log("\nGraphs validated. Run `qwen21 models` and then `qwen21 smoke` for a real generation.")
    return 0


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def cmd_models(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    settings = config.resolve(root, args.profile, args.comfy_path)
    head(f"Models ({settings.profile.name})")
    models_mod.install_models(settings.comfy, settings.profile.models, force=args.force)
    return 0


def cmd_keygen(args: argparse.Namespace) -> int:
    key = generate_api_key()
    if not getattr(args, "write", False):
        log(key)
        log()
        log("add it to .env as:  API_KEY=" + key)
        return 0
    target = env_path()
    if not target.is_file():
        shutil.copy2(root_env_example(), target)
    text = target.read_text(encoding="utf-8")
    lines = [
        f"API_KEY={key}" if line.strip().startswith("API_KEY=") else line
        for line in text.splitlines()
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"✓ API_KEY written to {target}")
    return 0


def root_env_example() -> Path:
    return Path(__file__).resolve().parent.parent / ".env.example"


def cmd_export(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    settings = config.resolve(root, args.profile, args.comfy_path)
    sys.path.insert(0, str(root))
    from qwen_api.workflows import ModelSpec, build_edit_prompt, build_generate_prompt

    spec = ModelSpec(
        unet=settings.profile.model("diffusion_models").name,
        clip=settings.profile.model("text_encoders").name,
        vae=settings.profile.model("vae").name,
        unet_gguf=settings.profile.loader == "gguf",
        encoder_device=settings.profile.encoder_device,
        cache_device=settings.profile.cache_device,
        cache_dtype=settings.profile.cache_dtype,
    )
    graphs = {
        "edit": build_edit_prompt(
            spec,
            "Replace the lettering on the cup with 'ROASTED IN TOKYO'",
            ["put your input image in ComfyUI/input and name it here"],
            resolution=1024,
            steps=25,
            seed=1234567890,  # fixed so the committed files never churn
            prefix="qwen21/example",
        ),
        "generate": build_generate_prompt(
            spec,
            "A cinematic coffee cart at dusk",
            width=1024,
            height=1024,
            steps=25,
            seed=1234567890,
            prefix="qwen21/example",
        ),
    }
    out_dir = root / "workflows" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, graph in graphs.items():
        target = out_dir / f"qwen21-{settings.profile.name}-{name}.json"
        target.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        log(f"✓ {target.relative_to(root)}  ({len(graph)} nodes)")
    log()
    log("These are API-format graphs. Paste them into ComfyUI's")
    log("'Load' > advanced 'API format' loader, or post them to /prompt.")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    load_env(env_path())
    lines = describe_unfilled()
    head(f"Keys ({env_path().name})")
    log("\n".join(lines) if lines else "  all set")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen21",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical first run:\n"
            "  python -m qwen21 install        # clone ComfyUI, venvs, models, .env\n"
            "  python -m qwen21 start          # ComfyUI + API + public tunnel\n"
            "  python -m qwen21 status         # where is it reachable?\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_common(p: argparse.ArgumentParser, *, profile_default: str = "auto") -> None:
        p.add_argument("--profile", default=profile_default, choices=["auto", *profiles.profile_names()])
        p.add_argument("--comfy-path", default=os.environ.get("QWEN_COMFY_PATH"),
                       help="use an existing ComfyUI instead of the submodule")

    p_install = sub.add_parser("install", help="Set up ComfyUI, venvs, models and .env")
    with_common(p_install)
    p_install.add_argument("--force", action="store_true", help="re-download model files")
    p_install.add_argument("--skip-models", action="store_true")
    p_install.add_argument("--recreate-env", action="store_true", help="rebuild the virtualenvs")
    p_install.add_argument("--ref", default=None, help=f"ComfyUI ref (default {profiles.RECOMMENDED_COMFY_REF})")
    p_install.set_defaults(func=cmd_install)

    p_doctor = sub.add_parser("doctor", help="Check hardware, files, nodes and keys")
    with_common(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    p_start = sub.add_parser("start", help="Run ComfyUI, the API and the tunnel")
    with_common(p_start)
    p_start.add_argument("--tunnel", choices=["quick", "named", "off"], default=None)
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop everything this tool started")
    p_stop.set_defaults(func=cmd_stop)

    p_restart = sub.add_parser("restart", help="Stop everything, then start it again")
    with_common(p_restart)
    p_restart.add_argument("--tunnel", choices=["quick", "named", "off"], default=None)
    p_restart.set_defaults(func=cmd_restart)

    p_status = sub.add_parser("status", help="What is running, and where")
    with_common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Tail the service logs")
    p_logs.add_argument("name", nargs="?", choices=["comfy", "api", "tunnel"])
    p_logs.add_argument("-n", "--lines", type=int, default=40)
    p_logs.set_defaults(func=cmd_logs)

    p_tunnel = sub.add_parser("tunnel", help="Start/stop the public tunnel on its own")
    with_common(p_tunnel)
    p_tunnel.add_argument("--mode", choices=["quick", "named", "off"], default=None)
    p_tunnel.set_defaults(func=cmd_tunnel)

    p_serve = sub.add_parser("serve", help="Run only the API in the foreground")
    with_common(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_comfy = sub.add_parser("comfy", help="Run only ComfyUI in the foreground")
    with_common(p_comfy)
    p_comfy.add_argument("extra", nargs=argparse.REMAINDER, help="extra ComfyUI arguments")
    p_comfy.set_defaults(func=cmd_comfy)

    p_smoke = sub.add_parser("smoke", help="End-to-end test of the running stack")
    with_common(p_smoke)
    p_smoke.add_argument("--direct", action="store_true",
                         help="only validate the graph against ComfyUI's /prompt")
    p_smoke.add_argument("--resolution", type=int, default=512)
    p_smoke.add_argument("--steps", type=int, default=4)
    p_smoke.add_argument("--timeout", type=float, default=1800)
    p_smoke.set_defaults(func=cmd_smoke)

    p_models = sub.add_parser("models", help="Download or verify the model files")
    with_common(p_models)
    p_models.add_argument("--force", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_keys = sub.add_parser("keys", help="List the keys you still need to paste in")
    p_keys.set_defaults(func=cmd_keys)

    p_keygen = sub.add_parser("keygen", help="Generate a local API key")
    p_keygen.add_argument("--write", action="store_true", help="write it into .env")
    p_keygen.set_defaults(func=cmd_keygen)

    p_export = sub.add_parser("export", help="Write API-format workflow JSON files")
    with_common(p_export)
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InstallError as exc:
        return die(str(exc))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
