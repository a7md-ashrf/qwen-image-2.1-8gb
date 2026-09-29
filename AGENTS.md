# Project memory: qwen-image-2.1-8gb

## Goal
Serve Qwen-Image-2.1 for **image editing** (`image + text prompt → edited image`)
over HTTP on **8GB hardware** (RTX 4060 8GB or MacBook Air M3 8GB), and make that
endpoint reachable from the internet through a tunnel — without a per-request
user touching ComfyUI.

## Current architecture (2.0)

```
client ──https──▶ cloudflared ──▶ qwen_api (FastAPI, own venv) ──HTTP──▶ ComfyUI v0.37.0
                   (quick or named)   Bearer auth, job queue,           (submodule, own venv)
                                       200-or-202 contract              loopback only
```

Two processes on purpose: the installer runs on a bare Python (no deps), and the
API runs in its own venv so a Windows portable `python_embeded` ComfyUI and a
macOS venv can sit on the same repo without dependency conflicts.

- `qwen21/` — installer + supervisor, **stdlib only**: `cli.py` (14 subcommands),
  `profiles.py` (model/flag table per hardware), `models.py` (resumable download
  + size/sha verify), `comfy_install.py` (submodule, venvs, ComfyUI-GGUF + its
  `gguf` requirement, workflows), `runner.py` (pid files, log tail, health
  probes, no systemd/launchd), `tunnel.py` (cloudflared quick/named).
- `qwen_api/` — the service: `app.py` (routes), `workflows.py` (builds the
  **API-format** ComfyUI graph in code), `comfy_client.py` (5 ComfyUI
  endpoints), `jobs.py` (SQLite-backed job table), `registry.py` (device →
  public link in MongoDB), `config.py`, `schemas.py`.
- `ComfyUI/` — git submodule, pinned `v0.37.0` (first stable release with
  `TextEncodeQwenImage21` + `QwenImage21Cache`).
- `workflows/` — UI templates regenerated from Comfy-Org's by
  `scripts/make_gguf_workflow.py` (loaders rewritten to GGUF/w4a8, prompt
  enhancement off), plus `workflows/api/*.json` = the deterministic graphs the
  service posts, one pair per profile.

## The 8GB budget (do not "improve" this away)
- **DiT:** `qwen_image_2.1-Q4_K.gguf` (4.2 GB) via `UnetLoaderGGUF`.
- **Encoder:** `qwen3vl_8b_w4a8.safetensors` (6.3 GB) with `CLIPLoader.device="cpu"`.
- **VAE:** `qwen_image_2.1_vae_bf16.safetensors` (0.68 GB), 4-channel (alpha).
- The official int8 stack (7.3 GB transformer + 9.4 GB encoder) cannot coexist in
  8 GB of anything. That is why we ship GGUF + 4-bit encoder.
- Mac: Q3_K (Q2_K in `mac-8gb-lite`) because 8 GB *unified* memory also has to
  hold macOS. MPS cannot run int8_convrot.
- The KV cache node (`QwenImage21Cache`, `device=auto`, `dtype=default`) is most
  of an edit's speed; `dtype=int8` halves it, `device=off` disables it.
- Sampler defaults match ComfyUI's official 2.1 templates: 25 steps, cfg 1.0,
  `euler` / `simple`.

## API contract
`POST /v1/edit` (multipart image+prompt), `POST /v1/generate` (JSON prompt),
`GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/image/{n}`, `POST /v1/jobs/{id}/cancel`,
`GET /v1/models`, `GET /healthz` (no auth), `GET /docs`.
Bearer `API_KEY` on everything else; `MAX_CONCURRENT=1`, `MAX_QUEUE=8`, per-key
rate limit, SQLite job table in `runtime/outputs/`, 24 h retention sweep.

**A request never blocks past the tunnel's patience:** wait up to
`SYNC_MAX_WAIT` (90 s), else return `202` + job id. Cloudflare's proxy read
timeout is 125 s; a 25-step edit takes longer than that on this hardware.

## Verified (on a real M3, ComfyUI 0.37.0)
- install chain: submodule → venvs → torch → ComfyUI-GGUF (+`gguf`) → workflows
- `start/stop/restart/status/logs`; ComfyUI is MPS-capable, API health is green
- **both generated graphs accepted by ComfyUI's `/prompt`**, including the
  autogrow reference-image slots `images: {"image_1": ["100", 0]}`
- 72 tests (`python -m unittest discover -s tests`); ruff clean (F, E9, I001, UP)

## NOT verified — do not claim otherwise
- **A real image has never been generated**: the model weights are not installed
  by default. `qwen21 models && qwen21 smoke` is the acceptance test.
- **Windows**: `install.ps1` and the nvidia-8gb profile have never been executed
  on Windows. Written to be OS-agnostic; that is a claim, not a result.
- **A real Cloudflare tunnel has never been published** (needs the user's token).
- `docs/HARDWARE.md` deliberately quotes **no timings** until someone measures.

## Known traps hit while building this
- ComfyUI returns 400 bodies as a **JSON string containing JSON**; both HTTP
  clients must unwrap twice or the error detail disappears.
- `--lowvram` is a no-op in current ComfyUI (dynamic VRAM). Don't re-add it.
- ComfyUI-GGUF's node needs its own `requirements.txt` installed, or model load
  fails on a missing `gguf` module.
- Children must be started with `start_new_session=True`, or `killpg` signals the
  CLI itself.
- The `TextEncodeQwenImage21` autogrow input is a **dict of slot → link** in API
  format, not a dotted `images.image_1` input.

## Device registry (optional)
A quick tunnel hostname is random per restart, so `HOST_NAME` (default: OS
hostname) is the stable key. `qwen21 start` POSTs the live link to
`POST /v1/internal/tunnel`; the service upserts `{link, device, updated_at}`
into MongoDB with a **unique index on `device`**, so a restart overwrites the
device's own row. `pymongo.AsyncMongoClient` — Motor hit EOL in May 2026.
Three rules: the device comes from the server config, never the request body;
the URI is never logged; and the registry is **best-effort**, so a Mongo outage
returns 503 on the publish route and never affects the endpoint.
`scripts/check_secrets.py` fails the build if a real credential reaches a
tracked file — its own test fixtures are assembled from parts for that reason.

## Pending, agreed but not implemented
Cloudflare request-size ceiling: Free/Pro cap uploads at **100 MB** (Business
200 MB), while the current defaults allow `MAX_UPLOAD_MB=25 × MAX_IMAGES=4` =
100 MB of image data + multipart overhead. Plan: a `MAX_REQUEST_MB` (90) total
ceiling enforced from `Content-Length` and while streaming, `MAX_UPLOAD_MB` down
to 20, a `response_format` (`b64` | `url`) option, filename mode to reference a
file already in `ComfyUI/input/`, and doc corrections (125 s read timeout,
30 s write timeout).

## Commands
```
./install.sh | .\install.ps1     # or: python -m qwen21 install
python -m qwen21 {doctor,start,stop,restart,status,logs,tunnel,serve,comfy,smoke,models,keys,keygen,export}
python -m unittest discover -s tests -v
```
