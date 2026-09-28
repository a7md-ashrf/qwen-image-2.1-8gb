# Qwen-Image-2.1 on 8GB — install, serve, expose

Run Qwen-Image-2.1 (7B, Apache-2.0) on an 8GB GPU **or** an 8GB Apple Silicon
Mac, and put a **public HTTP endpoint** in front of it that edits an image from
a text prompt.

One command installs everything: ComfyUI (pinned as a git submodule), the
right torch build, the GGUF model stack, a FastAPI service, and a Cloudflare
tunnel. No manual downloads, no YAML archaeology.

```bash
git clone --recurse-submodules https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb
./install.sh                 # Windows: .\install.ps1
qwen21 start                 # ComfyUI + API + public tunnel
```

```
ComfyUI   http://127.0.0.1:8188
API       http://127.0.0.1:8000  (interactive docs at /docs)
Public    https://<random>.trycloudflare.com/v1/edit
```

Edit an image from anywhere:

```bash
curl -X POST https://<tunnel>/v1/edit \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@before.png" \
  -F "prompt=Replace the lettering on the cup with 'ROASTED IN TOKYO', keep the ink texture"
```

JSON comes back with the rendered PNG as base64 plus a download URL. A full
reference is in [docs/API.md](docs/API.md).

## What "one command" does

`qwen21 install` (which `install.sh` / `install.ps1` bootstrap):

1. initialises the **ComfyUI submodule** pinned to `v0.37.0` — the first stable
   release with `TextEncodeQwenImage21` and `QwenImage21Cache`;
2. creates a virtualenv with the torch build for your hardware;
3. installs `city96/ComfyUI-GGUF` **and its `gguf` dependency** (the loader alone
   is not enough — it fails at load time without it);
4. downloads the profile's models with resume + size verification;
5. copies the bundled workflows into ComfyUI;
6. creates the API virtualenv and installs `api/requirements.txt`;
7. writes `.env` from `.env.example`.

Then it runs `doctor`, which tells you what is still missing.

## Hardware profiles

| profile | machine | diffusion model | text encoder | default |
|---|---|---|---|---|
| `nvidia-8gb` | RTX 4060 8GB, 2080 SUPER, 3060 | `qwen_image_2.1-Q4_K.gguf` (4.2 GB) | `qwen3vl_8b_w4a8` (6.3 GB, pinned to CPU) | 1024px, 25 steps |
| `mac-8gb` | MacBook Air M3 8GB | `qwen_image_2.1-Q3_K.gguf` (3.3 GB) | `qwen3vl_8b_w4a8` | 768px, 20 steps |
| `mac-8gb-lite` | 8GB and swapping badly | `Q2_K` (2.6 GB) | `qwen3vl_8b_w4a8` | 640px, 16 steps |
| `full` | 12GB+ VRAM / 16GB+ unified | `qwen_image_2.1_int8_convrot.safetensors` | `qwen3vl_8b_int8_convrot` | 1328px, 25 steps |

`auto` (the default) picks one from `nvidia-smi` / `sysctl hw.memsize`; override
with `--profile`.

Why GGUF and not the official int8 stack: the official template needs the
transformer (7.3 GB) *and* the encoder (9.4 GB) in memory, which no 8GB machine
has. A Q4 GGUF transformer (4.2 GB) plus a 4-bit encoder (6.3 GB) that runs on
the CPU fits, and ComfyUI's dynamic VRAM manager moves them in and out on its
own. Details and honest expectations: [docs/HARDWARE.md](docs/HARDWARE.md).

## Commands

| command | what it does |
|---|---|
| `qwen21 install` | full setup (`--skip-models`, `--force`, `--recreate-env`, `--ref`) |
| `qwen21 start` | ComfyUI + API + tunnel, in the background, in the right order |
| `qwen21 status` | what is up and where it is reachable |
| `qwen21 doctor` | hardware, files, node availability, keys; non-zero exit when broken |
| `qwen21 stop` / `restart` | stop everything / stop then start |
| `qwen21 logs api\|comfy\|tunnel` | tail a log |
| `qwen21 tunnel --mode quick\|named\|off` | manage the public URL on its own |
| `qwen21 models` | download or verify model files |
| `qwen21 smoke` | end-to-end test: uploads an image, edits it, reports bytes and time |
| `qwen21 smoke --direct` | validates the generated graph against ComfyUI's `/prompt` |
| `qwen21 export` | write the API-format workflow JSONs to `workflows/api/` |
| `qwen21 keygen [--write]` | generate the local API key |
| `qwen21 keys` | which secrets are still placeholders, and where to get them |
| `qwen21 serve` / `qwen21 comfy` | run just the API / just ComfyUI, in the foreground |

## Public endpoint

Default is a Cloudflare **quick tunnel**: no account, no DNS, a random
`*.trycloudflare.com` URL that changes on restart.

For a stable hostname, set `TUNNEL_MODE=named` and paste the token from
Cloudflare Zero Trust → Networks → Tunnels → Create into `.env`. Full walkthrough
and the security checklist: [docs/DEPLOY.md](docs/DEPLOY.md).

Two design details worth knowing:

* **The API is the only thing published.** ComfyUI stays on `127.0.0.1:8188`.
* **A request never blocks past the tunnel's patience.** Cloudflare drops an
  origin response after ~100 s of silence; a 25-step edit takes longer. So a call
  waits up to `SYNC_MAX_WAIT` (90 s) and otherwise returns `202` with a job id
  you can poll. One endpoint, no client-side retry logic.

## Keys

Only one is mandatory.

```bash
qwen21 keygen --write     # API_KEY in .env
```

`TUNNEL_TOKEN` (Cloudflare, only for a stable URL) and `HF_TOKEN` (Hugging Face,
only for rate limits or gated repos) are optional and documented in
`.env.example` with the exact page to get them from. `qwen21 doctor` lists
whatever is still a placeholder.

## Layout

```
qwen21/          installer, profile table, process supervisor, tunnel manager
qwen_api/        the FastAPI service (separate venv, talks HTTP to ComfyUI)
ComfyUI/         git submodule, pinned to v0.37.0
workflows/       UI workflows for ComfyUI + API-format graphs for the service
api/             API service requirements
scripts/         regenerates workflows/ from the official templates
docs/            API.md, DEPLOY.md, HARDWARE.md
runtime/         venvs, logs, pids, downloaded tools, job database (gitignored)
tests/           stdlib unit tests: python -m unittest discover -s tests
```

The service is a separate process talking to ComfyUI over its documented HTTP
API. That is what makes one codebase work on a Windows portable install (with
its embedded Python), a macOS venv, and a remote GPU box, without any of them
having to import the other's dependencies.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `doctor` says ComfyUI is too old | the submodule is pinned; `git -C ComfyUI fetch --depth 1 origin tag v0.37.0 && git -C ComfyUI checkout v0.37.0` |
| `Missing: gguf` at load time | `qwen21 install` installs the node's `requirements.txt`; if you added the node by hand, `runtime/venvs/comfy/bin/pip install gguf` |
| `524` from Cloudflare | expected for long jobs: use `wait=false` and poll, or a named tunnel |
| OOM on NVIDIA | close GPU-heavy apps, drop `resolution`/`steps`, keep the encoder on `device=cpu` |
| Mac swapping / very slow | expected on 8GB; use `--profile mac-8gb-lite`, lower `steps`, and `QwenImage21Cache(device=off)` |
| `invalid API key` | `qwen21 keygen --write`, then `qwen21 stop && qwen21 start` (the service reads `.env` at start) |

## Credits and licence

* **Qwen-Image-2.1** — <https://github.com/QwenLM/Qwen-Image> (Apache-2.0).
  This repository only handles local plumbing; the model belongs to the Qwen team.
* **ComfyUI** — <https://github.com/Comfy-Org/ComfyUI> (GPL-3.0) as a submodule.
* **ComfyUI-GGUF** — <https://github.com/city96/ComfyUI-GGUF> (Apache-2.0).
* GGUF weights — <https://huggingface.co/leejet/Qwen-Image-2.1-GGUF>;
  safetensors — <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>.

Helper code in this repository is MIT licensed ([LICENSE](LICENSE)). Model weights
are **not** redistributed here and stay under their upstream licences; review
them before serving generated images to anyone.

[简体中文](README_zh-CN.md)
