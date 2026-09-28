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
* **A request never blocks past the tunnel's patience.** Cloudflare's proxy read
  timeout is 125 s (`524`); a 25-step edit takes longer. So a call waits up to
  `SYNC_MAX_WAIT` (90 s) and otherwise returns `202` with a job id you can poll.
  One endpoint, no client-side retry logic.

## Keys

Only one is mandatory.

## Windows quick start

The installer handles all of this, but the manual path is here if you want it.

### Requirements

- Windows 10/11 (64-bit), NVIDIA GPU with a current driver
- [Git](https://git-scm.com/download/win) and [Python 3.11–3.12 (64-bit)](https://www.python.org/downloads/) —
  tick **"Add python.exe to PATH"** during install
- ~12 GB free disk for the `nvidia-8gb` profile (4.2 GB transformer + 6.3 GB
  encoder + 0.7 GB VAE + ~2 GB of Python environments)
- 16 GB system RAM is comfortable; 8 GB works but pages hard

> No CUDA toolkit install needed: the CUDA runtime ships inside the PyTorch wheels.

### 1. Clone and install

```powershell
git clone --recurse-submodules https://github.com/a7md-ashrf/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb
.\install.ps1
```

`install.ps1` finds a suitable Python, initialises the ComfyUI submodule, builds
the virtualenvs, installs the GGUF loader, downloads the models and writes
`.env`. If PowerShell refuses to run the script:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

### 2. Generate the API key and start

```powershell
qwen21 keygen --write
qwen21 start
qwen21 status
```

`start` waits for ComfyUI to answer and then for the API to answer, so the URL it
prints is already live.

### 3. Call it

Use `curl.exe`, not `curl` — PowerShell aliases `curl` to `Invoke-WebRequest`,
which does not accept `-F`:

```powershell
$env:KEY = (Select-String -Path .env -Pattern '^API_KEY=(.*)$').Matches[0].Groups[1].Value
curl.exe -H "Authorization: Bearer $env:KEY" http://127.0.0.1:8000/v1/models
curl.exe -H "Authorization: Bearer $env:KEY" -F "image=@input.png" -F "prompt=Change the background to a sunset" http://127.0.0.1:8000/v1/edit -o response.json
```

### 4. Firewall

Nothing to do: `API_HOST` and `COMFY_HOST` both default to `127.0.0.1`, so
Windows will not prompt and nothing is reachable from the network. The tunnel
publishes the API over outbound connections only. If you deliberately change
`API_HOST` to `0.0.0.0`, pre-approve the port:

```powershell
netsh advfirewall firewall add rule name="Qwen Image API" dir=in action=allow protocol=TCP localport=8000
```

### Troubleshooting

- **`doctor` says "ComfyUI does not know node …"** — the checkout is older than
  v0.37.0; the submodule is pinned, so this only happens on a stale clone.
- **Out of memory on first job** — close GPU-heavy apps, lower `resolution` to
  768 or `steps` to 15, keep the encoder on `device=cpu` (the profile already
  does), and `qwen21 stop && qwen21 start` so old weights leave VRAM.
- **`CUDA is not available` / torch picked a CPU build** — check `nvidia-smi`
  works in the same terminal, then reinstall torch from
  <https://pytorch.org/get-started/locally/> choosing the CUDA build.
- **A job sits at `queued`** — `MAX_CONCURRENT` is 1 and ComfyUI runs one job at
  a time; poll `qwen21 status` for the queue.
