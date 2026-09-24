# Project Memory: qwen-image-2.1-8gb → standalone image-edit API

## Goal
Serve Qwen-Image-2.1 for **image-to-image editing** (`image + text prompt → edited image`) over HTTP, **skipping the ComfyUI workflow/UI**, on an **NVIDIA 8GB GPU (RTX 2080 SUPER)**. The network API receives an image file + a prompt string and returns a modified PNG.

## What the repo originally was
`qwen-image-2.1-8gb` is a **ComfyUI plumbing helper only** (`qwen21.py`): downloads a Q4 GGUF DiT + quantized Qwen3-VL text encoder + VAE, installs `leejet/ComfyUI-GGUF`, copies JSON workflows, launches ComfyUI with `--lowvram`. It contains **no inference code**. The 8GB optimization lives almost entirely in two files:
- `qwen_image_2.1-Q4_K.gguf` (leejet, ~5GB) — the DiT
- `qwen3vl_8b_w4a8.safetensors` (Comfy-Org W4A8) — the text encoder
- `qwen_image_2.1_vae_bf16.safetensors` — unquantized VAE (same as official HF)

## Pivot decided
Use **only** the downloaded files' intent (GGUF DiT + official VAE) and run via **diffusers `QwenImage21Pipeline`** behind a **FastAPI** HTTP service, no ComfyUI.

## Model stack (as built)
- **DiT:** `qwen_image_2.1-Q4_K.gguf` loaded via `QwenImage21Transformer2DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(compute_dtype=dtype))`, auto-downloaded to `models/` (reuses `qwen21.PROFILE_8GB` / `qwen21.download`)
- **Text encoder:** official `Qwen/Qwen-Image-2.1` text_encoder via `AutoModelForImageTextToText.from_pretrained(..., BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))`
- **VAE / scheduler:** official HF repo (identical weights to the Comfy-Org VAE file)
- **Offload:** `pipe.enable_model_cpu_offload()`; `threading.RLock` serializes requests

## Constraints / conflicts resolved
1. **W4A8 TE cannot leave ComfyUI** — `qwen3vl_8b_w4a8.safetensors` is a ComfyUI-only quant format; replaced with bnb-4bit official TE (minor quality diff, unavoidable).
2. **GGUF load outside ComfyUI is unverified for 2.1** (release is days old) — two-stage try/catch: plain `from_single_file`, then with `config=MODEL_ID, subfolder="transformer"`; clear RuntimeError with `--bf16-transformer` fallback (Comfy-Org bf16, ~14GB, tighter on 8GB).
3. **VAE has no conflict** — same weights, official diffusers-format copy.
4. **Turing (sm7.5) has no BF16 compute** — dtype auto-chooses `bf16` only on Ampere+ (`capability[0] >= 8`), else `fp16`; expose `--dtype {auto,bf16,fp16}`.
5. **No CUDA → error at load time** (CUDA required).
6. **~32GB system RAM recommended** (matches repo README) for CPU offload of the ~21GB bf16 components.

## Architecture decisions
- `api.py`: single file, FastAPI app created at module top; heavy imports (`torch`, `diffusers`, `transformers`) inside `load_pipeline()` so the module imports without torch.
- `dtype_name="auto"` stored in `_config`; **resolved at load time** in `load_pipeline()` (not CLI parse), so `--no-warmup` boots without torch.
- `run_edit(...)` builds `torch.Generator("cpu").manual_seed(seed)`; seed returned in `X-Seed` header.
- Defaults mirror the Comfy workflow: `steps=25`, `guidance_scale=1.0` (no CFG), `euler/simple` scheduler, `width=height=1024`, dimensions snapped to multiple of 16, 16–4096 range, prompt ≤10000 chars, image converted RGB (RGBA preserved by pipeline).
- Health endpoint reports `model_loaded` without triggering a load.

## API contract
- `POST /edit` — multipart `image` (file) + form fields `prompt` (required), `negative_prompt`, `steps`, `guidance_scale`, `seed`, `width`, `height`. Returns `image/png` + `X-Seed`, `X-Width`, `X-Height`.
- `GET /health` → `{status, model_loaded, model_id, dtype_name}`.
- Run: `python api.py --host 0.0.0.0 --port 8000` (auto-downloads missing GGUF on first boot; `--no-download`/`--no-warmup`/`--bf16-transformer`/`--dtype`/`--gguf PATH`/`--model-id`/`--port`/`--host` supported).

## Files
- `api.py` — service (+ `#!/usr/bin/env python3`, executable)
- `requirements-api.txt` — `fastapi, uvicorn[standard], python-multipart, pillow, httpx, torch>=2.4.0, transformers>=5.17, diffusers@git+https://github.com/huggingface/diffusers, accelerate, bitsandbytes, gguf`
- `tests/test_api.py` — 12 tests; skip-when-deps-missing; validates helpers + endpoint behavior (mocked pipeline)
- `.gitignore` — added `models/`
- Pre-existing untouched: `qwen21.py`, `workflows/`, `tests/test_qwen21.py`, `README.md`, install scripts
- **No lint/typecheck config exists** in the repo (no ruff/mypy/flake8 config); ruff default flags pre-existing `qwen21.py` findings too; remaining warnings in new code are FastAPI idioms (`File(...)/Form(...)` defaults) + a deliberate catch-all fallback — acceptable.

## Testing status
17 tests pass (`python -m unittest discover -s tests`); boot smoke test confirmed uvicorn starts with `--no-warmup --no-download` and `/health` → 200, `/edit` validation → 422.

## Known risks (not yet validated on target hardware)
- GGUF tensor-name mapping into `QwenImage21Transformer2DModel.from_single_file` for the 2.1 release — verified at first warmup on the GPU box; fallback flag ready.
- bnb-4bit TE + `enable_model_cpu_offload` compatibility — surfaces at warmup; if it misbehaves, report the traceback.
- fp16 on Turing can NaN for some Qwen-Image checkpoints — `--dtype bf16` requires Ampere+; on Turing fp16 is the default and matches how ComfyUI runs this stack.
- Python 3.14 on this dev box has no torch; target runtime is the NVIDIA Linux/Windows box (install `torch` CUDA build from https://pytorch.org separately; diffusers pinned to `git+` for day-0 2.1 support).
