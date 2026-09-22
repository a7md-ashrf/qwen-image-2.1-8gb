# Qwen-Image-2.1 on 8GB VRAM

[English](README.md) | [简体中文](README_zh-CN.md)

**Run Qwen-Image-2.1 on an 8GB NVIDIA GPU with ComfyUI + GGUF.**

A tiny, no-framework helper that installs the low-VRAM model stack, checks your GPU/files, drops in ready-to-import workflows, and launches ComfyUI with `--lowvram`.

> Tested as a development profile on an **RTX 2080 SUPER 8GB** with Q4-class GGUF diffusion weights, the W4A8 Qwen3-VL text encoder, and the dedicated Qwen-Image-2.1 VAE.

## Why this repo exists

Qwen-Image-2.1 is unusually capable for a new open-weight image model, but the default files are still large enough to make an 8GB card look impossible at first glance. The trick is **quantization + CPU offload**, not pretending the whole stack fits in VRAM at once.

This repo turns that setup into three commands.

## 30-second setup

You need:

- Windows or Linux
- NVIDIA GPU (8GB target profile)
- a recent **ComfyUI** checkout / portable install
- Git
- enough disk space for the model files (the downloads are much larger than 8GB in total)

### Windows

```powershell
git clone https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb

.\install.ps1 -ComfyUI "D:\ComfyUI_windows_portable\ComfyUI"
```

### Linux

```bash
git clone https://github.com/toyhank/qwen-image-2.1-8gb
cd qwen-image-2.1-8gb

./install.sh /path/to/ComfyUI
```

The installer:

1. installs/updates the maintained `leejet/ComfyUI-GGUF` node;
2. downloads a Q4-class Qwen-Image-2.1 GGUF diffusion model;
3. downloads `qwen3vl_8b_w4a8.safetensors`;
4. downloads the **Qwen-Image-2.1-specific VAE**;
5. copies the bundled text-to-image and image-edit workflows into ComfyUI.

## Check before downloading

```bash
python qwen21.py doctor --comfy /path/to/ComfyUI
```

Example output:

```text
Qwen-Image-2.1 8GB doctor
=================================
GPU: NVIDIA GeForce RTX 2080 SUPER — 8.0 GiB VRAM
OK: ComfyUI-GGUF
OK: compatible GGUF found: qwen_image_2.1_Q4_K_M.gguf
OK: qwen3vl_8b_w4a8.safetensors
OK: qwen_image_2.1_vae_bf16.safetensors

Ready. Import one of the workflows/ JSON files in ComfyUI.
```

## Launch low-VRAM mode

```bash
python qwen21.py launch --comfy /path/to/ComfyUI
```

Anything after the command is passed to ComfyUI:

```bash
python qwen21.py launch --comfy /path/to/ComfyUI -- --listen 0.0.0.0
```

## What gets downloaded

| Component | File | Why |
|---|---|---|
| Diffusion model | `qwen_image_2.1-Q4_K.gguf` | Q4-class GGUF cuts GPU memory pressure |
| Text encoder | `qwen3vl_8b_w4a8.safetensors` | lower-memory Qwen3-VL encoder |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | required 2.1 VAE; old Qwen Image VAE is not interchangeable |
| Custom node | `leejet/ComfyUI-GGUF` | GGUF loader with Qwen-Image-2.1 support |

## Workflows

- `workflows/qwen-image-2.1-8gb-t2i.json` — text to image
- `workflows/qwen-image-2.1-8gb-edit.json` — image editing

Drag a JSON file into ComfyUI, or run:

```bash
python qwen21.py workflows --comfy /path/to/ComfyUI
```

## 8GB expectations

**8GB is a target VRAM profile, not a claim that all weights total 8GB.** ComfyUI may offload the text encoder / model blocks to system RAM. For a good experience, 32GB system RAM is a practical target.

Start with 1024×1024 and a single image. If you hit OOM:

- close browsers/games using VRAM;
- keep `--lowvram` enabled;
- reduce resolution;
- restart ComfyUI after changing models;
- avoid loading unrelated checkpoints at the same time.

## Why Qwen-Image-2.1 is worth trying

The upstream model unifies text-to-image and editing, supports multiple reference images, improves text rendering, and adds native transparency. This repo only handles the local low-VRAM plumbing; **Qwen-Image-2.1 itself belongs to the Qwen team**.

## Upstream / credits

- Qwen-Image-2.1: https://github.com/QwenLM/Qwen-Image-2.1
- ComfyUI model files: https://huggingface.co/Comfy-Org/Qwen-Image-2.1
- GGUF weights: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF
- ComfyUI-GGUF: https://github.com/leejet/ComfyUI-GGUF

## License

The helper code in this repository is MIT licensed. **Model weights are not redistributed** and remain under their upstream licenses. Review the Qwen / model repository license terms before use.

---

If this saved you an evening of CUDA / VRAM debugging, a ⭐ helps other 8GB GPU owners find it.
