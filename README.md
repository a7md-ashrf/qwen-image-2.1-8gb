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

[73 more lines in file. Use offset=61 to continue.]