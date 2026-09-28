"""API-format workflow builders.

ComfyUI's `/prompt` endpoint wants a flat JSON graph keyed by node id, which is
a different shape from the UI workflows in `workflows/`. Building the graph in
code (instead of shipping hand-wired link JSON) is what lets one codebase serve
the GGUF profile and the official safetensors profile, and lets every knob be
overridable per request.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# Fixed node ids: the graph is tiny, and stable ids make ComfyUI's validation
# errors ("node 40: ...") readable. LoadImage nodes get 100+.
UNET = "1"
CLIP = "2"
VAE = "3"
CACHE = "4"
LATENT = "10"
ENCODE = "30"
SAMPLER = "40"
DECODE = "50"
SAVE = "60"

IMAGE_SLOTS = 16  # TextEncodeQwenImage21 exposes image_1 .. image_16


def round_to(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _title(node: dict[str, Any], title: str) -> dict[str, Any]:
    node["_meta"] = {"title": title}
    return node


@dataclass(frozen=True)
class ModelSpec:
    unet: str
    clip: str
    vae: str
    unet_gguf: bool = False
    encoder_device: str = "cpu"
    cache_device: str = "auto"
    cache_dtype: str = "default"


def _loaders(spec: ModelSpec) -> dict[str, dict[str, Any]]:
    loader = "UnetLoaderGGUF" if spec.unet_gguf else "UNETLoader"
    unet_inputs: dict[str, Any] = {"unet_name": spec.unet}
    if not spec.unet_gguf:
        unet_inputs["weight_dtype"] = "default"
    return {
        UNET: _title({"class_type": loader, "inputs": unet_inputs}, "Load diffusion model"),
        CLIP: _title(
            {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": spec.clip,
                    "type": "qwen_image",
                    "device": spec.encoder_device,
                },
            },
            "Load text encoder",
        ),
        VAE: _title(
            {"class_type": "VAELoader", "inputs": {"vae_name": spec.vae}}, "Load VAE"
        ),
        CACHE: _title(
            {
                "class_type": "QwenImage21Cache",
                "inputs": {
                    "model": [UNET, 0],
                    "device": spec.cache_device,
                    "dtype": spec.cache_dtype,
                },
            },
            "Qwen 2.1 KV cache",
        ),
    }


def _encode(
    prompt: str, negative_prompt: str, resolution: int, images: dict[str, Any]
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "clip": [CLIP, 0],
        "vae": [VAE, 0],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": int(resolution),
    }
    if images:
        inputs["images"] = images
    return _title(
        {"class_type": "TextEncodeQwenImage21", "inputs": inputs}, "Text Encode Qwen Image 2.1"
    )


def _tail(seed: int, steps: int, cfg: float, sampler: str, scheduler: str, prefix: str,
          latent: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        SAMPLER: _title(
            {
                "class_type": "KSampler",
                "inputs": {
                    "model": [CACHE, 0],
                    "seed": int(seed),
                    "steps": int(steps),
                    "cfg": float(cfg),
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "positive": [ENCODE, 0],
                    "negative": [ENCODE, 1],
                    "latent_image": latent,
                    "denoise": 1.0,
                },
            },
            "KSampler",
        ),
        DECODE: _title(
            {"class_type": "VAEDecode", "inputs": {"samples": [SAMPLER, 0], "vae": [VAE, 0]}},
            "VAE Decode",
        ),
        SAVE: _title(
            {
                "class_type": "SaveImage",
                "inputs": {"images": [DECODE, 0], "filename_prefix": prefix},
            },
            "Save Image",
        ),
    }


def build_edit_prompt(
    spec: ModelSpec,
    prompt: str,
    image_names: list[str],
    negative_prompt: str = "",
    resolution: int = 1024,
    seed: int | None = None,
    steps: int = 25,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    prefix: str = "qwen21/api",
) -> dict[str, dict[str, Any]]:
    """Image + instruction -> edited image. The node output follows the first
    reference image's aspect ratio, scaled to about `resolution` x `resolution`."""
    if not image_names:
        raise ValueError("edit requires at least one image")
    if len(image_names) > IMAGE_SLOTS:
        raise ValueError(f"at most {IMAGE_SLOTS} reference images are supported")

    graph = _loaders(spec)
    images: dict[str, Any] = {}
    for index, name in enumerate(image_names):
        node_id = str(100 + index)  # keep LoadImage ids clear of the fixed ones
        graph[node_id] = _title(
            {"class_type": "LoadImage", "inputs": {"image": name}}, f"Load image {index + 1}"
        )
        images[f"image_{index + 1}"] = [node_id, 0]

    graph[ENCODE] = _encode(prompt, negative_prompt, resolution, images)
    graph.update(
        _tail(
            seed if seed is not None else random.randrange(2**31),
            steps,
            cfg,
            sampler,
            scheduler,
            prefix,
            [ENCODE, 2],
        )
    )
    return graph


def build_generate_prompt(
    spec: ModelSpec,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    steps: int = 25,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    prefix: str = "qwen21/api",
) -> dict[str, dict[str, Any]]:
    """Prompt -> image. The output size is whatever width/height say."""
    graph = _loaders(spec)
    graph[LATENT] = _title(
        {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": round_to(width),
                "height": round_to(height),
                "batch_size": 1,
            },
        },
        "Empty Latent Image",
    )
    graph[ENCODE] = _encode(prompt, negative_prompt, max(width, height), {})
    graph.update(
        _tail(
            seed if seed is not None else random.randrange(2**31),
            steps,
            cfg,
            sampler,
            scheduler,
            prefix,
            [LATENT, 0],
        )
    )
    return graph


def describe(spec: ModelSpec) -> dict[str, Any]:
    return {
        "unet": spec.unet,
        "unet_loader": "UnetLoaderGGUF" if spec.unet_gguf else "UNETLoader",
        "clip": spec.clip,
        "vae": spec.vae,
        "encoder_device": spec.encoder_device,
        "cache": {"device": spec.cache_device, "dtype": spec.cache_dtype},
    }
