#!/usr/bin/env python3
"""Qwen-Image-2.1 image-edit API served outside ComfyUI.

Licensed under the MIT License (see LICENSE).

Uses the same 8GB-oriented Q4_K GGUF diffusion weights this repo downloads,
with a bitsandbytes 4-bit Qwen3-VL text encoder and CPU offload.

Install (on the CUDA machine):
  pip install -r requirements-api.txt
  # plus the CUDA build of torch from https://pytorch.org if needed

Run:
  python api.py --host 0.0.0.0 --port 8000

Call:
  curl -F "image=@input.png" -F "prompt=Change the background to a sunset" \
    http://HOST:8000/edit -o output.png
"""

from __future__ import annotations

import argparse
import io
import random
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image

from qwen21 import PROFILE_8GB, download

MODEL_ID = "Qwen/Qwen-Image-2.1"
BF16_TRANSFORMER_URL = (
    "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/"
    "diffusion_models/qwen_image_2.1_bf16.safetensors"
)
GGUF_NAME, GGUF_URL = next(
    (name, url) for _, name, url in PROFILE_8GB if name.endswith(".gguf")
)
DEFAULT_GGUF = Path(__file__).resolve().parent / "models" / GGUF_NAME

app = FastAPI(title="Qwen-Image-2.1 Edit API", version="1.0")

_lock = threading.RLock()
_pipe: Any = None
_config: dict[str, Any] = {}


def snap_dim(value: int) -> int:
    return max(16, (int(value) // 16) * 16)


def choose_dtype_name(capability: tuple[int, int] | None) -> str:
    if capability and capability[0] >= 8:
        return "bf16"
    return "fp16"


def resolve_gguf(path: Path, download_enabled: bool = True) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.stat().st_size > 1024 * 1024:
        return path
    if not download_enabled:
        raise FileNotFoundError(f"GGUF weights not found: {path}")
    download(GGUF_URL, path)
    return path


def load_pipeline(
    gguf: Path,
    model_id: str,
    dtype_name: str,
    bf16_transformer: bool,
) -> Any:
    import torch
    from diffusers import (
        GGUFQuantizationConfig,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required to run Qwen-Image-2.1 with this API."
        )

    capability = tuple(torch.cuda.get_device_capability())
    dtype_name = resolve_dtype_name(dtype_name, capability)
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    if bf16_transformer:
        transformer = QwenImage21Transformer2DModel.from_single_file(
            BF16_TRANSFORMER_URL,
            dtype=dtype,
        )
    else:
        try:
            transformer = QwenImage21Transformer2DModel.from_single_file(
                str(gguf),
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                dtype=dtype,
            )
        except Exception as first_error:
            try:
                transformer = QwenImage21Transformer2DModel.from_single_file(
                    str(gguf),
                    config=model_id,
                    subfolder="transformer",
                    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                    dtype=dtype,
                )
            except Exception as second_error:
                raise RuntimeError(
                    f"Failed to load GGUF transformer {gguf}: {second_error} "
                    f"(first attempt: {first_error}). "
                    "Retry with --bf16-transformer to use the Comfy-Org bf16 "
                    "checkpoint instead."
                ) from second_error

    text_encoder = AutoModelForImageTextToText.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
    )

    pipe = QwenImage21Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_pipeline() -> Any:
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                if not _config:
                    raise HTTPException(
                        status_code=503,
                        detail="Model is not configured. Start the service via api.py main().",
                    )
                try:
                    _pipe = load_pipeline(**_config)
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(
                        status_code=503, detail=f"Model failed to load: {exc}"
                    ) from exc
    return _pipe


def run_edit(
    pipe: Any,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    import torch

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": image,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
    }
    if negative_prompt.strip() and guidance_scale > 1.0:
        kwargs["negative_prompt"] = negative_prompt
    return pipe(**kwargs).images[0]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "model_id": _config.get("model_id"),
        "dtype": _config.get("dtype_name"),
    }


@app.post("/edit")
def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(25),
    guidance_scale: float = Form(1.0),
    seed: int | None = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
) -> Response:
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    if len(prompt) > 10_000:
        raise HTTPException(status_code=422, detail="prompt is too long")
    if not 1 <= steps <= 100:
        raise HTTPException(status_code=422, detail="steps must be between 1 and 100")
    if guidance_scale < 1.0:
        raise HTTPException(status_code=422, detail="guidance_scale must be >= 1.0")
    if width < 16 or height < 16 or width > 4096 or height > 4096:
        raise HTTPException(
            status_code=422, detail="width and height must be between 16 and 4096"
        )

    try:
        source = Image.open(image.file)
        source.load()
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="image file could not be decoded"
        ) from exc
    if source.mode not in ("RGB", "RGBA"):
        source = source.convert("RGB")

    out_width = snap_dim(width)
    out_height = snap_dim(height)
    used_seed = seed if seed is not None else random.randint(0, 2**63 - 1)

    with _lock:
        pipe = get_pipeline()
        result = run_edit(
            pipe,
            source,
            prompt.strip(),
            negative_prompt,
            steps,
            guidance_scale,
            out_width,
            out_height,
            used_seed,
        )

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-Seed": str(used_seed),
            "X-Width": str(result.width),
            "X-Height": str(result.height),
        },
    )


def resolve_dtype_name(choice: str, capability: tuple[int, int] | None) -> str:
    if choice != "auto":
        return choice
    return choose_dtype_name(capability)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen21-api",
        description="Serve Qwen-Image-2.1 image editing over HTTP without ComfyUI.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument(
        "--bf16-transformer",
        action="store_true",
        help="Use the Comfy-Org bf16 checkpoint instead of the Q4_K GGUF.",
    )
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading missing GGUF weights.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip loading the model before binding the port.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    gguf = args.gguf
    if not args.bf16_transformer:
        try:
            gguf = resolve_gguf(args.gguf, download_enabled=not args.no_download)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    _config.clear()
    _config.update(
        gguf=gguf,
        model_id=args.model_id,
        dtype_name=args.dtype,
        bf16_transformer=args.bf16_transformer,
    )

    if not args.no_warmup:
        print("Loading Qwen-Image-2.1 ...")
        get_pipeline()
        print("Ready.")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
