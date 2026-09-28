"""Hardware profiles: which models, which torch build, which ComfyUI flags.

A profile is a complete, self-consistent answer to "what do I download and how
do I launch ComfyUI on this machine". Everything else in the package reads it.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

HF = "https://huggingface.co"

# Verified against the upstream repositories (2026-09):
#   Comfy-Org/Qwen-Image-2.1   -> diffusion_models/, text_encoders/, vae/
#   leejet/Qwen-Image-2.1-GGUF -> qwen_image_2.1-Q{2,3,4,6,8}_*.gguf
GGUF_REPO = f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main"
QWEN_REPO = f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main"

# ComfyUI v0.37.0 is the first stable release carrying TextEncodeQwenImage21
# and QwenImage21Cache. Older checkouts cannot run the bundled workflows.
MIN_COMFY_VERSION = (0, 37, 0)
RECOMMENDED_COMFY_REF = "v0.37.0"

GGUF_NODE_REPO = "https://github.com/city96/ComfyUI-GGUF.git"


@dataclass(frozen=True)
class ModelFile:
    folder: str
    name: str
    url: str
    size: int  # bytes, used for pre-flight disk checks and doctor output
    sha256: str | None = None
    note: str = ""

    @property
    def rel(self) -> str:
        return f"{self.folder}/{self.name}"


def _gguf(quant: str) -> ModelFile:
    sizes = {"Q2_K": 2_560_000_000, "Q3_K": 3_270_000_000, "Q4_K": 4_200_000_000}
    return ModelFile(
        folder="diffusion_models",
        name=f"qwen_image_2.1-{quant}.gguf",
        url=f"{GGUF_REPO}/qwen_image_2.1-{quant}.gguf",
        size=sizes[quant],
        note="GGUF diffusion transformer, dequantized on the fly by ComfyUI-GGUF",
    )


VAE = ModelFile(
    folder="vae",
    name="qwen_image_2.1_vae_bf16.safetensors",
    url=f"{QWEN_REPO}/vae/qwen_image_2.1_vae_bf16.safetensors",
    size=683_000_000,
    note="2.1-specific VAE (4 channels -> real alpha). Not interchangeable with Qwen-Image 1.x.",
)

CLIP_W4A8 = ModelFile(
    folder="text_encoders",
    name="qwen3vl_8b_w4a8.safetensors",
    url=f"{QWEN_REPO}/text_encoders/qwen3vl_8b_w4a8.safetensors",
    size=6_310_000_000,
    note="4-bit weight-only Qwen3-VL encoder. Smallest official encoder, the only one that "
    "can be loaded whole on 8GB.",
)

CLIP_INT8 = ModelFile(
    folder="text_encoders",
    name="qwen3vl_8b_int8_convrot.safetensors",
    url=f"{QWEN_REPO}/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    size=9_350_000_000,
    note="Official template encoder. Higher quality, needs CPU offload or >=10GB to stay resident.",
)

UNET_INT8 = ModelFile(
    folder="diffusion_models",
    name="qwen_image_2.1_int8_convrot.safetensors",
    url=f"{QWEN_REPO}/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    size=7_260_000_000,
    note="Official template diffusion model (int8 convrot). Needs a CUDA/ROCm GPU with comfy_kitchen.",
)


@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    loader: str  # "gguf" | "unet"
    needs_gguf_node: bool
    comfy_args: tuple[str, ...]
    comfy_env: dict[str, str] = field(default_factory=dict)
    models: tuple[ModelFile, ...] = ()
    resolution: int = 1024
    steps: int = 25
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    encoder_device: str = "cpu"  # CLIPLoader device widget: "default" | "cpu"
    cache_device: str = "auto"  # QwenImage21Cache device: auto|gpu|cpu|off
    cache_dtype: str = "default"  # QwenImage21Cache dtype: default|int8|int4
    torch_index_url: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(m.size for m in self.models)

    def model(self, folder: str) -> ModelFile | None:
        return next((m for m in self.models if m.folder == folder), None)


PROFILES: dict[str, Profile] = {
    "nvidia-8gb": Profile(
        name="nvidia-8gb",
        summary="NVIDIA 6-8GB VRAM (RTX 4060 8GB, 2080 SUPER, 3060)",
        loader="gguf",
        needs_gguf_node=True,
        comfy_args=("--preview-method", "none"),
        models=(_gguf("Q4_K"), CLIP_W4A8, VAE),
        resolution=1024,
        steps=25,
        encoder_device="cpu",
        cache_device="auto",
        warnings=(
            "The text encoder is pinned to device=cpu. 8GB cannot hold the encoder and the "
            "diffusion transformer at the same time; swap cost is paid once per job.",
            "Close GPU-heavy apps (browsers with hardware acceleration, games) before starting.",
        ),
    ),
    "mac-8gb": Profile(
        name="mac-8gb",
        summary="Apple Silicon with 8GB unified memory (MacBook Air M3)",
        loader="gguf",
        needs_gguf_node=True,
        comfy_args=("--preview-method", "none", "--force-fp16"),
        comfy_env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        models=(_gguf("Q3_K"), CLIP_W4A8, VAE),
        resolution=768,
        steps=20,
        encoder_device="cpu",
        cache_device="cpu",
        warnings=(
            "8GB unified memory is the hard limit: encoder (6.3GB) + transformer (3.3GB) do not "
            "both fit, so macOS will swap. Expect minutes per image, not seconds.",
            "MPS has no native int4 kernel for the w4a8 encoder; PYTORCH_ENABLE_MPS_FALLBACK=1 "
            "and device=cpu are set. If loading still fails, run doctor --profile mac-8gb-lite.",
            "The int8_convrot path needs a third-party MPS backend and more RAM than this machine "
            "has. It is deliberately not part of this profile.",
        ),
    ),
    "mac-8gb-lite": Profile(
        name="mac-8gb-lite",
        summary="Apple Silicon 8GB, smallest possible footprint",
        loader="gguf",
        needs_gguf_node=True,
        comfy_args=("--preview-method", "none", "--force-fp16"),
        comfy_env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        models=(_gguf("Q2_K"), CLIP_W4A8, VAE),
        resolution=640,
        steps=16,
        encoder_device="cpu",
        cache_device="off",
        warnings=(
            "Q2_K visibly degrades text rendering and fine detail. Use mac-8gb unless the "
            "machine is already swapping to death.",
        ),
    ),
    "full": Profile(
        name="full",
        summary="12GB+ VRAM or 16GB+ unified memory - official safetensors path",
        loader="unet",
        needs_gguf_node=False,
        comfy_args=("--preview-method", "none"),
        models=(UNET_INT8, CLIP_INT8, VAE),
        resolution=1328,
        steps=25,
        encoder_device="default",
        cache_device="auto",
    ),
}


def profile_names() -> list[str]:
    return list(PROFILES)


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"Unknown profile {name!r}. Choose one of: {', '.join(PROFILES)}"
        ) from None


# --------------------------------------------------------------------------- #
# hardware detection
# --------------------------------------------------------------------------- #


def parse_nvidia_smi(output: str) -> tuple[str, int] | None:
    line = next((x.strip() for x in output.splitlines() if x.strip()), "")
    if not line:
        return None
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def gpu_info() -> tuple[str, int] | None:
    """(name, total VRAM in MiB) for NVIDIA, else None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_nvidia_smi(result.stdout)


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def mac_total_memory_gib() -> float:
    """Total unified memory in GiB, or 0.0 when it cannot be determined."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True
        )
        return int(out.stdout.strip()) / (1024**3)
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def system_memory_gib() -> float:
    if apple_silicon():
        return mac_total_memory_gib()
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def detect_profile() -> tuple[str, list[str]]:
    """Pick a profile for this machine. Returns (name, reasons)."""
    reasons: list[str] = []

    info = gpu_info()
    if info:
        name, mib = info
        gib = mib / 1024
        reasons.append(f"NVIDIA GPU: {name} ({gib:.1f} GiB VRAM)")
        if gib >= 11:
            return "full", reasons
        if gib >= 5.5:
            return "nvidia-8gb", reasons
        reasons.append(f"{gib:.1f} GiB is below the 5.5 GiB floor for the 8GB profile")
        return "nvidia-8gb", reasons

    if apple_silicon():
        mem = mac_total_memory_gib()
        reasons.append(f"Apple Silicon, {mem:.0f} GiB unified memory" if mem else "Apple Silicon")
        if mem and mem >= 15:
            return "full", reasons
        return "mac-8gb", reasons

    reasons.append("No NVIDIA GPU and no Apple Silicon detected")
    return "nvidia-8gb", reasons
