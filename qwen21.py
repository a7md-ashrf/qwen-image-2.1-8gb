#!/usr/bin/env python3
"""Low-VRAM helper for Qwen-Image-2.1 + ComfyUI.

This project does not redistribute model weights. It downloads files from
their upstream repositories and keeps their original licenses.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


GGUF_NODE_REPO = "https://github.com/leejet/ComfyUI-GGUF.git"

PROFILE_8GB = (
    (
        "diffusion_models",
        "qwen_image_2.1-Q4_K.gguf",
        "https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q4_K.gguf",
    ),
    (
        "text_encoders",
        "qwen3vl_8b_w4a8.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors",
    ),
    (
        "vae",
        "qwen_image_2.1_vae_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
    ),
)


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


def normalize_comfy(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "main.py").is_file():
        return p
    if (p / "ComfyUI" / "main.py").is_file():
        return p / "ComfyUI"
    raise SystemExit(f"ComfyUI not found at: {p}")


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "qwen-image-2.1-8gb/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    print(f"↓ {target.name}" + (f" (resume at {human_bytes(offset)})" if offset else ""))

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists():
            partial.replace(target)
            print(f"✓ {target}")
            return
        raise

    status = getattr(response, "status", None)
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0

    remaining = response.headers.get("Content-Length")
    total = offset + int(remaining) if remaining and remaining.isdigit() else None
    done = offset

    with response, partial.open(mode) as fh:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = min(100.0, done * 100.0 / total)
                print(
                    f"  {human_bytes(done)} / {human_bytes(total)} ({pct:5.1f}%)",
                    end="\r",
                )
    if total:
        print()
    partial.replace(target)
    print(f"✓ {target}")


def install_gguf_node(comfy: Path) -> None:
    target = comfy / "custom_nodes" / "ComfyUI-GGUF"
    target.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if not git:
        raise SystemExit("git is required to install ComfyUI-GGUF.")
    if (target / ".git").is_dir():
        print("↻ Updating ComfyUI-GGUF")
        subprocess.run([git, "-C", str(target), "pull", "--ff-only"], check=True)
    elif target.exists():
        raise SystemExit(
            f"{target} exists but is not a git checkout. Move/remove it and retry."
        )
    else:
        print("↓ Installing ComfyUI-GGUF")
        subprocess.run(
            [git, "clone", "--depth", "1", GGUF_NODE_REPO, str(target)],
            check=True,
        )


def install_models(comfy: Path, force: bool = False) -> None:
    model_root = comfy / "models"
    for folder, name, url in PROFILE_8GB:
        target = model_root / folder / name
        if target.exists() and target.stat().st_size > 1024 * 1024 and not force:
            print(f"✓ {name} already exists ({human_bytes(target.stat().st_size)})")
            continue
        download(url, target)


def install_workflows(comfy: Path) -> None:
    source = Path(__file__).resolve().parent / "workflows"
    target = comfy / "user" / "default" / "workflows"
    target.mkdir(parents=True, exist_ok=True)
    for src in source.glob("*.json"):
        dst = target / src.name
        shutil.copy2(src, dst)
        print(f"✓ workflow: {dst.name}")


def doctor(comfy: Path) -> int:
    problems = 0
    print("Qwen-Image-2.1 8GB doctor")
    print("=" * 33)

    info = gpu_info()
    if info:
        name, mib = info
        gib = mib / 1024
        print(f"GPU: {name} — {gib:.1f} GiB VRAM")
        if gib < 7.5:
            print("WARN: below the target 8GB profile; expect heavier CPU offload.")
    else:
        print("WARN: NVIDIA GPU not detected through nvidia-smi.")

    print(f"ComfyUI: {comfy}")
    node = comfy / "custom_nodes" / "ComfyUI-GGUF"
    if node.is_dir():
        print("OK: ComfyUI-GGUF")
    else:
        print("MISSING: ComfyUI-GGUF")
        problems += 1

    for folder, name, _ in PROFILE_8GB:
        path = comfy / "models" / folder / name
        if path.is_file():
            print(f"OK: {name} ({human_bytes(path.stat().st_size)})")
        else:
            alt = None
            if name == "qwen_image_2.1-Q4_K.gguf":
                candidates = list(
                    (comfy / "models" / folder).glob("*qwen*2.1*Q4*K*.gguf")
                )
                alt = candidates[0] if candidates else None
            if alt:
                print(
                    f"OK: compatible GGUF found: {alt.name} "
                    f"({human_bytes(alt.stat().st_size)})"
                )
            else:
                print(f"MISSING: {path}")
                problems += 1

    if problems:
        print(f"\n{problems} required component(s) missing. Run the install command.")
        return 1
    print("\nReady. Import one of the workflows/ JSON files in ComfyUI.")
    return 0


def find_python(comfy: Path) -> tuple[Path, bool]:
    portable = comfy.parent / "python_embeded" / "python.exe"
    if portable.is_file():
        return portable, True
    return Path(sys.executable), False


def launch(comfy: Path, extra: list[str]) -> int:
    python, portable = find_python(comfy)
    cmd = [str(python), "-s", str(comfy / "main.py"), "--lowvram"]
    if portable:
        cmd.append("--windows-standalone-build")
    cmd.extend(extra)
    print("Launching:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(comfy))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen21",
        description="Install and validate a low-VRAM Qwen-Image-2.1 ComfyUI setup.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Check GPU, node, and model files.")
    p_doctor.add_argument("--comfy", required=True, help="Path to ComfyUI or its parent.")

    p_install = sub.add_parser(
        "install",
        help="Install GGUF node, models, and workflows.",
    )
    p_install.add_argument("--comfy", required=True, help="Path to ComfyUI or its parent.")
    p_install.add_argument("--force", action="store_true", help="Redownload model files.")

    p_workflow = sub.add_parser(
        "workflows",
        help="Copy bundled workflows into ComfyUI.",
    )
    p_workflow.add_argument("--comfy", required=True)

    p_launch = sub.add_parser("launch", help="Launch ComfyUI with --lowvram.")
    p_launch.add_argument("--comfy", required=True)
    p_launch.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra ComfyUI arguments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comfy = normalize_comfy(args.comfy)

    if args.command == "doctor":
        return doctor(comfy)
    if args.command == "install":
        install_gguf_node(comfy)
        install_models(comfy, force=args.force)
        install_workflows(comfy)
        return doctor(comfy)
    if args.command == "workflows":
        install_workflows(comfy)
        return 0
    if args.command == "launch":
        return launch(comfy, args.extra)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
