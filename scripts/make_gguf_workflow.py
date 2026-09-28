#!/usr/bin/env python3
"""Build the bundled UI workflows from the official ComfyUI templates.

The upstream templates (`Comfy-Org/workflow_templates`) target the int8_convrot
safetensors stack, which needs ~17GB of RAM/VRAM across the encoder and the
transformer. The 8GB profiles here use a GGUF diffusion model and the 4-bit
encoder instead, so this script rewrites the loader nodes and turns the optional
prompt-enhancement branch off.

Run it when you want to refresh the bundled workflows:

    python scripts/make_gguf_workflow.py

It writes:
    workflows/qwen-image-2.1-t2i-8gb.json
    workflows/qwen-image-2.1-edit-8gb.json
    workflows/qwen-image-2.1-bg-removal-8gb.json
"""
from __future__ import annotations

import copy
import json
import sys
import urllib.request
from pathlib import Path

TEMPLATES = {
    "t2i": "image_qwen_image_2_1_t2i.json",
    "edit": "image_qwen_image_2_1_image_edit.json",
    "bg-removal": "image_qwen_image_2_1_background_removal.json",
}
BASE = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates"

UNET_FROM = "qwen_image_2.1_int8_convrot.safetensors"
UNET_TO = "qwen_image_2.1-Q4_K.gguf"
UNET_TO_MAC = "qwen_image_2.1-Q3_K.gguf"
CLIP_FROM = "qwen3vl_8b_int8_convrot.safetensors"
CLIP_TO = "qwen3vl_8b_w4a8.safetensors"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "workflows"


def fetch(name: str, cache: Path) -> dict:
    if not cache.is_file():
        url = f"{BASE}/{name}"
        print(f"  ↓ {name}")
        with urllib.request.urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    return json.loads(cache.read_text(encoding="utf-8"))


def all_nodes(workflow: dict) -> list[dict]:
    nodes = list(workflow.get("nodes", []))
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        nodes.extend(subgraph.get("nodes", []))
    return nodes


def patch(workflow: dict, unet: str) -> dict:
    """Swap the loaders over to the 8GB stack and disable prompt enhancement."""
    workflow = copy.deepcopy(workflow)
    for node in all_nodes(workflow):
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list) or not widgets:
            continue
        name = widgets[0] if isinstance(widgets[0], str) else ""
        if node.get("type") == "UNETLoader" and name == UNET_FROM:
            # UnetLoaderGGUF only takes unet_name; drop the weight_dtype widget.
            node["type"] = "UnetLoaderGGUF"
            node["widgets_values"] = [unet]
        elif node.get("type") == "CLIPLoader" and (
            name == CLIP_FROM or name.startswith("qwen3.5_9b_qwen_image_2.1_pe_")
        ):
            # The prompt-enhancer branch is switched off below, but its loader
            # still has to name a file that exists or ComfyUI refuses the graph.
            widgets[0] = CLIP_TO
    # The only boolean True on the subgraph instance is `refine_prompt`, whose
    # branch loads a 9.5GB prompt-enhancer text encoder. Off by default here.
    for node in workflow.get("nodes", []):
        widgets = node.get("widgets_values")
        if isinstance(widgets, list) and True in widgets:
            widgets[widgets.index(True)] = False
    return workflow


def main() -> int:
    cache_dir = ROOT / "runtime" / "templates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for key, template in TEMPLATES.items():
        workflow = patch(fetch(template, cache_dir / template), UNET_TO)
        target = OUT / f"qwen-image-2.1-{key}-8gb.json"
        target.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"  ✓ {target.relative_to(ROOT)}")
        # A second, smaller quantisation for 8GB unified memory machines.
        mac = OUT / f"qwen-image-2.1-{key}-8gb-mac.json"
        mac.write_text(
            json.dumps(patch(fetch(template, cache_dir / template), UNET_TO_MAC), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ {mac.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
