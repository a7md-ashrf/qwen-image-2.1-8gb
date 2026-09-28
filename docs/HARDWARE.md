# Hardware notes

Two target machines, and an honest account of what each can do. **No timing
numbers are quoted here yet**, because none have been measured: the models are
not installed by default (`qwen21 install --skip-models` leaves them out, and
the repo ships no weights). Run `qwen21 smoke` on your own machine to get real
seconds-per-image, then paste those numbers back in here or into your issue —
an unreproducible benchmark is worse than none.

## The two targets

| | RTX 4060 8GB (Windows) | MacBook Air M3 8GB (macOS) |
|---|---|---|
| profile | `nvidia-8gb` | `mac-8gb` |
| diffusion model | `qwen_image_2.1-Q4_K.gguf` — 4.2 GB | `qwen_image_2.1-Q3_K.gguf` — 3.3 GB |
| text encoder | `qwen3vl_8b_w4a8` — 6.3 GB, `device=cpu` | same, `device=cpu` |
| VAE | `qwen_image_2.1_vae_bf16` — 0.68 GB | same |
| ComfyUI flags | `--preview-method none` | `--preview-method none --force-fp16`, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| default size / steps | 1024px, 25 | 768px, 20 |
| downloads | ~11 GB | ~10 GB |

Run `qwen21 doctor` and it prints the profile, the resolved hardware and whether
every file is present at the right size.

## Why the encoder is pinned to the CPU

`TextEncodeQwenImage21` needs the Qwen3-VL text encoder in memory to turn your
prompt into conditioning. The official 9.4 GB int8 encoder plus a 7.3 GB
int8 transformer cannot coexist in 8 GB of anything. So the 8GB profiles use:

* the **4-bit encoder** (6.3 GB) instead of the 9.4 GB int8 one, and
* the **GGUF transformer** (4.2 GB) instead of the 7.3 GB int8 one, and
* `CLIPLoader.device = "cpu"`, so the encoder lives in system RAM while the
  transformer sits in VRAM/unified memory.

ComfyUI's dynamic VRAM manager loads and unloads them around the sampling loop on
its own; `--lowvram` is a no-op in current ComfyUI and the installer does not
pass it. What *does* help is `--preview-method none` (the default, passed
explicitly for clarity) and, on Apple Silicon, `--force-fp16`.

The transformer is dequantized on the fly by ComfyUI-GGUF, which is why it needs
that custom node **and its `gguf` Python package** — the installer installs both.

## RTX 4060 8GB expectations

The comfortable configuration. Expect roughly:

* model load + first encode: 10-30 s (the encoder is read from disk);
* 1024×1024, 25 steps: tens of seconds per image;
* a typical edit (one reference image) inside a minute.

If you hit an out-of-memory error:

1. close anything using the GPU (browsers with hardware acceleration, editors,
   games, Discord);
2. lower `resolution` to 768 or 640, or `steps` to 15;
3. keep the encoder on `device=cpu` (the `nvidia-8gb` profile already does);
4. restart ComfyUI after switching models — old weights linger in VRAM until the
   process is restarted (`qwen21 stop && qwen21 start`).

`QwenImage21Cache` (`device=auto`, `dtype=default`) keeps the prompt and image
prefix between sampling steps, which is most of the speedup in an edit. Setting
`dtype=int8` halves that cache; `device=off` disables it and recomputes every
step (slower but a good way to rule it out when debugging).

## MacBook Air M3 8GB expectations

Be realistic: this is a *swapping* machine, not a fast one.

* 6.3 GB encoder + 3.3 GB transformer in 8 GB of shared memory means macOS
  pages to disk. First job pays the load cost; later jobs reuse the page cache
  if you have the disk space.
* `--force-fp16` avoids the MPS float32 path. Some ops (including parts of the
  4-bit encoder) have no MPS kernel, hence `PYTORCH_ENABLE_MPS_FALLBACK=1`;
  those fall back to CPU and slow the encode down.
* Expect **minutes**, not seconds, per image at 768px. Smaller is much better
  than fewer steps: drop to 512-640px rather than 8 steps.
* The int8_convrot path is not an option here: it needs an MPS backend that does
  not ship with ComfyUI and more memory than this machine has.

If the machine is unusable:

```bash
qwen21 install --profile mac-8gb-lite   # Q2_K, 640px, 16 steps, cache off
qwen21 stop && qwen21 start --profile mac-8gb-lite
```

Q2_K is visibly weaker at text rendering. If text quality matters more than
throughput, keep `mac-8gb` and be patient, or use `full` on a bigger machine.

Free memory before a run: quit Safari tabs, stop Docker/VMs, and make sure
"App Nap"/heavy Spotlight indexing is not running during the first job.

## Measuring your own numbers

```bash
qwen21 smoke --resolution 1024 --steps 25
```

prints the wall-clock time and output size for a real edit on your hardware.
Put those numbers in your own README/issue — a reproducible measurement is worth
more than any benchmark table.

Peak memory on NVIDIA while a job runs:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5
```

## Other profiles

* `full` (12GB+ VRAM or 16GB+ unified memory): the official int8_convrot
  safetensors with the int8 encoder, loaded through the core `UNETLoader` — no
  GGUF node needed, better fidelity, 1328px default.
* Anything bigger: install ComfyUI yourself and point this tool at it with
  `--comfy-path /path/to/ComfyUI` (a Windows portable layout works: the
  `python_embeded` interpreter is used automatically).
