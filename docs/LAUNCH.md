# Launch plan: optimize for Stars without shipping vaporware

This repo is intentionally built around a **fresh model + painful constraint + searchable solution**:

> Qwen-Image-2.1 + 8GB VRAM + one-command setup + a public edit endpoint

## Repository settings

Suggested repository name:

`qwen-image-2.1-8gb`

Suggested description:

`Run Qwen-Image-2.1 on an 8GB GPU or Mac — one-command ComfyUI + GGUF install, image-edit REST API, public tunnel.`

Suggested topics:

`qwen`, `qwen-image`, `qwen-image-2-1`, `comfyui`, `gguf`, `image-generation`, `low-vram`, `rtx`, `local-ai`, `fastapi`, `cloudflare-tunnel`, `apple-silicon`

## README headline

Use a concrete claim people search for:

**Run Qwen-Image-2.1 on 8GB VRAM**

Do not use a fake benchmark. Add measured timing / peak VRAM as soon as you have it.

## First screenshot to add

Make one 1024×1024 image that visibly tests text rendering. Example prompt:

`A cinematic coffee cart at dusk. A clear paper cup has crisp bold black handwritten text: "QWEN 2.1 / 8GB VRAM". Shallow depth of field, realistic droplets, 85mm lens.`

Then add:

- output image;
- GPU model;
- generation time;
- peak VRAM;
- workflow / quantization.

That single proof image will do more for credibility than another page of prose.

## Launch posts

### Hacker News

Title:

`Show HN: Run Qwen-Image-2.1 on an 8GB GPU`

Body: 4–6 factual sentences. Lead with the 8GB constraint and the exact hardware used.

### Reddit

Use model / local-AI / ComfyUI communities where self-promotion is allowed. Post the actual benchmark image, not just a link.

Title idea:

`Qwen-Image-2.1 running on my 8GB RTX card — setup + workflow`

### X / Bluesky

Post the generated image first. Put the repo link in the text/reply. Include the measured VRAM and seconds/image.
## Star conversion

Keep these above the README fold:

1. exact problem solved;
2. one command;
3. screenshot;
4. real 8GB hardware result;
5. no mystery dependencies.

A little headline aggression is useful. A false performance claim is not: one
reproducible screenshot will convert better and survive scrutiny.

## Added in 2.0: the public endpoint

The original pitch was "run it on 8GB". The 2.0 pitch is one line longer and
harder to copy:

> Run Qwen-Image-2.1 on an 8GB GPU **and put it behind a public HTTP endpoint**
> that edits an image from a text prompt.

`./install.sh && qwen21 start` gives you ComfyUI, an authenticated REST API and
a Cloudflare tunnel. That is a complete product, not a model card, and it is the
part worth showing: a `curl` that edits a photo from a phone, through a random
`*.trycloudflare.com` URL, on hardware that was never supposed to run it.

The demo that converts best is still a before/after image pair. Produce it with
`qwen21 smoke` and paste the real numbers (seconds, resolution, steps) next to
it. Two demos worth posting:

* the edit that changes lettering in a photo (typography is Qwen-Image's
  strength, and it is visible at a glance);
* a background removal with transparency, straight from the bundled workflow.
