# API

The service is a small custom REST API (not OpenAI-compatible by design) with one
idea: **a request never blocks longer than the tunnel can survive**.

```
POST /v1/edit      image(s) + instruction  ->  edited image
POST /v1/generate  prompt                  ->  image
GET  /v1/jobs/{id}                         ->  status of an async job
GET  /v1/jobs/{id}/image/{index}           ->  the rendered PNG
POST /v1/jobs/{id}/cancel                  ->  stop it
GET  /v1/models                            ->  which model files are in use
GET  /healthz                              ->  no auth, for probes
GET  /docs                                 ->  interactive OpenAPI browser
```

Base URL: `https://<your tunnel>/` (public) or `http://127.0.0.1:8000/` (local).

## Auth

Every route except `/healthz` needs the bearer token from `.env`:

```bash
curl -H "Authorization: Bearer $API_KEY" https://img.example.com/v1/models
```

`X-API-Key: $API_KEY` works too. Requests are rate limited per key
(`RATE_LIMIT_PER_MIN`, default 10/min → `429`), and the queue is capped
(`MAX_QUEUE`, default 8 → `503` with `Retry-After`).

## POST /v1/edit

`multipart/form-data`.

| field | type | default | meaning |
|---|---|---|---|
| `image` | file (1..4) | required | PNG or JPEG. Repeat the field for multiple references. |
| `prompt` | string | required | the edit instruction, e.g. `"change the lettering to ROASTED IN TOKYO"` |
| `negative_prompt` | string | `""` | what to avoid |
| `resolution` | int | profile default (1024 / 768) | longest edge; output follows the **first** image's aspect ratio |
| `steps` | int | 25 (20 on Mac) | sampling steps |
| `cfg` | float | 1.0 | Qwen-Image-2.1 is trained for cfg 1.0; higher values wash out |
| `seed` | int | random | pass a fixed seed to reproduce a result |
| `sampler` / `scheduler` | string | `euler` / `simple` | ComfyUI sampler names |
| `n` | int | 1 | run N times; images come back in order |
| `wait` | bool | true | see "Sync vs async" below |

Multiple references are addressed by index, exactly like the ComfyUI workflow:
`"put this shirt from image 2 on the person in image 1"` becomes the prompt, and
the encoder sees them as `image_1`, `image_2`, … (written `<image1>`, `<image2>`
in the prompt text if you prefer).

```bash
curl -X POST https://img.example.com/v1/edit \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@before.png" \
  -F "prompt=Replace the lettering on the cup with 'ROASTED IN TOKYO', keep the ink texture" \
  -F "steps=25" \
  -F "seed=1234"
```

Response `200`:

```json
{
  "id": "0f1c9a7c2b3d4e5f6a7b8c9d",
  "kind": "edit",
  "status": "succeeded",
  "seed": 1234,
  "duration_s": 41.7,
  "images": [
    {
      "index": 0,
      "media_type": "image/png",
      "bytes": 1834221,
      "url": "https://img.example.com/v1/jobs/0f1c…/image/0",
      "b64_json": "iVBORw0KGgoAAAANSUhEUg…"
    }
  ]
}
```

## POST /v1/generate

`application/json`, same knobs minus the images, plus explicit `width`/`height`:

```bash
curl -X POST https://img.example.com/v1/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red apple on a wooden table, studio photo","width":1024,"height":1024,"steps":25}'
```

## Sync vs async (read this once)

A 25-step edit on an RTX 4060 takes ~30-60 s; on a MacBook Air M3 with 8GB it can
take many minutes. Cloudflare's proxy drops an origin response that stays silent
for ~100 seconds (`524`), so:

* the service waits up to `SYNC_MAX_WAIT` seconds (default **90**);
* if the job finished → `200` with base64 + a download URL;
* if it is still running → **`202`** with a job id, `Location` header and
  `Retry-After: 2`.

So a client that only handles `200` still works for quick jobs, and a client that
understands `202` never depends on a long-lived HTTP request. Poll it:

```bash
curl -H "Authorization: Bearer $API_KEY" https://img.example.com/v1/jobs/$ID
# {"status":"running","elapsed":12.3,...}
# {"status":"succeeded","images":[{"index":0,"url":"...","bytes":1834221}]}
curl -o after.png -H "Authorization: Bearer $API_KEY" https://img.example.com/v1/jobs/$ID/image/0
```

Send `wait=false` to always get the `202` immediately (the right choice for
batching many edits behind the tunnel).

## Jobs

* Jobs are persisted in SQLite (`runtime/outputs/jobs.sqlite3`) and survive a
  service restart. A job that was mid-flight when the process died comes back as
  `failed`, never as a zombie.
* Rendered files live in `runtime/outputs/<job id>/` and are deleted after
  `JOB_TTL_HOURS` (default 24 h) by a background sweep.
* `POST /v1/jobs/{id}/cancel` cancels the local job and asks ComfyUI to interrupt.
  ComfyUI only stops at a node/step boundary, so a cancel can take a few seconds
  to take effect.
* `GET /v1/jobs?limit=50` lists recent jobs.

## Limits

| setting | default | notes |
|---|---|---|
| `MAX_UPLOAD_MB` | 25 | per file |
| `MAX_IMAGES` | 4 | reference images per request (`image_1`..`image_16` exist in the node) |
| `MAX_QUEUE` | 8 | in-flight + queued jobs, then `503` |
| `RATE_LIMIT_PER_MIN` | 10 | per API key, `0` disables |
| `JOB_TIMEOUT` | 3600 | give up on one generation |

## How it maps to ComfyUI

| API field | ComfyUI node input |
|---|---|
| `image` | `/upload/image` (multipart) → `LoadImage.image` |
| `prompt`, `negative_prompt` | `TextEncodeQwenImage21.prompt` / `.negative_prompt` |
| multiple images | `TextEncodeQwenImage21.images` = `{"image_1": ["<node>", 0], …}` |
| `resolution` | `TextEncodeQwenImage21.resolution` |
| `width`/`height` | `EmptyLatentImage.width/height` (generate only) |
| `steps`, `cfg`, `seed`, `sampler`, `scheduler` | `KSampler` |
| profile | `UnetLoaderGGUF` or `UNETLoader`, `CLIPLoader.device`, `QwenImage21Cache` |
| output | `SaveImage` → `/history/{id}` → `/view`, cached under `runtime/outputs` |

The exact graphs are generated by `qwen21 export` into `workflows/api/`, so you
can inspect or hand-edit them.
