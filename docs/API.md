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
| `response_format` | `b64` \| `url` | `b64` | `url` returns only the download URL |

Instead of a file you can send **the name of a file that is already in
`ComfyUI/input/`** (form field `image`, value a plain filename, no `@`). Nothing
crosses the tunnel, so this is the way past the upload ceiling below:

```bash
scp big-scan.tiff me@box:~/qwen-image-2.1-8gb/runtime/comfy/input/
curl -X POST https://img.example.com/v1/edit \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=big-scan.tiff" \
  -F "prompt=remove the background"
```

Names are matched against `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, must exist, and
may not mix with uploads in the same request. A file you reference is **yours**:
the service never deletes it (unlike the files it uploads itself).

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

## Images are never written to disk

Rendered images go straight from ComfyUI into the response. Nothing survives a
request on the filesystem:

* the service holds image bytes **in RAM only** (`IMAGE_CACHE_MB`, 64 MB) and
  writes nothing — the job database (`runtime/jobs.sqlite3`) is the only file it
  creates;
* ComfyUI's rendered file is deleted the moment the service reads it, and any
  file the service uploaded is deleted when the job ends, whatever the outcome
  (succeeded, failed, cancelled, or a 413/415 part-way through a request);
* `qwen21 start` sweeps by age anything a crash left behind, so a sweep can never
  race a render in flight.

Two consequences:

* **`GET /v1/jobs/{id}/image/{n}` answers `410 Gone`** once the bytes have left
  the cache (or after a restart, since memory does not survive it). The message
  says so. Job status reports `retained: true|false` per image so you can check
  before fetching.
* `response_format=b64` always carries the bytes **in the response itself**, so a
  client that polls within the window is unaffected; the `url` is for re-fetching
  later, and may expire.

`KEEP_IMAGE_FILES=1` stops the deletions and leaves ComfyUI's output in place —
for debugging what it actually produced. Never enable it on a machine that must
not keep images.

## Size limits (and the tunnel)

| setting | default | notes |
|---|---|---|
| `MAX_REQUEST_MB` | 90 | whole request body, checked from `Content-Length` and again while streaming |
| `MAX_UPLOAD_MB` | 20 | per file |
| `MAX_IMAGES` | 4 | reference images per request (`image_1`..`image_16` exist in the node) |
| `IMAGE_CACHE_MB` | 64 | in-RAM retention for the async path; `0` retains nothing |

The ceiling exists because **Cloudflare caps request bodies by account plan**:
100 MB on Free and Pro, 200 MB on Business, up to 5 GB on Enterprise (all
2026 figures). Responses have no size limit. An over-cap request is refused by
this service with a `413` that names the number, rather than by Cloudflare with
an error page. `qwen21 doctor` fails if `MAX_UPLOAD_MB × MAX_IMAGES` would
exceed `MAX_REQUEST_MB`.

Two other Cloudflare limits shape the contract, from the same connection-limits
table:

* **Proxy read timeout: 125 s** → `524`. That is why `SYNC_MAX_WAIT` is 90 s and
  long jobs become `202` + polling.
* **Proxy write timeout: 30 s** → `524`. A large upload on a slow uplink can be
  cut mid-transfer, another reason not to sit near the ceiling. JPEG beats PNG
  for reference photos (a 12 MP JPEG is ~4 MB where the same image as PNG is
  ~20 MB).

## Jobs

* Job metadata is persisted in SQLite (`runtime/jobs.sqlite3`) and survives a
  service restart. A job that was mid-flight when the process died comes back as
  `failed`, never as a zombie. Image bytes do not survive a restart by design.
* Jobs and their cached bytes are forgotten after `JOB_TTL_HOURS` (24 h).
* `POST /v1/jobs/{id}/cancel` cancels the local job and asks ComfyUI to interrupt.
  ComfyUI only stops at a node/step boundary, so a cancel can take a few seconds
  to take effect.
* `GET /v1/jobs?limit=50` lists recent jobs.

## Limits

| setting | default | notes |
|---|---|---|
| `MAX_QUEUE` | 8 | in-flight + queued jobs, then `503` |
| `RATE_LIMIT_PER_MIN` | 10 | per API key, `0` disables |
| `JOB_TIMEOUT` | 3600 | give up on one generation |
| `JOB_TTL_HOURS` | 24 | forgotten jobs and their cached bytes |

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
