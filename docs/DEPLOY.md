# Deployment, keys and the public endpoint

Everything here is optional except one thing: **the API service only accepts
requests carrying `API_KEY`**, so that key is the only secret you must set
before exposing anything.

## 1. The local API key

```bash
qwen21 keygen --write        # writes API_KEY=… into .env
qwen21 keygen                # or just print one
```

`qwen21 doctor` fails while it is still the placeholder from `.env.example`.

## 2. The public endpoint

`qwen21 start` runs ComfyUI on `127.0.0.1:8188`, the API on `127.0.0.1:8000`,
and cloudflared in front of the API. ComfyUI itself is never published.

### Mode `quick` (default, no account needed)

```env
TUNNEL_MODE=quick
```

`cloudflared` prints a random hostname, e.g.
`https://olive-pans-shout.trycloudflare.com`. It changes every restart, so it is
for demos and testing. `qwen21 status` shows the current URL. Quick tunnels do
not support Server-Sent Events, which is one more reason the API uses polling
(`202` + job id) instead of a long-lived stream.

### Mode `named` (stable URL, needs a Cloudflare account)

1. <https://dash.cloudflare.com> → **Zero Trust** → **Networks** → **Tunnels**.
2. **Create a tunnel** → name it (e.g. `qwen-image`) → pick the OS → copy the
   install command. It looks like:
   ```
   cloudflared.exe service install <BASE64-BLOB>
   ```
   The long base64 blob at the end of that command is the **tunnel token**. It is a
   base64 JSON blob that already contains your account, tunnel id and secret -
   treat it like a password.
3. In **Public Hostname** add a route: subdomain (e.g. `img`) + domain (a zone
   already on Cloudflare) + service `http://127.0.0.1:8000`.
4. Put both values in `.env`:
   ```env
   TUNNEL_MODE=named
   TUNNEL_TOKEN=<paste-the-base64-blob-here>
   TUNNEL_HOSTNAME=img.example.com
   PUBLIC_BASE_URL=https://img.example.com   # so responses contain absolute image URLs
   ```
5. `qwen21 start` (or `qwen21 tunnel --mode named`). You do **not** need to run
   `cloudflared service install`; the token is enough, and this tool manages the
   process.

### Mode `off`

`TUNNEL_MODE=off` keeps everything on `127.0.0.1`. Use it when the machine is
already behind your own reverse proxy or VPN.

## 3. Optional: device registry (find your machines by name)

A quick tunnel gets a **new random hostname on every restart**, so the URL is not
an identifier — the device is. When the tunnel comes up, `qwen21 start` publishes
this device's public link to MongoDB and **overwrites the row for the same
device**, so one collection answers "where is each machine right now":

```json
{
  "link":    "https://olive-pans-shout.trycloudflare.com/v1/edit",
  "device":  "rtx4060-box",
  "updated_at": "2026-09-29T18:04:11.512873+00:00"
}
```

### Setup

1. Atlas → **Database Access** → **Add New Database User**. Set *Database
   Access* to the single database you want (`tunnels`) with **Read and write**.
   A user with cluster-wide access is a much bigger credential than this needs.
2. Atlas → **Network Access** → **IP Access List** → add your own IP. Without it
   those credentials work from anywhere on the internet.
3. **Connect** → copy the SRV string and put it in `.env`:

```env
HOST_NAME=rtx4060-box            # stable name; unset means the OS hostname
MONGODB=mongodb+srv://USER:PASSWORD@cluster0.example.net/tunnels
MONGODB_COLLECTION=devices       # optional, this is the default
```

`.env` is gitignored and the value is never logged — error messages name only
the host and database. `python scripts/check_secrets.py` fails the build if a
real credential ever reaches a tracked file; run it before committing anything.

### Behaviour

| Situation | What happens |
|---|---|
| `qwen21 start` | publishes `{link, device, updated_at}` for this device |
| Tunnel restarts, new random hostname | same row, `link` overwritten |
| `qwen21 tunnel` alone | re-publishes, so a tunnel-only restart stays listed |
| MongoDB unreachable | a warning; **the endpoint still works** and `start` still succeeds |
| `MONGODB` unset | nothing is published, nothing is contacted |
| `qwen21 stop` | the row is **left** alone — a stopped-but-intentional machine should stay listed |

Because there is no heartbeat, a machine that dies for good leaves its last row
behind; `updated_at` is how you spot it. Prune with:

```js
db.devices.deleteMany({ updated_at: { $lt: ISODate("2026-09-01T00:00:00Z") } })
```

Add a periodic heartbeat later if you want the collection to be a liveness list
rather than a "last known address" list.

### Checking what is registered

```bash
qwen21 doctor              # device name, whether MONGODB is set, what is published
curl -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/v1/registry
```

## 4. Optional: Hugging Face token

Not needed for the public Qwen-Org and leejet repositories. Set `HF_TOKEN` if
downloads get rate limited (a 429 during `qwen21 models`) or if you later add a
gated model. Create one at <https://huggingface.co/settings/tokens>.

## 5. Keys checklist

`qwen21 keys` prints the current state; `qwen21 doctor` blocks on the required
one.

| key | required | where it comes from |
|---|---|---|
| `API_KEY` | yes | `qwen21 keygen` |
| `TUNNEL_TOKEN` | only for `named` | Cloudflare Zero Trust → Tunnels → Create |
| `TUNNEL_HOSTNAME` | no | a hostname on a zone you control |
| `HF_TOKEN` | no | huggingface.co/settings/tokens |

`.env` is gitignored. Never commit it, never paste a tunnel token into an issue
or a chat.

## 6. Security checklist before you point a domain at it

- [ ] `API_KEY` generated, long, not the placeholder.
- [ ] `RATE_LIMIT_PER_MIN` set to something a human would not hit by accident.
- [ ] `MAX_QUEUE` left small (8 is fine). Each queued job is a GPU-minute.
- [ ] `COMFY_HOST=127.0.0.1` and `API_HOST=127.0.0.1`. Do not set `0.0.0.0`
      unless you understand that your router/ISP may expose the raw service.
- [ ] Your router does not forward 8188 or 8000 (nothing needs it: the tunnel
      makes outbound connections only).
- [ ] Anyone with the key can spend your GPU.
- [ ] If you use the device registry: the Atlas user is scoped to one database
      and your IP is in the access list, and the password has been rotated if it
      was ever pasted into a chat, an issue or a commit. Hand out per-client keys? The
      service supports one key today; put Cloudflare Access in front if you need
      real identity (`docs.comfy.org` → Zero Trust → Access → self-hosted app).
- [ ] Consider `SYNC_MAX_WAIT` + `202` polling for anything long-running.

## 7. Operating it

```bash
qwen21 start        # ComfyUI + API + tunnel, in the background
qwen21 status       # what is up, where it is reachable
qwen21 logs api -n 200
qwen21 logs comfy
qwen21 stop         # stops all three, in the right order
qwen21 restart      # stop + start
```

`qwen21 start` waits for ComfyUI to answer `/system_stats` and the API to answer
`/healthz` before starting the tunnel, so the URL it prints is already live.
PIDs and logs live in `runtime/pids/` and `runtime/logs/`.

There is no systemd unit or launchd plist on purpose: the same commands work on
Windows and macOS. If you want it to survive a reboot, point Task Scheduler
(Windows) or a launchd agent (macOS) at `qwen21 start`.

## 8. Two GPUs, one machine

`MAX_CONCURRENT=1` because ComfyUI serialises work anyway. If you ever move the
API to a different machine than ComfyUI, point `COMFY_URL` at it and consider
raising `MAX_CONCURRENT` only if ComfyUI is started with enough VRAM for it.
