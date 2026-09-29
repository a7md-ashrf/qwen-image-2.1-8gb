"""End-to-end tests for the HTTP service against a fake ComfyUI server.

These run without a GPU, a model file or ComfyUI itself: a threaded
`http.server` implements the five ComfyUI endpoints the service uses, and the
real FastAPI app is driven through Starlette's TestClient.

Needs the API dependencies (`pip install -r api/requirements.txt`); the whole
module skips itself when they are not importable.

    pip install -r api/requirements.txt
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from starlette.testclient import TestClient
except ImportError:  # api dependencies are not installed
    TestClient = None

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc00000030101003f2d0e4b00000000"
    "49454e44ae426082"
)

MODELS = {
    "diffusion_models": ["qwen_image_2.1-Q4_K.gguf", "qwen2_video_vae.safetensors"],
    "text_encoders": ["qwen3vl_8b_w4a8.safetensors"],
    "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
}


class FakeState:
    """Knobs the tests turn to simulate a slow or a broken ComfyUI."""

    def __init__(self, root: Path | None = None) -> None:
        self.history_calls = 0
        self.history_delay = 0
        self.fail_with: dict | None = None
        self.last_prompt: dict = {}
        self.prompts: list[dict] = []
        self.uploads = 0
        self.empty_models = False
        # ComfyUI is file-based: it writes uploads to input/ and renders to
        # output/. The fake does the same so the tests can prove the service
        # cleans up after itself.
        self.root = root

    def write(self, folder: str, name: str) -> str:
        assert self.root is not None
        target = self.root / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(PNG_1PX)
        return name


class FakeComfyUI(BaseHTTPRequestHandler):
    state = FakeState()

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the test output clean
        pass

    def _send(self, payload, content_type="application/json", status=200):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/models/"):
            folder = path.split("/")[-1]
            return self._send([] if self.state.empty_models else MODELS.get(folder, []))
        if path.startswith("/object_info/"):
            return self._send({path.split("/")[-1]: {}})
        if path.startswith("/history/"):
            self.state.history_calls += 1
            if self.state.history_calls <= self.state.history_delay:
                return self._send({})
            name = self.state.write("output", f"render-{self.state.history_calls}.png")
            return self._send(
                {
                    path.split("/")[-1]: {
                        "status": {"status_str": "success", "messages": []},
                        "outputs": {"60": {"images": [{"filename": name, "subfolder": "",
                                                      "type": "output"}]}},
                    }
                }
            )
        if path == "/view":
            return self._send(PNG_1PX, "image/png")
        if path == "/system_stats":
            return self._send({"devices": [{"name": "Fake GPU", "vram_total": 8 * 2**30}]})
        if path == "/queue":
            return self._send({"queue_running": [], "queue_pending": []})
        return self._send({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path == "/upload/image":
            self.state.uploads += 1
            return self._send({"name": f"upload{self.state.uploads}.png", "subfolder": "",
                               "type": "input"})
        if self.path == "/prompt":
            payload = json.loads(raw)
            self.state.prompts.append(payload["prompt"])
            self.state.last_prompt = payload["prompt"]
            if self.state.fail_with:
                # ComfyUI really does wrap the error dict in a JSON string.
                # Pass bytes so _send does not encode it a second time.
                return self._send(json.dumps(json.dumps(self.state.fail_with)).encode(), status=400)
            return self._send({"prompt_id": f"pid-{len(self.state.prompts)}", "number": 1})
        if self.path in ("/interrupt",):
            return self._send({})
        return self._send({"error": "not found"}, status=404)


RESET_KEYS = (
    "MONGODB",
    "MONGODB_COLLECTION",
    "HOST_NAME",
    "COMFY_URL",
    "API_KEY",
    "QWEN_OUTPUT_DIR",
    "SYNC_MAX_WAIT",
    "RATE_LIMIT_PER_MIN",
    "MAX_QUEUE",
    "MAX_IMAGES",
    "MAX_REQUEST_MB",
    "MAX_UPLOAD_MB",
    "RESPONSE_FORMAT",
    "IMAGE_CACHE_MB",
    "KEEP_IMAGE_FILES",
    "QWEN_COMFY_DIR",
    "QWEN_COMFY_INPUT_DIR",
    "QWEN_COMFY_OUTPUT_DIR",
    "QWEN_COMFY_TEMP_DIR",
    "ALLOW_ANONYMOUS",
    "QWEN_UNET",
    "QWEN_CLIP",
    "QWEN_VAE",
    "QWEN_ENCODER_DEVICE",
    "PUBLIC_BASE_URL",
)


def load_app(comfy_url: str, output_dir: str, comfy_root: str, **env: str):
    """Import a fresh copy of the app with exactly this environment."""
    previous = sys.modules.get("qwen_api.app")
    if previous is not None and hasattr(previous, "store"):
        previous.store.close()  # do not leak a sqlite handle per reload
    for module in [m for m in list(sys.modules) if m.startswith("qwen_api")]:
        del sys.modules[module]
    for key in RESET_KEYS:
        os.environ.pop(key, None)
    os.environ.update(
        {
            "COMFY_URL": comfy_url,
            "API_KEY": "test-key",
            "QWEN_JOB_DB": str(Path(output_dir) / "jobs.sqlite3"),
            "QWEN_COMFY_INPUT_DIR": str(Path(comfy_root) / "input"),
            "QWEN_COMFY_OUTPUT_DIR": str(Path(comfy_root) / "output"),
            "QWEN_COMFY_TEMP_DIR": str(Path(comfy_root) / "temp"),
            "SYNC_MAX_WAIT": "5",
            "RATE_LIMIT_PER_MIN": "0",
            **env,
        }
    )
    return importlib.import_module("qwen_api.app")


@unittest.skipIf(TestClient is None, "api dependencies not installed (pip install -r api/requirements.txt)")
class ApiTestCase(unittest.TestCase):
    env: ClassVar[dict] = {}
    auth = {"Authorization": "Bearer test-key"}

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyUI)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.comfy_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.tmp = tempfile.TemporaryDirectory()
        cls.comfy = tempfile.TemporaryDirectory()
        cls.state = FakeState(Path(cls.comfy.name))
        FakeComfyUI.state = cls.state
        cls.module = load_app(cls.comfy_url, cls.tmp.name, cls.comfy.name, **cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def setUp(self):
        # Per-test dirs: the job database is a file, and ComfyUI's image tree
        # must not leak between tests.
        self.tmp = tempfile.TemporaryDirectory()
        self.comfy = tempfile.TemporaryDirectory()
        self.state = FakeState(Path(self.comfy.name))
        FakeComfyUI.state = self.state
        self.module = load_app(
            self.comfy_url, self.tmp.name, self.comfy.name, **self.env
        )
        # Entering the context manager starts the lifespan and keeps ONE event
        # loop for the whole test, so background jobs survive between requests
        # exactly as they do under uvicorn.
        self._client = TestClient(self.module.app)
        self.client = self._client.__enter__()

    def tearDown(self):
        self._client.__exit__(None, None, None)
        self.tmp.cleanup()
        self.comfy.cleanup()

    def comfy_files(self) -> list[Path]:
        return sorted(p for p in Path(self.comfy.name).rglob("*") if p.is_file())

    def post_edit(self, **fields):
        return self.client.post(
            "/v1/edit",
            files={"image": ("in.png", PNG_1PX, "image/png")},
            data=fields,
            headers=self.auth,
        )


class TestEditFlow(ApiTestCase):
    def test_edit_returns_base64_and_a_url(self):
        response = self.post_edit(prompt="make it red", steps=8)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "succeeded")
        self.assertEqual(body["images"][0]["media_type"], "image/png")
        self.assertEqual(body["images"][0]["bytes"], len(PNG_1PX))
        import base64

        self.assertEqual(base64.b64decode(body["images"][0]["b64_json"]), PNG_1PX)
        self.assertIn("/v1/jobs/", body["images"][0]["url"])

    def test_the_graph_reaches_comfyui_with_the_prompt_and_image(self):
        self.post_edit(prompt="make it red", steps=8, resolution=768, seed=99)
        graph = self.state.last_prompt
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "qwen3vl_8b_w4a8.safetensors")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "qwen_image_2.1-Q4_K.gguf")
        self.assertEqual(graph["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(graph["30"]["inputs"]["prompt"], "make it red")
        self.assertEqual(graph["30"]["inputs"]["resolution"], 768)
        self.assertEqual(graph["30"]["inputs"]["images"], {"image_1": ["100", 0]})
        self.assertEqual(graph["40"]["inputs"]["steps"], 8)
        self.assertEqual(graph["40"]["inputs"]["seed"], 99)

    def test_two_images_become_two_slots(self):
        response = self.client.post(
            "/v1/edit",
            files=[("image", ("a.png", PNG_1PX, "image/png")),
                   ("image", ("b.png", PNG_1PX, "image/png"))],
            data={"prompt": "swap"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.state.last_prompt["30"]["inputs"]["images"],
            {"image_1": ["100", 0], "image_2": ["101", 0]},
        )
        self.assertEqual(self.state.uploads, 2)

    def test_image_can_be_fetched_back_by_url(self):
        body = self.post_edit(prompt="x").json()
        response = self.client.get(f"/v1/jobs/{body['id']}/image/0", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, PNG_1PX)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_n_runs_the_graph_twice(self):
        response = self.post_edit(prompt="x", n="2")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["images"]), 2)
        self.assertEqual(len(self.state.prompts), 2)

    def test_job_listing_and_cancellation(self):
        body = self.post_edit(prompt="x").json()
        listing = self.client.get("/v1/jobs", headers=self.auth).json()
        self.assertEqual([j["id"] for j in listing["jobs"]], [body["id"]])
        cancel = self.client.post(f"/v1/jobs/{body['id']}/cancel", headers=self.auth)
        self.assertEqual(cancel.json()["cancelled"], False)  # already finished

    def test_a_comfyui_rejection_becomes_a_502(self):
        self.state.fail_with = {
            "error": {"type": "prompt_outputs_failed_validation", "message": "bad"},
            "node_errors": {"40": {"errors": [{"type": "value_not_in_list",
                                              "message": "Value not in list"}]}},
        }
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 502)
        # The double-encoded body must still yield the node detail.
        self.assertIn("node 40", response.json()["detail"])
        self.assertIn("Value not in list", response.json()["detail"])


class TestGenerateFlow(ApiTestCase):
    def test_generate_uses_an_empty_latent(self):
        response = self.client.post(
            "/v1/generate", json={"prompt": "a cat", "width": 800, "height": 600}, headers=self.auth
        )
        self.assertEqual(response.status_code, 200, response.text)
        graph = self.state.last_prompt
        self.assertEqual(graph["10"]["class_type"], "EmptyLatentImage")
        self.assertNotIn("images", graph["30"]["inputs"])

    def test_models_endpoint_reports_the_resolved_stack(self):
        body = self.client.get("/v1/models", headers=self.auth).json()
        self.assertEqual(body["unet"], "qwen_image_2.1-Q4_K.gguf")
        self.assertEqual(body["loader"], "gguf")
        self.assertEqual(body["vae"], "qwen_image_2.1_vae_bf16.safetensors")

    def test_healthz_needs_no_key(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


class TestAuthAndLimits(ApiTestCase):
    def test_everything_but_healthz_needs_the_key(self):
        for method, path in (("get", "/v1/models"), ("get", "/v1/jobs"),
                             ("post", "/v1/generate"), ("get", "/v1/jobs/abc")):
            response = getattr(self.client, method)(path, **({"json": {"prompt": "x"}}
                                                              if method == "post" else {}))
            self.assertEqual(response.status_code, 401, path)
            self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

    def test_wrong_key_is_rejected(self):
        response = self.client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        self.assertEqual(response.status_code, 401)

    def test_x_api_key_header_also_works(self):
        response = self.client.get("/v1/models", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)

    def test_a_full_queue_is_refused_before_uploading(self):
        module = load_app(self.comfy_url, self.tmp.name, self.comfy.name,
                          MAX_QUEUE="0", **self.env)
        with TestClient(module.app) as client:
            response = client.post(
                "/v1/edit",
                files={"image": ("a.png", PNG_1PX, "image/png")},
                data={"prompt": "x"},
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers.get("retry-after"), "10")
        self.assertEqual(self.state.uploads, 0)

    def test_rate_limit_returns_429(self):
        module = load_app(self.comfy_url, self.tmp.name, self.comfy.name,
                          RATE_LIMIT_PER_MIN="1", **self.env)
        with TestClient(module.app) as client:
            first = client.get("/v1/models", headers=self.auth)
            second = client.get("/v1/models", headers=self.auth)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.headers.get("retry-after"), "30")

    def test_non_images_are_refused(self):
        response = self.client.post(
            "/v1/edit",
            files={"image": ("notes.txt", b"hello", "text/plain")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 415)

    def test_an_empty_prompt_is_refused(self):
        response = self.post_edit(prompt="   ")
        self.assertEqual(response.status_code, 422)


class FakeCollection:
    """The slice of a pymongo collection the registry actually uses."""

    def __init__(self, fail: bool = False, duplicate: bool = False) -> None:
        self.calls: list[tuple] = []
        self.documents: dict[str, dict] = {}
        self.fail = fail
        self.duplicate = duplicate
        self.indexes: list = []

    async def create_index(self, field, **kwargs):
        self.indexes.append((field, kwargs))

    async def update_one(self, query, update, upsert=False):
        self.calls.append((query, update, upsert))
        if self.fail:
            from pymongo.errors import ServerSelectionTimeoutError

            raise ServerSelectionTimeoutError("simulated MongoDB outage")
        if self.duplicate and upsert:
            from pymongo.errors import DuplicateKeyError

            raise DuplicateKeyError("simulated duplicate")
        key = query["device"]
        self.documents.setdefault(key, {}).update(update["$set"])
        return None

    async def find_one(self, query, projection=None):
        return self.documents.get(query["device"])


class TestNoImageFilesAreLeftBehind(ApiTestCase):
    """ComfyUI writes images to disk; the service must leave nothing behind."""

    def test_a_completed_edit_leaves_no_files_anywhere(self):
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 200, response.text)
        # the fake really did write an upload and a render
        self.assertEqual(self.state.uploads, 1)
        self.assertEqual(self.comfy_files(), [], "image files survived the request")

    def test_no_output_directory_is_created_at_all(self):
        self.post_edit(prompt="x")
        self.assertFalse((Path(self.tmp.name) / "outputs").exists())

    def test_the_response_still_carries_the_bytes(self):
        import base64

        body = self.post_edit(prompt="x").json()
        self.assertEqual(base64.b64decode(body["images"][0]["b64_json"]), PNG_1PX)
        self.assertTrue(body["images"][0]["retained"])
        self.assertEqual(self.comfy_files(), [])

    def test_a_failed_job_leaves_nothing_behind(self):
        self.state.fail_with = {"error": {"message": "no"}}
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.comfy_files(), [])

    def test_a_cancelled_job_leaves_nothing_behind(self):
        self.state.history_delay = 50  # still "running" when we cancel
        response = self.post_edit(prompt="x", wait="false")
        job_id = response.json()["id"]
        self.client.post(f"/v1/jobs/{job_id}/cancel", headers=self.auth)
        for _ in range(50):
            if not self.comfy_files():
                break
            import time as _t

            _t.sleep(0.05)
        self.assertEqual(self.comfy_files(), [])

    def test_many_images_leave_nothing_behind(self):
        response = self.post_edit(prompt="x", n="3")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["images"]), 3)
        self.assertEqual(self.comfy_files(), [])

    def test_a_part_refused_halfway_through_cleans_up_its_predecessors(self):
        response = self.client.post(
            "/v1/edit",
            files=[
                ("image", ("ok.png", PNG_1PX, "image/png")),
                ("image", ("notes.txt", b"not an image", "text/plain")),
            ],
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 415, response.text)
        self.assertEqual(self.state.uploads, 1, "the first part really was uploaded")
        self.assertEqual(self.comfy_files(), [], "the upload must be cleaned up")

    def test_the_model_lookup_happens_before_any_upload(self):
        # No model files at all: the service must fail before touching the wire,
        # so there is no upload to leak and no bandwidth wasted.
        self.state.empty_models = True
        response = self.client.post(
            "/v1/edit",
            files={"image": ("a.png", PNG_1PX, "image/png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("no model file for", response.json()["detail"])
        self.assertEqual(self.state.uploads, 0)
        self.assertEqual(self.comfy_files(), [])

    def test_an_orphan_from_a_crash_is_swept_by_age(self):
        import os
        import time

        old = Path(self.comfy.name) / "output"
        old.mkdir(parents=True, exist_ok=True)
        stale = old / "leftover.png"
        stale.write_bytes(PNG_1PX)
        fresh = old / "in-flight.png"
        fresh.write_bytes(PNG_1PX)
        os.utime(stale, (time.time() - 7200, time.time() - 7200))

        self.module.storage.sweep(3600)
        self.assertFalse(stale.exists(), "an old orphan should be removed")
        self.assertTrue(fresh.exists(), "a file from a live render must survive")

    def test_reference_mode_touches_nothing(self):
        target = Path(self.comfy.name) / "input"
        target.mkdir(parents=True, exist_ok=True)
        (target / "already-there.png").write_bytes(PNG_1PX)
        response = self.client.post(
            "/v1/edit",
            files={"image": (None, "already-there.png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.state.uploads, 0, "reference mode must not upload")
        self.assertTrue((target / "already-there.png").exists(), "a file we did not create stays")


class TestResponseFormat(ApiTestCase):
    def test_url_format_omits_the_base64_payload(self):
        body = self.post_edit(prompt="x", response_format="url").json()
        self.assertNotIn("b64_json", body["images"][0])
        self.assertTrue(body["images"][0]["url"].endswith("/image/0"))

    def test_b64_is_the_default(self):
        self.assertIn("b64_json", self.post_edit(prompt="x").json()["images"][0])

    def test_an_invalid_format_is_refused(self):
        self.assertEqual(self.post_edit(prompt="x", response_format="raw").status_code, 422)

    def test_generate_accepts_url_too(self):
        response = self.client.post(
            "/v1/generate", json={"prompt": "x", "response_format": "url"}, headers=self.auth
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("b64_json", response.json()["images"][0])


class TestRequestSizeCeiling(ApiTestCase):
    env = {"MAX_REQUEST_MB": "1", "MAX_UPLOAD_MB": "1", "MAX_IMAGES": "4"}

    def test_content_length_over_the_ceiling_is_refused_before_anything_happens(self):
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024)
        response = self.client.post(
            "/v1/edit",
            files={"image": ("big.png", big, "image/png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 413, response.text)
        self.assertIn("this service accepts", response.json()["detail"])
        self.assertEqual(self.state.uploads, 0, "nothing should be uploaded")
        self.assertEqual(self.comfy_files(), [])

    def test_the_message_names_the_cloudflare_cap(self):
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024)
        response = self.client.post(
            "/v1/edit",
            files={"image": ("big.png", big, "image/png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertIn("100 MB", response.json()["detail"])

    def test_a_request_under_the_ceiling_still_works(self):
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 200, response.text)


class TestFilenameMode(ApiTestCase):
    def test_an_existing_file_is_referenced_not_uploaded(self):
        target = Path(self.comfy.name) / "input"
        target.mkdir(parents=True, exist_ok=True)
        (target / "photo.png").write_bytes(PNG_1PX)
        response = self.client.post(
            "/v1/edit",
            files={"image": (None, "photo.png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.state.uploads, 0)
        self.assertEqual(
            self.state.last_prompt["30"]["inputs"]["images"], {"image_1": ["100", 0]}
        )

    def test_a_file_that_is_not_there_is_refused(self):
        response = self.client.post(
            "/v1/edit",
            files={"image": (None, "nope.png")},
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("input directory", response.json()["detail"])

    def test_path_traversal_is_refused(self):
        for name in ("../secrets.png", "/etc/passwd", "..\\secrets.png", "a/b.png"):
            with self.subTest(name=name):
                response = self.client.post(
                    "/v1/edit",
                    files={"image": (None, name)},
                    data={"prompt": "x"},
                    headers=self.auth,
                )
                self.assertEqual(response.status_code, 400, name)

    def test_mixing_uploads_and_names_is_refused(self):
        target = Path(self.comfy.name) / "input"
        target.mkdir(parents=True, exist_ok=True)
        (target / "photo.png").write_bytes(PNG_1PX)
        response = self.client.post(
            "/v1/edit",
            files=[
                ("image", (None, "photo.png")),
                ("image", ("upload.png", PNG_1PX, "image/png")),
            ],
            data={"prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("mixing", response.json()["detail"])

    def test_reference_mode_needs_the_api_key_too(self):
        response = self.client.post(
            "/v1/edit", files={"image": (None, "photo.png")}, data={"prompt": "x"}
        )
        self.assertEqual(response.status_code, 401)


class TestImageCacheEviction(ApiTestCase):
    env = {"IMAGE_CACHE_MB": "0.00002"}  # ~20 bytes: nothing will fit

    def test_a_response_still_succeeds_when_nothing_fits(self):
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        # inline bytes come from the request's own buffer, not the cache
        self.assertIsNone(body["images"][0]["b64_json"])
        self.assertFalse(body["images"][0]["retained"])

    def test_the_image_endpoint_answers_410_once_evicted(self):
        body = self.post_edit(prompt="x").json()
        response = self.client.get(f"/v1/jobs/{body['id']}/image/0", headers=self.auth)
        self.assertEqual(response.status_code, 410)
        self.assertIn("no longer retained", response.json()["detail"])

    def test_status_reports_that_the_image_is_gone(self):
        body = self.post_edit(prompt="x").json()
        status = self.client.get(f"/v1/jobs/{body['id']}", headers=self.auth).json()
        self.assertEqual(status["status"], "succeeded")
        self.assertFalse(status["images"][0]["retained"])


class TestDeviceRegistry(ApiTestCase):
    env = {"HOST_NAME": "rtx4060-box", "MONGODB": "mongodb+srv://u:p@cluster.example.net/tunnels"}

    def _stub(self, **kwargs):
        """Replace the app's registry with a fake collection."""
        collection = FakeCollection(**kwargs)
        self.module.registry._collection = collection
        self.module.registry._index_ready = False
        return collection

    def _publish(self, link="https://olive-pans.trycloudflare.com/v1/edit"):
        return self.client.post(
            "/v1/internal/tunnel", json={"link": link}, headers=self.auth
        )

    def test_publish_writes_link_device_and_updated_at(self):
        collection = self._stub()
        response = self._publish()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["published"])
        self.assertEqual(body["device"], "rtx4060-box")
        self.assertEqual(body["link"], "https://olive-pans.trycloudflare.com/v1/edit")
        self.assertTrue(body["updated_at"])
        query, update, upsert = collection.calls[0]
        self.assertEqual(query, {"device": "rtx4060-box"})
        self.assertTrue(upsert, "must upsert so a reconnect overwrites the old row")
        self.assertEqual(set(update["$set"]), {"link", "device", "updated_at"})
        self.assertEqual(collection.indexes, [("device", {"unique": True,
                                                           "name": "device_unique"})])

    def test_the_device_comes_from_the_server_not_the_body(self):
        self._stub()
        response = self.client.post(
            "/v1/internal/tunnel",
            json={"link": "https://x.trycloudflare.com/v1/edit", "device": "someone-else"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device"], "rtx4060-box")

    def test_reconnecting_overwrites_the_same_row(self):
        collection = self._stub()
        self._publish("https://first.trycloudflare.com/v1/edit")
        self._publish("https://second.trycloudflare.com/v1/edit")
        self.assertEqual(len(collection.documents), 1, "one row per device")
        stored = collection.documents["rtx4060-box"]
        self.assertEqual(stored["link"], "https://second.trycloudflare.com/v1/edit")

    def test_duplicate_key_race_retries_without_upsert(self):
        collection = self._stub(duplicate=True)
        response = self._publish()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([call[2] for call in collection.calls], [True, False])

    def test_mongodb_outage_is_503_and_never_touches_the_endpoint(self):
        self._stub(fail=True)
        response = self._publish()
        self.assertEqual(response.status_code, 503)
        self.assertIn("registry unavailable", response.json()["detail"])
        # the service is still fully functional
        self.assertEqual(self.client.get("/healthz").json()["status"], "ok")
        self.assertEqual(self.client.get("/v1/models", headers=self.auth).status_code, 200)

    def test_non_https_links_are_refused(self):
        self._stub()
        for bad in ("http://x.trycloudflare.com/v1/edit", "https://", "ftp://a/b"):
            with self.subTest(bad=bad):
                self.assertEqual(self._publish(bad).status_code, 400)

    def test_publish_needs_the_api_key(self):
        self._stub()
        response = self.client.post(
            "/v1/internal/tunnel",
            json={"link": "https://x.trycloudflare.com/v1/edit"},
        )
        self.assertEqual(response.status_code, 401)

    def test_registry_read_reports_what_is_stored(self):
        self._stub()
        self._publish("https://olive-pans.trycloudflare.com/v1/edit")
        body = self.client.get("/v1/registry", headers=self.auth).json()
        self.assertTrue(body["published"])
        self.assertEqual(body["link"], "https://olive-pans.trycloudflare.com/v1/edit")
        self.assertIn("tunnels", body["target"])
        self.assertNotIn("p@", body["target"])

    def test_registry_read_reports_nothing_yet(self):
        self._stub()
        body = self.client.get("/v1/registry", headers=self.auth).json()
        self.assertFalse(body["published"])
        self.assertEqual(body["device"], "rtx4060-box")

class TestRegistryDisabled(ApiTestCase):
    env = {"HOST_NAME": "some-box", "MONGODB": ""}

    def test_publish_is_503_when_unconfigured(self):
        response = self.client.post(
            "/v1/internal/tunnel",
            json={"link": "https://x.trycloudflare.com/v1/edit"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("MONGODB is not set", response.json()["detail"])
        self.assertEqual(self.client.get("/healthz").json()["status"], "ok")

    def test_the_service_works_with_no_registry_at_all(self):
        body = self.post_edit(prompt="x").json()
        self.assertEqual(body["status"], "succeeded")


class TestSlowJobsBecomeAsync(ApiTestCase):
    env = {"SYNC_MAX_WAIT": "0.3"}

    def test_a_slow_job_returns_202_and_can_be_polled(self):
        self.state.history_delay = 2  # first two /history calls come back empty
        response = self.post_edit(prompt="x")
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertIn("status_url", body)
        self.assertEqual(response.headers["location"], f"/v1/jobs/{body['id']}")

        deadline = time.time() + 10
        status = {}
        while time.time() < deadline:
            status = self.client.get(f"/v1/jobs/{body['id']}", headers=self.auth).json()
            if status["status"] == "succeeded":
                break
            time.sleep(0.1)
        self.assertEqual(status["status"], "succeeded")
        image = self.client.get(f"/v1/jobs/{body['id']}/image/0", headers=self.auth)
        self.assertEqual(image.content, PNG_1PX)

    def test_wait_false_is_always_async(self):
        response = self.post_edit(prompt="x", wait="false")
        self.assertEqual(response.status_code, 202)
        self.assertIn("id", response.json())


if __name__ == "__main__":
    unittest.main()
