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

    def __init__(self) -> None:
        self.history_calls = 0
        self.history_delay = 0
        self.fail_with: dict | None = None
        self.last_prompt: dict = {}
        self.prompts: list[dict] = []
        self.uploads = 0


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
        if path == "/system_stats":
            return self._send({"devices": [{"name": "Fake GPU", "vram_total": 8 * 2**30}]})
        if path.startswith("/models/"):
            return self._send(MODELS.get(path.split("/")[-1], []))
        if path.startswith("/object_info/"):
            return self._send({path.split("/")[-1]: {}})
        if path.startswith("/history/"):
            self.state.history_calls += 1
            if self.state.history_calls <= self.state.history_delay:
                return self._send({})
            return self._send(
                {
                    path.split("/")[-1]: {
                        "status": {"status_str": "success", "messages": []},
                        "outputs": {
                            "60": {
                                "images": [
                                    {
                                        "filename": "out.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
        if path == "/view":
            return self._send(PNG_1PX, "image/png")
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
    "COMFY_URL",
    "API_KEY",
    "QWEN_OUTPUT_DIR",
    "SYNC_MAX_WAIT",
    "RATE_LIMIT_PER_MIN",
    "MAX_QUEUE",
    "MAX_IMAGES",
    "ALLOW_ANONYMOUS",
    "QWEN_UNET",
    "QWEN_CLIP",
    "QWEN_VAE",
    "QWEN_ENCODER_DEVICE",
    "PUBLIC_BASE_URL",
)


def load_app(comfy_url: str, output_dir: str, **env: str):
    """Import a fresh copy of the app with exactly this environment."""
    for module in [m for m in list(sys.modules) if m.startswith("qwen_api")]:
        del sys.modules[module]
    for key in RESET_KEYS:
        os.environ.pop(key, None)
    os.environ.update(
        {
            "COMFY_URL": comfy_url,
            "API_KEY": "test-key",
            "QWEN_OUTPUT_DIR": output_dir,
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
        cls.state = FakeState()
        FakeComfyUI.state = cls.state
        cls.module = load_app(cls.comfy_url, cls.tmp.name, **cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def setUp(self):
        FakeComfyUI.state = self.state.__class__()
        self.state = FakeComfyUI.state
        # A per-test output dir: the job database is a file, and a shared one
        # would leak jobs between tests.
        self.tmp = tempfile.TemporaryDirectory()
        self.module = load_app(self.comfy_url, self.tmp.name, **self.env)
        # Entering the context manager starts the lifespan and keeps ONE event
        # loop for the whole test, so background jobs survive between requests
        # exactly as they do under uvicorn.
        self._client = TestClient(self.module.app)
        self.client = self._client.__enter__()

    def tearDown(self):
        self._client.__exit__(None, None, None)
        self.tmp.cleanup()

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
        module = load_app(self.comfy_url, self.tmp.name, MAX_QUEUE="0", **self.env)
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
        module = load_app(self.comfy_url, self.tmp.name, RATE_LIMIT_PER_MIN="1", **self.env)
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
