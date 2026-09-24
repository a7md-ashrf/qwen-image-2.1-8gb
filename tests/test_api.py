import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from fastapi.testclient import TestClient
    from PIL import Image

    import api

    HAS_API_DEPS = True
except ImportError:
    HAS_API_DEPS = False


@unittest.skipUnless(HAS_API_DEPS, "api dependencies not installed")
class ApiHelperTests(unittest.TestCase):
    def test_snap_dim(self):
        self.assertEqual(api.snap_dim(1024), 1024)
        self.assertEqual(api.snap_dim(1000), 992)
        self.assertEqual(api.snap_dim(10), 16)
        self.assertEqual(api.snap_dim(4096), 4096)

    def test_choose_dtype_name(self):
        self.assertEqual(api.choose_dtype_name((8, 6)), "bf16")
        self.assertEqual(api.choose_dtype_name((12, 0)), "bf16")
        self.assertEqual(api.choose_dtype_name((7, 5)), "fp16")
        self.assertEqual(api.choose_dtype_name(None), "fp16")

    def test_resolve_dtype_name_explicit_wins(self):
        self.assertEqual(api.resolve_dtype_name("fp16", (8, 6)), "fp16")
        self.assertEqual(api.resolve_dtype_name("bf16", (7, 5)), "bf16")
        self.assertEqual(api.resolve_dtype_name("auto", (8, 0)), "bf16")
        self.assertEqual(api.resolve_dtype_name("auto", (7, 5)), "fp16")

    def test_resolve_gguf_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            self.assertEqual(
                api.resolve_gguf(path, download_enabled=False),
                path.resolve(),
            )

    def test_resolve_gguf_missing_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            with self.assertRaises(FileNotFoundError):
                api.resolve_gguf(path, download_enabled=False)

    def test_gguf_profile_entry_exists(self):
        self.assertTrue(api.GGUF_NAME.endswith(".gguf"))
        self.assertTrue(api.GGUF_URL.startswith("https://"))


@unittest.skipUnless(HAS_API_DEPS, "api dependencies not installed")
class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self._old_pipe = api._pipe
        api._pipe = None

    def tearDown(self):
        api._pipe = self._old_pipe

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model_loaded", payload)

    def test_edit_returns_png(self):
        source = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(source, format="PNG")

        def fake_run_edit(
            pipe,
            image,
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
            width,
            height,
            seed,
        ):
            self.assertEqual(prompt, "make it blue")
            self.assertEqual(steps, 25)
            self.assertEqual(seed, 42)
            return Image.new("RGB", (width, height), "blue")

        with (
            mock.patch.object(api, "get_pipeline", return_value=object()),
            mock.patch.object(api, "run_edit", side_effect=fake_run_edit),
        ):
            response = self.client.post(
                "/edit",
                files={"image": ("input.png", source.getvalue(), "image/png")},
                data={"prompt": "make it blue", "seed": "42"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["x-seed"], "42")
        self.assertEqual(response.headers["x-width"], "1024")
        self.assertEqual(response.headers["x-height"], "1024")
        result = Image.open(io.BytesIO(response.content))
        self.assertEqual(result.size, (1024, 1024))

    def test_edit_generates_seed_when_missing(self):
        source = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(source, format="PNG")

        with (
            mock.patch.object(api, "get_pipeline", return_value=object()),
            mock.patch.object(
                api, "run_edit", return_value=Image.new("RGB", (256, 256), "blue")
            ),
        ):
            response = self.client.post(
                "/edit",
                files={"image": ("input.png", source.getvalue(), "image/png")},
                data={"prompt": "x", "width": "256", "height": "256"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("x-seed", response.headers)

    def test_edit_rejects_empty_prompt(self):
        source = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(source, format="PNG")
        response = self.client.post(
            "/edit",
            files={"image": ("input.png", source.getvalue(), "image/png")},
            data={"prompt": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_edit_rejects_bad_steps(self):
        source = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(source, format="PNG")
        response = self.client.post(
            "/edit",
            files={"image": ("input.png", source.getvalue(), "image/png")},
            data={"prompt": "x", "steps": "0"},
        )
        self.assertEqual(response.status_code, 422)

    def test_edit_rejects_undecodable_image(self):
        response = self.client.post(
            "/edit",
            files={"image": ("input.png", b"not-a-png", "image/png")},
            data={"prompt": "x"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
