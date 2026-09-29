"""Unit tests for the installer side (stdlib only, no torch required).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen21 import cli, comfy_install, env, models, profiles, runner, tunnel
from qwen_api import workflows
from qwen_api.workflows import ModelSpec

SPEC = ModelSpec(
    unet="qwen_image_2.1-Q4_K.gguf",
    clip="qwen3vl_8b_w4a8.safetensors",
    vae="qwen_image_2.1_vae_bf16.safetensors",
    unet_gguf=True,
)


class TestProfiles(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        self.assertEqual(
            profiles.parse_nvidia_smi("NVIDIA GeForce RTX 2080 SUPER, 8192\n"),
            ("NVIDIA GeForce RTX 2080 SUPER", 8192),
        )

    def test_parse_nvidia_smi_bad_input(self):
        self.assertIsNone(profiles.parse_nvidia_smi(""))
        self.assertIsNone(profiles.parse_nvidia_smi("GPU only"))

    def test_every_profile_has_the_dedicated_vae(self):
        for name, profile in profiles.PROFILES.items():
            with self.subTest(profile=name):
                vae = profile.model("vae")
                self.assertIsNotNone(vae, "profile has no VAE")
                self.assertIn("2.1_vae", vae.name)

    def test_every_profile_has_a_diffusion_model_and_encoder(self):
        for name, profile in profiles.PROFILES.items():
            with self.subTest(profile=name):
                self.assertIsNotNone(profile.model("diffusion_models"))
                self.assertIsNotNone(profile.model("text_encoders"))

    def test_gguf_profiles_use_the_gguf_loader(self):
        for name, profile in profiles.PROFILES.items():
            with self.subTest(profile=name):
                if profile.model("diffusion_models").name.endswith(".gguf"):
                    self.assertEqual(profile.loader, "gguf")
                    self.assertTrue(profile.needs_gguf_node)
                else:
                    self.assertEqual(profile.loader, "unet")
                    self.assertFalse(profile.needs_gguf_node)

    def test_encoder_stays_off_the_gpu_on_small_machines(self):
        for name in ("nvidia-8gb", "mac-8gb", "mac-8gb-lite"):
            self.assertEqual(profiles.PROFILES[name].encoder_device, "cpu")

    def test_known_model_urls(self):
        """Every model must come from one of the two upstream repositories."""
        allowed = (
            "https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/",
            "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/",
        )
        for profile in profiles.PROFILES.values():
            for model in profile.models:
                with self.subTest(profile=profile.name, model=model.name):
                    self.assertTrue(model.url.startswith(allowed), model.url)
                    self.assertGreater(model.size, 0)

    def test_macs_use_a_smaller_quantisation_than_the_nvidia_profile(self):
        mac = profiles.PROFILES["mac-8gb"].model("diffusion_models")
        nvidia = profiles.PROFILES["nvidia-8gb"].model("diffusion_models")
        self.assertLess(mac.size, nvidia.size)
        self.assertTrue(mac.name.endswith("Q3_K.gguf"))
        self.assertTrue(nvidia.name.endswith("Q4_K.gguf"))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(SystemExit):
            profiles.get_profile("rtx-4090")


class TestModels(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(models.human_bytes(1024), "1.0 KB")
        self.assertEqual(models.human_bytes(1024 * 1024 * 4), "4.0 MB")

    def test_verify_detects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = profiles.VAE
            ok, detail = models.verify(Path(tmp) / model.name, model)
            self.assertFalse(ok)
            self.assertEqual(detail, "missing")

    def test_verify_rejects_truncated_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = profiles.VAE
            target = Path(tmp) / model.name
            target.write_bytes(b"tiny")
            ok, detail = models.verify(target, model)
            self.assertFalse(ok)
            self.assertIn("size", detail)

    def test_verify_accepts_a_plausible_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = profiles.VAE
            target = Path(tmp) / model.name
            target.write_bytes(b"x" * (model.size + 1))
            ok, detail = models.verify(target, model)
            self.assertTrue(ok, detail)


class TestEnv(unittest.TestCase):
    def test_load_env_parses_quotes_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nAPI_KEY=abc\nQUOTED=\"hello world\"\nEMPTY=\nBAD_LINE\n",
                encoding="utf-8",
            )
            parsed = env.load_env(path, override=True)
            self.assertEqual(parsed["API_KEY"], "abc")
            self.assertEqual(parsed["QUOTED"], "hello world")
            self.assertEqual(parsed["EMPTY"], "")
            self.assertNotIn("BAD_LINE", parsed)

    def test_placeholders_are_detected(self):
        self.assertTrue(env.is_placeholder("TODO-run-qwen21-keygen-write"))
        self.assertTrue(env.is_placeholder("sk-xxxxx"))
        self.assertTrue(env.is_placeholder(""))
        self.assertFalse(env.is_placeholder(env.generate_api_key()))

    def test_generated_keys_differ(self):
        self.assertNotEqual(env.generate_api_key(), env.generate_api_key())

    def test_every_required_key_explains_where_to_get_it(self):
        for spec in env.KEYS:
            with self.subTest(key=spec.name):
                self.assertTrue(spec.why)
                self.assertTrue(spec.where)


class TestComfyInstall(unittest.TestCase):
    def test_normalize_accepts_a_portable_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("# test", encoding="utf-8")
            self.assertEqual(comfy_install.normalize_comfy(root), root.resolve())
        with tempfile.TemporaryDirectory() as tmp:
            # ComfyUI_windows_portable/<name>/ComfyUI/main.py
            root = Path(tmp)
            nested = root / "ComfyUI"
            nested.mkdir()
            (nested / "main.py").write_text("# test", encoding="utf-8")
            self.assertEqual(comfy_install.normalize_comfy(root), nested.resolve())

    def test_normalize_rejects_a_directory_without_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(comfy_install.InstallError):
                comfy_install.normalize_comfy(Path(tmp))

    def test_version_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy = Path(tmp)
            (comfy / "comfyui_version.py").write_text('__version__ = "0.37.0"\n', "utf-8")
            self.assertEqual(comfy_install.comfy_version(comfy), (0, 37, 0))
            self.assertTrue(comfy_install.version_ok(comfy))
            (comfy / "comfyui_version.py").write_text('__version__ = "0.36.0"\n', "utf-8")
            self.assertFalse(comfy_install.version_ok(comfy))

    def test_workflows_are_copied_into_comfy(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "workflows"
            source.mkdir()
            (source / "a.json").write_text("{}", encoding="utf-8")
            comfy = root / "ComfyUI"
            comfy.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                written = comfy_install.install_workflows(comfy, root)
            self.assertEqual([p.name for p in written], ["a.json"])
            self.assertTrue((comfy / "user" / "default" / "workflows" / "a.json").is_file())


class TestWorkflows(unittest.TestCase):
    def test_edit_graph_shape(self):
        graph = workflows.build_edit_prompt(SPEC, "make it red", ["in.png"], resolution=768, steps=10)
        self.assertEqual(graph["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], SPEC.unet)
        self.assertEqual(graph["2"]["inputs"]["device"], "cpu")
        encode = graph["30"]["inputs"]
        self.assertEqual(encode["resolution"], 768)
        # the autogrow reference images arrive as a {slot: [node, output]} dict
        self.assertEqual(encode["images"], {"image_1": ["100", 0]})
        self.assertEqual(graph["40"]["inputs"]["steps"], 10)
        self.assertEqual(graph["40"]["inputs"]["latent_image"], ["30", 2])
        self.assertEqual(graph["40"]["inputs"]["denoise"], 1.0)

    def test_generate_graph_uses_an_empty_latent(self):
        graph = workflows.build_generate_prompt(SPEC, "a cat", width=800, height=608)
        self.assertEqual(graph["10"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["10"]["inputs"]["width"], 800)
        self.assertEqual(graph["10"]["inputs"]["height"], 608)
        self.assertNotIn("images", graph["30"]["inputs"])
        self.assertEqual(graph["40"]["inputs"]["latent_image"], ["10", 0])

    def test_dimensions_are_rounded_to_the_vae_grid(self):
        graph = workflows.build_generate_prompt(SPEC, "x", width=1000, height=999)
        self.assertEqual(graph["10"]["inputs"]["width"] % 16, 0)
        self.assertEqual(graph["10"]["inputs"]["height"] % 16, 0)
        self.assertEqual(workflows.build_generate_prompt(SPEC, "x", width=800, height=600)["10"]
                         ["inputs"]["height"], 608)

    def test_safetensors_profile_uses_the_core_loader(self):
        spec = ModelSpec(
            unet="qwen_image_2.1_int8_convrot.safetensors",
            clip="qwen3vl_8b_int8_convrot.safetensors",
            vae=SPEC.vae,
        )
        graph = workflows.build_generate_prompt(spec, "x")
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["weight_dtype"], "default")

    def test_multiple_reference_images_get_distinct_slots(self):
        graph = workflows.build_edit_prompt(SPEC, "swap", ["a.png", "b.png"])
        self.assertEqual(graph["30"]["inputs"]["images"], {"image_1": ["100", 0], "image_2": ["101", 0]})

    def test_seeds_are_explicit_and_optional(self):
        self.assertEqual(workflows.build_generate_prompt(SPEC, "x", seed=7)["40"]["inputs"]["seed"], 7)
        self.assertIsNotNone(workflows.build_generate_prompt(SPEC, "x")["40"]["inputs"]["seed"])

    def test_edit_requires_an_image(self):
        with self.assertRaises(ValueError):
            workflows.build_edit_prompt(SPEC, "x", [])

    def test_too_many_reference_images_are_rejected(self):
        with self.assertRaises(ValueError):
            workflows.build_edit_prompt(SPEC, "x", [f"{i}.png" for i in range(17)])

    def test_graph_is_json_serialisable(self):
        graph = workflows.build_edit_prompt(SPEC, "x", ["a.png"])
        self.assertEqual(json.loads(json.dumps(graph)), graph)

    def test_every_link_points_at_an_existing_node(self):
        for graph in (
            workflows.build_edit_prompt(SPEC, "x", ["a.png", "b.png"]),
            workflows.build_generate_prompt(SPEC, "x"),
        ):
            for node in graph.values():
                for value in node["inputs"].values():
                    links = value.values() if isinstance(value, dict) else [value]
                    for link in links:
                        if isinstance(link, list):
                            self.assertIn(link[0], graph)


class TestApiConfig(unittest.TestCase):
    """Settings come from the environment, so build them the real way."""

    def _load(self, **env):
        import os

        from qwen_api import config

        previous = {k: os.environ.get(k) for k in env}
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            return config.load()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_sync_wait_defaults_under_the_cloudflare_read_timeout(self):
        settings = self._load()
        # Cloudflare's proxy read timeout is 125 s (524). We must give up first.
        self.assertLess(settings.sync_max_wait, 125.0)

    def test_upload_defaults_stay_under_the_tunnel_ceiling(self):
        settings = self._load()
        mb = 1024 * 1024
        # Free/Pro cap a request body at 100 MB, so the default ceiling is 90
        # and the per-file/count defaults must fit inside it.
        self.assertEqual(settings.max_request_bytes, 90 * mb)
        self.assertEqual(settings.max_upload_mb, 20.0)
        self.assertLessEqual(settings.worst_case_request_bytes, settings.max_request_bytes)
        self.assertLessEqual(settings.worst_case_request_bytes, 100 * mb)

    def test_registry_settings_come_from_the_environment(self):
        settings = self._load(HOST_NAME="rtx4060-box", MONGODB="mongodb+srv://u:p@h/db",
                              MONGODB_COLLECTION="boxes")
        self.assertEqual(settings.host_name, "rtx4060-box")
        self.assertEqual(settings.mongodb_uri, "mongodb+srv://u:p@h/db")
        self.assertEqual(settings.mongodb_collection, "boxes")

    def test_host_name_falls_back_to_the_os_hostname(self):
        import socket

        self.assertEqual(self._load(HOST_NAME="").host_name, socket.gethostname())

    def test_mongodb_is_off_until_it_is_configured(self):
        settings = self._load(MONGODB="")
        self.assertEqual(settings.mongodb_uri, "")
        self.assertEqual(settings.mongodb_collection, "devices")


class TestJobs(unittest.TestCase):
    def setUp(self):
        from qwen_api.jobs import JobStore

        self.tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.tmp.name) / "jobs.sqlite3", ttl_hours=1.0)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_jobs_survive_a_store_reopen(self):
        from qwen_api.jobs import SUCCEEDED, JobStore

        job = self.store.create("edit", {"prompt": "x"})
        self.store.update(job, status=SUCCEEDED, images=[{"index": 0, "bytes": 3}])
        reopened = JobStore(Path(self.tmp.name) / "jobs.sqlite3", ttl_hours=1.0)
        try:
            restored = reopened.get(job.id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.status, SUCCEEDED)
            self.assertEqual(len(restored.images), 1)
        finally:
            reopened.close()

    def test_in_flight_jobs_fail_after_a_restart(self):
        from qwen_api.jobs import FAILED, JobStore

        self.store.create("edit")
        reopened = JobStore(Path(self.tmp.name) / "jobs.sqlite3", ttl_hours=1.0)
        try:
            job = reopened.list()[0]
            self.assertEqual(job.status, FAILED)
            self.assertIn("restarted", job.error)
        finally:
            reopened.close()

    def test_sweep_removes_expired_jobs(self):
        import time

        job = self.store.create("edit")
        self.store.update(job)
        job.updated_at = time.time() - 7200  # pretend it finished two hours ago
        self.assertEqual(self.store.sweep(), 1)
        self.assertIsNone(self.store.get(job.id))


class TestTunnel(unittest.TestCase):
    def test_quick_tunnel_command(self):
        cmd = tunnel.build_cmd(Path("/tmp/cloudflared"), "quick", "http://127.0.0.1:8000", None)
        self.assertIn("--url", cmd)
        self.assertIn("http://127.0.0.1:8000", cmd)

    def test_named_tunnel_uses_the_token(self):
        token = "TEST-NOT-A-REAL-TOKEN"
        cmd = tunnel.build_cmd(Path("/tmp/cloudflared"), "named", "http://x", token)
        self.assertIn("run", cmd)
        self.assertIn(token, cmd)

    def test_named_tunnel_without_a_token_explains_itself(self):
        with self.assertRaises(SystemExit) as ctx:
            tunnel.build_cmd(Path("/tmp/cloudflared"), "named", "http://x", None)
        self.assertIn("TUNNEL_TOKEN", str(ctx.exception))

    def test_token_detection_ignores_placeholders(self):
        self.assertIsNone(tunnel.load_token({}))
        self.assertIsNone(tunnel.load_token({"TUNNEL_TOKEN": "TODO-paste-the-token"}))
        # a real tunnel token is the base64 blob Cloudflare prints
        blob = "eyJhIjoiYWJjIiwidCI6ImRlZiJ9"
        self.assertEqual(tunnel.load_token({"TUNNEL_TOKEN": blob}), blob)

    def test_public_url(self):
        self.assertEqual(tunnel.public_url("named", "img.example.com"), "https://img.example.com")
        self.assertEqual(tunnel.public_url("named", None), None)
        self.assertEqual(tunnel.public_url("quick", None, "https://x.trycloudflare.com"),
                         "https://x.trycloudflare.com")

    def test_hostname_is_scraped_from_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "tunnel.log"
            log.write_text(
                "starting\nhttps://brave-words-here.trycloudflare.com\n", encoding="utf-8"
            )
            self.assertEqual(
                tunnel.quick_tunnel_url(log, timeout=1), "https://brave-words-here.trycloudflare.com"
            )


class TestCliSurface(unittest.TestCase):
    def test_every_documented_command_exists(self):
        parser = cli.build_parser()
        actions = [a for a in parser._actions if a.dest == "command"]
        self.assertTrue(actions)
        commands = set(actions[0].choices)
        for expected in ("install", "doctor", "start", "stop", "status", "tunnel", "smoke",
                         "keygen", "export", "models", "logs", "keys", "serve", "comfy"):
            self.assertIn(expected, commands)

    def test_keygen_prints_a_usable_key(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(cli.cmd_keygen(argparse_namespace()), 0)
        printed = buffer.getvalue().splitlines()[0]
        self.assertTrue(printed.startswith("qk-"))
        self.assertGreater(len(printed), 20)

    def test_sample_png_is_a_real_png(self):
        blob = cli._sample_png()
        self.assertTrue(blob.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", blob[:32])
        self.assertGreater(len(blob), 100)


class TestComfyResponses(unittest.TestCase):
    """ComfyUI double-encodes 400 bodies; both clients must unwrap them."""

    def test_decode_unwraps_a_json_string(self):
        body = json.dumps(json.dumps({"error": {"message": "nope"}}))
        self.assertEqual(cli._decode(body), {"error": {"message": "nope"}})

    def test_decode_passes_through_plain_json(self):
        self.assertEqual(cli._decode('{"a": 1}'), {"a": 1})

    def test_decode_survives_garbage(self):
        self.assertEqual(cli._decode("not json at all"), "not json at all")

    def test_classify_treats_only_missing_models_as_blocked(self):
        blocked = {"node_errors": {"1": {"errors": [{"type": "value_not_in_list",
                                                    "message": "Value not in list"}]}}}
        verdict, messages = cli._classify(400, blocked)
        self.assertEqual(verdict, "blocked")
        self.assertTrue(any("value_not_in_list" in m for m in messages))

    def test_classify_flags_real_graph_problems(self):
        broken = {"node_errors": {"30": {"errors": [
            {"type": "custom_validation_failed", "message": "images slot is wrong"}]}}}
        verdict, messages = cli._classify(400, broken)
        self.assertEqual(verdict, "invalid")
        self.assertIn("images slot is wrong", messages[0])

    def test_classify_accepts_a_queued_prompt(self):
        verdict, messages = cli._classify(200, {"prompt_id": "abc", "number": 1})
        self.assertEqual(verdict, "accepted")
        self.assertEqual(messages, [])

    def test_classify_rejects_a_detail_free_error(self):
        verdict, messages = cli._classify(500, "boom")
        self.assertEqual(verdict, "invalid")
        self.assertTrue(messages)


class TestNoSecretsCommitted(unittest.TestCase):
    """The guard that keeps credentials out of git, tested against itself."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import check_secrets

        self.check = check_secrets

    def test_this_repository_passes(self):
        self.assertEqual(self.check.main(), 0)

    @staticmethod
    def _uri(user: str, password: str, host: str = "cluster0.mongodb.net") -> str:
        # Assembled from parts on purpose: see check_secrets.py - a literal here
        # would be a real leak and the guard would (correctly) fail on this file.
        return f"mongodb+srv://{user}:{password}" + f"@{host}/tunnels"

    def test_a_real_uri_in_a_tracked_file_is_caught(self):
        secret = self._uri("realuser", "realpassword123")
        problems = self.check.scan(secret, "leak.md", {"MONGODB": secret})
        self.assertTrue(any("MONGODB" in p for p in problems), problems)
        self.assertTrue(any("non-placeholder" in p for p in problems), problems)

    def test_a_verbatim_secret_is_caught_even_without_a_uri(self):
        key = "qk-" + "abcdefghijklmnopqrstuvwxyz012345"
        problems = self.check.scan(f'token = "{key}"', "leak.py", {"API_KEY": key})
        self.assertTrue(any("API_KEY" in p for p in problems), problems)

    def test_placeholder_credentials_are_allowed(self):
        for uri in (self._uri("u", "p", "h") + "",
                    self._uri("USER", "PASSWORD", "cluster.example.net"),
                    self._uri("user", "password", "cluster.example.net"),
                    self._uri("user", "TODO", "cluster.example.net")):
            with self.subTest(uri=uri):
                self.assertEqual(self.check.scan(uri, "docs.md", {}), [])

    def test_a_short_secret_is_not_treated_as_a_secret(self):
        # 8 characters is the floor; below that, matches are noise.
        self.assertNotIn("x", self.check.local_secrets())

    def test_redaction_never_leaks_the_password(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from qwen_api.registry import redact

        uri = self._uri("user", "supersecret")
        redacted = redact(uri)
        self.assertNotIn("supersecret", redacted)
        self.assertIn("cluster0.mongodb.net", redacted)


class TestRegistryLinks(unittest.TestCase):
    @staticmethod
    def _uri(user: str, password: str, host: str) -> str:
        return f"mongodb+srv://{user}:{password}" + f"@{host}/tunnels"

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from qwen_api import registry

        self.registry = registry

    def test_only_https_links_are_accepted(self):
        self.assertEqual(
            self.registry.validate_link("https://x.trycloudflare.com/v1/edit"),
            "https://x.trycloudflare.com/v1/edit",
        )
        for bad in ("http://x.trycloudflare.com", "https://", "", "   ",
                    "https://x.com/\r\nX-Evil: 1", "https://" + "a" * 600):
            with self.subTest(bad=bad[:30]):
                with self.assertRaises(ValueError):
                    self.registry.validate_link(bad)

    def test_database_comes_from_the_uri_path(self):
        self.assertEqual(
            self.registry.Registry("mongodb+srv://u:p@h/tunnels", "devices", "d").device_database,
            "tunnels",
        )
        self.assertEqual(
            self.registry.Registry("mongodb+srv://u:p@h/", "devices", "d").device_database,
            "test",
        )

    def test_unconfigured_registry_refuses_publish_instead_of_connecting(self):
        import asyncio

        entry = self.registry.Registry("", "devices", "macm3")
        self.assertFalse(entry.configured)
        with self.assertRaises(self.registry.RegistryError):
            asyncio.run(entry.publish("https://x.trycloudflare.com/v1/edit"))
        self.assertIn("not configured", entry.target)

    def test_target_never_contains_credentials(self):
        entry = self.registry.Registry(self._uri("u", "hunter2", "h"), "devices", "d")
        self.assertNotIn("hunter2", entry.target)
        self.assertIn("tunnels", entry.target)


class TestRunner(unittest.TestCase):
    def test_pids_and_logs_live_under_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = runner.Runner(Path(tmp))
            self.assertTrue(r.pids.is_dir())
            self.assertTrue(r.logs.is_dir())
            self.assertTrue(str(r.log_file("api")).endswith("api.log"))
            self.assertIsNone(r.pid("nothing-here"))
            self.assertEqual(r.running(), {})

    def test_tail_of_a_missing_log_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = runner.Runner(Path(tmp))
            self.assertIn("no log", r.tail("comfy"))


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


if __name__ == "__main__":
    unittest.main()
