import tempfile
import unittest
from pathlib import Path

import qwen21


class Qwen21Tests(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        self.assertEqual(
            qwen21.parse_nvidia_smi("NVIDIA GeForce RTX 2080 SUPER, 8192\n"),
            ("NVIDIA GeForce RTX 2080 SUPER", 8192),
        )

    def test_parse_nvidia_smi_bad_input(self):
        self.assertIsNone(qwen21.parse_nvidia_smi(""))
        self.assertIsNone(qwen21.parse_nvidia_smi("GPU only"))

    def test_human_bytes(self):
        self.assertEqual(qwen21.human_bytes(1024), "1.0 KB")
        self.assertEqual(qwen21.human_bytes(1024 * 1024 * 4), "4.0 MB")

    def test_normalize_comfy_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("# test", encoding="utf-8")
            self.assertEqual(qwen21.normalize_comfy(root), root.resolve())

    def test_profile_contains_dedicated_vae(self):
        names = [name for _, name, _ in qwen21.PROFILE_8GB]
        self.assertIn("qwen_image_2.1_vae_bf16.safetensors", names)


if __name__ == "__main__":
    unittest.main()
