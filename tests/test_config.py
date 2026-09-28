"""
tests/test_config.py
Tests for configuration parsing and persistence:
- KEY=VALUE parsing with DEVICE_INDEX=auto
- Legacy line 0 device index backwards compatibility
- WHISPER_MODEL and WHISPER_LANGUAGE settings
- Preservation of other custom lines (e.g. FREELLMAPI_DIR)
"""

import unittest
import tempfile
import os
from unittest.mock import patch

import gui_app
from ai_brain import _read_app_config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.txt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_config_with_device_index_auto(self):
        """DEVICE_INDEX=auto should be parsed as device_index=None without ValueError."""
        content = "DEVICE_INDEX=auto\nWHISPER_MODEL=base\nWHISPER_LANGUAGE=auto\nFREELLMAPI_DIR=C:\\test\n"
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(content)

        with patch("gui_app.os.path.join", return_value=self.config_path):
            cfg = gui_app._read_config()
            self.assertIsNone(cfg["device_index"])
            self.assertEqual(cfg["whisper_model"], "base")
            self.assertEqual(cfg["whisper_language"], "auto")
            self.assertEqual(cfg["freellmapi_dir"], "C:\\test")

    def test_read_config_legacy_line0(self):
        """Legacy line 0 integer should be parsed properly."""
        content = "2\nFREELLMAPI_DIR=C:\\my_freellm\n"
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(content)

        with patch("gui_app.os.path.join", return_value=self.config_path):
            cfg = gui_app._read_config()
            self.assertEqual(cfg["device_index"], 2)
            self.assertEqual(cfg["freellmapi_dir"], "C:\\my_freellm")

    def test_read_config_legacy_line0_auto(self):
        """Legacy line 0 with 'auto' should parse device_index as None."""
        content = "auto\nFREELLMAPI_DIR=C:\\my_freellm\n"
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(content)

        with patch("gui_app.os.path.join", return_value=self.config_path):
            cfg = gui_app._read_config()
            self.assertIsNone(cfg["device_index"])

    def test_write_config_preserves_existing_keys(self):
        """Writing config should update device/whisper settings while preserving FREELLMAPI_DIR."""
        initial = "0\nFREELLMAPI_DIR=C:\\Users\\test\\freellmapi\nCUSTOM_SETTING=123\n"
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(initial)

        with patch("gui_app.os.path.join", return_value=self.config_path), \
             patch("gui_app.HAS_KEYRING", False):
            gui_app._write_config(
                api_key="test_key",
                device_index=1,
                whisper_model="small",
                whisper_language="hi",
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            saved_content = f.read()

        self.assertIn("FREELLMAPI_DIR=C:\\Users\\test\\freellmapi", saved_content)
        self.assertIn("CUSTOM_SETTING=123", saved_content)
        self.assertIn("WHISPER_MODEL=small", saved_content)
        self.assertIn("WHISPER_LANGUAGE=hi", saved_content)

    def test_ai_brain_read_app_config(self):
        """ai_brain._read_app_config should extract uppercase dict correctly."""
        content = "0\nFREELLMAPI_DIR=C:\\freellm\nWHISPER_MODEL=medium\n"
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(content)

        with patch("ai_brain.os.path.join", return_value=self.config_path):
            cfg = _read_app_config()
            self.assertEqual(cfg.get("DEVICE_INDEX"), "0")
            self.assertEqual(cfg.get("FREELLMAPI_DIR"), "C:\\freellm")
            self.assertEqual(cfg.get("WHISPER_MODEL"), "medium")


if __name__ == "__main__":
    unittest.main()
