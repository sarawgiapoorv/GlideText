"""
tests/test_platform_compat.py -- Tests for platform abstraction layer and compatibility helpers.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import platform_compat
from platform_compat.base import FallbackBackend, PlatformBackend
from platform_compat.macos_backend import MacOSBackend
from platform_compat.windows_backend import WindowsBackend


class TestPlatformCompat(unittest.TestCase):
    """Test platform detection, backend instantiation, and path resolution."""

    def test_current_platform_matches_sys_platform(self):
        backend = platform_compat.get_platform_backend()
        self.assertIsInstance(backend, PlatformBackend)
        if sys.platform == "win32":
            self.assertTrue(platform_compat.IS_WINDOWS)
            self.assertFalse(platform_compat.IS_MACOS)
            self.assertEqual(platform_compat.get_platform(), "windows")
            self.assertIsInstance(backend, WindowsBackend)
        elif sys.platform == "darwin":
            self.assertTrue(platform_compat.IS_MACOS)
            self.assertFalse(platform_compat.IS_WINDOWS)
            self.assertEqual(platform_compat.get_platform(), "macos")
            self.assertIsInstance(backend, MacOSBackend)

    def test_windows_paths(self):
        win_backend = WindowsBackend()
        with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\MockAppData"}):
            app_dir = win_backend.get_app_data_dir()
            self.assertEqual(app_dir, os.path.join("C:\\MockAppData", "GlideText"))
            logs_dir = win_backend.get_logs_dir()
            self.assertEqual(logs_dir, os.path.join("C:\\MockAppData", "GlideText", "logs"))
            self.assertEqual(win_backend.get_font_family(), "Segoe UI")

    def test_macos_paths(self):
        mac_backend = MacOSBackend()
        with patch("os.path.expanduser", side_effect=lambda p: p.replace("~", "/Users/mockuser")):
            app_dir = mac_backend.get_app_data_dir()
            self.assertEqual(os.path.normpath(app_dir), os.path.normpath("/Users/mockuser/Library/Application Support/GlideText"))
            logs_dir = mac_backend.get_logs_dir()
            self.assertEqual(os.path.normpath(logs_dir), os.path.normpath("/Users/mockuser/Library/Logs/GlideText"))
            self.assertEqual(mac_backend.get_font_family(), "Helvetica Neue")

    def test_fallback_backend_paths(self):
        fallback = FallbackBackend()
        with patch("os.path.expanduser", side_effect=lambda p: p.replace("~", "/home/fallback")):
            app_dir = fallback.get_app_data_dir()
            self.assertEqual(os.path.normpath(app_dir), os.path.normpath("/home/fallback/.glidetext"))
            logs_dir = fallback.get_logs_dir()
            self.assertEqual(os.path.normpath(logs_dir), os.path.normpath("/home/fallback/.glidetext/logs"))

    def test_macos_launch_agent_plist_generation_and_removal(self):
        mac_backend = MacOSBackend()
        temp_plist = os.path.join(mac_backend.get_temp_dir(), "test_com.glidetext.app.plist")
        mac_backend._plist_path = temp_plist

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ok = mac_backend.set_auto_boot(True)
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(temp_plist))

            # Verify plist structure
            import plistlib
            with open(temp_plist, "rb") as fp:
                data = plistlib.load(fp)
            self.assertEqual(data["Label"], "com.glidetext.app")
            self.assertTrue(data["RunAtLoad"])
            self.assertIn("--silent", data["ProgramArguments"])

            # Verify removal
            del_ok = mac_backend.set_auto_boot(False)
            self.assertTrue(del_ok)
            self.assertFalse(os.path.isfile(temp_plist))

    def test_macos_permissions_graceful_fallbacks_when_no_pyobjc(self):
        mac_backend = MacOSBackend()
        # When pyobjc is not installed (e.g. on Windows or non-Mac environment)
        # Permission checks must degrade gracefully and return clean tuples, never raise
        with patch.dict("sys.modules", {"AVFoundation": None, "ApplicationServices": None, "Quartz": None}):
            mic_ok, mic_msg = mac_backend.check_microphone_permission()
            self.assertIsInstance(mic_ok, bool)
            self.assertIsInstance(mic_msg, str)

            acc_ok, acc_msg = mac_backend.check_accessibility_permission()
            self.assertIsInstance(acc_ok, bool)
            self.assertIsInstance(acc_msg, str)

            inp_ok, inp_msg = mac_backend.check_input_monitoring_permission()
            self.assertIsInstance(inp_ok, bool)
            self.assertIsInstance(inp_msg, str)

    def test_main_required_libs_per_platform(self):
        import main
        # Verify REQUIRED_LIBS contains appropriate platform libs
        if sys.platform == "win32":
            req_names = [pip_name for _, pip_name in main.REQUIRED_LIBS]
            self.assertIn("keyboard", req_names)
            self.assertIn("pycaw", req_names)
            self.assertIn("comtypes", req_names)
            self.assertNotIn("pynput", req_names)
            self.assertNotIn("pyobjc-core", req_names)


if __name__ == "__main__":
    unittest.main()
