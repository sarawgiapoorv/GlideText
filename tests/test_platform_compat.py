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

    def test_macos_app_classification(self):
        from context_snapshot import AppCategory, classify_application

        test_cases = [
            ("Code", "com.microsoft.VSCode", AppCategory.IDE_CODE_EDITOR, "VS Code"),
            ("Cursor", "com.todesktop.230313mzl4w4u92", AppCategory.IDE_CODE_EDITOR, "Cursor"),
            ("Terminal", "com.apple.Terminal", AppCategory.TERMINAL, "Terminal"),
            ("iTerm2", "com.googlecode.iterm2", AppCategory.TERMINAL, "iTerm2"),
            ("Warp", "dev.warp.warp-stable", AppCategory.TERMINAL, "Warp"),
            ("Slack", "com.tinyspeck.slackmacgap", AppCategory.SLACK_CHAT, "Slack"),
            ("Microsoft Teams", "com.microsoft.teams", AppCategory.TEAMS_CHAT, "Teams"),
            ("Microsoft Outlook", "com.microsoft.outlook", AppCategory.EMAIL, "Outlook"),
            ("Notes", "com.apple.Notes", AppCategory.DOCUMENT_EDITOR, "Notes"),
            ("Safari", "com.apple.Safari", AppCategory.BROWSER_GENERAL, "Safari"),
            ("Google Chrome", "com.google.Chrome", AppCategory.BROWSER_GENERAL, "Chrome"),
            ("Arc", "company.thebrowser.browser", AppCategory.BROWSER_GENERAL, "Arc"),
        ]

        for exe_name, bundle_id, expected_cat, expected_name_sub in test_cases:
            cat, name, sensitive = classify_application(
                exe_name=exe_name,
                window_title="Sample Document",
                app_hint=bundle_id,
            )
            self.assertEqual(cat, expected_cat, f"Mismatch category for {exe_name} / {bundle_id}")
            self.assertIn(expected_name_sub, name, f"Mismatch name for {exe_name} / {bundle_id}")
            self.assertFalse(sensitive, f"Unexpected sensitivity for {exe_name} / {bundle_id}")

    def test_vs_code_integrated_terminal_classification(self):
        from context_snapshot import AppCategory, classify_application

        cat, name, sensitive = classify_application(
            exe_name="Code",
            window_title="bash - Terminal 1",
            app_hint="com.microsoft.VSCode",
        )
        self.assertEqual(cat, AppCategory.TERMINAL)
        self.assertIn("Terminal", name)
        self.assertFalse(sensitive)

    def test_macos_sensitive_app_deny_list(self):
        from context_snapshot import classify_application, is_sensitive_window

        sensitive_cases = [
            ("1Password", "com.1password.1password"),
            ("Bitwarden", "com.bitwarden.desktop"),
            ("Dashlane", "com.dashlane.dashlane"),
            ("Keychain Access", "com.apple.keychainaccess"),
            ("SecurityAgent", "com.apple.securityagent"),
            ("loginwindow", "com.apple.loginwindow"),
            ("coreauthd", "com.apple.coreauthd"),
            ("pinentry-mac", "org.gpgtools.pinentry-mac"),
            ("KeePassXC", "org.keepassxc.keepassxc"),
        ]

        for exe_name, bundle_id in sensitive_cases:
            self.assertTrue(
                is_sensitive_window(exe_name=exe_name, app_hint=bundle_id),
                f"Expected sensitive window for {exe_name} / {bundle_id}",
            )
            _, _, sensitive = classify_application(exe_name=exe_name, app_hint=bundle_id)
            self.assertTrue(
                sensitive,
                f"Expected classify_application sensitive=True for {exe_name} / {bundle_id}",
            )

    def test_secure_input_refusal_in_injector(self):
        from text_injector import TextInjector

        injector = TextInjector()
        with patch("platform_compat.is_secure_input_enabled", return_value=True):
            res = injector.inject("secret text", target_hwnd=(1234, 5678))
            self.assertFalse(res.success)
            self.assertEqual(res.method, "refused_secure_input")
            self.assertIn("Secure Input", res.error)

    def test_macos_window_token_comparison_and_restoration(self):
        mac_backend = MacOSBackend()
        token = (9999, 101)  # (pid, window_id)

        mock_front = MagicMock()
        mock_front.processIdentifier.return_value = 9999

        mock_appkit = MagicMock()
        mock_appkit.NSWorkspace.sharedWorkspace.return_value.frontmostApplication.return_value = mock_front

        with patch("sys.platform", "darwin"), \
             patch("platform_compat.is_secure_input_enabled", return_value=False), \
             patch.dict("sys.modules", {"AppKit": mock_appkit}):
            verified = mac_backend.verify_and_restore_target_window(token)
            self.assertTrue(verified)

        # When frontmost app PID differs, tries reactivation
        mock_diff_front = MagicMock()
        mock_diff_front.processIdentifier.return_value = 1111

        mock_running_app = MagicMock()
        mock_appkit_reactivate = MagicMock()
        mock_appkit_reactivate.NSWorkspace.sharedWorkspace.return_value.frontmostApplication.side_effect = [mock_diff_front, mock_front]
        mock_appkit_reactivate.NSRunningApplication.runningApplicationWithProcessIdentifier_.return_value = mock_running_app

        with patch("sys.platform", "darwin"), \
             patch("platform_compat.is_secure_input_enabled", return_value=False), \
             patch.dict("sys.modules", {"AppKit": mock_appkit_reactivate}):
            restored = mac_backend.verify_and_restore_target_window(token)
            self.assertTrue(restored)
            mock_running_app.activateWithOptions_.assert_called_once()

    def test_logical_action_combos_translation(self):
        """Logical editing actions must translate to Ctrl on Windows and Cmd/Option on macOS."""
        from platform_compat.input_backend import (
            LogicalAction,
            get_action_combo,
            get_word_modifier,
        )

        with patch("sys.platform", "win32"):
            self.assertEqual(get_action_combo(LogicalAction.COPY), "ctrl+c")
            self.assertEqual(get_action_combo(LogicalAction.PASTE), "ctrl+v")
            self.assertEqual(get_action_combo(LogicalAction.UNDO), "ctrl+z")
            self.assertEqual(get_action_combo(LogicalAction.SELECT_ALL), "ctrl+a")
            self.assertEqual(get_action_combo(LogicalAction.SELECT_WORD_LEFT), "ctrl+shift+left")
            self.assertEqual(get_word_modifier(), "ctrl")

        with patch("sys.platform", "darwin"):
            self.assertEqual(get_action_combo(LogicalAction.COPY), "cmd+c")
            self.assertEqual(get_action_combo(LogicalAction.PASTE), "cmd+v")
            self.assertEqual(get_action_combo(LogicalAction.UNDO), "cmd+z")
            self.assertEqual(get_action_combo(LogicalAction.SELECT_ALL), "cmd+a")
            self.assertEqual(get_action_combo(LogicalAction.SELECT_WORD_LEFT), "option+shift+left")
            self.assertEqual(get_word_modifier(), "option")

    def test_macos_input_backend_synthetic_keystroke(self):
        """MacOSInputBackend must generate Quartz CGEvents with correct keycode and flag mask."""
        from platform_compat.input_backend import MacOSInputBackend

        mock_quartz = MagicMock()
        mock_event = MagicMock()
        mock_quartz.CGEventCreateKeyboardEvent.return_value = mock_event

        backend = MacOSInputBackend()
        with patch.dict("sys.modules", {"Quartz": mock_quartz}):
            # Press and release Cmd+V (keycode for 'v' is 9, cmd flag is 0x00100000)
            backend.press_and_release("cmd+v")
            self.assertTrue(mock_quartz.CGEventCreateKeyboardEvent.called)
            mock_quartz.CGEventSetFlags.assert_called()
            mock_quartz.CGEventPost.assert_called()

    def test_macos_input_backend_write_unicode(self):
        """MacOSInputBackend.write must use CGEventKeyboardSetUnicodeString for layout independence."""
        from platform_compat.input_backend import MacOSInputBackend

        mock_quartz = MagicMock()
        mock_event = MagicMock()
        mock_quartz.CGEventCreateKeyboardEvent.return_value = mock_event

        backend = MacOSInputBackend()
        with patch.dict("sys.modules", {"Quartz": mock_quartz}):
            backend.write("Hi", delay=0)
            self.assertEqual(mock_quartz.CGEventKeyboardSetUnicodeString.call_count, 2)

    def test_macos_input_backend_pynput_listener_dispatch(self):
        """MacOSInputBackend must dispatch push-to-talk holds and multi-key combos."""
        from platform_compat.input_backend import MacOSInputBackend

        backend = MacOSInputBackend()

        press_called = []
        release_called = []
        hotkey_called = []

        backend.on_press_key("right option", lambda ev: press_called.append(True))
        backend.on_release_key("right option", lambda ev: release_called.append(True))
        backend.add_hotkey("ctrl+shift+a", lambda: hotkey_called.append(True))

        mock_alt_r = MagicMock()
        mock_alt_r.name = "alt_r"
        del mock_alt_r.char

        mock_ctrl = MagicMock()
        mock_ctrl.name = "ctrl_l"
        del mock_ctrl.char

        mock_shift = MagicMock()
        mock_shift.name = "shift_l"
        del mock_shift.char

        mock_a = MagicMock()
        mock_a.char = "a"
        mock_a.name = None

        # Simulate Right Option press & release
        backend._on_pynput_press(mock_alt_r)
        self.assertEqual(len(press_called), 1)

        backend._on_pynput_release(mock_alt_r)
        self.assertEqual(len(release_called), 1)

        # Simulate Ctrl+Shift+A combo
        backend._on_pynput_press(mock_ctrl)
        backend._on_pynput_press(mock_shift)
        self.assertEqual(len(hotkey_called), 0)

        backend._on_pynput_press(mock_a)
        self.assertEqual(len(hotkey_called), 1)

        backend.unhook_all()
        self.assertEqual(len(backend._press_callbacks), 0)

    def test_permissions_check_diagnostics_tool(self):
        """Test permissions diagnostic tool on non-macOS and mocked macOS."""
        from platform_compat.permissions_check import run_diagnostics

        # On Windows, diagnostics returns 0
        with patch("sys.platform", "win32"):
            res = run_diagnostics(prompt=False)
            self.assertEqual(res, 0)

        # On macOS with all granted
        with patch("sys.platform", "darwin"), \
             patch("platform_compat.check_microphone_permission", return_value=(True, "Granted")), \
             patch("platform_compat.check_accessibility_permission", return_value=(True, "Granted")), \
             patch("platform_compat.check_input_monitoring_permission", return_value=(True, "Granted")):
            res = run_diagnostics(prompt=False)
            self.assertEqual(res, 0)

        # On macOS with missing permission
        with patch("sys.platform", "darwin"), \
             patch("platform_compat.check_microphone_permission", return_value=(True, "Granted")), \
             patch("platform_compat.check_accessibility_permission", return_value=(False, "Missing")), \
             patch("platform_compat.check_input_monitoring_permission", return_value=(True, "Granted")):
            res = run_diagnostics(prompt=False)
            self.assertEqual(res, 1)

    def test_local_llm_ollama_paths_and_spawn_macos(self):
        """LocalLLMEngine on macOS must discover Homebrew Ollama and spawn with start_new_session."""
        from local_llm import LocalLLMEngine

        with patch("sys.platform", "darwin"), \
             patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=lambda p: p == "/opt/homebrew/bin/ollama"):
            bin_path = LocalLLMEngine._find_ollama_executable()
            self.assertEqual(bin_path, "/opt/homebrew/bin/ollama")

        engine = LocalLLMEngine()
        with patch("sys.platform", "darwin"), \
             patch.object(engine, "is_server_running", side_effect=[False, False, True]), \
             patch.object(engine, "_find_ollama_executable", return_value="/opt/homebrew/bin/ollama"), \
             patch("local_llm.subprocess.Popen") as mock_popen:
            success = engine.ensure_server_running(timeout_seconds=2.0)
            self.assertTrue(success)
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            self.assertTrue(kwargs.get("start_new_session"))
            self.assertNotIn("creationflags", kwargs)

    def test_freellm_manager_npm_search_and_shutdown_macos(self):
        """freellm_manager on macOS must search Homebrew paths and killpg process group."""
        import freellm_manager

        target_bin = os.path.normpath("/opt/homebrew/bin/npm")
        with patch("sys.platform", "darwin"), \
             patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=lambda p: os.path.normpath(p) == target_bin), \
             patch("os.access", return_value=True):
            npm_path = freellm_manager._find_npm()
            self.assertEqual(os.path.normpath(npm_path), target_bin)

        mock_proc = MagicMock()
        mock_proc.pid = 4321
        with patch("sys.platform", "darwin"), \
             patch.object(freellm_manager, "_process", mock_proc), \
             patch("os.getpgid", return_value=4321, create=True), \
             patch("os.killpg", create=True) as mock_killpg:
            freellm_manager.shutdown()
            mock_killpg.assert_called_once()
            pgid, sig = mock_killpg.call_args[0]
            self.assertEqual(pgid, 4321)

    def test_dock_icon_policy_macos(self):
        """platform_compat on macOS must set NSApplication activation policy for Dock icon."""
        import platform_compat
        from platform_compat.macos_backend import MacOSBackend

        mock_appkit = MagicMock()
        mock_app = MagicMock()
        mock_appkit.NSApplication.sharedApplication.return_value = mock_app
        mock_appkit.NSApplicationActivationPolicyAccessory = 1
        mock_appkit.NSApplicationActivationPolicyRegular = 0

        backend = MacOSBackend()
        with patch("sys.platform", "darwin"), \
             patch.dict("sys.modules", {"AppKit": mock_appkit}):
            backend.hide_dock_icon()
            mock_app.setActivationPolicy_.assert_called_with(1)

            backend.show_dock_icon()
            mock_app.setActivationPolicy_.assert_called_with(0)

    def test_autoboot_status_and_toggle_macos(self):
        """MacOSBackend must manage LaunchAgent plist without crashing."""
        from platform_compat.macos_backend import MacOSBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = MacOSBackend()
            test_plist = os.path.join(tmp_dir, "com.glidetext.app.plist")
            backend._plist_path = test_plist

            self.assertFalse(backend.is_auto_boot_enabled())

            # Enable autoboot
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                enabled = backend.set_auto_boot(True)
                self.assertTrue(enabled)
                self.assertTrue(os.path.isfile(test_plist))
                self.assertTrue(backend.is_auto_boot_enabled())

                # Disable autoboot
                disabled = backend.set_auto_boot(False)
                self.assertTrue(disabled)
                self.assertFalse(os.path.isfile(test_plist))
                self.assertFalse(backend.is_auto_boot_enabled())


if __name__ == "__main__":
    unittest.main()
