"""
tests/test_text_injector.py
Tests for text injection reliability:
- InjectionResult structured output
- ASCII vs Unicode / Multiline routing
- Target window verification and focus mismatch protection
- Clipboard preservation
"""

import unittest
from unittest.mock import MagicMock, patch
import ctypes

from text_injector import TextInjector, InjectionResult


class TestTextInjector(unittest.TestCase):
    def setUp(self):
        self.injector = TextInjector()

    def test_ascii_single_line_routes_to_keyboard(self):
        """Single-line ASCII text should use keyboard.write for natural typing."""
        with patch("text_injector.keyboard.write") as mock_write, \
             patch.object(self.injector, "verify_and_restore_target_window", return_value=True):
            
            res = self.injector.inject("Hello world! This is a simple test.")
            self.assertIsInstance(res, InjectionResult)
            self.assertTrue(res.success)
            self.assertEqual(res.method, "keyboard")
            mock_write.assert_called_once()

    def test_unicode_hindi_routes_to_clipboard(self):
        """Non-ASCII Unicode text (e.g. Hindi) must route to clipboard paste to avoid garbled text."""
        hindi_text = "नमस्ते दुनिया! यह एक परीक्षण है।"
        mock_result = InjectionResult(success=True, method="clipboard", injected_text=hindi_text)
        with patch.object(self.injector, "_inject_via_clipboard", return_value=mock_result) as mock_clip, \
             patch.object(self.injector, "verify_and_restore_target_window", return_value=True):

            res = self.injector.inject(hindi_text)
            self.assertIsInstance(res, InjectionResult)
            self.assertTrue(res.success)
            self.assertEqual(res.method, "clipboard")
            mock_clip.assert_called_once_with(hindi_text)

    def test_multiline_text_routes_to_clipboard(self):
        """Multiline text must route to clipboard paste to avoid newline trigger issues in terminals/editors."""
        multiline = "Line 1\nLine 2\nLine 3"
        mock_result = InjectionResult(success=True, method="clipboard", injected_text=multiline)
        with patch.object(self.injector, "_inject_via_clipboard", return_value=mock_result) as mock_clip, \
             patch.object(self.injector, "verify_and_restore_target_window", return_value=True):

            res = self.injector.inject(multiline)
            self.assertIsInstance(res, InjectionResult)
            self.assertTrue(res.success)
            self.assertEqual(res.method, "clipboard")
            mock_clip.assert_called_once_with(multiline)

    def test_target_window_mismatch_prevents_injection(self):
        """If target_hwnd is passed and foreground window does not match, injection must be refused."""
        with patch.object(self.injector, "verify_and_restore_target_window", return_value=False), \
             patch("text_injector.keyboard.write") as mock_write, \
             patch.object(self.injector, "_inject_via_clipboard") as mock_clip:

            res = self.injector.inject("Confidential text", target_hwnd=12345)
            self.assertIsInstance(res, InjectionResult)
            self.assertFalse(res.success)
            self.assertEqual(res.method, "refused_window_mismatch")
            self.assertIn("Target window changed", res.error)

            # Assert neither keyboard nor clipboard injection was executed
            mock_write.assert_not_called()
            mock_clip.assert_not_called()

    def test_keyboard_failure_falls_back_to_clipboard(self):
        """If keyboard.write throws an exception, it should fall back to clipboard paste."""
        mock_result = InjectionResult(success=True, method="clipboard", injected_text="Simple text")
        with patch("text_injector.keyboard.write", side_effect=RuntimeError("Keyboard driver error")), \
             patch.object(self.injector, "_inject_via_clipboard", return_value=mock_result) as mock_clip, \
             patch.object(self.injector, "verify_and_restore_target_window", return_value=True):

            res = self.injector.inject("Simple text")
            self.assertTrue(res.success)
            self.assertEqual(res.method, "clipboard")
            mock_clip.assert_called_once()

    def test_empty_text_returns_success_empty(self):
        """Empty or whitespace text should return success with empty method without doing work."""
        res = self.injector.inject("   ")
        self.assertTrue(res.success)
        self.assertEqual(res.method, "empty")
        self.assertEqual(res.injected_text, "")


if __name__ == "__main__":
    unittest.main()
