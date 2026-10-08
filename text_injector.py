"""
text_injector.py — OS-level text injection for GlideText.

Types text directly at the active Windows cursor position using simulated
keystrokes or Windows clipboard (Ctrl+V) paste with clipboard restoration.

Features:
  • Snippet expansion: voice macros from snippets.json
  • Robust Injection: Primary keyboard typing for ASCII; Fallback to clipboard
    for Unicode/multiline or on typing failure.
  • Clipboard preservation: Restores original clipboard content.
  • Target Window Verification: Prevents typing into the wrong window if
    the user switched windows during processing.
  • Structured InjectionResult reporting actual success/failure and method used.
"""

from __future__ import annotations

import sys
if sys.platform == "win32":
    try:
        import ctypes
        import ctypes.wintypes
    except Exception:
        pass
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from platform_compat.input_backend import LogicalAction, get_action_combo

if sys.platform == "win32":
    try:
        import keyboard
        HAS_KEYBOARD = True
    except ImportError:
        HAS_KEYBOARD = False
else:
    try:
        from platform_compat.input_backend import get_input_backend
        keyboard = get_input_backend()
        HAS_KEYBOARD = True
    except Exception:
        HAS_KEYBOARD = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEYSTROKE_DELAY = 0.012  # ~12 ms between keystrokes


@dataclass
class InjectionResult:
    """Structured result from an injection attempt."""
    success: bool
    method: str  # "keyboard" | "clipboard" | "none"
    injected_text: str = ""
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        return self.injected_text if self.success else ""


class TextInjector:
    """Simulates typing at the current cursor position with snippet expansion."""

    def __init__(self, delay: float = KEYSTROKE_DELAY):
        self.delay = delay
        self._snippets: dict[str, str] = self._load_snippets()

    @staticmethod
    def _load_snippets() -> dict[str, str]:
        """Load snippet macros from snippets.json."""
        snippets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "snippets.json"
        )
        if os.path.isfile(snippets_path):
            try:
                with open(snippets_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"[Injector] Failed to load snippets.json: {e}")
        return {}

    def reload_snippets(self) -> None:
        """Hot-reload snippets from disk."""
        self._snippets = self._load_snippets()

    def expand_snippets(self, text: str) -> str:
        """
        Scan text for snippet trigger phrases and expand them.
        Matching is case-insensitive.
        """
        if not self._snippets or not text:
            return text

        text_lower = text.strip().lower()
        sorted_snippets = sorted(self._snippets.items(), key=lambda x: len(x[0]), reverse=True)
        for trigger, expansion in sorted_snippets:
            if text_lower == trigger.lower().strip():
                logging.info(f"[Injector] Snippet expanded: '{trigger}'")
                return expansion

        result = text
        for trigger, expansion in sorted_snippets:
            idx = result.lower().find(trigger.lower())
            while idx != -1:
                result = result[:idx] + expansion + result[idx + len(trigger):]
                logging.info(f"[Injector] Snippet expanded inline: '{trigger}'")
                idx = result.lower().find(trigger.lower(), idx + len(expansion))

        return result

    @staticmethod
    def is_safe_for_keyboard(text: str) -> bool:
        """
        Check if text can be safely typed via keyboard.write.
        Text containing newlines, tabs, or non-ASCII characters (e.g. Hindi, Unicode,
        emojis) should use clipboard injection to prevent dropped/corrupted characters.
        """
        if "\n" in text or "\r" in text or "\t" in text:
            return False
        # ASCII printable characters: space (32) to tilde (126)
        return all(32 <= ord(c) <= 126 for c in text)

    @staticmethod
    def verify_and_restore_target_window(target_hwnd: Any) -> bool:
        """
        Verify that the current foreground window matches target_hwnd.
        If it does not, attempt to restore and focus target_hwnd.
        Returns True if the target window is verified and focused.
        """
        if not target_hwnd:
            return True
        import platform_compat
        return platform_compat.verify_and_restore_target_window(target_hwnd)

    def _inject_via_clipboard(self, text: str) -> InjectionResult:
        """Inject text via Windows clipboard + Ctrl+V, then restore original clipboard."""
        if not HAS_PYPERCLIP:
            return InjectionResult(
                success=False,
                method="clipboard",
                error="pyperclip is not available for clipboard injection",
            )
        if not HAS_KEYBOARD:
            return InjectionResult(
                success=False,
                method="clipboard",
                error="keyboard library is not available for pasting",
            )

        # 1. Backup existing clipboard content
        clipboard_backup = ""
        try:
            clipboard_backup = pyperclip.paste()
        except Exception:
            clipboard_backup = ""

        try:
            # 2. Set new content
            pyperclip.copy(text)
            time.sleep(0.03)

            # 3. Trigger Paste (Ctrl+V on Windows, Cmd+V on macOS)
            paste_combo = get_action_combo(LogicalAction.PASTE)
            keyboard.press_and_release(paste_combo)
            time.sleep(0.08)

            return InjectionResult(
                success=True,
                method="clipboard",
                injected_text=text,
            )
        except Exception as e:
            logging.error(f"[Injector] Clipboard injection failed: {e}")
            return InjectionResult(
                success=False,
                method="clipboard",
                error=str(e),
            )
        finally:
            # 4. Asynchronously restore user's original clipboard after a safe delay
            def _restore():
                time.sleep(0.25)
                try:
                    pyperclip.copy(clipboard_backup)
                except Exception:
                    pass

            threading.Thread(target=_restore, daemon=True).start()

    def inject(self, text: str, target_hwnd: Any = None) -> InjectionResult:
        """
        Inject text into the target active window.

        Args:
            text: Text to type or paste.
            target_hwnd: Optional target HWND / window token. If provided, target is verified/restored.
                         Injection aborts if target window cannot be verified.

        Returns:
            InjectionResult describing whether injection succeeded and method used.
        """
        if not text:
            return InjectionResult(success=True, method="empty", injected_text="")

        cleaned = text.strip().replace("\r\n", "\n")
        if not cleaned:
            return InjectionResult(success=True, method="empty", injected_text="")

        # Secure input check (e.g. macOS password fields)
        import platform_compat
        if platform_compat.is_secure_input_enabled():
            err = "Secure Input is active (password field detected). Injection refused."
            logging.warning(f"[Injector] {err}")
            return InjectionResult(
                success=False,
                method="refused_secure_input",
                error=err,
            )

        # Small pause before typing to let the user's key-release register
        time.sleep(0.12)

        # Target window verification
        if target_hwnd is not None and target_hwnd != 0:
            if not self.verify_and_restore_target_window(target_hwnd):
                err = f"Target window changed and could not be verified (target_hwnd={target_hwnd}). Injection cancelled."
                logging.warning(f"[Injector] {err}")
                return InjectionResult(
                    success=False,
                    method="refused_window_mismatch",
                    error=err,
                )

        # Decide injection method:
        # Use keyboard if text is ASCII-only and single-line
        if self.is_safe_for_keyboard(cleaned) and HAS_KEYBOARD:
            try:
                keyboard.write(cleaned, delay=self.delay)
                return InjectionResult(
                    success=True,
                    method="keyboard",
                    injected_text=cleaned,
                )
            except Exception as e:
                logging.warning(
                    f"[Injector] Keyboard typing failed ({e}), falling back to clipboard paste..."
                )
                return self._inject_via_clipboard(cleaned)
        else:
            # Unicode, multiline, or keyboard unsupported: use clipboard injection directly
            return self._inject_via_clipboard(cleaned)

    def inject_with_newline(self, text: str, target_hwnd: Any = None) -> InjectionResult:
        """Type or paste the text followed by Enter."""
        res = self.inject(text, target_hwnd=target_hwnd)
        if res.success and text and text.strip() and HAS_KEYBOARD:
            time.sleep(0.05)
            try:
                keyboard.press_and_release("enter")
            except Exception as e:
                logging.warning(f"[Injector] Failed to press enter: {e}")
        return res
