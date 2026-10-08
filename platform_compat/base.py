"""
platform_compat/base.py -- Common interface and base definitions for GlideText platform backends.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class PlatformBackend(ABC):
    """Abstract base class defining OS-level integration points for GlideText."""

    @abstractmethod
    def get_app_data_dir(self) -> str:
        """Return the platform-appropriate application support / data directory."""
        pass

    @abstractmethod
    def get_logs_dir(self) -> str:
        """Return the platform-appropriate directory for persistent log files."""
        pass

    def get_temp_dir(self) -> str:
        """Return the platform-appropriate temporary audio capture directory."""
        temp_dir = os.path.join(tempfile.gettempdir(), "glidetext")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    @abstractmethod
    def get_active_window_info(self) -> dict:
        """
        Detect currently focused application and window context.
        Must return dict with keys:
            - 'title': str
            - 'exe_name': str
            - 'app_hint': str
            - 'app_category': str
            - 'is_sensitive_context': bool
            - 'hwnd': Any (opaque window token)
        """
        pass

    @abstractmethod
    def verify_and_restore_target_window(self, target_token: Any) -> bool:
        """
        Verify that target window token is still focused, or attempt to restore focus.
        Returns True if target window is verified and active.
        """
        pass

    def is_secure_input_enabled(self) -> bool:
        """Check if OS-level Secure Input is enabled (macOS password fields)."""
        return False

    def duck_audio(self) -> None:
        """Lower system or application playback volume during recording."""
        pass

    def restore_audio(self) -> None:
        """Restore playback volume to previous levels."""
        pass

    def set_thread_priority(self, priority_level: int) -> None:
        """Set caller thread scheduling priority if supported by OS."""
        pass

    @abstractmethod
    def is_auto_boot_enabled(self) -> bool:
        """Check if launch-at-login is configured."""
        pass

    @abstractmethod
    def set_auto_boot(self, enabled: bool) -> bool:
        """Enable or disable launch-at-login."""
        pass

    def get_font_family(self) -> str:
        """Return primary UI font family."""
        return "Helvetica Neue"

    def hide_dock_icon(self) -> None:
        """Hide application from taskbar/dock for background/silent mode."""
        pass

    def show_dock_icon(self) -> None:
        """Show application in taskbar/dock."""
        pass

    def check_microphone_permission(self) -> Tuple[bool, str]:
        """Check microphone access permission. Returns (granted, status_message)."""
        return True, "Microphone access allowed."

    def check_accessibility_permission(self, prompt: bool = False) -> Tuple[bool, str]:
        """Check accessibility/synthetic keystroke permission. Returns (granted, status_message)."""
        return True, "Accessibility access allowed."

    def check_input_monitoring_permission(self, prompt: bool = False) -> Tuple[bool, str]:
        """Check global hotkey/input monitoring permission. Returns (granted, status_message)."""
        return True, "Input monitoring access allowed."


class FallbackBackend(PlatformBackend):
    """Safe no-op fallback backend for unsupported operating systems."""

    def get_app_data_dir(self) -> str:
        d = os.path.join(os.path.expanduser("~"), ".glidetext")
        os.makedirs(d, exist_ok=True)
        return d

    def get_logs_dir(self) -> str:
        d = os.path.join(self.get_app_data_dir(), "logs")
        os.makedirs(d, exist_ok=True)
        return d

    def get_active_window_info(self) -> dict:
        return {
            "title": "",
            "exe_name": "",
            "app_hint": "",
            "app_category": "unknown",
            "is_sensitive_context": False,
            "hwnd": None,
        }

    def verify_and_restore_target_window(self, target_token: Any) -> bool:
        return True

    def is_auto_boot_enabled(self) -> bool:
        return False

    def set_auto_boot(self, enabled: bool) -> bool:
        return False
