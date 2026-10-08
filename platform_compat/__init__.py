"""
platform_compat -- Thin cross-platform abstraction layer for GlideText.
Supports Windows, macOS (Apple Silicon M1-M5 and Intel), and Linux (fallback).
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Tuple

from platform_compat.base import FallbackBackend, PlatformBackend

IS_WINDOWS: bool = sys.platform == "win32"
IS_MACOS: bool = sys.platform == "darwin"
IS_LINUX: bool = sys.platform.startswith("linux")


def get_platform() -> str:
    """Return canonical platform string ('windows', 'macos', 'linux', or 'unknown')."""
    if IS_WINDOWS:
        return "windows"
    elif IS_MACOS:
        return "macos"
    elif IS_LINUX:
        return "linux"
    return "unknown"


_BACKEND_INSTANCE: Optional[PlatformBackend] = None


def get_platform_backend() -> PlatformBackend:
    """Return the singleton PlatformBackend instance for the current host OS."""
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is not None:
        return _BACKEND_INSTANCE

    if IS_WINDOWS:
        from platform_compat.windows_backend import WindowsBackend
        _BACKEND_INSTANCE = WindowsBackend()
    elif IS_MACOS:
        from platform_compat.macos_backend import MacOSBackend
        _BACKEND_INSTANCE = MacOSBackend()
    else:
        _BACKEND_INSTANCE = FallbackBackend()

    return _BACKEND_INSTANCE


# Convenience function proxies
def get_app_data_dir() -> str:
    return get_platform_backend().get_app_data_dir()


def get_logs_dir() -> str:
    return get_platform_backend().get_logs_dir()


def get_temp_dir() -> str:
    return get_platform_backend().get_temp_dir()


def get_active_window_info() -> dict:
    return get_platform_backend().get_active_window_info()


def verify_and_restore_target_window(target_token: Any) -> bool:
    return get_platform_backend().verify_and_restore_target_window(target_token)


def is_secure_input_enabled() -> bool:
    return get_platform_backend().is_secure_input_enabled()


def duck_audio() -> None:
    get_platform_backend().duck_audio()


def restore_audio() -> None:
    get_platform_backend().restore_audio()


def set_thread_priority(priority_level: int) -> None:
    get_platform_backend().set_thread_priority(priority_level)


def is_auto_boot_enabled() -> bool:
    return get_platform_backend().is_auto_boot_enabled()


def set_auto_boot(enabled: bool) -> bool:
    return get_platform_backend().set_auto_boot(enabled)


def get_font_family() -> str:
    return get_platform_backend().get_font_family()


def hide_dock_icon() -> None:
    get_platform_backend().hide_dock_icon()


def show_dock_icon() -> None:
    get_platform_backend().show_dock_icon()


def check_microphone_permission() -> Tuple[bool, str]:
    return get_platform_backend().check_microphone_permission()


def check_accessibility_permission(prompt: bool = False) -> Tuple[bool, str]:
    return get_platform_backend().check_accessibility_permission(prompt=prompt)


def check_input_monitoring_permission(prompt: bool = False) -> Tuple[bool, str]:
    return get_platform_backend().check_input_monitoring_permission(prompt=prompt)
