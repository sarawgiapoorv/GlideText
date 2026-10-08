"""
platform_compat/windows_backend.py -- Existing Windows logic encapsulated behind PlatformBackend.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

from platform_compat.base import PlatformBackend


class WindowsBackend(PlatformBackend):
    """Windows-native backend utilizing Win32 APIs, Registry, and COM/pycaw."""

    def __init__(self):
        self._duck_lock = threading.Lock()
        self._is_ducked = False
        self._original_volumes: dict[int, float] = {}

    def get_app_data_dir(self) -> str:
        base = os.getenv("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local"))
        app_dir = os.path.join(base, "GlideText")
        os.makedirs(app_dir, exist_ok=True)
        return app_dir

    def get_logs_dir(self) -> str:
        logs_dir = os.path.join(self.get_app_data_dir(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def get_active_window_info(self) -> dict:
        """Detect the currently focused Windows application using Win32 API."""
        from context_snapshot import AppCategory, classify_application

        result = {
            "title": "",
            "exe_name": "",
            "app_hint": "",
            "app_category": AppCategory.UNKNOWN.value,
            "is_sensitive_context": False,
            "hwnd": None,
        }

        if sys.platform != "win32":
            return result

        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            result["hwnd"] = hwnd

            # Window title
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            raw_title = buf.value or ""

            # Process executable
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            h_process = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
            )
            exe_name = ""
            if h_process:
                exe_buf = ctypes.create_unicode_buffer(512)
                size = ctypes.wintypes.DWORD(512)
                kernel32.QueryFullProcessImageNameW(
                    h_process, 0, exe_buf, ctypes.byref(size)
                )
                kernel32.CloseHandle(h_process)
                exe_path = exe_buf.value
                exe_name = os.path.basename(exe_path) if exe_path else ""
                result["exe_name"] = exe_name

            category, safe_app_name, is_sensitive = classify_application(
                exe_name=exe_name,
                window_title=raw_title,
                app_hint="",
            )
            result["app_category"] = category.value
            result["is_sensitive_context"] = is_sensitive
            if category != AppCategory.UNKNOWN:
                result["app_hint"] = safe_app_name
            else:
                result["app_hint"] = exe_name.replace(".exe", "") if exe_name else ""
            result["title"] = "" if is_sensitive else raw_title
        except Exception as e:
            logging.debug(f"[WindowsBackend] Window detection error: {e}")

        return result

    def verify_and_restore_target_window(self, target_token: Any) -> bool:
        """Verify foreground HWND matches target_token and restore if minimized/changed."""
        if sys.platform != "win32" or not target_token:
            return True

        target_hwnd = target_token
        try:
            import ctypes
            user32 = ctypes.windll.user32

            if not user32.IsWindow(target_hwnd):
                logging.warning(f"[WindowsBackend] Target HWND {target_hwnd} is no longer a valid window.")
                return False

            curr_hwnd = user32.GetForegroundWindow()
            if curr_hwnd == target_hwnd:
                return True

            logging.info(
                f"[WindowsBackend] Foreground window changed ({curr_hwnd} != target {target_hwnd}). "
                "Attempting to restore target window..."
            )

            kernel32 = ctypes.windll.kernel32
            curr_thread = kernel32.GetCurrentThreadId()
            target_thread = user32.GetWindowThreadProcessId(target_hwnd, None)

            attached = False
            if curr_thread != target_thread:
                attached = bool(user32.AttachThreadInput(curr_thread, target_thread, True))

            try:
                SW_RESTORE = 9
                if user32.IsIconic(target_hwnd):
                    user32.ShowWindow(target_hwnd, SW_RESTORE)

                user32.SetForegroundWindow(target_hwnd)
                time.sleep(0.06)

                confirmed_hwnd = user32.GetForegroundWindow()
                if confirmed_hwnd == target_hwnd:
                    logging.info(f"[WindowsBackend] Successfully restored target HWND {target_hwnd}.")
                    return True
                else:
                    logging.warning(
                        f"[WindowsBackend] Target window restoration failed. "
                        f"Foreground is {confirmed_hwnd}, expected {target_hwnd}."
                    )
                    return False
            finally:
                if attached:
                    user32.AttachThreadInput(curr_thread, target_thread, False)
        except Exception as e:
            logging.warning(f"[WindowsBackend] verify_and_restore_target_window failed: {e}")
            return False

    def is_secure_input_enabled(self) -> bool:
        return False

    def duck_audio(self) -> None:
        """Duck active application volumes via pycaw / comtypes asynchronously."""
        if sys.platform != "win32":
            return
        threading.Thread(target=self._duck_impl, daemon=True).start()

    def _duck_impl(self) -> None:
        with self._duck_lock:
            if self._is_ducked:
                return
            try:
                import comtypes
                from pycaw.pycaw import AudioUtilities
                comtypes.CoInitialize()
                try:
                    sessions = AudioUtilities.GetAllSessions()
                    self._original_volumes.clear()
                    for session in sessions:
                        volume = session.SimpleAudioVolume
                        if session.Process:
                            proc_id = session.Process.pid
                            vol = volume.GetMasterVolume()
                            self._original_volumes[proc_id] = vol
                            ducked_vol = max(0.0, vol * 0.10)
                            volume.SetMasterVolume(ducked_vol, None)
                    self._is_ducked = True
                finally:
                    try:
                        comtypes.CoUninitialize()
                    except Exception:
                        pass
            except Exception as e:
                logging.info(f"[WindowsBackend] Audio ducking failed: {e}")

    def restore_audio(self) -> None:
        """Restore application volumes asynchronously."""
        if sys.platform != "win32":
            return
        threading.Thread(target=self._restore_impl, daemon=True).start()

    def _restore_impl(self) -> None:
        with self._duck_lock:
            if not self._is_ducked:
                return
            try:
                import comtypes
                from pycaw.pycaw import AudioUtilities
                comtypes.CoInitialize()
                try:
                    sessions = AudioUtilities.GetAllSessions()
                    for session in sessions:
                        volume = session.SimpleAudioVolume
                        if session.Process:
                            proc_id = session.Process.pid
                            if proc_id in self._original_volumes:
                                volume.SetMasterVolume(self._original_volumes[proc_id], None)
                    self._original_volumes.clear()
                    self._is_ducked = False
                finally:
                    try:
                        comtypes.CoUninitialize()
                    except Exception:
                        pass
            except Exception as e:
                logging.info(f"[WindowsBackend] Audio volume restore failed: {e}")

    def set_thread_priority(self, priority_level: int) -> None:
        """Set caller thread priority using SetThreadPriority."""
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                h_thread = kernel32.GetCurrentThread()
                kernel32.SetThreadPriority(h_thread, priority_level)
            except Exception as e:
                logging.error(f"[WindowsBackend] Failed to set thread priority to {priority_level}: {e}")

    def is_auto_boot_enabled(self) -> bool:
        """Check Windows Registry Run key for GlideText auto-boot."""
        if sys.platform != "win32":
            return False
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_READ,
            )
            try:
                val, _ = winreg.QueryValueEx(key, "GlideText")
                return bool(val)
            except FileNotFoundError:
                try:
                    val, _ = winreg.QueryValueEx(key, "LocalFlow")
                    return bool(val)
                except FileNotFoundError:
                    return False
            finally:
                winreg.CloseKey(key)
        except Exception as e:
            logging.error(f"[WindowsBackend] Failed to query registry autoboot: {e}")
            return False

    def set_auto_boot(self, enabled: bool) -> bool:
        """Configure Windows Registry Run key for GlideText."""
        if sys.platform != "win32":
            return False
        try:
            import winreg
            proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            main_path = os.path.join(proj_dir, "main.py")
            pyw_exe = sys.executable.lower().replace("python.exe", "pythonw.exe")
            if not os.path.isfile(pyw_exe):
                pyw_exe = sys.executable
            cmd_string = f'"{pyw_exe}" "{main_path}" --silent'

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            )
            try:
                # Clean up legacy LocalFlow registry key if present
                try:
                    winreg.DeleteValue(key, "LocalFlow")
                except FileNotFoundError:
                    pass

                if enabled:
                    winreg.SetValueEx(key, "GlideText", 0, winreg.REG_SZ, cmd_string)
                    logging.info(f"[WindowsBackend] Set GlideText autoboot: {cmd_string}")
                else:
                    try:
                        winreg.DeleteValue(key, "GlideText")
                        logging.info("[WindowsBackend] Removed GlideText from boot.")
                    except FileNotFoundError:
                        pass
                return True
            finally:
                winreg.CloseKey(key)
        except Exception as e:
            logging.error(f"[WindowsBackend] Failed to update autoboot status: {e}")
            return False

    def get_font_family(self) -> str:
        return "Segoe UI"
