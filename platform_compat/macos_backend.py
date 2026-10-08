"""
platform_compat/macos_backend.py -- macOS native implementations for GlideText.
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
import threading
import time
from typing import Any, Optional, Tuple

from platform_compat.base import PlatformBackend


class MacOSBackend(PlatformBackend):
    """macOS-native backend using Cocoa/AppKit, Quartz, CoreAudio, and LaunchAgents."""

    def __init__(self):
        self._duck_lock = threading.Lock()
        self._is_ducked = False
        self._original_volume: Optional[int] = None
        self._plist_label = "com.glidetext.app"
        self._plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{self._plist_label}.plist")

    def get_app_data_dir(self) -> str:
        d = os.path.expanduser("~/Library/Application Support/GlideText")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    def get_logs_dir(self) -> str:
        d = os.path.expanduser("~/Library/Logs/GlideText")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    def is_secure_input_enabled(self) -> bool:
        """
        Check if macOS Secure Event Input is active.
        When active, password/PIN fields have focus and keystroke sniffing/injection is blocked.
        """
        if sys.platform != "darwin":
            return False
        try:
            import ctypes
            carbon = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Carbon.framework/Carbon")
            return bool(carbon.IsSecureEventInputEnabled())
        except Exception as e:
            logging.debug(f"[MacOSBackend] Could not check IsSecureEventInputEnabled: {e}")
            return False

    def get_active_window_info(self) -> dict:
        """
        Detect currently focused macOS application using NSWorkspace and Quartz.
        Returns the standard dict format expected across GlideText.
        """
        from context_snapshot import AppCategory, classify_application

        result = {
            "title": "",
            "exe_name": "",
            "bundle_id": "",
            "app_hint": "",
            "app_category": AppCategory.UNKNOWN.value,
            "is_sensitive_context": False,
            "hwnd": None,
        }

        if sys.platform != "darwin":
            return result

        # Check Secure Input first
        if self.is_secure_input_enabled():
            result["is_sensitive_context"] = True
            result["app_category"] = AppCategory.UNKNOWN.value
            result["app_hint"] = "Secure Password Input"
            return result

        try:
            from AppKit import NSWorkspace
            front_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if not front_app:
                return result

            app_name = str(front_app.localizedName() or "")
            bundle_id = str(front_app.bundleIdentifier() or "")
            pid = int(front_app.processIdentifier())
            result["exe_name"] = app_name
            result["bundle_id"] = bundle_id

            # Stable opaque window token: encode pid and front window id if available
            window_id = 0
            raw_title = ""

            try:
                from Quartz import (
                    CGWindowListCopyWindowInfo,
                    kCGWindowListOptionOnScreenOnly,
                    kCGWindowListExcludeDesktopElements,
                    kCGNullWindowID,
                )
                window_list = CGWindowListCopyWindowInfo(
                    kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                    kCGNullWindowID,
                )
                if window_list:
                    for win in window_list:
                        win_pid = win.get("kCGWindowOwnerPID")
                        if win_pid == pid:
                            w_id = win.get("kCGWindowNumber")
                            if w_id and not window_id:
                                window_id = int(w_id)
                            t = win.get("kCGWindowName")
                            if t and not raw_title:
                                raw_title = str(t)
                            if window_id and raw_title:
                                break
            except Exception as q_err:
                logging.debug(f"[MacOSBackend] Quartz window list query failed: {q_err}")

            # Window token as stable tuple (pid, window_id) or packed int
            # Pack as integer (pid << 32 | window_id) if window_id fits, or tuple
            token = (pid, window_id) if window_id else (pid, 0)
            result["hwnd"] = token

            category, safe_app_name, is_sensitive = classify_application(
                exe_name=app_name,
                window_title=raw_title,
                app_hint=bundle_id,
            )
            result["app_category"] = category.value
            result["is_sensitive_context"] = is_sensitive
            if category != AppCategory.UNKNOWN:
                result["app_hint"] = safe_app_name
            else:
                result["app_hint"] = app_name or bundle_id
            result["title"] = "" if is_sensitive else raw_title

        except Exception as e:
            logging.warning(f"[MacOSBackend] get_active_window_info failed: {e}")

        return result

    def verify_and_restore_target_window(self, target_token: Any) -> bool:
        """
        Verify target window/app on macOS.
        target_token can be a tuple of (pid, win_id) or an int pid.
        """
        if sys.platform != "darwin" or target_token is None:
            return True

        if self.is_secure_input_enabled():
            logging.warning("[MacOSBackend] Secure Input is enabled; refusing window injection.")
            return False

        try:
            target_pid = target_token[0] if isinstance(target_token, (tuple, list)) else int(target_token)
        except (ValueError, IndexError, TypeError):
            logging.debug(f"[MacOSBackend] Invalid target_token format: {target_token}")
            return True

        try:
            from AppKit import NSWorkspace, NSRunningApplication, NSApplicationActivateIgnoringOtherApps
            front_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if front_app and int(front_app.processIdentifier()) == target_pid:
                return True

            logging.info(
                f"[MacOSBackend] Frontmost app changed (current={front_app.processIdentifier() if front_app else None} != "
                f"target={target_pid}). Attempting to reactivate target app..."
            )

            target_running_app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target_pid)
            if target_running_app:
                target_running_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                time.sleep(0.08)
                recheck = NSWorkspace.sharedWorkspace().frontmostApplication()
                if recheck and int(recheck.processIdentifier()) == target_pid:
                    logging.info(f"[MacOSBackend] Successfully reactivated target app PID {target_pid}.")
                    return True
                else:
                    logging.warning(f"[MacOSBackend] Target app reactivation failed for PID {target_pid}.")
                    return False
            else:
                logging.warning(f"[MacOSBackend] Target app PID {target_pid} is no longer running.")
                return False
        except Exception as e:
            logging.warning(f"[MacOSBackend] verify_and_restore_target_window exception: {e}")
            return False

    def duck_audio(self) -> None:
        """Duck system output volume using osascript (best-effort, non-blocking)."""
        if sys.platform != "darwin":
            return
        threading.Thread(target=self._duck_impl, daemon=True).start()

    def _duck_impl(self) -> None:
        with self._duck_lock:
            if self._is_ducked:
                return
            try:
                # Query current volume
                res = subprocess.run(
                    ["osascript", "-e", "output volume of (get volume settings)"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res.returncode == 0 and res.stdout.strip().isdigit():
                    vol = int(res.stdout.strip())
                    self._original_volume = vol
                    ducked = max(0, int(vol * 0.15))
                    subprocess.run(
                        ["osascript", "-e", f"set volume output volume {ducked}"],
                        capture_output=True,
                        timeout=1.5,
                    )
                    self._is_ducked = True
            except Exception as e:
                logging.info(f"[MacOSBackend] Audio ducking failed (continuing without ducking): {e}")

    def restore_audio(self) -> None:
        """Restore system output volume to original level."""
        if sys.platform != "darwin":
            return
        threading.Thread(target=self._restore_impl, daemon=True).start()

    def _restore_impl(self) -> None:
        with self._duck_lock:
            if not self._is_ducked:
                return
            try:
                if self._original_volume is not None:
                    subprocess.run(
                        ["osascript", "-e", f"set volume output volume {self._original_volume}"],
                        capture_output=True,
                        timeout=1.5,
                    )
                self._original_volume = None
                self._is_ducked = False
            except Exception as e:
                logging.info(f"[MacOSBackend] Audio volume restore failed: {e}")

    def set_thread_priority(self, priority_level: int) -> None:
        # Thread priority setting on macOS via POSIX / pthreads is generally a no-op for non-root
        pass

    def is_auto_boot_enabled(self) -> bool:
        """Check if LaunchAgent plist exists."""
        return os.path.isfile(self._plist_path)

    def set_auto_boot(self, enabled: bool) -> bool:
        """Configure launch-at-login via ~/Library/LaunchAgents plist."""
        try:
            if enabled:
                proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                main_path = os.path.join(proj_dir, "main.py")
                python_bin = sys.executable

                plist_content = {
                    "Label": self._plist_label,
                    "ProgramArguments": [python_bin, main_path, "--silent"],
                    "RunAtLoad": True,
                    "KeepAlive": False,
                    "StandardOutPath": os.path.join(self.get_logs_dir(), "launchagent.log"),
                    "StandardErrorPath": os.path.join(self.get_logs_dir(), "launchagent.err"),
                }

                os.makedirs(os.path.dirname(self._plist_path), exist_ok=True)
                with open(self._plist_path, "wb") as fp:
                    plistlib.dump(plist_content, fp)

                if sys.platform == "darwin":
                    # Attempt launchctl bootstrap or load
                    uid = os.getuid() if hasattr(os, "getuid") else 501
                    cmd_bootstrap = ["launchctl", "bootstrap", f"gui/{uid}", self._plist_path]
                    res = subprocess.run(cmd_bootstrap, capture_output=True, text=True)
                    if res.returncode != 0:
                        # Legacy fallback
                        subprocess.run(["launchctl", "load", "-w", self._plist_path], capture_output=True)

                logging.info(f"[MacOSBackend] Enabled LaunchAgent at {self._plist_path}")
                return True
            else:
                if os.path.isfile(self._plist_path):
                    if sys.platform == "darwin":
                        uid = os.getuid() if hasattr(os, "getuid") else 501
                        cmd_bootout = ["launchctl", "bootout", f"gui/{uid}/{self._plist_label}"]
                        res = subprocess.run(cmd_bootout, capture_output=True, text=True)
                        if res.returncode != 0:
                            subprocess.run(["launchctl", "unload", "-w", self._plist_path], capture_output=True)
                    try:
                        os.remove(self._plist_path)
                    except OSError:
                        pass
                logging.info(f"[MacOSBackend] Disabled LaunchAgent at {self._plist_path}")
                return True
        except Exception as e:
            logging.error(f"[MacOSBackend] Failed to update LaunchAgent status: {e}")
            return False

    def get_font_family(self) -> str:
        return "Helvetica Neue"

    def hide_dock_icon(self) -> None:
        """Hide dock icon using NSApplicationActivationPolicyAccessory for --silent mode."""
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception as e:
            logging.debug(f"[MacOSBackend] Could not hide Dock icon: {e}")

    def show_dock_icon(self) -> None:
        """Restore regular dock icon policy."""
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyRegular
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        except Exception as e:
            logging.debug(f"[MacOSBackend] Could not show Dock icon: {e}")

    def check_microphone_permission(self) -> Tuple[bool, str]:
        """Check microphone access on macOS via AVFoundation."""
        if sys.platform != "darwin":
            return True, "Microphone access allowed."
        try:
            import objc
            from AVFoundation import AVCaptureDevice, AVMediaTypeAudio, AVAuthorizationStatusAuthorized, AVAuthorizationStatusNotDetermined
            status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
            if status == AVAuthorizationStatusAuthorized:
                return True, "Microphone permission granted."
            elif status == AVAuthorizationStatusNotDetermined:
                return False, "Microphone permission not yet requested. System prompt will appear when dictating."
            else:
                return False, "Microphone permission denied. Enable in System Settings > Privacy & Security > Microphone."
        except Exception as e:
            logging.debug(f"[MacOSBackend] Microphone permission check via AVFoundation unavailable: {e}")
            return True, "Microphone permission check bypassed."

    def check_accessibility_permission(self, prompt: bool = False) -> Tuple[bool, str]:
        """Check Accessibility permission via ApplicationServices AXIsProcessTrusted."""
        if sys.platform != "darwin":
            return True, "Accessibility access allowed."
        try:
            from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
            if prompt:
                trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
            else:
                trusted = AXIsProcessTrusted()
            if trusted:
                return True, "Accessibility permission granted."
            return False, "Accessibility permission missing. Enable in System Settings > Privacy & Security > Accessibility."
        except Exception as e:
            logging.debug(f"[MacOSBackend] Accessibility check unavailable: {e}")
            return True, "Accessibility check bypassed."

    def check_input_monitoring_permission(self, prompt: bool = False) -> Tuple[bool, str]:
        """Check Input Monitoring permission via Quartz CGPreflightListenEventAccess."""
        if sys.platform != "darwin":
            return True, "Input monitoring access allowed."
        try:
            from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
            trusted = CGPreflightListenEventAccess()
            if trusted:
                return True, "Input monitoring permission granted."
            if prompt:
                CGRequestListenEventAccess()
            return False, "Input monitoring permission missing. Enable in System Settings > Privacy & Security > Input Monitoring."
        except Exception as e:
            logging.debug(f"[MacOSBackend] Input monitoring check unavailable: {e}")
            return True, "Input monitoring check bypassed."
