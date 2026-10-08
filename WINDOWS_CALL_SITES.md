# Phase 0 Baseline & Windows-Specific Call Site Checklist

## Test Suite Baseline
- Command: `python -m unittest discover tests`
- Platform: Windows (Windows PowerShell)
- Date: 2026-10-08
- Result: **Ran 136 tests in 4.848s -- OK (0 failures, 0 errors)**

---

## Windows-Specific Call Sites Audit

### 1. `main.py`
- Line 9: Docstring reference to `pythonw main.py`.
- Lines 31-34: `REQUIRED_LIBS` includes Windows-only packages: `pycaw`, `comtypes`, `keyboard`.
- Line 48: `get_app_dir()` references `os.getenv("LOCALAPPDATA", ...)`.
- Lines 148, 155, 166, 306: `creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0` in subprocess invocations.
- Lines 207, 210, 213: Font family `"Segoe UI"` in Tkinter setup splash screen.
- Lines 285-289: Windows process priority class setting via `ctypes.windll.kernel32.SetPriorityClass`.

### 2. `gui_app.py`
- Line 22: Top-level `import keyboard`.
- Line 31: `_log_dir` uses `os.getenv("LOCALAPPDATA", ...)`.
- Lines 60-67: `set_thread_priority` uses `ctypes.windll.kernel32.SetThreadPriority`.
- Lines 94-98: `import winreg` / `HAS_WINREG`.
- Line 276: `FONT = "Segoe UI"`.
- Lines 1112-1136: `_capture_lookback_context` calls `keyboard.press("ctrl")`, `keyboard.press("shift")`, `keyboard.press_and_release("left")`, `keyboard.press_and_release("ctrl+c")`, `keyboard.press_and_release("right")`.
- Lines 1214-1268: `_apply_spoken_correction` calls `keyboard.press("shift")`, `keyboard.press_and_release("left")`, `keyboard.press_and_release("ctrl+c")`, `keyboard.write(...)`, `keyboard.press_and_release("backspace")`, `keyboard.press_and_release("right")`.
- Lines 1284-1301: `_is_auto_boot_enabled` queries Windows Registry `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.
- Lines 1307-1341: `_set_auto_boot` writes/deletes Windows Registry key for `pythonw.exe main.py --silent`.
- Line 1466: `ImageFont.truetype("segoeui.ttf", 22)` for system tray icon.
- Lines 1434-1456: `_setup_tray` uses `pystray.Icon.run` in a background thread (conflicts with macOS Tkinter mainloop).
- Line 1489: `_on_window_close` minimize-to-tray logic.
- Line 1497: `_quit_app` calls `keyboard.unhook_all()`.
- Lines 1535-1556: `_initialize_backend` registers hotkeys via `keyboard.on_press_key("right alt")`, `keyboard.on_release_key("right alt")`, `keyboard.add_hotkey("ctrl + shift + a")`, and blocks via `keyboard.wait()`.
- Lines 1766, 1771, 2024, 2131, 2140: Window handle tracking via `hwnd` and `target_hwnd`.
- Lines 1860-1897: `_execute_editing_command` uses `keyboard.press_and_release` ("ctrl+z", "ctrl+a", "delete", "enter") and `keyboard.write`.

### 3. `audio_recorder.py`
- Lines 40-44: `from pycaw.pycaw import AudioUtilities`, `HAS_PYCAW`.
- Lines 49-53: `import ctypes`, `import ctypes.wintypes`, `HAS_WIN32`.
- Lines 121-167: `get_active_window_info()` uses Win32 API (`GetForegroundWindow`, `GetWindowTextW`, `GetWindowThreadProcessId`, `OpenProcess`, `QueryFullProcessImageNameW`).
- Lines 184-253: `AudioDucker` uses `comtypes.CoInitialize`, `AudioUtilities.GetAllSessions()`, `SimpleAudioVolume`.
- Lines 327-333: `_warmup_impl` calls `ctypes.windll.kernel32.SetThreadPriority`.
- Lines 403-409: `_vad_monitor` calls `ctypes.windll.kernel32.SetThreadPriority`.

### 4. `text_injector.py`
- Lines 19-20: `import ctypes`, `import ctypes.wintypes`.
- Lines 31-34: `import keyboard`, `HAS_KEYBOARD`.
- Lines 128-185: `verify_and_restore_target_window(target_hwnd)` uses `ctypes.windll.user32` (`IsWindow`, `GetForegroundWindow`, `GetWindowThreadProcessId`, `AttachThreadInput`, `IsIconic`, `ShowWindow`, `SetForegroundWindow`) and `ctypes.windll.kernel32.GetCurrentThreadId()`.
- Line 214: `_inject_via_clipboard` uses `keyboard.press_and_release("ctrl+v")`.
- Line 277: `inject` uses `keyboard.write(cleaned, delay=self.delay)`.
- Line 298: `inject_with_newline` uses `keyboard.press_and_release("enter")`.

### 5. `context_snapshot.py`
- Lines 138-155: `_SENSITIVE_EXECUTABLES` Windows `.exe` blacklist (1password.exe, bitwarden.exe, logonui.exe, etc.).
- Lines 181-281: `_IDE_EXECUTABLES`, `_TERMINAL_EXECUTABLES`, `_EMAIL_EXECUTABLES`, `_SLACK_CHAT_EXECUTABLES`, `_TEAMS_CHAT_EXECUTABLES`, `_DOCUMENT_EDITOR_EXECUTABLES`, `_BROWSER_EXECUTABLES` tables keyed by Windows `.exe` names.
- Lines 526, 540, 580, 598-599, 670, 716: `target_hwnd` / `hwnd` typed as `Optional[int]`.

### 6. `voice_commands.py`
- Lines 27-30: `import keyboard`, `HAS_KEYBOARD`.
- Lines 74, 109, 137, 238, 269: `target_hwnd` window token verification.
- Line 427: `_send_backspaces` uses `keyboard.press_and_release("backspace")`.

### 7. `freellm_manager.py`
- Lines 172-173: `locate_freellmapi_dir` includes `USERPROFILE` and Windows OneDrive paths.
- Line 207: `_find_npm` searches `npm.cmd` on Windows.
- Lines 250-251: `_spawn` sets `creationflags=0x08000000` (CREATE_NO_WINDOW) and `shell=True` on Windows.
- Line 253: Log directory references `LOCALAPPDATA`.
- Lines 448-456: `shutdown` uses Windows `taskkill /F /T /PID <pid>`.

### 8. `local_llm.py`
- Lines 96-106: `_find_ollama_executable` checks `LOCALAPPDATA` and `ProgramFiles` for `ollama.exe`.
- Lines 141-143: `ensure_server_running` uses `creationflags=0x08000000` (CREATE_NO_WINDOW).

### 9. `create_shortcut.py`
- Entire file is Windows-specific: creates `.lnk` shortcut using `WScript.Shell` ComObject via PowerShell.

### 10. `requirements.txt`
- Lines 1, 10, 11: Unconditioned dependencies on `keyboard`, `pycaw`, `comtypes`.

### 11. `tests/`
- Direct references to `hwnd` as integers.
- Direct patches to `text_injector.keyboard.write`.
