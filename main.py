"""
main.py -- Zero-Touch Bootstrapper and Entry point for GlideText.

Launches the CustomTkinter desktop GUI after performing a silent dependency setup.
All coordination, hotkey handling, and backend logic live in gui_app.py.

Usage:
    python main.py          (from an elevated terminal)
    pythonw main.py         (silent, no console window)
    Launch_GlideText.bat    (auto-elevates & silent)
"""

import sys
import os
import subprocess
import threading
import time
import re
import importlib.util
import traceback

# We don't import gui_app or any other third-party dependencies at the top level
# to prevent import crashes during pre-flight checks.

REQUIRED_LIBS = [
    ("customtkinter", "customtkinter"),
    ("sounddevice", "sounddevice"),
    ("keyring", "keyring"),
    ("faster_whisper", "faster-whisper"),
    ("wavio", "wavio"),
    ("pycaw", "pycaw"),
    ("comtypes", "comtypes"),
    ("pyperclip", "pyperclip"),
    ("keyboard", "keyboard"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("pystray", "pystray"),
    ("PIL", "Pillow"),
]

OPTIONAL_LIBS = [
    ("noisereduce", "noisereduce"),
    ("webrtcvad", "webrtcvad"),
]


def get_app_dir() -> str:
    base = os.getenv("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local"))
    app_dir = os.path.join(base, "GlideText")
    os.makedirs(app_dir, exist_ok=True)
    return app_dir


def get_logs_dir() -> str:
    logs_dir = os.path.join(get_app_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def log_setup(msg: str):
    try:
        setup_log = os.path.join(get_logs_dir(), "setup.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(setup_log, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


def is_package_installed_on_disk(module_name: str) -> bool:
    """Check if package spec is found on disk, indicating pip already installed it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False


def check_dependencies():
    """Verify dependencies. Returns list of pip_names for REQUIRED packages genuinely missing from disk.
    
    Catches ANY Exception during import and logs real error message to setup.log.
    If a package is already installed on disk but fails to import, logs error once and does NOT return it for pip re-install.
    """
    missing_required = []

    for module_name, pip_name in REQUIRED_LIBS:
        try:
            __import__(module_name)
        except Exception as e:
            err_detail = traceback.format_exc()
            log_setup(f"REQUIRED module '{module_name}' failed to import: {e}\n{err_detail}")
            if is_package_installed_on_disk(module_name):
                log_setup(f"REQUIRED package '{pip_name}' is already installed on disk but failed to import. Skipping pip reinstall loop.")
            else:
                missing_required.append(pip_name)

    for module_name, pip_name in OPTIONAL_LIBS:
        try:
            __import__(module_name)
        except Exception as e:
            err_detail = traceback.format_exc()
            log_setup(f"OPTIONAL module '{module_name}' failed to import: {e}\n{err_detail}")
            log_setup(f"OPTIONAL package '{pip_name}' import failed; continuing without it.")

    return missing_required


def get_deps_ok_marker_path() -> str:
    return os.path.join(get_app_dir(), "deps_ok")


def should_skip_splash() -> bool:
    return os.path.isfile(get_deps_ok_marker_path())


def should_retry_optional() -> bool:
    marker = get_deps_ok_marker_path()
    if not os.path.isfile(marker):
        return True
    try:
        mtime = os.path.getmtime(marker)
        return (time.time() - mtime) > 86400  # 24 hours
    except Exception:
        return True


def write_deps_ok_marker():
    try:
        marker = get_deps_ok_marker_path()
        with open(marker, "w", encoding="utf-8") as f:
            f.write(f"ok={time.time()}\n")
        log_setup("Successfully wrote deps_ok marker file.")
    except Exception as e:
        log_setup(f"Failed to write deps_ok marker: {e}")


def run_installer(status_label, root, missing_required):
    """Run pip install silently in a background thread for missing required dependencies."""
    try:
        if missing_required:
            status_label.config(text="Downloading required AI libraries...")
            req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
            if os.path.isfile(req_path):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"],
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
            else:
                for pip_name in missing_required:
                    status_label.config(text=f"Installing {pip_name}...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )

        # Retry optional packages at most once per day
        if should_retry_optional():
            status_label.config(text="Checking optional voice components...")
            for module_name, pip_name in OPTIONAL_LIBS:
                if not is_package_installed_on_disk(module_name):
                    log_setup(f"Attempting installation of optional package '{pip_name}'...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )

        write_deps_ok_marker()
        status_label.config(text="Environment ready! Starting GlideText...")
        time.sleep(0.5)

    except Exception as e:
        log_setup(f"Setup installer exception: {e}")
        status_label.config(text=f"Setup warning: {e}")
        time.sleep(1.5)
    finally:
        root.after(0, root.destroy)


def show_splash_screen(missing_required):
    """Display a custom borderless dark Tkinter setup screen while installing."""
    import tkinter as tk
    
    root = tk.Tk()
    root.title("GlideText Setup")
    
    bg_color = "#0f172a"      # Slate 900
    text_color = "#f1f5f9"    # Slate 50
    sec_color = "#94a3b8"     # Slate 400
    accent_color = "#6366f1"  # Indigo 500
    border_color = "#1e293b"  # Slate 800
    
    root.overrideredirect(True)
    w = 420
    h = 240
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
    root.configure(bg=bg_color)
    
    frame = tk.Frame(root, bg=bg_color, bd=1, relief="solid", highlightbackground=border_color, highlightthickness=1)
    frame.pack(fill="both", expand=True)
    
    title_label = tk.Label(frame, text="GlideText", bg=bg_color, fg=accent_color, font=("Segoe UI", 26, "bold"))
    title_label.pack(pady=(35, 10))
    
    sub_label = tk.Label(frame, text="Privacy-First Voice Dictation Engine", bg=bg_color, fg=sec_color, font=("Segoe UI", 10, "italic"))
    sub_label.pack(pady=(0, 20))
    
    status_label = tk.Label(frame, text="Checking environment status...", bg=bg_color, fg=text_color, font=("Segoe UI", 11))
    status_label.pack(pady=10)
    
    progress_bg = tk.Frame(frame, bg="#1e293b", height=4, width=320)
    progress_bg.pack(pady=(10, 0))
    progress_bg.pack_propagate(False)
    
    progress_bar = tk.Frame(progress_bg, bg=accent_color, height=4, width=0)
    progress_bar.pack(side="left")
    
    def animate(step=0):
        if root.winfo_exists():
            new_width = (step % 32) * 10
            progress_bar.config(width=new_width)
            root.after(80, animate, step + 1)
            
    animate()
    
    t = threading.Thread(target=run_installer, args=(status_label, root, missing_required), daemon=True)
    t.start()
    
    root.mainloop()


def sanitize_traceback(tb_str: str) -> str:
    """Purge API keys and sensitive tokens from traceback to prevent exfiltration."""
    tb_str = re.sub(r'AIzaSy[a-zA-Z0-9_-]+', '[API_KEY_SANITIZED]', tb_str)
    tb_str = re.sub(r'AQ\.[a-zA-Z0-9_-]+', '[API_KEY_SANITIZED]', tb_str)
    return tb_str


def _get_manager():
    """Lazily import freellm_manager to avoid circular import at module load time."""
    try:
        import freellm_manager
        return freellm_manager
    except Exception as e:
        print(f"[FreeLLMAPI] Could not load freellm_manager: {e}")
        return None


def locate_freellmapi_dir() -> str | None:
    """Find the FreeLLMAPI directory. Delegates to freellm_manager."""
    mgr = _get_manager()
    return mgr.locate_freellmapi_dir() if mgr else None


def is_freellmapi_running(timeout: float = 1.0) -> bool:
    """Return True if FreeLLMAPI is currently accepting TCP connections."""
    mgr = _get_manager()
    return mgr.is_running() if mgr else False


def ensure_freellmapi_running(poll_timeout: float = 8.0) -> bool:
    """Start FreeLLMAPI in the background without blocking the GUI."""
    mgr = _get_manager()
    if mgr is None:
        return False
    mgr.start_async(poll_timeout=poll_timeout)
    return True


def terminate_freellmapi() -> None:
    """Cleanly shut down the background FreeLLMAPI process tree."""
    mgr = _get_manager()
    if mgr:
        mgr.shutdown()


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
            print("[Process] Process priority set to ABOVE_NORMAL")
        except Exception as e:
            print(f"[Process] Failed to set process priority class: {e}")

    # Dependency Flow
    skip_splash = should_skip_splash()
    missing_required = []

    if not skip_splash:
        missing_required = check_dependencies()
        if missing_required:
            try:
                show_splash_screen(missing_required)
            except Exception as gui_err:
                log_setup(f"Failed to display splash screen: {gui_err}")
                req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
                if os.path.isfile(req_path):
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"],
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                write_deps_ok_marker()
        else:
            write_deps_ok_marker()
    else:
        log_setup("deps_ok marker exists -- skipping splash screen.")
        # Perform lightweight check for logs without showing splash
        check_dependencies()

    start_silent = "--silent" in sys.argv
    try:
        ensure_freellmapi_running()

        from gui_app import GlideTextApp
        app = GlideTextApp(start_silent=start_silent)
        app.mainloop()
    except Exception as e:
        print(f"[Main Crash] {e}")
        log_setup(f"Main startup crash: {e}\n{traceback.format_exc()}")
        try:
            with open("crash_report.txt", "w", encoding="utf-8") as f:
                sanitized_tb = sanitize_traceback(traceback.format_exc())
                f.write(sanitized_tb)
        except Exception as write_err:
            print(f"Failed to write crash report to disk: {write_err}")
            
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "GlideText Startup Crash",
                f"GlideText failed to start.\n\n"
                f"Error: {e}\n\n"
                f"The full traceback has been written to 'crash_report.txt' in the application directory."
            )
            root.destroy()
        except Exception as gui_err:
            print(f"Failed to show GUI error message: {gui_err}")
            
        sys.exit(1)
    finally:
        terminate_freellmapi()


if __name__ == "__main__":
    main()
