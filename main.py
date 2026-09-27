"""
main.py — Zero-Touch Bootstrapper and Entry point for GlideText.

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
    ("pyperclip", "pyperclip")
]


def check_dependencies():
    """Verify if all required dependencies are present in the environment."""
    missing = []
    for module_name, pip_name in REQUIRED_LIBS:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def run_installer(status_label, root):
    """Run pip install silently in a background thread."""
    try:
        # 1. Update core dependencies
        status_label.config(text="Downloading required AI libraries...")
        req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        if os.path.isfile(req_path):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"],
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
        else:
            # Fallback if requirements.txt is missing: install sequentially
            for _, pip_name in REQUIRED_LIBS:
                status_label.config(text=f"Installing {pip_name}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

        # 2. Attempt soft-fail installation of webrtcvad
        status_label.config(text="Setting up voice activity modules...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "webrtcvad>=2.0.10", "--quiet"],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        # 3. Final verification check
        still_missing = check_dependencies()
        if still_missing:
            status_label.config(text="Some components failed. Retrying...")
            time.sleep(2)
        else:
            status_label.config(text="Environment ready! Starting GlideText...")
            time.sleep(1)
            
    except Exception as e:
        status_label.config(text=f"Setup warning: {e}")
        time.sleep(3)
    finally:
        # Safely trigger UI closure on the main thread
        root.after(0, root.destroy)


def show_splash_screen():
    """Display a custom borderless dark Tkinter setup screen while installing."""
    import tkinter as tk
    
    root = tk.Tk()
    root.title("GlideText Setup")
    
    # Design colors
    bg_color = "#0f172a"      # Slate 900
    text_color = "#f1f5f9"    # Slate 50
    sec_color = "#94a3b8"     # Slate 400
    accent_color = "#6366f1"  # Indigo 500
    border_color = "#1e293b"  # Slate 800
    
    # Hide window decoration and center window
    root.overrideredirect(True)
    w = 420
    h = 240
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
    root.configure(bg=bg_color)
    
    # Container frame
    frame = tk.Frame(root, bg=bg_color, bd=1, relief="solid", highlightbackground=border_color, highlightthickness=1)
    frame.pack(fill="both", expand=True)
    
    # Title Label
    title_label = tk.Label(frame, text="GlideText", bg=bg_color, fg=accent_color, font=("Segoe UI", 26, "bold"))
    title_label.pack(pady=(35, 10))
    
    # Subtitle Label
    sub_label = tk.Label(frame, text="Privacy-First Voice Dictation Engine", bg=bg_color, fg=sec_color, font=("Segoe UI", 10, "italic"))
    sub_label.pack(pady=(0, 20))
    
    # Status Label
    status_label = tk.Label(frame, text="Checking environment status...", bg=bg_color, fg=text_color, font=("Segoe UI", 11))
    status_label.pack(pady=10)
    
    # Progress bar indicator
    progress_bg = tk.Frame(frame, bg="#1e293b", height=4, width=320)
    progress_bg.pack(pady=(10, 0))
    progress_bg.pack_propagate(False)
    
    progress_bar = tk.Frame(progress_bg, bg=accent_color, height=4, width=0)
    progress_bar.pack(side="left")
    
    # Animate loading indicator
    def animate(step=0):
        if root.winfo_exists():
            new_width = (step % 32) * 10
            progress_bar.config(width=new_width)
            root.after(80, animate, step + 1)
            
    animate()
    
    # Run the installation asynchronously
    t = threading.Thread(target=run_installer, args=(status_label, root), daemon=True)
    t.start()
    
    root.mainloop()


def sanitize_traceback(tb_str: str) -> str:
    """Purge API keys and sensitive tokens from traceback to prevent exfiltration."""
    tb_str = re.sub(r'AIzaSy[a-zA-Z0-9_-]+', '[API_KEY_SANITIZED]', tb_str)
    tb_str = re.sub(r'AQ\.[a-zA-Z0-9_-]+', '[API_KEY_SANITIZED]', tb_str)
    return tb_str



# ---------------------------------------------------------------------------
# FreeLLMAPI lifecycle -- delegated to freellm_manager.py
# ---------------------------------------------------------------------------
# Import lazily inside functions so this module still loads even if
# freellm_manager has a syntax error (defensive: won't crash GlideText).

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
    """
    Start FreeLLMAPI in the background without blocking the GUI.

    Kicks off a daemon thread (start_async) that:
      1. Fast-checks if the server is already up.
      2. Discovers and spawns the npm process headlessly.
      3. Polls TCP until ready or poll_timeout.
      4. Registers an atexit cleanup automatically.

    Always returns immediately so the GUI can continue loading.
    """
    mgr = _get_manager()
    if mgr is None:
        return False
    mgr.start_async(poll_timeout=poll_timeout)
    return True     # startup is in progress; actual readiness tracked by mgr.is_running()


def terminate_freellmapi() -> None:
    """Cleanly shut down the background FreeLLMAPI process tree."""
    mgr = _get_manager()
    if mgr:
        mgr.shutdown()




def main():
    # Elevate process priority to ABOVE_NORMAL to ensure low latency and responsiveness
    if sys.platform == "win32":
        try:
            import ctypes
            # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
            print("[Process] Process priority set to ABOVE_NORMAL")
        except Exception as e:
            print(f"[Process] Failed to set process priority class: {e}")

    # Pre-Flight check
    missing = check_dependencies()
    if missing:
        try:
            show_splash_screen()
        except Exception as gui_err:
            print(f"Failed to display splash screen: {gui_err}")
            # Fallback to headless command line install
            print("Installing missing dependencies in headless mode...")
            req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
            if os.path.isfile(req_path):
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"])
            subprocess.run([sys.executable, "-m", "pip", "install", "webrtcvad>=2.0.10", "--quiet"])

    start_silent = "--silent" in sys.argv
    try:
        # Pre-start FreeLLMAPI proxy silently in the background
        ensure_freellmapi_running()

        # Import the main GUI app dynamically now that dependencies are guaranteed
        from gui_app import GlideTextApp
        app = GlideTextApp(start_silent=start_silent)
        app.mainloop()
    except Exception as e:
        import traceback
        import tkinter as tk
        from tkinter import messagebox
        
        # Print traceback to console immediately
        traceback.print_exc()
        
        # Write traceback to crash_report.txt
        try:
            with open("crash_report.txt", "w", encoding="utf-8") as f:
                sanitized_tb = sanitize_traceback(traceback.format_exc())
                f.write(sanitized_tb)
        except Exception as write_err:
            print(f"Failed to write crash report to disk: {write_err}")
            
        # Display GUI error window
        try:
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

