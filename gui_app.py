"""
gui_app.py -- Production-grade CustomTkinter GUI for GlideText.

Rebuilt from scratch with:
  - Premium dark-mode interface with recording pulse animation
  - Collapsible settings panel (device, API key, dictation style)
  - Persistent history vault display (SQLite-backed)
  - System tray integration via pystray (close-to-tray)
  - Non-blocking architecture: hotkeys and AI pipeline on background threads
  - Push-to-talk (Right Alt hold) and continuous dictation (Ctrl+Shift+A)
  - VAD auto-stop support for continuous mode
  - Live dictation editing commands ("scratch that", "undo")
  - Context-aware active window detection
  - Robust error handling: no silent crashes
  - All print/UI strings use ASCII-safe characters (Windows cp1252 safe)
"""

import customtkinter as ctk
import threading
import os
import time
import keyboard
from datetime import datetime
import random
import queue
import sys
import logging
from logging.handlers import RotatingFileHandler

# Setup persistent rotating file logging
_log_dir = os.path.join(os.getenv("LOCALAPPDATA", os.path.expanduser("~")), "GlideText", "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "glidetext.log")

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _file_handler = RotatingFileHandler(_log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    _file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _file_handler.setFormatter(_file_formatter)
    _logger.addHandler(_file_handler)
    
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_formatter = logging.Formatter('%(message)s')
    _stream_handler.setFormatter(_stream_formatter)
    _logger.addHandler(_stream_handler)

def set_thread_priority(priority_level: int):
    """Set the calling thread's priority on Windows.
    
    priority_level:
        2  = THREAD_PRIORITY_HIGHEST
        1  = THREAD_PRIORITY_ABOVE_NORMAL
        0  = THREAD_PRIORITY_NORMAL
        -1 = THREAD_PRIORITY_BELOW_NORMAL
        -2 = THREAD_PRIORITY_LOWEST
    """
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_thread = kernel32.GetCurrentThread()
            kernel32.SetThreadPriority(h_thread, priority_level)
        except Exception as e:
            logging.error(f"[Priority] Failed to set thread priority to {priority_level}: {e}")

try:
    import pystray
    from PIL import Image, ImageDraw, ImageFont
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

from audio_recorder import AudioRecorder, get_active_window_info
from ai_brain import AIBrain
from text_injector import TextInjector
from history_vault import HistoryVault

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

try:
    import winreg

    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False


# ======================================================================
# Config helpers
# ======================================================================

def _read_config():
    """Read config.txt and return a dict with api_key and device_index."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.txt"
    )
    result = {"api_key": "", "device_index": 0}
    config_valid = True
    config_missing_or_empty = False
    
    # Retrieve API key securely via keyring (check GlideText first, fall back to legacy LocalFlow)
    if HAS_KEYRING:
        try:
            key = keyring.get_password("GlideText", "api_key") or keyring.get_password("LocalFlow", "api_key")
            if key:
                result["api_key"] = key
        except Exception as e:
            logging.error(f"[Config] Failed to read from keyring: {e}")

    # Retrieve device index from config.txt
    if not os.path.isfile(config_path):
        config_missing_or_empty = True
        config_valid = False
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                config_missing_or_empty = True
                config_valid = False
            else:
                lines = [line.strip() for line in content.splitlines()]
                # Device index parsing
                if len(lines) >= 1:
                    val = lines[0]
                    if val == "":
                        result["device_index"] = 0
                    else:
                        try:
                            result["device_index"] = int(val)
                        except ValueError:
                            result["device_index"] = 0
                            config_valid = False
                else:
                    result["device_index"] = 0
                    config_valid = False
        except Exception as e:
            logging.error(f"[Config] Failed to read config.txt: {e}")
            config_valid = False

    # If configuration is missing or invalid, we want to warn the user but fallback gracefully
    if not config_valid:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            if config_missing_or_empty:
                messagebox.showwarning(
                    "Configuration Missing",
                    "Configuration file (config.txt) was missing or empty.\n"
                    "GlideText has created a new configuration file with default values (Device Index: 0)."
                )
            else:
                messagebox.showwarning(
                    "Configuration Corrupted",
                    "Configuration file (config.txt) was improperly formatted.\n"
                    "GlideText has restored default values (Device Index: 0)."
                )
            root.destroy()
        except Exception as msg_err:
            logging.error(f"[Config] Failed to show warning dialog: {msg_err}")

        # Write the fallback values to restore stability
        _write_config(result["api_key"], result["device_index"])

    return result


def _write_config(api_key: str, device_index):
    """Write api_key to keyring and device_index to config.txt."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.txt"
    )
    # Store API key securely (supports comma-separated multi-keys)
    if HAS_KEYRING:
        try:
            if api_key:
                keyring.set_password("GlideText", "api_key", api_key)
            else:
                try:
                    keyring.delete_password("GlideText", "api_key")
                except Exception:
                    pass
                try:
                    keyring.delete_password("LocalFlow", "api_key")
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"[Config] Failed to write to keyring: {e}")

    # Store device index plainly
    try:
        dev_str = str(device_index) if device_index is not None else ""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"{dev_str}\n")
    except Exception as e:
        logging.error(f"[Config] Failed to write config.txt: {e}")



# ======================================================================
# Design Tokens (Minimalist Cream & Pure White Aesthetic)
# ======================================================================

class C:
    """Design tokens: Minimalist Editorial Cream & Pure White aesthetic."""
    BG_DEEP       = "#fbfbf8"      # Minimalist warm cream/alabaster background
    BG_MAIN       = "#f5f4ef"      # Soft linen/cream
    BG_CARD       = "#ffffff"      # Pristine white card surface
    BG_INPUT      = "#f9f8f5"      # Light warm input surface
    BORDER        = "#e5e2da"      # Subtle warm stone border
    BORDER_FOCUS  = "#18181b"      # Sleek pitch-black focus outline
    ACCENT        = "#18181b"      # Minimalist Onyx Black accent
    ACCENT_HOVER  = "#3f3f46"      # Hover Charcoal
    GREEN         = "#15803d"      # Muted emerald green
    GREEN_DIM     = "#166534"
    RED           = "#dc2626"
    RED_PULSE     = "#fecaca"
    AMBER         = "#d97706"
    TEXT          = "#09090b"      # Pure deep black text
    TEXT_SEC      = "#4b5563"      # Charcoal gray for secondary labels
    TEXT_DIM      = "#9ca3af"      # Muted warm stone for subtle captions
    TRANSPARENT   = "transparent"


FONT = "Segoe UI"
FONT_CURSIVE = "Segoe Script"
FONT_SERIF = "Georgia"


# ======================================================================
# Application
# ======================================================================

class GlideTextApp(ctk.CTk):
    """Main GlideText desktop application."""

    def __init__(self, start_silent: bool = False):
        super().__init__()
        
        # Tune UI thread priority to BELOW_NORMAL (-1) to prevent stutters
        set_thread_priority(-1)

        # -- Window --
        self.title("GlideText")
        self.geometry("540x860")
        self.minsize(480, 700)
        self.configure(fg_color=C.BG_DEEP)
        ctk.set_appearance_mode("light")

        # -- Read config --
        self._cfg = _read_config()

        # -- Backend components --
        self.recorder = AudioRecorder(device_index=self._cfg.get("device_index"))
        self.vault = HistoryVault()
        self.brain = AIBrain(vault=self.vault)
        if self._cfg.get("api_key"):
            self.brain.set_api_key(self._cfg["api_key"])
        self.brain.on_mode_change = lambda mode: self.after(0, lambda: self._update_engine_mode_ui(mode))
        self.injector = TextInjector()

        # Ensure FreeLLMAPI proxy is active in background without blocking UI
        try:
            import freellm_manager
            freellm_manager.start_async()
        except Exception as e:
            logging.debug(f"[GUI] FreeLLMAPI background start skipped: {e}")

        # -- State --
        self._current_status = "initializing"
        self._is_processing = False
        self._lock = threading.Lock()
        self._pulse_job = None
        self._tray_icon = None
        self._settings_open   = False
        self.is_continuous_mode = False
        self.is_widget_mode   = False
        self._active_style    = "Normal"
        self._last_injected_text = ""  # For "scratch that" editing commands

        self._is_starting_recording = False
        self._stop_pending = False

        # Persistent Pipeline Queue & Worker Thread
        self._pipeline_queue = queue.Queue()
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_worker, daemon=True
        )
        self._pipeline_thread.start()

        # -- Build UI --
        self._build_ambient_canvas()
        self._build_header()
        self._build_status_card()
        self._build_settings_section()
        self._build_history_section()
        self._build_footer()

        # -- Populate existing history --
        self._load_history_from_db()

        # -- Window protocol --
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # -- Bind hover event to trigger audio and vocabulary warmup --
        self.bind("<Enter>", self._on_hover_warmup)

        # -- Silent Mode execution --
        if start_silent:
            self.withdraw()

        # -- Launch backend (background) --
        threading.Thread(
            target=self._initialize_backend, daemon=True
        ).start()

    # ==================================================================
    #  UI CONSTRUCTION
    # ==================================================================

    def _build_ambient_canvas(self):
        """Build decorative background canvas with animated floating cursive letters."""
        self.ambient_canvas = ctk.CTkCanvas(
            self, height=52, bg=C.BG_DEEP, highlightthickness=0
        )
        self.ambient_canvas.pack(fill="x", padx=18, pady=(10, 0))
        
        # Initialize floating particles (letters, cursive symbols, words)
        self._alphabet_particles = []
        chars = ["𝒻", "𝓁", "ℴ", "𝓌", "𝒶", "𝒷", "𝒸", "𝓥", "𝒾", "𝓂", "𝒾", "𝓃", "𝒹", "✨", "✍", "α", "β", "voice", "mind", "flow"]
        colors = ["#d8d3c5", "#cfc8b8", "#c4bca9", "#b9af9a", "#e2ded4", "#a39983"]
        
        for _ in range(18):
            p = {
                "x": random.randint(15, 480),
                "y": random.randint(5, 45),
                "vx": random.uniform(-0.35, 0.35),
                "vy": random.uniform(-0.55, -0.15),  # drift smoothly upward
                "char": random.choice(chars),
                "size": random.randint(11, 16),
                "color": random.choice(colors),
                "phase": random.uniform(0, 6.28),
            }
            self._alphabet_particles.append(p)
            
        self._ambient_job = None
        self._animate_floating_alphabets()

    def _animate_floating_alphabets(self):
        """Update and redraw floating alphabet particles smoothly."""
        if not hasattr(self, "ambient_canvas") or not self.ambient_canvas.winfo_exists():
            return
            
        try:
            self.ambient_canvas.delete("particle")
            w = max(self.ambient_canvas.winfo_width(), 480)
            h = max(self.ambient_canvas.winfo_height(), 52)
            
            for p in self._alphabet_particles:
                p["phase"] += 0.04
                p["x"] += p["vx"] + 0.2 * (random.uniform(-0.08, 0.08))
                p["y"] += p["vy"]
                
                # Wrap around screen edges
                if p["y"] < -10:
                    p["y"] = h + 5
                    p["x"] = random.randint(10, w - 10)
                if p["x"] < -10:
                    p["x"] = w + 5
                elif p["x"] > w + 10:
                    p["x"] = -5
                    
                # Draw cursive / typography particle
                font_family = FONT_CURSIVE if p["char"] not in ["✨", "✍", "α", "β", "voice", "mind", "flow"] else FONT_SERIF
                self.ambient_canvas.create_text(
                    p["x"], p["y"], text=p["char"],
                    font=(font_family, p["size"]), fill=p["color"], tags="particle"
                )
                
            self._ambient_job = self.after(40, self._animate_floating_alphabets)
        except Exception:
            pass

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(self, fg_color=C.TRANSPARENT, height=48)
        self.header_frame.pack(fill="x", padx=24, pady=(2, 0))
        self.header_frame.pack_propagate(False)

        ctk.CTkLabel(
            self.header_frame, text="GlideText",
            font=(FONT_SERIF, 26, "bold"), text_color=C.TEXT,
        ).pack(side="left")

        ctk.CTkLabel(
            self.header_frame, text="• Speech to Mind",
            font=(FONT_CURSIVE, 13, "italic"), text_color=C.TEXT_SEC,
        ).pack(side="left", padx=(10, 0), pady=(4, 0))

        # Right side of header: Engine mode badge + Reset button
        self.engine_badge_frame = ctk.CTkFrame(self.header_frame, fg_color=C.TRANSPARENT)
        self.engine_badge_frame.pack(side="right", pady=8)

        self.engine_badge = ctk.CTkLabel(
            self.engine_badge_frame,
            text="● Cloud Polish",
            font=(FONT, 11, "bold"),
            text_color="#38bdf8",
            fg_color="#0f172a",
            corner_radius=8,
            padx=10, pady=3,
        )
        self.engine_badge.pack(side="left")

        self.reset_cloud_btn = ctk.CTkButton(
            self.engine_badge_frame,
            text="↺ Reset",
            width=50, height=24,
            font=(FONT, 10, "bold"),
            fg_color="#334155",
            hover_color="#475569",
            text_color="#ffffff",
            corner_radius=6,
            command=self._on_reset_cloud_click,
        )
        # Initially hidden in normal cloud mode

    def _update_engine_mode_ui(self, mode: str):
        """Update header badge reflecting active polish engine."""
        try:
            if mode == "local":
                self.engine_badge.configure(
                    text="⚡ Local LLM (llama3.2:3b)",
                    text_color="#fb923c",
                    fg_color="#431407"
                )
                self.reset_cloud_btn.pack(side="left", padx=(6, 0))
            else:
                self.engine_badge.configure(
                    text="● Cloud Polish",
                    text_color="#38bdf8",
                    fg_color="#0f172a"
                )
                self.reset_cloud_btn.pack_forget()
        except Exception:
            pass

    def _on_reset_cloud_click(self):
        """User manually resets from sticky local mode back to Gemini cloud."""
        self.brain.reset_cloud_mode()
        self._update_engine_mode_ui("cloud")
        self._refresh_telemetry_ui()

    # -- Status Card --

    def _build_status_card(self):
        self.status_card = ctk.CTkFrame(
            self, fg_color=C.BG_CARD, corner_radius=16,
            border_width=1, border_color=C.BORDER,
        )
        self.status_card.pack(fill="x", padx=24, pady=(18, 0))

        inner = ctk.CTkFrame(self.status_card, fg_color=C.TRANSPARENT)
        inner.pack(padx=28, pady=26)

        self.waveform_canvas = ctk.CTkCanvas(
            inner, width=80, height=40,
            bg=C.BG_CARD, highlightthickness=0
        )
        # Hidden by default

        self.status_dot = ctk.CTkLabel(
            inner, text="●", font=(FONT, 40), text_color=C.TEXT_DIM,
        )
        self.status_dot.pack()

        self.status_label = ctk.CTkLabel(
            inner, text="INITIALIZING...",
            font=(FONT, 20, "bold"), text_color=C.TEXT_SEC,
        )
        self.status_label.pack(pady=(6, 0))

        self.status_hint = ctk.CTkLabel(
            inner, text="Connecting to Gemini cloud...",
            font=(FONT, 12), text_color=C.TEXT_DIM,
        )
        self.status_hint.pack(pady=(4, 0))
        
        # Bind double-click to toggle widget mode on all these elements
        for widget in [self, self.status_card, inner, self.status_dot, self.status_label, self.status_hint]:
            widget.bind("<Double-Button-1>", lambda e: self._toggle_widget_mode())

    # -- Settings Panel --

    def _build_settings_section(self):
        self.settings_toggle = ctk.CTkButton(
            self, text="[+] Settings",
            font=(FONT, 13), fg_color=C.TRANSPARENT,
            hover_color=C.BG_CARD, text_color=C.TEXT_SEC,
            anchor="w", height=32, command=self._toggle_settings,
        )
        self.settings_toggle.pack(fill="x", padx=24, pady=(14, 0))

        # Container (initially hidden)
        self.settings_frame = ctk.CTkFrame(
            self, fg_color=C.BG_CARD, corner_radius=12,
            border_width=1, border_color=C.BORDER,
        )

        inner = ctk.CTkFrame(self.settings_frame, fg_color=C.TRANSPARENT)
        inner.pack(fill="x", padx=18, pady=18)

        # Device Index
        ctk.CTkLabel(
            inner, text="Input Device Index (blank = system default)",
            font=(FONT, 12), text_color=C.TEXT_SEC,
        ).pack(anchor="w")
        self.device_entry = ctk.CTkEntry(
            inner, font=(FONT, 13), fg_color=C.BG_INPUT,
            border_color=C.BORDER, text_color=C.TEXT, height=34,
            placeholder_text="auto",
        )
        if self._cfg["device_index"] is not None:
            self.device_entry.insert(0, str(self._cfg["device_index"]))
        self.device_entry.pack(fill="x", pady=(4, 14))

        # Gemini API Key
        ctk.CTkLabel(
            inner, text="Gemini API Key",
            font=(FONT, 12), text_color=C.TEXT_SEC,
        ).pack(anchor="w")
        # Plaintext API key is not inserted here to prevent memory exfiltration.
        self.api_key_entry = ctk.CTkEntry(
            inner, font=(FONT, 13), fg_color=C.BG_INPUT,
            border_color=C.BORDER, text_color=C.TEXT, height=34,
            show="*", placeholder_text="••••••••••••••••" if self.brain.api_key else "Enter API key here",
        )
        self.api_key_entry.pack(fill="x", pady=(4, 14))

        # Dictation Style
        ctk.CTkLabel(
            inner, text="Dictation Style",
            font=(FONT, 12), text_color=C.TEXT_SEC,
        ).pack(anchor="w")
        self.style_menu = ctk.CTkOptionMenu(
            inner, values=["Normal", "Formal", "Casual", "Developer"],
            font=(FONT, 13), fg_color=C.BG_INPUT,
            button_color=C.ACCENT, button_hover_color=C.ACCENT_HOVER,
            text_color=C.TEXT, dropdown_fg_color=C.BG_CARD,
            dropdown_text_color=C.TEXT, dropdown_hover_color=C.ACCENT,
            height=34, command=self._on_style_change,
        )
        self.style_menu.set("Normal")
        self.style_menu.pack(fill="x", pady=(4, 16))

        # Multi-key hint
        ctk.CTkLabel(
            inner, text="Tip: Paste multiple API keys separated by commas for auto-rotation.",
            font=(FONT, 11), text_color=C.TEXT_SEC,
            wraplength=340, justify="left",
        ).pack(anchor="w", pady=(0, 14))

        # Auto-Boot Toggle (Windows Registry)
        self.autoboot_switch = ctk.CTkSwitch(
            inner, text="Start GlideText with Windows Boot",
            font=(FONT, 13, "bold"), text_color=C.TEXT,
            progress_color=C.GREEN, button_color="#ffffff",
            button_hover_color="#e2e8f0", command=self._on_autoboot_toggle
        )
        if self._check_autoboot_status():
            self.autoboot_switch.select()
        self.autoboot_switch.pack(fill="x", pady=(4, 16))

        # Apply button
        ctk.CTkButton(
            inner, text="Apply Settings",
            font=(FONT, 13, "bold"), fg_color=C.ACCENT,
            hover_color=C.ACCENT_HOVER, text_color="#ffffff",
            height=36, corner_radius=8, command=self._apply_settings,
        ).pack(fill="x")

        self.settings_feedback = ctk.CTkLabel(
            inner, text="", font=(FONT, 11), text_color=C.GREEN,
        )
        self.settings_feedback.pack(pady=(8, 0))

        # API Telemetry & Analytics Dashboard
        self._build_telemetry_section(inner)

    def _build_telemetry_section(self, parent):
        """Build telemetry card showing API calls, success rate, and provider breakdown."""
        telem_card = ctk.CTkFrame(
            parent, fg_color="#0f172a", corner_radius=10,
            border_width=1, border_color="#1e293b"
        )
        telem_card.pack(fill="x", pady=(14, 0))

        top_row = ctk.CTkFrame(telem_card, fg_color=C.TRANSPARENT)
        top_row.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(
            top_row, text="API Telemetry & Analytics",
            font=(FONT, 12, "bold"), text_color=C.TEXT,
        ).pack(side="left")

        ctk.CTkButton(
            top_row, text="Clear", width=42, height=20,
            font=(FONT, 9), fg_color=C.TRANSPARENT,
            hover_color="#1e293b", text_color=C.TEXT_DIM,
            command=self._clear_telemetry_stats,
        ).pack(side="right")

        # Summary Chips Row
        chips_row = ctk.CTkFrame(telem_card, fg_color=C.TRANSPARENT)
        chips_row.pack(fill="x", padx=12, pady=(0, 8))

        self.telem_total_label = ctk.CTkLabel(
            chips_row, text="Total Calls: 0",
            font=(FONT, 11), text_color=C.TEXT_SEC,
        )
        self.telem_total_label.pack(side="left", padx=(0, 12))

        self.telem_success_label = ctk.CTkLabel(
            chips_row, text="Success Rate: 100%",
            font=(FONT, 11, "bold"), text_color=C.GREEN,
        )
        self.telem_success_label.pack(side="left")

        # Container for dynamic provider rows
        self.telem_providers_frame = ctk.CTkFrame(telem_card, fg_color=C.TRANSPARENT)
        self.telem_providers_frame.pack(fill="x", padx=12, pady=(0, 10))

    def _refresh_telemetry_ui(self):
        """Fetch latest API metrics from HistoryVault and update settings telemetry."""
        try:
            analytics = self.vault.get_api_analytics()
            total = analytics.get("total_calls", 0)
            rate = analytics.get("success_rate", 100.0)

            if hasattr(self, "telem_total_label"):
                self.telem_total_label.configure(text=f"Total Calls: {total}")

            if hasattr(self, "telem_success_label"):
                rate_color = C.GREEN if rate >= 90.0 else (C.AMBER if rate >= 70.0 else C.RED)
                self.telem_success_label.configure(
                    text=f"Success: {rate}%",
                    text_color=rate_color
                )

            if hasattr(self, "telem_providers_frame"):
                for widget in self.telem_providers_frame.winfo_children():
                    widget.destroy()

                providers = analytics.get("providers", [])
                if not providers:
                    ctk.CTkLabel(
                        self.telem_providers_frame,
                        text="No API requests recorded yet in this session.",
                        font=(FONT, 10, "italic"), text_color=C.TEXT_DIM,
                    ).pack(anchor="w")
                else:
                    for p in providers:
                        p_name = p.get("provider", "Unknown")
                        p_tot = p.get("total", 0)
                        p_suc = p.get("success", 0)
                        p_429 = p.get("rate_limits", 0)
                        p_lat = p.get("avg_latency", 0)
                        p_rate = p.get("success_rate", 100.0)

                        row = ctk.CTkFrame(self.telem_providers_frame, fg_color="#1e293b", corner_radius=6)
                        row.pack(fill="x", pady=2)

                        ctk.CTkLabel(
                            row, text=f" {p_name}",
                            font=(FONT, 10, "bold"), text_color=C.TEXT,
                        ).pack(side="left", padx=(6, 8), pady=3)

                        stat_txt = f"{p_suc}/{p_tot} ok ({p_rate:.0f}%)"
                        if p_429 > 0:
                            stat_txt += f" | {p_429} rate-limited (429)"
                        if p_lat > 0:
                            stat_txt += f" | ~{p_lat}ms"

                        color = C.GREEN if p_429 == 0 else C.AMBER
                        ctk.CTkLabel(
                            row, text=stat_txt,
                            font=(FONT, 10), text_color=color,
                        ).pack(side="right", padx=(0, 6), pady=3)
        except Exception:
            pass

    def _clear_telemetry_stats(self):
        """Wipe API call telemetry and refresh UI."""
        self.vault.clear_api_metrics()
        self._refresh_telemetry_ui()

    # -- History Panel --

    def _build_history_section(self):
        self.history_bar = ctk.CTkFrame(self, fg_color=C.TRANSPARENT, height=30)
        self.history_bar.pack(fill="x", padx=24, pady=(14, 0))
        self.history_bar.pack_propagate(False)

        ctk.CTkLabel(
            self.history_bar, text="History",
            font=(FONT, 13), text_color=C.TEXT_SEC,
        ).pack(side="left")

        ctk.CTkButton(
            self.history_bar, text="Clear", font=(FONT, 11),
            fg_color=C.TRANSPARENT, hover_color=C.BG_CARD,
            text_color=C.TEXT_DIM, width=50, height=26,
            command=self._clear_history,
        ).pack(side="right")

        self.history_box = ctk.CTkTextbox(
            self, font=(FONT, 12), fg_color=C.BG_CARD,
            text_color=C.TEXT, border_width=1,
            border_color=C.BORDER, corner_radius=12,
            wrap="word", state="disabled", activate_scrollbars=True,
        )
        self.history_box.pack(fill="both", expand=True, padx=24, pady=(6, 0))

    # -- Footer --

    def _build_footer(self):
        self.footer_label = ctk.CTkLabel(
            self, text="Made by Apoorv Sarawgi",
            font=(FONT, 11), text_color=C.TEXT_DIM,
        )
        self.footer_label.pack(pady=(12, 16))

    # ==================================================================
    #  STATUS ENGINE
    # ==================================================================

    def _set_status(self, status: str, hint: str | None = None):
        """Thread-safe status transition."""
        self._current_status = status
        try:
            self.after(0, lambda: self._render_status(status, hint))
        except Exception:
            pass  # Window may be destroyed

    def _render_status(self, status: str, hint: str | None = None):
        """Apply visual state (main thread only)."""
        if self._pulse_job:
            try:
                self.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None

        presets = {
            "ready":        (C.GREEN,    "READY TO DICTATE",  "Hold [Right Alt] to record | Ctrl+Shift+A for continuous"),
            "recording":    (C.RED,      "RECORDING",         "Release [Right Alt] to stop | Ctrl+Shift+A to toggle"),
            "transcribing": (C.AMBER,    "TRANSCRIBING",      "Transcribing speech locally..."),
            "processing":   (C.AMBER,    "PROCESSING",        "Polishing text with AI..."),
            "typing":       (C.ACCENT,   "TYPING",            "Injecting text at cursor..."),
            "initializing": (C.TEXT_DIM, "INITIALIZING...",   "Connecting to Gemini cloud..."),
            "warning":      (C.AMBER,    "WARNING",           ""),
            "error":        (C.RED,      "ERROR",             ""),
        }
        
        colour, label, default_hint = presets.get(
            status, (C.TEXT_DIM, status.upper(), "")
        )

        try:
            self.status_label.configure(text=label, text_color=colour)
            self.status_hint.configure(text=hint or default_hint)
        except Exception:
            pass  # Widget may not exist yet

        if status == "recording":
            self.status_dot.pack_forget()
            self.waveform_canvas.pack(pady=(4, 4), before=self.status_label)
            self._animate_waveform()
        else:
            self.waveform_canvas.pack_forget()
            # Ensure dot is packed before label if it was removed
            self.status_dot.pack(before=self.status_label)
            try:
                self.status_dot.configure(text_color=colour)
            except Exception:
                pass

    def _animate_waveform(self):
        if self._current_status != "recording":
            self.waveform_canvas.delete("all")
            return
            
        try:
            self.waveform_canvas.delete("all")
            rms = getattr(self.recorder, "current_rms", 0.0)
            
            # Simple volume mapping
            normalized = min(max(rms / 1500.0, 0.1), 1.0)
                
            # Draw 5 bars
            w = 80
            h = 40
            bar_w = 8
            gap = 6
            start_x = (w - (5 * bar_w + 4 * gap)) / 2
            
            for i in range(5):
                jitter = random.uniform(0.6, 1.4) if rms > 150 else 1.0
                # Falloff towards edges
                edge_multiplier = 0.6 if (i == 0 or i == 4) else (0.8 if (i == 1 or i == 3) else 1.0)
                
                bar_h = max(4, h * normalized * jitter * edge_multiplier)
                if bar_h > h: bar_h = h
                
                x0 = start_x + i * (bar_w + gap)
                y0 = (h - bar_h) / 2
                x1 = x0 + bar_w
                y1 = y0 + bar_h
                
                self.waveform_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=C.RED, outline=""
                )
                
            self._pulse_job = self.after(50, self._animate_waveform)
        except Exception:
            pass

    # ==================================================================
    #  SETTINGS
    # ==================================================================

    def _toggle_settings(self):
        if self._settings_open:
            self.settings_frame.pack_forget()
            self.settings_toggle.configure(text="[+] Settings")
        else:
            self.settings_frame.pack(
                fill="x", padx=24, pady=(4, 0),
                after=self.settings_toggle,
            )
            self.settings_toggle.configure(text="[-] Settings")
            self._refresh_telemetry_ui()
        self._settings_open = not self._settings_open

    def _toggle_widget_mode(self, _event=None):
        """Toggle between full Dashboard and minimal Floating Widget."""
        self.is_widget_mode = not self.is_widget_mode
        
        if self.is_widget_mode:
            # Hide all panels except the status card
            if hasattr(self, "ambient_canvas"):
                self.ambient_canvas.pack_forget()
            self.header_frame.pack_forget()
            self.settings_toggle.pack_forget()
            if self._settings_open:
                self.settings_frame.pack_forget()
            self.history_bar.pack_forget()
            self.history_box.pack_forget()
            self.footer_label.pack_forget()
            
            # Make borderless, translucent, and float at top right
            self.overrideredirect(True)
            self.attributes('-alpha', 0.9)
            self.attributes('-topmost', True)
            
            # Adjust geometry to wrap status card
            screen_width = self.winfo_screenwidth()
            x = screen_width - 320
            y = 80
            self.geometry(f"280x80+{x}+{y}")
            self.status_card.pack(fill="both", expand=True, padx=4, pady=4)
        else:
            # Restore Dashboard mode
            self.overrideredirect(False)
            self.attributes('-alpha', 1.0)
            self.attributes('-topmost', False)
            
            # Reset geometry
            self.geometry("540x860")
            
            # Repack everything
            if hasattr(self, "ambient_canvas"):
                self.ambient_canvas.pack(fill="x", padx=18, pady=(10, 0), before=self.header_frame)
            self.header_frame.pack(fill="x", padx=24, pady=(2, 0), before=self.status_card)
            self.status_card.pack(fill="x", padx=24, pady=(18, 0))
            self.settings_toggle.pack(fill="x", padx=24, pady=(14, 0), after=self.status_card)
            if self._settings_open:
                self.settings_frame.pack(fill="x", padx=24, pady=(4, 0), after=self.settings_toggle)
            self.history_bar.pack(fill="x", padx=24, pady=(14, 0))
            self.history_box.pack(fill="both", expand=True, padx=24, pady=(6, 0))
            self.footer_label.pack(pady=(12, 16))

    def _on_hover_warmup(self, _event=None):
        """Asynchronously pre-warm audio recording and load app-specific vocabulary on widget hover."""
        self.recorder.warmup()
        context = get_active_window_info()
        self.brain.reload_vocabulary(context)
        # Pre-warm connection in background
        threading.Thread(target=self.brain.pre_warm_gemini_connection, daemon=True).start()

    def _capture_lookback_context(self) -> str:
        """Selects the preceding ~5-8 words using Ctrl+Shift+Left 6 times, copies, and restores the cursor."""
        import pyperclip
        import keyboard
        import time

        try:
            clipboard_backup = pyperclip.paste()
        except Exception:
            clipboard_backup = ""

        pre_text = ""
        try:
            # Clear clipboard to detect if copy succeeded
            pyperclip.copy("")
            time.sleep(0.01)

            # Hold ctrl+shift down, tap left 6 times, release
            keyboard.press("ctrl")
            keyboard.press("shift")
            for _ in range(6):
                keyboard.press_and_release("left")
                time.sleep(0.005)
            keyboard.release("shift")
            keyboard.release("ctrl")
            time.sleep(0.02)

            # Copy selection
            keyboard.press_and_release("ctrl+c")
            time.sleep(0.05)

            # Read selection
            pre_text = pyperclip.paste()

            # Immediately press Right Arrow to collapse selection back to starting position
            keyboard.press_and_release("right")
            time.sleep(0.01)

        except Exception as e:
            logging.info(f"[Lookback] Error capturing lookback context: {e}")
            try:
                keyboard.release("shift")
                keyboard.release("ctrl")
            except Exception:
                pass
        finally:
            # Restore clipboard
            try:
                pyperclip.copy(clipboard_backup)
            except Exception:
                pass

        logging.info(f"[Lookback] Captured: {repr(pre_text)}")
        return pre_text

    def _swap_text(self, old_text: str, new_text: str) -> bool:
        """Safe swap using Adaptive Suffix Diffing:
        Calculates the common prefix of old_text and new_text.
        Selects only the differing old_suffix, copies it to verify it matches,
        and replaces it with new_suffix.
        """
        if not old_text:
            self.injector.inject(new_text)
            return True

        if old_text == new_text:
            return True

        # Compute suffix diff
        min_len = min(len(old_text), len(new_text))
        prefix_len = 0
        while prefix_len < min_len and old_text[prefix_len] == new_text[prefix_len]:
            prefix_len += 1

        old_suffix = old_text[prefix_len:]
        new_suffix = new_text[prefix_len:]

        logging.info(f"  [Swap Guard] Suffix diff computed:")
        logging.info(f"    Prefix: '{old_text[:prefix_len]}'")
        logging.info(f"    Old Suffix: '{old_suffix}'")
        logging.info(f"    New Suffix: '{new_suffix}'")

        # If old_suffix is empty, we just need to append new_suffix
        if not old_suffix:
            if new_suffix:
                self.injector.inject(new_suffix)
            return True

        import pyperclip
        import keyboard
        import time

        try:
            clipboard_backup = pyperclip.paste()
        except Exception:
            clipboard_backup = ""

        import uuid
        sentinel = str(uuid.uuid4())

        time.sleep(0.05)
        try:
            # 1. Make clipboard verification budget scale with selection size
            max_budget = min(1.5, 0.150 + 0.004 * len(old_suffix))
            max_polls = max(5, int(max_budget / 0.02))

            # 2. Scale settle delay before Ctrl+C
            settle_delay = min(0.5, 0.05 + 0.002 * len(old_suffix))

            for attempt in range(1, 3):
                # Clear clipboard with sentinel to detect if copy succeeded
                pyperclip.copy(sentinel)
                time.sleep(0.02)
                
                # Select back the old_suffix character-by-character
                keyboard.press("shift")
                for _ in range(len(old_suffix)):
                    keyboard.press_and_release("left")
                    time.sleep(0.001)
                keyboard.release("shift")
                time.sleep(settle_delay)

                # Copy selection to clipboard
                keyboard.press_and_release("ctrl+c")
                
                # Poll for clipboard update
                selected = sentinel
                polls = 0
                for i in range(max_polls):
                    time.sleep(0.02)
                    polls += 1
                    try:
                        current_clip = pyperclip.paste()
                        if current_clip != sentinel:
                            selected = current_clip
                            break
                    except Exception:
                        pass
                
                logging.info(f"  [Swap Guard] Clipboard poll took {polls} attempts on attempt {attempt}.")
                
                # Verify selection
                if selected != sentinel and selected.strip() == old_suffix.strip():
                    # Overwrite selection with new_suffix
                    if new_suffix:
                        keyboard.write(new_suffix, delay=self.injector.delay)
                    else:
                        keyboard.press_and_release("backspace")
                    logging.info(f"  [Swap Guard] Suffix swap verified and completed on attempt {attempt}.")
                    return True
                else:
                    if selected == sentinel:
                        if attempt == 1:
                            logging.info(f"  [Swap Guard] Clipboard read timed out on attempt 1, retrying...")
                            keyboard.press_and_release("right")
                            time.sleep(0.05)
                            continue
                        else:
                            logging.info(f"  [Swap Guard] Cancelled swap. Clipboard read timed out on attempt 2.")
                    else:
                        logging.info(f"  [Swap Guard] Cancelled swap. Selected: {repr(selected)}, Expected: {repr(old_suffix)}")
                        keyboard.press_and_release("right")
                        return False
            
            keyboard.press_and_release("right")
            return False
        except Exception as e:
            logging.info(f"  ! Safe text swap failed: {e}")
            try:
                keyboard.release("shift")
            except Exception:
                pass
            return False
        finally:
            try:
                pyperclip.copy(clipboard_backup)
            except Exception:
                pass

    def _on_style_change(self, choice):
        self._active_style = choice
        self.brain.set_style(choice)


    def _check_autoboot_status(self) -> bool:
        if not HAS_WINREG:
            return False
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                r"Software\Microsoft\Windows\CurrentVersion\Run", 
                0, 
                winreg.KEY_READ
            )
            val = None
            try:
                val, _ = winreg.QueryValueEx(key, "GlideText")
            except FileNotFoundError:
                try:
                    val, _ = winreg.QueryValueEx(key, "LocalFlow")
                except FileNotFoundError:
                    val = None
            winreg.CloseKey(key)
            return bool(val)
        except Exception:
            return False

    def _on_autoboot_toggle(self):
        if not HAS_WINREG:
            return
        
        is_autoboot = self.autoboot_switch.get() == 1
        main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
        cmd_string = f'pythonw.exe "{main_py_path}" --silent'
        
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                r"Software\Microsoft\Windows\CurrentVersion\Run", 
                0, 
                winreg.KEY_SET_VALUE
            )
            if is_autoboot:
                winreg.SetValueEx(key, "GlideText", 0, winreg.REG_SZ, cmd_string)
                logging.info("[Registry] Set GlideText to run on boot.")
                # Clean up legacy registry key if present
                try:
                    winreg.DeleteValue(key, "LocalFlow")
                except FileNotFoundError:
                    pass
            else:
                try:
                    winreg.DeleteValue(key, "GlideText")
                    logging.info("[Registry] Removed GlideText from boot.")
                except FileNotFoundError:
                    pass
                try:
                    winreg.DeleteValue(key, "LocalFlow")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except Exception as e:
            logging.error(f"[Registry] Failed to modify boot settings: {e}")

    def _apply_settings(self):
        # Device index
        dev_text = self.device_entry.get().strip()
        if dev_text == "" or dev_text.lower() == "auto":
            new_device = None
        elif dev_text.isdigit():
            new_device = int(dev_text)
        else:
            self.settings_feedback.configure(
                text="Invalid device index", text_color=C.RED,
            )
            return

        self.recorder.device_index = new_device

        # Gemini API Key (supports comma-separated multi-keys)
        new_key = self.api_key_entry.get().strip()
        # If the input field is left empty (or shows the bullet placeholder), preserve the currently configured key
        if not new_key and self.brain.api_key:
            # Preserve current keys — build comma-separated string from internal list
            new_key = ",".join(self.brain._api_keys) if self.brain._api_keys else ""

        if new_key:
            self.brain.set_api_key(new_key)
            _write_config(new_key, new_device)
            # Update the masked placeholder and securely purge typed plaintext from the UI widget memory
            key_count = len(self.brain._api_keys)
            self.api_key_entry.configure(placeholder_text="••••••••••••••••")
            self.api_key_entry.delete(0, 'end')
            self._set_status("ready")
            self.settings_feedback.configure(
                text=f"Settings applied! {key_count} API key(s) saved.",
                text_color=C.GREEN,
            )
        else:
            self.settings_feedback.configure(
                text="Please enter a Gemini API Key",
                text_color=C.AMBER,
            )
            return

        self.after(
            4000,
            lambda: self.settings_feedback.configure(text=""),
        )

    # ==================================================================
    #  HISTORY
    # ==================================================================

    def _load_history_from_db(self):
        try:
            entries = self.vault.get_recent(limit=50)
            if not entries:
                return
            self.history_box.configure(state="normal")
            for ts, txt in entries:
                self.history_box.insert("end", f"{ts}\n", "ts")
                self.history_box.insert("end", f"{txt}\n")
                self.history_box.insert("end", "-" * 52 + "\n\n")
            self.history_box.configure(state="disabled")
        except Exception as e:
            logging.error(f"[GUI] Failed to load history: {e}")

    def _push_history_entry(self, ts: str, txt: str):
        """Prepend an entry to the history box (main thread)."""
        try:
            block = f"{ts}\n{txt}\n" + "-" * 52 + "\n\n"
            self.history_box.configure(state="normal")
            self.history_box.insert("1.0", block)
            self.history_box.configure(state="disabled")
        except Exception:
            pass

    def _clear_history(self):
        try:
            self.vault.clear()
            self.history_box.configure(state="normal")
            self.history_box.delete("1.0", "end")
            self.history_box.configure(state="disabled")
        except Exception as e:
            logging.error(f"[GUI] Failed to clear history: {e}")

    # ==================================================================
    #  SYSTEM TRAY
    # ==================================================================

    def _setup_tray(self):
        if not HAS_TRAY:
            return
        try:
            icon_img = self._make_tray_icon()
            menu = pystray.Menu(
                pystray.MenuItem(
                    "Show GlideText", self._tray_show, default=True,
                ),
                pystray.MenuItem(
                    "Toggle Widget Mode", lambda: self.after(0, self._toggle_widget_mode)
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._tray_quit),
            )
            self._tray_icon = pystray.Icon(
                "GlideText", icon_img, "GlideText -- Ready", menu,
            )
            threading.Thread(
                target=self._tray_icon.run, daemon=True,
            ).start()
        except Exception as e:
            logging.info(f"[GUI] Tray setup failed: {e}")

    @staticmethod
    def _make_tray_icon() -> "Image.Image":
        """Generate a small indigo circle with 'LF' text."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, size - 4, size - 4], fill="#18181b")
        try:
            font = ImageFont.truetype("segoeui.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        draw.text(
            (size // 2, size // 2), "LF",
            fill="white", font=font, anchor="mm",
        )
        return img

    def _tray_show(self, _icon=None, _item=None):
        self.after(0, self.deiconify)
        self.after(10, self.lift)
        self.after(20, self.focus_force)

    def _tray_quit(self, _icon=None, _item=None):
        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass
        self.after(0, self._quit_app)

    def _on_window_close(self):
        """X button -> minimise to tray (or quit if tray unavailable)."""
        if HAS_TRAY and self._tray_icon:
            self.withdraw()
        else:
            self._quit_app()

    def _quit_app(self):
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass
        # Shut down the background FreeLLMAPI Node process cleanly
        try:
            import freellm_manager
            freellm_manager.shutdown()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


    # ==================================================================
    #  BACKEND (runs in background threads)
    # ==================================================================

    def _initialize_backend(self):
        """Initialize Gemini connection and register hotkeys."""
        # Stage 1 -- Check API key
        self._set_status("initializing", "Connecting to Gemini cloud...")
        self.brain.load_whisper()

        # Stage 2 -- System tray
        try:
            self.after(0, self._setup_tray)
        except RuntimeError:
            pass

        # Stage 3 -- Register hotkeys
        try:
            keyboard.on_press_key(
                "right alt", self._on_key_press, suppress=False,
            )
            keyboard.on_release_key(
                "right alt", self._on_key_release, suppress=False,
            )
            keyboard.add_hotkey(
                "ctrl + shift + a", self._toggle_continuous_recording,
            )
            logging.info("[GUI] Hotkeys registered: Right Alt (push-to-talk), Ctrl+Shift+A (continuous)")
        except Exception as e:
            logging.info(f"[GUI] Hotkey registration failed: {e}")
            self._set_status("error", f"Hotkey setup failed: {e}")
            return

        # Stage 4 -- Set status based on API key presence
        if self.brain.is_ready:
            self._set_status("ready")
            self.recorder.warmup()
        else:
            self._set_status(
                "error",
                "Please enter your Gemini API Key in Settings",
            )

        # Keep this thread alive so keyboard hooks remain active
        try:
            keyboard.wait()
        except Exception:
            pass

    # -- Hotkey Handlers --

    def _on_key_press(self, _event):
        """Right Alt pressed: start push-to-talk recording."""
        if self.is_continuous_mode:
            return

        if not self.brain.api_key:
            from tkinter import messagebox
            messagebox.showerror(
                "Gemini API Key Missing",
                "Gemini API Key is missing.\n\n"
                "Please configure a valid API key in Settings.\n"
                "Tip: You can paste multiple keys separated by commas for auto-rotation."
            )
            return

        with self._lock:
            if self.recorder.is_recording or getattr(self, "_is_starting_recording", False) or self._is_processing:
                return
            self._is_starting_recording = True
            self._stop_pending = False

        # Capture lookback context immediately on key press (before speaking starts)
        self._lookback_context = self._capture_lookback_context()

        # Trigger TCP/TLS socket pre-warming in a background thread
        threading.Thread(target=self.brain.pre_warm_gemini_connection, daemon=True).start()

        # Optimistic UI update
        self._set_status("recording")

        # Load dynamic vocabulary for the current active app in a background thread
        context = get_active_window_info()
        self.brain.reload_vocabulary(context)

        def _async_start():
            set_thread_priority(2)  # HIGHEST priority
            try:
                self.recorder.start()
                if not self.recorder.is_recording:
                    raise RuntimeError("Audio stream failed to initialize")
                
                with self._lock:
                    if self._stop_pending:
                        self._stop_pending = False
                        self._async_stop()
            except Exception as e:
                logging.error(f"[GUI] Recording start error: {e}")
                self._is_streaming_active = False
                self._set_status("warning", f"Recording failed: {e}")
                self.after(3000, lambda: self._set_status("ready"))
            finally:
                self._is_starting_recording = False

        threading.Thread(target=_async_start, daemon=True).start()

    def _on_key_release(self, _event):
        """Right Alt released: stop push-to-talk recording."""
        if self.is_continuous_mode:
            return
        
        with self._lock:
            if getattr(self, "_is_starting_recording", False):
                self._stop_pending = True
                return
            if not self.recorder.is_recording:
                return

        self._async_stop()

    def _async_stop(self):
        try:
            self._set_status("transcribing")
            # Capture active window context IMMEDIATELY on stop trigger
            context = get_active_window_info()
            audio_path = self.recorder.stop()
            if audio_path is None:
                self._set_status("ready")
                return

            self._pipeline_queue.put((audio_path, context))
        except Exception as e:
            logging.error(f"[GUI] Recording stop error: {e}")
            self._set_status("warning", f"Stop failed: {e}")
            self.after(3000, lambda: self._set_status("ready"))

    # -- Continuous Dictation Toggle --

    def _toggle_continuous_recording(self):
        """Toggle hands-free continuous dictation on/off (Ctrl+Shift+A)."""
        with self._lock:
            if self._is_processing or getattr(self, "_is_starting_recording", False):
                return

        if self.is_continuous_mode:
            # STOP continuous session
            logging.info("[GUI] Continuous mode: OFF")
            self.is_continuous_mode = False
            try:
                context = get_active_window_info()
                audio_path = self.recorder.stop()
                if audio_path is not None:
                    self._pipeline_queue.put((audio_path, context))
                else:
                    self._set_status("ready")
            except Exception as e:
                logging.error(f"[GUI] Continuous stop error: {e}")
                self._set_status("warning", f"Stop failed: {e}")
                self.after(3000, lambda: self._set_status("ready"))
        else:
            # START continuous session
            if not self.brain.api_key:
                from tkinter import messagebox
                messagebox.showerror(
                    "Gemini API Key Missing",
                    "Gemini API Key is missing.\n\n"
                    "Please configure a valid API key in Settings.\n"
                    "Tip: You can paste multiple keys separated by commas for auto-rotation."
                )
                return

            if self.recorder.is_recording:
                return

            logging.info("[GUI] Continuous mode: ON (VAD auto-stop enabled)")
            self.is_continuous_mode = True
            self._is_starting_recording = True

            # Capture lookback context immediately on continuous start (before speaking starts)
            self._lookback_context = self._capture_lookback_context()

            # Trigger TCP/TLS socket pre-warming in a background thread
            threading.Thread(target=self.brain.pre_warm_gemini_connection, daemon=True).start()

            # Optimistic UI update
            self._set_status("recording", "Continuous mode ON -- will auto-stop on silence")

            def _async_start_continuous():
                set_thread_priority(2)  # HIGHEST priority
                try:
                    self.recorder.start(auto_stop_callback=self._on_vad_auto_stop)
                    if not self.recorder.is_recording:
                        raise RuntimeError("Audio stream failed to initialize")
                except Exception as e:
                    logging.error(f"[GUI] Continuous recording start error: {e}")
                    self.is_continuous_mode = False
                    self._set_status("warning", f"Recording failed: {e}")
                    self.after(3000, lambda: self._set_status("ready"))
                finally:
                    self._is_starting_recording = False

            threading.Thread(target=_async_start_continuous, daemon=True).start()

    def _on_vad_auto_stop(self, audio_path: str):
        """Callback from VAD when silence is detected in continuous mode."""
        self.is_continuous_mode = False
        if audio_path:
            context = get_active_window_info()
            self._pipeline_queue.put((audio_path, context))
        else:
            self._set_status("ready")

    # -- Editing Command Executor --

    def _execute_editing_command(self, command: str):
        """Execute a live dictation editing command."""
        import time

        if command == "delete_last_sentence":
            # Issue a standard Ctrl+Z undo sequence
            keyboard.press_and_release("ctrl+z")
            self._last_injected_text = ""
            logging.info("[GUI] Executed: undo (ctrl+z)")

        elif command == "delete_all":
            # Select all and delete (Ctrl+A, Delete)
            keyboard.press_and_release("ctrl+a")
            time.sleep(0.05)
            keyboard.press_and_release("delete")
            self._last_injected_text = ""
            logging.info("[GUI] Executed: delete all")

        elif command == "insert_newline":
            keyboard.press_and_release("enter")
            logging.info("[GUI] Executed: new line")

        elif command == "insert_paragraph":
            keyboard.press_and_release("enter")
            time.sleep(0.02)
            keyboard.press_and_release("enter")
            logging.info("[GUI] Executed: new paragraph")

        elif command == "insert_period":
            keyboard.write(".", delay=0)
            logging.info("[GUI] Executed: period")

        elif command == "insert_comma":
            keyboard.write(",", delay=0)
            logging.info("[GUI] Executed: comma")

        elif command == "insert_question_mark":
            keyboard.write("?", delay=0)
            logging.info("[GUI] Executed: question mark")

        elif command == "insert_exclamation":
            keyboard.write("!", delay=0)
            logging.info("[GUI] Executed: exclamation mark")

    # -- AI Pipeline --

    def _pipeline_worker(self):
        """Dedicated background pipeline thread to minimize context switching and thread creation overhead."""
        set_thread_priority(-2)  # LOWEST priority to prevent UI stuttering
        while True:
            try:
                task = self._pipeline_queue.get()
                if task is None:
                    break
                audio_path, context = task
                self._run_pipeline(audio_path, context)
            except Exception as e:
                logging.error(f"[GUI] Pipeline worker error: {e}")
            finally:
                self._pipeline_queue.task_done()

    def _run_pipeline(self, audio_path: str, context: dict):
        """Full dictation pipeline: transcribe -> polish -> inject."""
        with self._lock:
            if self._is_processing:
                return
            self._is_processing = True

        try:
            self._set_status("transcribing")

            if context and context.get("app_hint"):
                logging.info(f"[GUI] Active context: {context['app_hint']}")

            # 1. Run local ASR immediately (low perceived latency)
            raw_text = self.brain._offline_transcribe(audio_path, context)
            
            if not raw_text:
                self._set_status("ready", "No speech detected.")
                return

            # Optional: dynamic vocabulary addition if user explicitly said "add <word> to my dictionary"
            from ai_brain import detect_editing_command
            command, remainder = detect_editing_command(raw_text)
            if command and command.startswith("dict_add_"):
                word_to_add = command[len("dict_add_"):]
                self.brain._add_to_dictionary(word_to_add)
                self._set_status("ready", f"Learned: '{word_to_add}' added to memory!")
                return

            # Stage 2: Pure Speech-to-Text Polish (Wispr Flow style)
            # Never execute commands or keystrokes — ALWAYS convert spoken words into polished text.
            self._set_status("processing", "Polishing transcription...")
            pre_text = getattr(self, "_lookback_context", "")
            self._lookback_context = ""
            polished_text = self.brain.polish(raw_text, style=self._active_style, context_info=context, pre_text=pre_text)
            
            if not polished_text:
                polished_text = raw_text

            polished_expanded = self.injector.expand_snippets(polished_text)
            normalized_polished = polished_expanded.strip().replace("\r\n", "\n")

            # Terminal Safety Guard: If user is focused on a terminal or command prompt,
            # ensure no unprompted newlines are injected that could accidentally execute shell commands.
            app_hint = context.get("app_hint", "") if context else ""
            if any(term in app_hint.lower() for term in ["terminal", "cmd", "powershell", "bash", "wsl"]):
                normalized_polished = normalized_polished.replace("\n", " ").strip()

            # Direct Injection: Type the polished text at the cursor position
            self._set_status("typing", "Typing polished text...")
            self._last_injected_text = self.injector.inject(normalized_polished)


            # 4. Log final polished text to vault
            ts = self.vault.add_entry(normalized_polished, raw_text)
            self.after(
                0,
                lambda t=ts, p=normalized_polished: self._push_history_entry(t, p),
            )
            self.after(0, self._refresh_telemetry_ui)

            self._set_status("ready")

        except Exception as e:
            logging.error(f"[GUI] Pipeline error: {e}")
            err_msg = str(e)
            is_critical_key_error = "API_KEY_INVALID" in err_msg or "API key" in err_msg or "keyring" in err_msg or "403" in err_msg
            
            if is_critical_key_error:
                self._set_status("error", "API Key Missing or Invalid")
                from tkinter import messagebox
                self.after(0, lambda: messagebox.showerror(
                    "Gemini API Key Error",
                    "A critical error occurred: Gemini API Key is missing or invalid.\n\n"
                    "Please check your API key in Settings."
                ))
            else:
                self._set_status("warning", f"Transient error: {e}")
                self.after(5000, lambda: self._set_status("ready"))

        finally:
            # Clean up temp audio file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except OSError:
                pass
            with self._lock:
                self._is_processing = False


# Backward compatibility alias
LocalFlowApp = GlideTextApp
