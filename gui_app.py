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
    
    if sys.stdout is not None:
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
from dictation_session import (
    DictationSession,
    DictationSessionCoordinator,
    SessionMode,
    SessionState,
)

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

def _read_config() -> dict:
    """Read config.txt and keyring and return configuration dictionary."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.txt"
    )
    result = {
        "api_key": "",
        "device_index": None,
        "whisper_model": "base",
        "whisper_language": "auto",
        "freellmapi_dir": "",
    }

    # 1. Retrieve API key securely via keyring
    if HAS_KEYRING:
        try:
            key = keyring.get_password("GlideText", "api_key") or keyring.get_password("LocalFlow", "api_key")
            if key:
                result["api_key"] = key
        except Exception as e:
            logging.error(f"[Config] Failed to read from keyring: {e}")

    # 2. Parse config.txt
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines()]

            for idx, line in enumerate(lines):
                if not line or line.startswith("#"):
                    continue
                # Line 0 legacy standalone device index (e.g., "0" or "auto")
                if idx == 0 and "=" not in line:
                    if line.lower() in ("auto", "none", ""):
                        result["device_index"] = None
                    else:
                        try:
                            result["device_index"] = int(line)
                        except ValueError:
                            result["device_index"] = None
                    continue

                if "=" in line:
                    key, val = line.split("=", 1)
                    k_upper = key.strip().upper()
                    v = val.strip()
                    if k_upper == "DEVICE_INDEX":
                        if v.lower() in ("auto", "none", ""):
                            result["device_index"] = None
                        else:
                            try:
                                result["device_index"] = int(v)
                            except ValueError:
                                result["device_index"] = None
                    elif k_upper == "WHISPER_MODEL":
                        result["whisper_model"] = v or "base"
                    elif k_upper == "WHISPER_LANGUAGE":
                        result["whisper_language"] = v or "auto"
                    elif k_upper == "FREELLMAPI_DIR":
                        result["freellmapi_dir"] = v
                    elif k_upper == "GEMINI_API_KEY" and not result["api_key"]:
                        result["api_key"] = v
        except Exception as e:
            logging.error(f"[Config] Failed to read config.txt: {e}")
    else:
        # Create default config.txt if missing
        _write_config(result["api_key"], result["device_index"])

    return result


def _write_config(api_key: str, device_index, whisper_model: str = "base", whisper_language: str = "auto"):
    """Write api_key to keyring and update config.txt preserving existing key-value pairs."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.txt"
    )
    # Store API key securely
    if HAS_KEYRING:
        try:
            if api_key:
                keyring.set_password("GlideText", "api_key", api_key)
            else:
                for service in ("GlideText", "LocalFlow"):
                    try:
                        keyring.delete_password(service, "api_key")
                    except Exception:
                        pass
        except Exception as e:
            logging.error(f"[Config] Failed to write to keyring: {e}")

    try:
        existing_lines = []
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                existing_lines = f.read().splitlines()

        dev_val = "auto" if (device_index is None or str(device_index).lower() == "auto") else str(device_index)

        # Check if line 0 was legacy format (no '=')
        has_legacy_first_line = len(existing_lines) > 0 and "=" not in existing_lines[0] and not existing_lines[0].startswith("#")

        seen_keys = set()
        new_lines = []

        if has_legacy_first_line:
            new_lines.append(dev_val)
            seen_keys.add("DEVICE_INDEX")
            remaining_lines = existing_lines[1:]
        else:
            remaining_lines = existing_lines

        for line in remaining_lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                k, _ = stripped.split("=", 1)
                ku = k.strip().upper()
                if ku == "DEVICE_INDEX":
                    new_lines.append(f"DEVICE_INDEX={dev_val}")
                    seen_keys.add("DEVICE_INDEX")
                elif ku == "WHISPER_MODEL":
                    new_lines.append(f"WHISPER_MODEL={whisper_model}")
                    seen_keys.add("WHISPER_MODEL")
                elif ku == "WHISPER_LANGUAGE":
                    new_lines.append(f"WHISPER_LANGUAGE={whisper_language}")
                    seen_keys.add("WHISPER_LANGUAGE")
                else:
                    new_lines.append(line)
                    seen_keys.add(ku)
            else:
                new_lines.append(line)

        if "DEVICE_INDEX" not in seen_keys:
            if not has_legacy_first_line:
                new_lines.insert(0, f"DEVICE_INDEX={dev_val}")
        if "WHISPER_MODEL" not in seen_keys:
            new_lines.append(f"WHISPER_MODEL={whisper_model}")
        if "WHISPER_LANGUAGE" not in seen_keys:
            new_lines.append(f"WHISPER_LANGUAGE={whisper_language}")

        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines).strip() + "\n")
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
        if self._cfg.get("whisper_model") or self._cfg.get("whisper_language"):
            self.brain.set_whisper_config(
                model_name=self._cfg.get("whisper_model", "base"),
                language=self._cfg.get("whisper_language", "auto"),
            )
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
        self._continuous_active_flag = False
        self.is_widget_mode   = False
        self._active_style    = "Normal"
        self._last_injected_text = ""  # For "scratch that" editing commands

        self._active_session: DictationSession | None = None
        self._processed_session_ids: set[str] = set()
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
            text="● FreeLLMAPI (Priority 1)",
            font=(FONT, 11, "bold"),
            text_color="#10b981",
            fg_color="#064e3b",
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
        # Initially hidden in normal FreeLLMAPI mode

    def _update_engine_mode_ui(self, provider: str, is_fallback: bool = False):
        """Update header badge reflecting active polish engine / provider."""
        try:
            p_lower = str(provider).lower()
            if "freellm" in p_lower:
                self.engine_badge.configure(
                    text="● FreeLLMAPI (Priority 1)",
                    text_color="#10b981",
                    fg_color="#064e3b"
                )
                self.reset_cloud_btn.pack_forget()
            elif "gemini" in p_lower:
                label = "● Gemini (Fallback)" if is_fallback else "● Gemini Cloud"
                self.engine_badge.configure(
                    text=label,
                    text_color="#38bdf8",
                    fg_color="#0f172a"
                )
                self.reset_cloud_btn.pack(side="left", padx=(6, 0))
            elif "local" in p_lower or "ollama" in p_lower:
                label = "⚡ Local LLM (Fallback)" if is_fallback else "⚡ Local LLM"
                self.engine_badge.configure(
                    text=label,
                    text_color="#fb923c",
                    fg_color="#431407"
                )
                self.reset_cloud_btn.pack(side="left", padx=(6, 0))
            elif "raw" in p_lower or "fail" in p_lower:
                self.engine_badge.configure(
                    text="⚠ Raw Transcript (Fallback)",
                    text_color="#f87171",
                    fg_color="#450a0a"
                )
                self.reset_cloud_btn.pack(side="left", padx=(6, 0))
            else:
                self.engine_badge.configure(
                    text=f"● {provider}",
                    text_color="#94a3b8",
                    fg_color="#1e293b"
                )
                self.reset_cloud_btn.pack_forget()
        except Exception:
            pass

    @property
    def is_continuous_mode(self) -> bool:
        """True if there is an active capturing session in continuous mode."""
        with self._lock:
            if self._active_session is not None:
                return (
                    self._active_session.mode == SessionMode.CONTINUOUS
                    and self._active_session.is_active_capture
                )
            return getattr(self, "_continuous_active_flag", False)

    @is_continuous_mode.setter
    def is_continuous_mode(self, value: bool) -> None:
        with self._lock:
            self._continuous_active_flag = bool(value)

    @property
    def is_recording(self) -> bool:
        """True if the session is currently capturing audio (RECORDING or PAUSED)."""
        with self._lock:
            return bool(
                self._active_session is not None
                and self._active_session.is_active_capture
            )

    def _on_reset_cloud_click(self):
        """User manually resets temporary cooldowns back to FreeLLMAPI (Priority 1)."""
        self.brain.reset_cloud_mode()
        self._update_engine_mode_ui("freellmapi")
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

        # Speech-to-Text Whisper Model
        ctk.CTkLabel(
            inner, text="Whisper STT Model",
            font=(FONT, 12), text_color=C.TEXT_SEC,
        ).pack(anchor="w")
        self.whisper_model_menu = ctk.CTkOptionMenu(
            inner, values=["base", "tiny", "small", "medium", "large-v3"],
            font=(FONT, 13), fg_color=C.BG_INPUT,
            button_color=C.ACCENT, button_hover_color=C.ACCENT_HOVER,
            text_color=C.TEXT, dropdown_fg_color=C.BG_CARD,
            dropdown_text_color=C.TEXT, dropdown_hover_color=C.ACCENT,
            height=34,
        )
        self.whisper_model_menu.set(self._cfg.get("whisper_model", "base"))
        self.whisper_model_menu.pack(fill="x", pady=(4, 14))

        # STT Language
        ctk.CTkLabel(
            inner, text="STT Language (auto, en, hi, es, fr, de...)",
            font=(FONT, 12), text_color=C.TEXT_SEC,
        ).pack(anchor="w")
        self.whisper_lang_entry = ctk.CTkEntry(
            inner, font=(FONT, 13), fg_color=C.BG_INPUT,
            border_color=C.BORDER, text_color=C.TEXT, height=34,
            placeholder_text="auto",
        )
        self.whisper_lang_entry.insert(0, self._cfg.get("whisper_language", "auto"))
        self.whisper_lang_entry.pack(fill="x", pady=(4, 14))

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
            "recording":    (C.RED,      "RECORDING",         "Listening... (Release [Right Alt] or Ctrl+Shift+A to finish)"),
            "paused":       (C.AMBER,    "PAUSED (THINKING)", "Paused — keep speaking when ready (Ctrl+Shift+A to finish)"),
            "transcribing": (C.AMBER,    "TRANSCRIBING",      "Transcribing speech locally..."),
            "processing":   (C.AMBER,    "PROCESSING",        "Polishing text with AI..."),
            "typing":       (C.ACCENT,   "TYPING",            "Injecting text at cursor..."),
            "initializing": (C.TEXT_DIM, "INITIALIZING...",   "Initializing speech engine..."),
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

        if status in ("recording", "paused"):
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
        if self._current_status not in ("recording", "paused"):
            self.waveform_canvas.delete("all")
            return
            
        try:
            self.waveform_canvas.delete("all")
            w = 80
            h = 40
            bar_w = 8
            gap = 6
            start_x = (w - (5 * bar_w + 4 * gap)) / 2

            if self._current_status == "paused":
                # Non-intrusive calm amber indicator during thinking pauses
                for i in range(5):
                    bar_h = 6 if (i == 0 or i == 4) else (8 if (i == 1 or i == 3) else 10)
                    x0 = start_x + i * (bar_w + gap)
                    y0 = (h - bar_h) / 2
                    x1 = x0 + bar_w
                    y1 = y0 + bar_h
                    self.waveform_canvas.create_rectangle(
                        x0, y0, x1, y1, fill=C.AMBER, outline=""
                    )
                self._pulse_job = self.after(200, self._animate_waveform)
                return

            rms = getattr(self.recorder, "current_rms", 0.0)
            
            # Simple volume mapping
            normalized = min(max(rms / 1500.0, 0.1), 1.0)
                
            # Draw 5 bars
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

    def _capture_lookback_context(self, context_info: dict | None = None) -> str:
        """Selects the preceding ~5-8 words using Ctrl+Shift+Left 6 times, copies, and restores the cursor.

        Privacy & security guarantees:
          - Skips terminals and sensitive/credential windows (password managers, login/2FA, .env files).
          - Strictly bounds and sanitizes captured text via `sanitize_and_bound_cursor_text`.
          - Logs only character length, never raw user text.
        """
        # Flag check: LOOKBACK_CONTEXT=0 by default
        cfg = _read_config()
        lookback_enabled = os.getenv("LOOKBACK_CONTEXT", str(cfg.get("lookback_context", 0))).strip() == "1"
        if not lookback_enabled:
            return ""

        from context_snapshot import (
            AppCategory,
            classify_application,
            sanitize_and_bound_cursor_text,
        )

        app_category = AppCategory.UNKNOWN
        is_sensitive = False
        if context_info:
            app_hint = context_info.get("app_hint", "")
            exe_name = context_info.get("exe_name", "")
            title = context_info.get("title", "")
            app_category, _safe_name, is_sensitive = classify_application(
                exe_name=exe_name,
                window_title=title,
                app_hint=app_hint,
            )
            if bool(context_info.get("is_sensitive_context")):
                is_sensitive = True

            if is_sensitive:
                logging.info("[Lookback] Sensitive/credential context active -- skipping lookback capture.")
                return ""

            if app_category == AppCategory.TERMINAL or any(
                term in app_hint.lower() or term in exe_name.lower()
                for term in ["terminal", "cmd", "powershell", "bash", "wsl"]
            ):
                logging.info("[Lookback] Terminal active -- skipping lookback context capture.")
                return ""

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

        bounded_text = sanitize_and_bound_cursor_text(
            pre_text,
            is_sensitive=is_sensitive,
            app_category=app_category,
        )
        logging.info(f"[Lookback] Captured bounded lookback ({len(bounded_text)} chars).")
        return bounded_text

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
        exe = sys.executable
        if exe.lower().endswith("python.exe"):
            pythonw_exe = exe[:-10] + "pythonw.exe"
        else:
            pythonw_exe = exe
        main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
        cmd_string = f'"{pythonw_exe}" "{main_py_path}" --silent'
        
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                r"Software\Microsoft\Windows\CurrentVersion\Run", 
                0, 
                winreg.KEY_SET_VALUE
            )
            # Always clean up legacy LocalFlow registry key
            try:
                winreg.DeleteValue(key, "LocalFlow")
            except FileNotFoundError:
                pass

            if is_autoboot:
                winreg.SetValueEx(key, "GlideText", 0, winreg.REG_SZ, cmd_string)
                logging.info(f"[Registry] Set GlideText autoboot: {cmd_string}")
            else:
                try:
                    winreg.DeleteValue(key, "GlideText")
                    logging.info("[Registry] Removed GlideText from boot.")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except Exception as e:
            logging.error(f"[Registry] Failed to update autoboot status: {e}")
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
        self._on_style_change(self.style_menu.get())

        # Gemini API Key (supports comma-separated multi-keys)
        new_key = self.api_key_entry.get().strip()
        if new_key:
            self.brain.set_api_key(new_key)
            self.api_key_entry.configure(placeholder_text="••••••••••••••••")
            self.api_key_entry.delete(0, 'end')

        current_key = ",".join(self.brain._api_keys) if getattr(self.brain, "_api_keys", None) else ""

        # Whisper model and language
        new_model = self.whisper_model_menu.get().strip() if hasattr(self, "whisper_model_menu") else "base"
        new_lang = self.whisper_lang_entry.get().strip() if hasattr(self, "whisper_lang_entry") else "auto"
        if not new_model:
            new_model = "base"
        if not new_lang:
            new_lang = "auto"
        self.brain.set_whisper_config(model_name=new_model, language=new_lang)

        _write_config(current_key, new_device, whisper_model=new_model, whisper_language=new_lang)
        self._set_status("ready")

        self.settings_feedback.configure(
            text="Settings saved successfully!",
            text_color=C.GREEN,
        )

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
        """Initialize speech engine and register hotkeys."""
        # Stage 1 -- Check speech engine
        self._set_status("initializing", "Initializing speech engine...")
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

        # Stage 4 -- Set status to ready
        self._set_status("ready")
        self.recorder.warmup()

        # Keep this thread alive so keyboard hooks remain active
        try:
            keyboard.wait()
        except Exception:
            pass

    # -- Hotkey Handlers --

    def _on_key_press(self, _event):
        """Right Alt pressed: start a push-to-talk DictationSession."""
        if self.is_continuous_mode:
            return

        with self._lock:
            if (
                self.recorder.is_recording
                or getattr(self, "_is_starting_recording", False)
                or self._is_processing
                or (self._active_session is not None and self._active_session.is_active_capture)
            ):
                return
            self._is_starting_recording = True
            self._stop_pending = False

        # Capture target window immediately at key-down
        self._lookback_context = ""
        context = get_active_window_info()
        self._target_hwnd = context.get("hwnd")

        # Create independent push-to-talk DictationSession
        session = DictationSession(
            mode=SessionMode.PUSH_TO_TALK,
            context_info=context,
            target_hwnd=self._target_hwnd,
            style=self._active_style,
        )
        session.start()
        with self._lock:
            self._active_session = session

        # Trigger TCP/TLS socket pre-warming in a background thread if key available
        if self.brain.api_key:
            threading.Thread(target=self.brain.pre_warm_gemini_connection, daemon=True).start()

        # Optimistic UI update
        self._set_status("recording", "Push-to-talk active -- release [Right Alt] to finish")

        # Load dynamic vocabulary for the current active app in a background thread
        self.brain.reload_vocabulary(context)

        def _async_start():
            set_thread_priority(2)  # HIGHEST priority
            try:
                self.recorder.start(
                    on_speech_callback=session.on_speech_detected,
                    chunk_callback=session.append_audio_chunk,
                )
                if not self.recorder.is_recording:
                    raise RuntimeError("Audio stream failed to initialize")

                # Capture lookback context AFTER recording starts (and on background thread)
                captured = self._capture_lookback_context(context)
                session.set_lookback_context(captured)
                self._lookback_context = session.lookback_context

                should_stop = False
                with self._lock:
                    if self._stop_pending:
                        self._stop_pending = False
                        should_stop = True
                if should_stop:
                    self._async_stop(session)
            except Exception as e:
                logging.error(f"[GUI] Recording start error: {e}")
                session.fail_session(f"Recording start error: {e}")
                with self._lock:
                    if self._active_session is session:
                        self._active_session = None
                self._set_status("warning", f"Recording failed: {e}")
                self.after(3000, lambda: self._set_status("ready"))
            finally:
                self._is_starting_recording = False

        threading.Thread(target=_async_start, daemon=True).start()

    def _on_key_release(self, _event):
        """Right Alt released: finalize push-to-talk DictationSession."""
        if self.is_continuous_mode:
            return
        
        with self._lock:
            if getattr(self, "_is_starting_recording", False):
                self._stop_pending = True
                return
            if not self.recorder.is_recording and (
                self._active_session is None or not self._active_session.is_active_capture
            ):
                return
            session = self._active_session

        self._async_stop(session)

    def _async_stop(self, session: DictationSession | None = None):
        """Finalize the active push-to-talk session and enqueue it for processing."""
        try:
            with self._lock:
                target_session = session or self._active_session
                if target_session is not None and not target_session.is_active_capture:
                    return
                if self._active_session is target_session:
                    self._active_session = None

            self._set_status("transcribing")
            # Capture active window context on stop trigger
            context = get_active_window_info()
            if getattr(self, "_target_hwnd", None):
                context["target_hwnd"] = self._target_hwnd
            self._target_hwnd = None

            audio_path = self.recorder.stop()

            if target_session is None:
                target_session = DictationSession(
                    mode=SessionMode.PUSH_TO_TALK,
                    context_info=context,
                    style=self._active_style,
                )
                target_session.start()

            target_session.refresh_context_snapshot(updated_context=context)

            if not target_session.finalize(audio_path=audio_path):
                return

            if audio_path is None:
                target_session.complete_session(reason="no_audio")
                self._set_status("ready")
                return

            self._pipeline_queue.put(target_session)
        except Exception as e:
            logging.error(f"[GUI] Recording stop error: {e}")
            self._set_status("warning", f"Stop failed: {e}")
            self.after(3000, lambda: self._set_status("ready"))

    # -- Continuous Dictation Toggle --

    def _toggle_continuous_recording(self):
        """Toggle hands-free continuous dictation session on/off (Ctrl+Shift+A).

        - If no continuous session exists -> start a new continuous DictationSession
        - If a continuous session exists -> explicitly finalize the session and process
        """
        with self._lock:
            if self._is_processing or getattr(self, "_is_starting_recording", False):
                return

        if self.is_continuous_mode:
            # EXPLICIT USER STOP: Finalize continuous session
            logging.info("[GUI] Continuous mode: OFF (explicit user stop)")
            self.is_continuous_mode = False
            try:
                with self._lock:
                    session = self._active_session
                    self._active_session = None

                if session is not None and not session.is_active_capture:
                    return

                self._set_status("transcribing")
                context = get_active_window_info()
                if getattr(self, "_target_hwnd", None):
                    context["target_hwnd"] = self._target_hwnd
                self._target_hwnd = None

                audio_path = self.recorder.stop()

                if session is None:
                    session = DictationSession(
                        mode=SessionMode.CONTINUOUS,
                        context_info=context,
                        style=self._active_style,
                    )
                    session.start()

                session.refresh_context_snapshot(updated_context=context)

                if not session.finalize(audio_path=audio_path):
                    return

                if audio_path is not None:
                    self._pipeline_queue.put(session)
                else:
                    session.complete_session(reason="no_audio")
                    self._set_status("ready")
            except Exception as e:
                logging.error(f"[GUI] Continuous stop error: {e}")
                self._set_status("warning", f"Stop failed: {e}")
                self.after(3000, lambda: self._set_status("ready"))
        else:
            # START continuous session
            if self.recorder.is_recording or (
                self._active_session is not None and self._active_session.is_active_capture
            ):
                return

            logging.info("[GUI] Continuous mode: ON (VAD speech/pause activity detection enabled)")
            self.is_continuous_mode = True
            self._is_starting_recording = True
            self._lookback_context = ""

            context = get_active_window_info()
            self._target_hwnd = context.get("hwnd")

            session = DictationSession(
                mode=SessionMode.CONTINUOUS,
                context_info=context,
                target_hwnd=self._target_hwnd,
                style=self._active_style,
            )
            session.start()
            with self._lock:
                self._active_session = session

            if self.brain.api_key:
                threading.Thread(target=self.brain.pre_warm_gemini_connection, daemon=True).start()

            self.brain.reload_vocabulary(context)

            # Optimistic UI update
            self._set_status("recording", "Continuous session active -- press Ctrl+Shift+A to finish")

            def _async_start_continuous():
                set_thread_priority(2)  # HIGHEST priority
                try:
                    self.recorder.start(
                        auto_stop_callback=self._on_vad_auto_stop,
                        on_speech_callback=self._on_vad_speech_activity,
                        on_silence_callback=self._on_vad_silence_activity,
                        chunk_callback=session.append_audio_chunk,
                    )
                    if not self.recorder.is_recording:
                        raise RuntimeError("Audio stream failed to initialize")

                    # Capture lookback context AFTER recording starts (and on background thread)
                    captured = self._capture_lookback_context(context)
                    session.set_lookback_context(captured)
                    self._lookback_context = session.lookback_context
                except Exception as e:
                    logging.error(f"[GUI] Continuous recording start error: {e}")
                    self.is_continuous_mode = False
                    session.fail_session(f"Continuous recording start error: {e}")
                    with self._lock:
                        if self._active_session is session:
                            self._active_session = None
                    self._set_status("warning", f"Recording failed: {e}")
                    self.after(3000, lambda: self._set_status("ready"))
                finally:
                    self._is_starting_recording = False

            threading.Thread(target=_async_start_continuous, daemon=True).start()

    def _on_vad_speech_activity(self):
        """Callback from VAD when speech starts or resumes during a session."""
        session = self._active_session
        if session is not None and session.is_active_capture:
            if session.on_speech_detected() and self.is_continuous_mode:
                self._set_status(
                    "recording",
                    "Continuous session: Listening... (Ctrl+Shift+A to finish)",
                )

    def _on_vad_silence_activity(self, silence_duration: float = 0.0):
        """Callback from VAD when silence/thinking pause occurs after speech."""
        session = self._active_session
        if session is not None and session.is_active_capture:
            if session.on_silence_detected(silence_duration) and self.is_continuous_mode:
                self._set_status(
                    "paused",
                    "Paused (thinking)... Speak to continue, or Ctrl+Shift+A to finish",
                )

    def _on_vad_auto_stop(self, audio_path: str | None = None):
        """Replaced semantic role: VAD silence activity notification in continuous mode.

        CRITICAL: Silence in continuous mode means the user is thinking/paused,
        NOT that the dictation session has ended. This method updates the session
        and UI to the PAUSED state and NEVER stops or finalizes the session.
        """
        session = self._active_session
        if session is not None and session.is_active_capture:
            session.on_silence_detected()
        if self.is_continuous_mode:
            self._set_status(
                "paused",
                "Paused (thinking)... Speak to continue, or Ctrl+Shift+A to finish",
            )

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
                if isinstance(task, DictationSession):
                    self._run_session_pipeline(task)
                else:
                    audio_path, context = task
                    self._run_pipeline(audio_path, context)
            except Exception as e:
                logging.error(f"[GUI] Pipeline worker error: {e}")
            finally:
                self._pipeline_queue.task_done()

    def _run_pipeline(self, audio_path: str, context: dict):
        """Compatibility wrapper that wraps a raw (audio_path, context) tuple in a DictationSession."""
        session = DictationSession(
            mode=SessionMode.PUSH_TO_TALK,
            context_info=context,
            style=self._active_style,
            lookback_context=getattr(self, "_lookback_context", ""),
        )
        self._lookback_context = ""
        session.start()
        session.finalize(audio_path=audio_path)
        self._run_session_pipeline(session)

    def _run_session_pipeline(self, session: DictationSession):
        """Full session dictation pipeline: transcribe -> polish -> inject (strictly once per session)."""
        if session is None:
            return

        with self._lock:
            if (
                session.session_id in self._processed_session_ids
                or session.is_cancelled
                or session.is_injected
                or session.state in (SessionState.DONE, SessionState.ERROR)
            ):
                logging.info(
                    f"[GUI] Session {session.session_id[:8]} already processed or invalid; preventing duplicate execution."
                )
                return
            self._is_processing = True
            self._processed_session_ids.add(session.session_id)

        audio_path = session.audio_path
        snapshot = session.refresh_context_snapshot(
            style=session.style or self._active_style
        )
        context = session.context_info
        self._lookback_context = ""
        in_memory_audio: Optional[np.ndarray] = None

        try:
            if not audio_path:
                in_memory_audio = session.get_concatenated_audio()
                if in_memory_audio is None or len(in_memory_audio) < 4800:
                    session.complete_session(reason="no_audio")
                    self._set_status("ready")
                    return

            if session.is_cancelled:
                return

            # 1. TRANSCRIBING (guarded to run once)
            if not session.begin_transcription():
                return

            self._set_status("transcribing")

            logging.info(
                f"[GUI] Active context: {snapshot.app_name} "
                f"(category={snapshot.app_category.value}, coding_mode={snapshot.coding_mode})"
            )

            # Run local ASR immediately (privacy-first: raw audio stays local)
            try:
                raw_text = self.brain._offline_transcribe(
                    audio_path=audio_path,
                    context_info=context,
                    session_id=session.session_id,
                    context_snapshot=snapshot,
                    audio_array=in_memory_audio,
                    cancel_check=lambda: session.is_cancelled,
                )
            except TypeError:
                try:
                    raw_text = self.brain._offline_transcribe(
                        audio_path,
                        context,
                        session_id=session.session_id,
                        context_snapshot=snapshot,
                    )
                except TypeError:
                    try:
                        raw_text = self.brain._offline_transcribe(
                            audio_path, context, session_id=session.session_id
                        )
                    except TypeError:
                        raw_text = self.brain._offline_transcribe(audio_path, context)

            if session.is_cancelled:
                return

            session.complete_transcription(raw_text)

            # Free in-memory audio buffers immediately to bound RAM
            session.release_audio_buffers()
            in_memory_audio = None
            
            if not raw_text:
                session.complete_session(reason="no_speech_detected")
                self._set_status("ready", "No speech detected.")
                return

            # Check safe Voice Command layer (scratch that, delete that, undo last dictation, etc.)
            from voice_commands import parse_voice_command, GLOBAL_VOICE_COMMAND_EXECUTOR, GLOBAL_INSERTION_HISTORY
            voice_cmd = parse_voice_command(raw_text)
            if voice_cmd:
                target_hwnd = context.get("target_hwnd") or context.get("hwnd")
                cmd_res = GLOBAL_VOICE_COMMAND_EXECUTOR.execute_command(voice_cmd, target_hwnd=target_hwnd)
                session.complete_session(
                    reason="voice_command",
                    command=voice_cmd.command_type.value,
                    executed=cmd_res.executed,
                    deleted_len=len(cmd_res.deleted_text),
                )
                if cmd_res.executed:
                    self._set_status("ready", cmd_res.message)
                else:
                    self._set_status("ready", cmd_res.message or "Voice command skipped")
                return

            # Optional: dynamic vocabulary addition if user explicitly said "add <word> to my dictionary"
            from ai_brain import detect_editing_command
            command, remainder = detect_editing_command(raw_text)
            if command and command.startswith("dict_add_"):
                word_to_add = command[len("dict_add_"):]
                self.brain._add_to_dictionary(word_to_add)
                session.complete_session(reason="dictionary_command")
                self._set_status("ready", f"Learned: '{word_to_add}' added to memory!")
                return

            if session.is_cancelled:
                return

            # 2. POLISHING (guarded to run once)
            # Priority 1: FreeLLMAPI -> Priority 2: Gemini -> Priority 3: Local LLM -> Raw transcript
            if not session.begin_polishing():
                return

            self._set_status("processing", "Polishing transcription...")
            snapshot = session.refresh_context_snapshot(
                style=session.style or self._active_style
            )
            pre_text = snapshot.bounded_cursor_text

            try:
                try:
                    pipeline_res = self.brain.polish_with_provider_fallbacks(
                        raw_text=raw_text,
                        style=session.style or self._active_style,
                        context_info=context,
                        pre_text=pre_text,
                        context_snapshot=snapshot,
                    )
                except TypeError:
                    pipeline_res = self.brain.polish_with_provider_fallbacks(
                        raw_text=raw_text,
                        style=session.style or self._active_style,
                        context_info=context,
                        pre_text=pre_text,
                    )
                polished_text = pipeline_res.text or raw_text
                provider_used = pipeline_res.provider
                is_fallback = pipeline_res.is_fallback
            except Exception as llm_err:
                logging.warning(
                    f"[GUI] LLM polishing failed ({llm_err}); preserving user's spoken words with raw fallback."
                )
                polished_text = session.corrected_transcript or raw_text
                provider_used = "raw_fallback"
                is_fallback = True

            if session.is_cancelled:
                return

            session.complete_polishing(
                polished_text=polished_text,
                provider=provider_used,
                is_fallback=is_fallback,
            )

            # Update engine mode badge to reflect the actual provider used
            self.after(
                0,
                lambda p=provider_used, fb=is_fallback: self._update_engine_mode_ui(p, is_fallback=fb)
            )

            polished_expanded = self.injector.expand_snippets(polished_text)
            normalized_polished = polished_expanded.strip().replace("\r\n", "\n")

            # Terminal Safety Guard: If user is focused on a terminal or command prompt,
            # ensure no unprompted newlines are injected that could accidentally execute shell commands.
            app_hint = context.get("app_hint", "") if context else ""
            if snapshot.single_line_output or any(
                term in app_hint.lower() for term in ["terminal", "cmd", "powershell", "bash", "wsl"]
            ):
                normalized_polished = normalized_polished.replace("\n", " ").strip()

            session.polished_text = normalized_polished

            # 3. INJECTING (guarded to run once)
            # Before injection verify:
            # 1. session exists
            # 2. session has valid final text
            # 3. session is not already injected
            # 4. session is in an injectable state
            if not session.is_injectable or not session.begin_injection():
                logging.debug(
                    f"[GUI] Session {session.session_id[:8]} cannot inject: "
                    f"is_injectable={session.is_injectable}, state={session.state.value}."
                )
                return

            self._set_status("typing", "Typing polished text...")
            target_hwnd = context.get("target_hwnd") or context.get("hwnd")
            inject_res = self.injector.inject(normalized_polished, target_hwnd=target_hwnd)

            if inject_res.success:
                session.mark_injected(True)
                self._last_injected_text = inject_res.injected_text
                GLOBAL_INSERTION_HISTORY.record_insertion(
                    session_id=session.session_id,
                    text=normalized_polished,
                    target_hwnd=target_hwnd,
                )
            else:
                logging.warning(f"[GUI] Text injection skipped or failed: {inject_res.error}")
                self._set_status("warning", f"Injection: {inject_res.error}")

            # 4. Log final polished text to vault
            ts = self.vault.add_entry(normalized_polished, raw_text)
            self.after(
                0,
                lambda t=ts, p=normalized_polished: self._push_history_entry(t, p),
            )
            self.after(0, self._refresh_telemetry_ui)

            session.complete_session(
                reason="ok",
                injected=bool(inject_res.success),
                method=getattr(inject_res, "method", "unknown"),
            )
            self._set_status("ready")

        except Exception as e:
            logging.error(f"[GUI] Pipeline error: {e}")
            session.fail_session(str(e))
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
            session.cleanup_audio()
            with self._lock:
                self._is_processing = False


# Backward compatibility alias
LocalFlowApp = GlideTextApp
