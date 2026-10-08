"""
platform_compat/input_backend.py
Cross-platform keyboard and input abstraction layer for GlideText.

On Windows:
  Delegates directly to the `keyboard` package for hotkeys, holds, and synthetic keystrokes.

On macOS:
  - Uses `pynput` (global listener) for push-to-talk hold and hotkeys.
  - Uses Quartz `CGEvent` (via pyobjc-framework-Quartz) for synthetic keystrokes, shortcuts, and layout-independent typing.
  - Translates logical actions (COPY, PASTE, UNDO, SELECT_ALL, SELECT_WORD_LEFT) to macOS key conventions (Cmd / Option instead of Ctrl).
"""

from abc import ABC, abstractmethod
from enum import Enum
import logging
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class LogicalAction(Enum):
    """Logical editing and navigation actions that translate to OS-specific key combos."""
    COPY = "copy"
    PASTE = "paste"
    UNDO = "undo"
    SELECT_ALL = "select_all"
    SELECT_WORD_LEFT = "select_word_left"
    SELECT_WORD_RIGHT = "select_word_right"
    SELECT_LINE_LEFT = "select_line_left"
    BACKSPACE = "backspace"
    DELETE = "delete"
    ENTER = "enter"


def get_action_combo(action: Union[LogicalAction, str]) -> str:
    """Translate a logical editing action to an OS-specific key combo string.

    macOS uses Command for standard editing shortcuts and Option for word navigation.
    Windows uses Control for standard editing and word navigation.
    """
    act_str = action.value if isinstance(action, LogicalAction) else str(action).lower()
    if sys.platform == "darwin":
        combo_map = {
            "copy": "cmd+c",
            "paste": "cmd+v",
            "undo": "cmd+z",
            "select_all": "cmd+a",
            "select_word_left": "option+shift+left",
            "select_word_right": "option+shift+right",
            "select_line_left": "cmd+shift+left",
            "backspace": "backspace",
            "delete": "delete",
            "enter": "enter",
        }
    else:
        combo_map = {
            "copy": "ctrl+c",
            "paste": "ctrl+v",
            "undo": "ctrl+z",
            "select_all": "ctrl+a",
            "select_word_left": "ctrl+shift+left",
            "select_word_right": "ctrl+shift+right",
            "select_line_left": "shift+home",
            "backspace": "backspace",
            "delete": "delete",
            "enter": "enter",
        }
    return combo_map.get(act_str, act_str)


def get_word_modifier() -> str:
    """Return the modifier key used for word-by-word selection/navigation.

    'option' on macOS, 'ctrl' on Windows/Linux.
    """
    return "option" if sys.platform == "darwin" else "ctrl"


class BaseInputBackend(ABC):
    """Abstract base class for keyboard listener and synthetic input backends."""

    @abstractmethod
    def on_press_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        """Register a callback when a specific key is pressed down."""
        pass

    @abstractmethod
    def on_release_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        """Register a callback when a specific key is released."""
        pass

    @abstractmethod
    def add_hotkey(self, combo: str, callback: Callable[[], None], suppress: bool = False) -> Any:
        """Register a callback when a combination of keys is pressed."""
        pass

    @abstractmethod
    def unhook_all(self) -> None:
        """Unregister all hotkeys and listeners."""
        pass

    @abstractmethod
    def wait(self) -> None:
        """Block current thread until unhook_all is called or process exits."""
        pass

    @abstractmethod
    def press(self, key: str) -> None:
        """Send a synthetic key-down event."""
        pass

    @abstractmethod
    def release(self, key: str) -> None:
        """Send a synthetic key-up event."""
        pass

    @abstractmethod
    def press_and_release(self, combo: str) -> None:
        """Send a synthetic key press and release (single key or combo like 'ctrl+c')."""
        pass

    @abstractmethod
    def write(self, text: str, delay: float = 0.0) -> None:
        """Type characters sequentially with optional inter-keystroke delay."""
        pass

    def send_action(self, action: Union[LogicalAction, str]) -> None:
        """Send an OS-translated logical editing shortcut."""
        combo = get_action_combo(action)
        self.press_and_release(combo)

    def select_words_left(self, count: int = 1) -> None:
        """Select `count` words to the left of the cursor."""
        word_mod = get_word_modifier()
        self.press(word_mod)
        self.press("shift")
        for _ in range(count):
            self.press_and_release("left")
            time.sleep(0.002)
        self.release("shift")
        self.release(word_mod)


class WindowsInputBackend(BaseInputBackend):
    """Windows input backend delegating directly to the `keyboard` package."""

    def __init__(self):
        try:
            import keyboard
            self._keyboard = keyboard
        except ImportError:
            self._keyboard = None
            logging.warning("[InputBackend] Windows 'keyboard' package not available.")

    def on_press_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        if self._keyboard:
            return self._keyboard.on_press_key(key, callback, suppress=suppress)

    def on_release_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        if self._keyboard:
            return self._keyboard.on_release_key(key, callback, suppress=suppress)

    def add_hotkey(self, combo: str, callback: Callable[[], None], suppress: bool = False) -> Any:
        if self._keyboard:
            return self._keyboard.add_hotkey(combo, callback, suppress=suppress)

    def unhook_all(self) -> None:
        if self._keyboard:
            self._keyboard.unhook_all()

    def wait(self) -> None:
        if self._keyboard:
            self._keyboard.wait()

    def press(self, key: str) -> None:
        if self._keyboard:
            self._keyboard.press(key)

    def release(self, key: str) -> None:
        if self._keyboard:
            self._keyboard.release(key)

    def press_and_release(self, combo: str) -> None:
        if self._keyboard:
            self._keyboard.press_and_release(combo)

    def write(self, text: str, delay: float = 0.0) -> None:
        if self._keyboard:
            self._keyboard.write(text, delay=delay)


# macOS Quartz Virtual Key Codes
_MAC_KEY_CODES: Dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
    "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
    "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
    "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29,
    "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37,
    "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47,
    "return": 36, "enter": 36, "tab": 48, "space": 49,
    "backspace": 51, "delete": 51, "forward_delete": 117,
    "escape": 53, "esc": 53,
    "command": 55, "cmd": 55, "shift": 56, "option": 58, "alt": 58,
    "control": 59, "ctrl": 59,
    "right shift": 60, "right option": 61, "right alt": 61,
    "right control": 62, "right ctrl": 62,
    "left": 123, "right": 124, "down": 125, "up": 126,
}

# Quartz CGEventFlagMask bits
_MAC_FLAG_MASKS: Dict[str, int] = {
    "cmd": 0x00100000,      # kCGEventFlagMaskCommand
    "command": 0x00100000,
    "shift": 0x00020000,    # kCGEventFlagMaskShift
    "alt": 0x00080000,      # kCGEventFlagMaskAlternate (Option)
    "option": 0x00080000,
    "ctrl": 0x00040000,     # kCGEventFlagMaskControl
    "control": 0x00040000,
}


class MacOSInputBackend(BaseInputBackend):
    """macOS input backend using `pynput` for hotkeys and Quartz `CGEvent` for synthetic input."""

    def __init__(self):
        self._listener = None
        self._listener_lock = threading.Lock()
        self._press_callbacks: Dict[str, List[Callable[[Any], None]]] = {}
        self._release_callbacks: Dict[str, List[Callable[[Any], None]]] = {}
        self._hotkey_callbacks: List[Tuple[Set[str], Callable[[], None]]] = []
        self._currently_pressed: Set[str] = set()
        self._triggered_hotkeys: Set[frozenset] = set()
        self._stop_event = threading.Event()
        self._active_modifier_flags: int = 0

    def _ensure_listener(self) -> None:
        """Start pynput keyboard listener if not already running."""
        with self._listener_lock:
            if self._listener is not None:
                return
            try:
                from pynput import keyboard as pynput_keyboard
                self._listener = pynput_keyboard.Listener(
                    on_press=self._on_pynput_press,
                    on_release=self._on_pynput_release,
                )
                self._listener.daemon = True
                self._listener.start()
                logging.info("[InputBackend] macOS pynput global keyboard listener started.")
            except Exception as e:
                logging.warning(
                    f"[InputBackend] Failed to start pynput listener "
                    f"(Input Monitoring permission may be required): {e}"
                )

    def _normalize_pynput_key(self, key: Any) -> Set[str]:
        """Normalize a pynput key event into canonical alias names."""
        aliases: Set[str] = set()
        name = getattr(key, "name", None)
        if name:
            name_low = name.lower()
            aliases.add(name_low)
            if name_low in ("alt_r", "option_r"):
                aliases.update(["right alt", "right option", "alt_r", "option_r", "alt", "option"])
            elif name_low in ("alt_l", "option_l"):
                aliases.update(["left alt", "left option", "alt_l", "option_l", "alt", "option"])
            elif name_low in ("alt", "option"):
                aliases.update(["alt", "option"])
            elif name_low in ("ctrl_r", "control_r"):
                aliases.update(["right ctrl", "right control", "ctrl", "control"])
            elif name_low in ("ctrl_l", "control_l", "ctrl", "control"):
                aliases.update(["left ctrl", "left control", "ctrl", "control"])
            elif name_low in ("cmd_r", "command_r"):
                aliases.update(["right cmd", "right command", "cmd", "command"])
            elif name_low in ("cmd_l", "command_l", "cmd", "command"):
                aliases.update(["left cmd", "left command", "cmd", "command"])
            elif name_low in ("shift_r",):
                aliases.update(["right shift", "shift"])
            elif name_low in ("shift_l", "shift"):
                aliases.update(["left shift", "shift"])
            elif name_low in ("enter", "return"):
                aliases.update(["enter", "return"])
            elif name_low in ("esc", "escape"):
                aliases.update(["esc", "escape"])

        char = getattr(key, "char", None)
        if char:
            aliases.add(char.lower())

        return aliases

    def _on_pynput_press(self, key: Any) -> None:
        """Internal callback when a key is pressed."""
        aliases = self._normalize_pynput_key(key)
        self._currently_pressed.update(aliases)

        # 1. Fire on_press callbacks
        key_name = next(iter(aliases), str(key))
        event = type("KeyEvent", (), {"name": key_name})()
        for alias in aliases:
            for cb in list(self._press_callbacks.get(alias, [])):
                try:
                    cb(event)
                except Exception as e:
                    logging.error(f"[InputBackend] Error in on_press callback for {alias}: {e}")

        # 2. Check combo hotkeys
        for combo_set, cb in list(self._hotkey_callbacks):
            if combo_set.issubset(self._currently_pressed):
                f_combo = frozenset(combo_set)
                if f_combo not in self._triggered_hotkeys:
                    self._triggered_hotkeys.add(f_combo)
                    try:
                        cb()
                    except Exception as e:
                        logging.error(f"[InputBackend] Error in hotkey callback for {combo_set}: {e}")

    def _on_pynput_release(self, key: Any) -> None:
        """Internal callback when a key is released."""
        aliases = self._normalize_pynput_key(key)

        # 1. Fire on_release callbacks
        key_name = next(iter(aliases), str(key))
        event = type("KeyEvent", (), {"name": key_name})()
        for alias in aliases:
            for cb in list(self._release_callbacks.get(alias, [])):
                try:
                    cb(event)
                except Exception as e:
                    logging.error(f"[InputBackend] Error in on_release callback for {alias}: {e}")

        # 2. Update currently pressed keys
        self._currently_pressed.difference_update(aliases)

        # 3. Clean up triggered hotkeys
        self._triggered_hotkeys = {
            f_combo for f_combo in self._triggered_hotkeys
            if set(f_combo).issubset(self._currently_pressed)
        }

    def on_press_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        self._ensure_listener()
        k_norm = key.strip().lower()
        if k_norm not in self._press_callbacks:
            self._press_callbacks[k_norm] = []
        self._press_callbacks[k_norm].append(callback)
        return callback

    def on_release_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        self._ensure_listener()
        k_norm = key.strip().lower()
        if k_norm not in self._release_callbacks:
            self._release_callbacks[k_norm] = []
        self._release_callbacks[k_norm].append(callback)
        return callback

    def add_hotkey(self, combo: str, callback: Callable[[], None], suppress: bool = False) -> Any:
        self._ensure_listener()
        parts = {p.strip().lower() for p in combo.split("+") if p.strip()}
        self._hotkey_callbacks.append((parts, callback))
        return callback

    def unhook_all(self) -> None:
        with self._listener_lock:
            if self._listener is not None:
                try:
                    self._listener.stop()
                except Exception:
                    pass
                self._listener = None
            self._press_callbacks.clear()
            self._release_callbacks.clear()
            self._hotkey_callbacks.clear()
            self._currently_pressed.clear()
            self._triggered_hotkeys.clear()
            self._stop_event.set()

    def wait(self) -> None:
        self._stop_event.wait()

    def _post_quartz_key(self, keycode: int, key_down: bool, flags: int = 0) -> None:
        """Post a Quartz CGEvent for a keypress/release."""
        try:
            import Quartz
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, key_down)
            if flags:
                Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        except Exception as e:
            logging.warning(f"[InputBackend] Quartz synthetic key event failed: {e}")

    def press(self, key: str) -> None:
        k = key.strip().lower()
        flag = _MAC_FLAG_MASKS.get(k, 0)
        if flag:
            self._active_modifier_flags |= flag
        code = _MAC_KEY_CODES.get(k, 0)
        self._post_quartz_key(code, True, self._active_modifier_flags)

    def release(self, key: str) -> None:
        k = key.strip().lower()
        flag = _MAC_FLAG_MASKS.get(k, 0)
        if flag:
            self._active_modifier_flags &= ~flag
        code = _MAC_KEY_CODES.get(k, 0)
        self._post_quartz_key(code, False, self._active_modifier_flags)

    def press_and_release(self, combo: str) -> None:
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        flags = self._active_modifier_flags
        target_key = None
        for p in parts:
            if p in _MAC_FLAG_MASKS:
                flags |= _MAC_FLAG_MASKS[p]
            else:
                target_key = p

        if not target_key and parts:
            target_key = parts[-1]

        code = _MAC_KEY_CODES.get(target_key, 0)
        self._post_quartz_key(code, True, flags)
        time.sleep(0.01)
        self._post_quartz_key(code, False, self._active_modifier_flags)

    def write(self, text: str, delay: float = 0.0) -> None:
        if not text:
            return
        try:
            import Quartz
            for char in text:
                ev_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
                Quartz.CGEventKeyboardSetUnicodeString(ev_down, len(char), char)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)

                ev_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
                if delay > 0:
                    time.sleep(delay)
        except Exception as e:
            logging.warning(f"[InputBackend] Quartz write failed: {e}")


class FallbackInputBackend(BaseInputBackend):
    """Safe no-op input backend for headless environments or unsupported platforms."""

    def on_press_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        return None

    def on_release_key(self, key: str, callback: Callable[[Any], None], suppress: bool = False) -> Any:
        return None

    def add_hotkey(self, combo: str, callback: Callable[[], None], suppress: bool = False) -> Any:
        return None

    def unhook_all(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def press(self, key: str) -> None:
        pass

    def release(self, key: str) -> None:
        pass

    def press_and_release(self, combo: str) -> None:
        pass

    def write(self, text: str, delay: float = 0.0) -> None:
        pass


_INPUT_BACKEND_INSTANCE: Optional[BaseInputBackend] = None


def get_input_backend() -> BaseInputBackend:
    """Return the platform-appropriate input backend singleton."""
    global _INPUT_BACKEND_INSTANCE
    if _INPUT_BACKEND_INSTANCE is None:
        if sys.platform == "win32":
            _INPUT_BACKEND_INSTANCE = WindowsInputBackend()
        elif sys.platform == "darwin":
            _INPUT_BACKEND_INSTANCE = MacOSInputBackend()
        else:
            _INPUT_BACKEND_INSTANCE = FallbackInputBackend()
    return _INPUT_BACKEND_INSTANCE


def set_input_backend(backend: Optional[BaseInputBackend]) -> None:
    """Explicitly set or reset the input backend singleton (useful for testing)."""
    global _INPUT_BACKEND_INSTANCE
    _INPUT_BACKEND_INSTANCE = backend
