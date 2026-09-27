"""
text_injector.py — OS-level text injection for GlideText.

Types text directly at the active Windows cursor position using simulated
keystrokes. Works in any application: browsers, editors, chat apps, terminals.

Features:
  • Snippet expansion: voice macros from snippets.json
  • Unicode-safe keystroke injection
"""

import logging
import json
import os
import time
import keyboard


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEYSTROKE_DELAY = 0.012  # ~12 ms between keystrokes


class TextInjector:
    """Simulates typing at the current cursor position with snippet expansion."""

    def __init__(self, delay: float = KEYSTROKE_DELAY):
        self.delay = delay
        self._snippets: dict[str, str] = self._load_snippets()

    @staticmethod
    def _load_snippets() -> dict[str, str]:
        """Load snippet macros from snippets.json."""
        snippets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "snippets.json"
        )
        if os.path.isfile(snippets_path):
            try:
                with open(snippets_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"  ⚠  Failed to load snippets.json: {e}")
        return {}

    def reload_snippets(self) -> None:
        """Hot-reload snippets from disk."""
        self._snippets = self._load_snippets()

    def expand_snippets(self, text: str) -> str:
        """
        Scan text for snippet trigger phrases and expand them.

        Matching is case-insensitive. The entire text is replaced if it
        matches a trigger exactly, or individual trigger phrases within
        the text are expanded inline.

        Args:
            text: The polished text from the AI pipeline.

        Returns:
            Text with any matching snippets expanded.
        """
        if not self._snippets or not text:
            return text

        # Check for exact full-text match first (case-insensitive)
        text_lower = text.strip().lower()
        sorted_snippets = sorted(self._snippets.items(), key=lambda x: len(x[0]), reverse=True)
        for trigger, expansion in sorted_snippets:
            if text_lower == trigger.lower().strip():
                logging.info(f"  > Snippet expanded: '{trigger}'")
                return expansion

        # Inline replacement for triggers found within the text
        result = text
        for trigger, expansion in sorted_snippets:
            # Case-insensitive search and replace
            idx = result.lower().find(trigger.lower())
            while idx != -1:
                result = result[:idx] + expansion + result[idx + len(trigger):]
                logging.info(f"  > Snippet expanded inline: '{trigger}'")
                idx = result.lower().find(trigger.lower(), idx + len(expansion))

        return result

    def inject(self, text: str) -> str:
        """
        Type the given text at the current cursor position.

        Args:
            text: The polished text to type out.

        Returns:
            The exact normalized string that was typed.
        """
        if not text:
            return ""

        cleaned = text.strip().replace("\r\n", "\n")

        if not cleaned:
            return ""

        # Small pause before typing to let the user's key-release register
        time.sleep(0.15)

        try:
            keyboard.write(cleaned, delay=self.delay)
        except Exception as e:
            logging.info(f"  ! Keyboard write failed: {e}")
        
        return cleaned

    def inject_with_newline(self, text: str) -> None:
        """
        Type the text followed by a newline (Enter key).

        Args:
            text: The polished text to type out.
        """
        self.inject(text)
        if text and text.strip():
            time.sleep(0.05)
            keyboard.press_and_release("enter")
