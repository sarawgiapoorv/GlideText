"""
voice_commands.py -- Safe VoiceCommand layer for GlideText.

Architecture & Safety Guarantees:
  - Voice commands are detected BEFORE LLM polishing and executed locally.
  - Commands operate strictly on GlideText's OWN insertion history stack.
  - Zero unrestricted autonomous document editing.
  - Strict command recognition: phrases embedded in longer spoken sentences
    (e.g., "I scratched that idea from the document") are treated as ordinary text.
  - Non-destructive fallbacks: if there is no previous insertion or the active window
    has changed, no destructive keys are sent.
  - Recovery buffer: deleted text is preserved in history for audit & recovery.
"""

from __future__ import annotations

import enum
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False


class VoiceCommandType(str, enum.Enum):
    """Supported voice commands."""
    SCRATCH_THAT = "scratch_that"
    DELETE_THAT = "delete_that"
    DELETE_PREVIOUS_SENTENCE = "delete_previous_sentence"
    DELETE_PREVIOUS_PARAGRAPH = "delete_previous_paragraph"
    UNDO_LAST_DICTATION = "undo_last_dictation"
    CLEAR_LAST_DICTATION = "clear_last_dictation"


@dataclass(frozen=True)
class VoiceCommand:
    """Represents a recognized standalone voice command."""
    command_type: VoiceCommandType
    raw_spoken_text: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class VoiceCommandResult:
    """Outcome of a voice command execution."""
    success: bool
    command_type: Optional[VoiceCommandType] = None
    executed: bool = False
    deleted_text: str = ""
    message: str = ""
    error: Optional[str] = None


@dataclass
class InsertionRecord:
    """Record of text inserted by GlideText into a target window."""
    session_id: str
    text: str
    target_hwnd: Optional[int]
    timestamp: float
    char_count: int
    word_count: int

    @property
    def sentences(self) -> list[str]:
        """Split text into sentences preserving content."""
        if not self.text.strip():
            return []
        parts = re.split(r'(?<=[.!?])\s+', self.text.strip())
        return [p for p in parts if p]

    @property
    def paragraphs(self) -> list[str]:
        """Split text into paragraphs."""
        if not self.text.strip():
            return []
        parts = re.split(r'\n+', self.text.strip())
        return [p for p in parts if p]


class InsertionHistory:
    """Thread-safe bounded stack of GlideText insertions for safe undo/scratch operations."""

    def __init__(self, max_history: int = 25):
        self.max_history = max_history
        self._history: list[InsertionRecord] = []
        self._recovery_stack: list[InsertionRecord] = []
        self._lock = threading.RLock()

    def record_insertion(
        self,
        session_id: str,
        text: str,
        target_hwnd: Optional[int] = None,
    ) -> Optional[InsertionRecord]:
        """Add a successful GlideText injection to the insertion history."""
        if not text or not text.strip():
            return None

        record = InsertionRecord(
            session_id=session_id,
            text=text,
            target_hwnd=target_hwnd,
            timestamp=time.time(),
            char_count=len(text),
            word_count=len(text.split()),
        )

        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history.pop(0)
            return record

    def peek(self) -> Optional[InsertionRecord]:
        """Return the most recent insertion without removing it."""
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def pop_last(self, target_hwnd: Optional[int] = None) -> Optional[InsertionRecord]:
        """Remove and return the most recent insertion, optionally matching target_hwnd."""
        with self._lock:
            if not self._history:
                return None
            if target_hwnd is not None:
                # Find most recent record for this specific window
                for i in range(len(self._history) - 1, -1, -1):
                    rec = self._history[i]
                    if rec.target_hwnd == target_hwnd or rec.target_hwnd is None:
                        popped = self._history.pop(i)
                        self._recovery_stack.append(popped)
                        return popped
                return None

            popped = self._history.pop()
            self._recovery_stack.append(popped)
            return popped

    def get_recovery_records(self) -> list[InsertionRecord]:
        """Return list of recovered/undone insertions."""
        with self._lock:
            return list(self._recovery_stack)

    def clear(self) -> None:
        """Clear all history and recovery records."""
        with self._lock:
            self._history.clear()
            self._recovery_stack.clear()


# ---------------------------------------------------------------------------
# Strict Voice Command Parser
# ---------------------------------------------------------------------------

# Strict standalone exact phrases mapped to VoiceCommandType
_STANDALONE_COMMAND_PATTERNS: list[tuple[re.Pattern, VoiceCommandType]] = [
    (re.compile(r"^(?:please\s+)?scratch\s+that(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.SCRATCH_THAT),
    (re.compile(r"^(?:please\s+)?delete\s+that(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.DELETE_THAT),
    (re.compile(r"^(?:please\s+)?(?:delete|remove)\s+(?:the\s+)?(?:previous|last)\s+sentence(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.DELETE_PREVIOUS_SENTENCE),
    (re.compile(r"^(?:please\s+)?(?:delete|remove)\s+(?:the\s+)?(?:previous|last)\s+paragraph(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH),
    (re.compile(r"^(?:please\s+)?undo\s+(?:the\s+)?(?:last\s+)?dictation(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.UNDO_LAST_DICTATION),
    (re.compile(r"^(?:please\s+)?clear\s+(?:the\s+)?(?:last\s+)?dictation(?:\s+please)?$", re.IGNORECASE), VoiceCommandType.CLEAR_LAST_DICTATION),
    (re.compile(r"^undo\s+that$", re.IGNORECASE), VoiceCommandType.UNDO_LAST_DICTATION),
]


def parse_voice_command(spoken_text: str) -> Optional[VoiceCommand]:
    """Strictly parse spoken audio transcript for an intentional standalone voice command.

    Safety & False-Positive Immunity:
      - Embedded phrases (e.g. "I scratched that idea from the document", "Don't delete that")
        return None and are dictated as ordinary text.
      - Transcripts with more than 6 words are never parsed as single-phrase commands.
      - Punctuation (periods, commas, quotes) from ASR are stripped cleanly before matching.

    Returns:
      VoiceCommand if spoken text matches a standalone command, None otherwise.
    """
    if not spoken_text:
        return None

    cleaned = spoken_text.strip().strip('"\'`').strip().rstrip(".!?,:;").strip()
    if not cleaned:
        return None

    words = cleaned.split()
    # Safety guard: standalone voice commands are concise (<= 6 words)
    if len(words) > 6:
        return None

    # Disallow conversational/narrative prefixes (e.g. "I said scratch that", "we can scratch that")
    narrative_prefixes = ["i ", "we ", "he ", "she ", "they ", "you ", "don't ", "dont ", "do not "]
    cleaned_lower = cleaned.lower()
    if any(cleaned_lower.startswith(p) for p in narrative_prefixes):
        return None

    for pattern, cmd_type in _STANDALONE_COMMAND_PATTERNS:
        if pattern.match(cleaned):
            return VoiceCommand(
                command_type=cmd_type,
                raw_spoken_text=spoken_text,
            )

    return None


# ---------------------------------------------------------------------------
# Voice Command Executor
# ---------------------------------------------------------------------------

class VoiceCommandExecutor:
    """Executes validated voice commands safely against GlideText's insertion history."""

    def __init__(self, insertion_history: InsertionHistory):
        self.history = insertion_history
        self._lock = threading.RLock()

    def execute_command(
        self,
        command: VoiceCommand,
        target_hwnd: Optional[int] = None,
    ) -> VoiceCommandResult:
        """Execute a parsed VoiceCommand safely.

        Guarantees:
          - If there is no known GlideText insertion, nothing is sent to the OS.
          - If the target window has changed, no destructive keys are sent.
          - Preserves deleted text in the recovery stack.
        """
        if command is None:
            return VoiceCommandResult(
                success=False,
                error="Null voice command",
            )

        with self._lock:
            cmd_type = command.command_type
            last_record = self.history.peek()

            if last_record is None:
                logging.info(
                    f"[VoiceCommand] Ignored '{cmd_type.value}': No previous GlideText insertion in history."
                )
                return VoiceCommandResult(
                    success=False,
                    command_type=cmd_type,
                    executed=False,
                    message="No previous dictation to undo",
                )

            # Target HWND safety verification
            if target_hwnd is not None and last_record.target_hwnd is not None:
                if target_hwnd != last_record.target_hwnd:
                    logging.warning(
                        f"[VoiceCommand] Refused '{cmd_type.value}': Target window mismatch "
                        f"(current={target_hwnd}, insertion={last_record.target_hwnd})."
                    )
                    return VoiceCommandResult(
                        success=False,
                        command_type=cmd_type,
                        executed=False,
                        message="Active window changed -- undo skipped for safety",
                    )

            if cmd_type in (
                VoiceCommandType.SCRATCH_THAT,
                VoiceCommandType.DELETE_THAT,
                VoiceCommandType.UNDO_LAST_DICTATION,
                VoiceCommandType.CLEAR_LAST_DICTATION,
            ):
                return self._undo_full_insertion(last_record)

            elif cmd_type == VoiceCommandType.DELETE_PREVIOUS_SENTENCE:
                return self._delete_previous_sentence(last_record)

            elif cmd_type == VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH:
                return self._delete_previous_paragraph(last_record)

            return VoiceCommandResult(
                success=False,
                command_type=cmd_type,
                error=f"Unsupported command type: {cmd_type}",
            )

    def _undo_full_insertion(self, record: InsertionRecord) -> VoiceCommandResult:
        """Safely delete the entirety of the last inserted text."""
        popped = self.history.pop_last()
        if popped is None:
            return VoiceCommandResult(
                success=False,
                message="No previous dictation to undo",
            )

        char_count = popped.char_count
        deleted_text = popped.text

        success = self._send_backspaces(char_count)
        if success:
            logging.info(f"[VoiceCommand] Undid full insertion ({char_count} chars): '{deleted_text[:40]}...'")
            return VoiceCommandResult(
                success=True,
                command_type=VoiceCommandType.SCRATCH_THAT,
                executed=True,
                deleted_text=deleted_text,
                message=f"Undid: '{deleted_text}'",
            )
        else:
            return VoiceCommandResult(
                success=False,
                command_type=VoiceCommandType.SCRATCH_THAT,
                executed=False,
                deleted_text=deleted_text,
                error="Keyboard driver unavailable to send backspaces",
            )

    def _delete_previous_sentence(self, record: InsertionRecord) -> VoiceCommandResult:
        """Delete the last sentence of the most recent insertion."""
        sentences = record.sentences
        if not sentences or len(sentences) <= 1:
            # Single sentence insertion -> undo entire insertion
            return self._undo_full_insertion(record)

        # Multiple sentences in this insertion: delete the last sentence
        last_sentence = sentences[-1]
        remaining_sentences = sentences[:-1]
        remaining_text = " ".join(remaining_sentences)

        # Pop old record and replace with updated record
        self.history.pop_last()
        self.history.record_insertion(
            session_id=record.session_id,
            text=remaining_text,
            target_hwnd=record.target_hwnd,
        )

        char_count = len(last_sentence)
        # Account for preceding space if present
        if len(record.text) > len(remaining_text) + char_count:
            char_count += (len(record.text) - (len(remaining_text) + char_count))

        success = self._send_backspaces(char_count)
        if success:
            logging.info(f"[VoiceCommand] Deleted previous sentence ({char_count} chars): '{last_sentence}'")
            return VoiceCommandResult(
                success=True,
                command_type=VoiceCommandType.DELETE_PREVIOUS_SENTENCE,
                executed=True,
                deleted_text=last_sentence,
                message=f"Deleted sentence: '{last_sentence}'",
            )
        else:
            return VoiceCommandResult(
                success=False,
                command_type=VoiceCommandType.DELETE_PREVIOUS_SENTENCE,
                error="Keyboard driver unavailable to send backspaces",
            )

    def _delete_previous_paragraph(self, record: InsertionRecord) -> VoiceCommandResult:
        """Delete the last paragraph of the most recent insertion."""
        paragraphs = record.paragraphs
        if not paragraphs or len(paragraphs) <= 1:
            # Single paragraph -> undo entire insertion
            return self._undo_full_insertion(record)

        last_paragraph = paragraphs[-1]
        remaining_paragraphs = paragraphs[:-1]
        remaining_text = "\n\n".join(remaining_paragraphs)

        self.history.pop_last()
        self.history.record_insertion(
            session_id=record.session_id,
            text=remaining_text,
            target_hwnd=record.target_hwnd,
        )

        char_count = len(last_paragraph)
        if len(record.text) > len(remaining_text) + char_count:
            char_count += (len(record.text) - (len(remaining_text) + char_count))

        success = self._send_backspaces(char_count)
        if success:
            logging.info(f"[VoiceCommand] Deleted previous paragraph ({char_count} chars): '{last_paragraph}'")
            return VoiceCommandResult(
                success=True,
                command_type=VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH,
                executed=True,
                deleted_text=last_paragraph,
                message=f"Deleted paragraph: '{last_paragraph}'",
            )
        else:
            return VoiceCommandResult(
                success=False,
                command_type=VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH,
                error="Keyboard driver unavailable to send backspaces",
            )

    @staticmethod
    def _send_backspaces(count: int) -> bool:
        """Simulate sending `count` backspaces to remove inserted characters."""
        if count <= 0:
            return True
        if not HAS_KEYBOARD:
            return False

        try:
            # For small to moderate text, send backspaces with minimal delay
            # If large (>100 chars), send in rapid sequence
            delay = 0.002 if count > 50 else 0.005
            for _ in range(count):
                keyboard.press_and_release("backspace")
                time.sleep(delay)
            return True
        except Exception as e:
            logging.error(f"[VoiceCommand] Failed to send backspaces: {e}")
            return False


# Global singleton instance for app-wide insertion tracking
GLOBAL_INSERTION_HISTORY = InsertionHistory()
GLOBAL_VOICE_COMMAND_EXECUTOR = VoiceCommandExecutor(GLOBAL_INSERTION_HISTORY)
