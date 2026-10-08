"""
dictation_session.py -- First-class DictationSession state machine for GlideText.

Architecture:
  - A user dictation is a continuous session (continuous mode or push-to-talk),
    NOT a sequence of VAD-delimited silence chunks.
  - Silence is an activity state (RECORDING <-> PAUSED), never a session boundary
    in continuous mode.
  - Every session has a unique, immutable session_id.
  - Strict idempotency guards prevent duplicate finalization, transcription,
    polishing, or text injection.
  - Structured session lifecycle logging without exposing raw dictated text or
    sensitive audio payloads.
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


class SessionState(str, enum.Enum):
    """Lifecycle states of a DictationSession."""
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    FINALIZING = "FINALIZING"
    TRANSCRIBING = "TRANSCRIBING"
    POLISHING = "POLISHING"
    INJECTING = "INJECTING"
    DONE = "DONE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class SessionMode(str, enum.Enum):
    """Trigger modes for a DictationSession."""
    PUSH_TO_TALK = "push_to_talk"
    CONTINUOUS = "continuous"


class SessionEvent(str, enum.Enum):
    """Structured session lifecycle events for privacy-safe logging."""
    SESSION_STARTED = "SESSION_STARTED"
    SPEECH_STARTED = "SPEECH_STARTED"
    SPEECH_PAUSED = "SPEECH_PAUSED"
    SPEECH_RESUMED = "SPEECH_RESUMED"
    SESSION_FINALIZED = "SESSION_FINALIZED"
    SESSION_TRANSCRIPTION_STARTED = "SESSION_TRANSCRIPTION_STARTED"
    SESSION_TRANSCRIPTION_COMPLETED = "SESSION_TRANSCRIPTION_COMPLETED"
    SESSION_POLISH_STARTED = "SESSION_POLISH_STARTED"
    SESSION_POLISH_COMPLETED = "SESSION_POLISH_COMPLETED"
    SESSION_INJECTION_STARTED = "SESSION_INJECTION_STARTED"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    SESSION_CANCELLED = "SESSION_CANCELLED"
    SESSION_ERROR = "SESSION_ERROR"


# Valid state transitions
_VALID_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {SessionState.RECORDING, SessionState.CANCELLED, SessionState.ERROR},
    SessionState.RECORDING: {
        SessionState.PAUSED,
        SessionState.FINALIZING,
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.PAUSED: {
        SessionState.RECORDING,
        SessionState.FINALIZING,
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.FINALIZING: {
        SessionState.TRANSCRIBING,
        SessionState.DONE,   # e.g. empty/too-short audio discarded cleanly
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.TRANSCRIBING: {
        SessionState.POLISHING,
        SessionState.DONE,   # e.g. no speech detected or dictionary meta-command
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.POLISHING: {
        SessionState.INJECTING,
        SessionState.DONE,
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.INJECTING: {
        SessionState.DONE,
        SessionState.CANCELLED,
        SessionState.ERROR,
    },
    SessionState.DONE: set(),
    SessionState.CANCELLED: set(),
    SessionState.ERROR: set(),
}


@dataclass
class SessionMetrics:
    """Performance, duration, and memory metrics for a DictationSession."""
    session_id: str
    audio_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    silence_duration_sec: float = 0.0
    processing_time_sec: float = 0.0
    transcription_time_sec: float = 0.0
    polish_time_sec: float = 0.0
    injection_time_sec: float = 0.0
    peak_memory_bytes: int = 0
    peak_memory_mb: float = 0.0
    transcript_length: int = 0
    transcript_word_count: int = 0
    final_text_length: int = 0
    final_text_word_count: int = 0
    pause_count: int = 0
    speech_segment_count: int = 0
    provider_used: Optional[str] = None
    is_fallback: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class LifecycleLogEntry:
    """Privacy-safe record of a lifecycle event on a session."""
    session_id: str
    event: SessionEvent
    state: SessionState
    timestamp: float
    metadata: dict = field(default_factory=dict)


class DictationSession:
    """Represents a single user-initiated dictation session.

    Guarantees:
      - Unique, immutable `session_id`
      - Silence transitions between RECORDING and PAUSED without ending the session
      - Audio is preserved across any number of silence/thinking intervals
      - At most one finalization, one transcription, one polish, and one injection
      - Safe memory bounds (zero unneeded duplicates, immediate release after transcription)
    """

    def __init__(
        self,
        mode: SessionMode | str = SessionMode.PUSH_TO_TALK,
        context_info: Optional[dict] = None,
        target_hwnd: Optional[Any] = None,
        style: str = "Normal",
        session_id: Optional[str] = None,
        lookback_context: str = "",
    ):
        if isinstance(mode, str):
            mode = SessionMode(mode)
        self._session_id: str = session_id or uuid.uuid4().hex
        self._mode: SessionMode = mode
        self._state: SessionState = SessionState.IDLE
        self._lock = threading.RLock()

        # Context & configuration captured strictly for this session (never leaked across sessions)
        self.context_info: dict = dict(context_info) if context_info else {}
        self.target_hwnd: Optional[Any] = (
            target_hwnd if target_hwnd is not None else self.context_info.get("hwnd")
        )
        if self.target_hwnd is not None:
            self.context_info["target_hwnd"] = self.target_hwnd
        self.style: str = style

        from context_snapshot import ContextSnapshot, build_context_snapshot
        self._context_snapshot: ContextSnapshot = build_context_snapshot(
            session_id=self._session_id,
            context_info=self.context_info,
            raw_lookback=lookback_context,
            user_style=self.style,
        )
        self._lookback_context: str = self._context_snapshot.bounded_cursor_text

        # Timestamps
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finalized_at: Optional[float] = None
        self.completed_at: Optional[float] = None

        # Stage timings for telemetry & metrics
        self.transcription_started_at: Optional[float] = None
        self.transcription_completed_at: Optional[float] = None
        self.polish_started_at: Optional[float] = None
        self.polish_completed_at: Optional[float] = None
        self.injection_started_at: Optional[float] = None
        self.injection_completed_at: Optional[float] = None

        # Speech / VAD activity tracking within the session
        self._has_had_speech: bool = False
        self._speech_segment_count: int = 0
        self._pause_count: int = 0
        self._total_silence_sec: float = 0.0
        self._total_audio_samples: int = 0

        # Audio storage (in-memory chunks + consolidated WAV path)
        self._audio_chunks: list[np.ndarray] = []
        self.audio_path: Optional[str] = None

        # Idempotency flags (strictly once per session)
        self._finalized_once: bool = False
        self._transcribed_once: bool = False
        self._polished_once: bool = False
        self._injected_once: bool = False

        # Results (kept in memory for session processing, never logged raw in lifecycle events)
        self.raw_transcript: str = ""
        self.corrected_transcript: str = ""
        self.polished_text: str = ""
        self.provider_used: Optional[str] = None
        self.is_fallback: bool = False
        self.error: Optional[str] = None

        # Audit trail of lifecycle events
        self._events: list[LifecycleLogEntry] = []

    # ------------------------------------------------------------------
    # Immutable & context properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Unique immutable identifier for this dictation session."""
        return self._session_id

    @property
    def start_time(self) -> float:
        """Timestamp when recording began."""
        return self.started_at or self.created_at

    @property
    def duration(self) -> float:
        """Duration of audio capture in seconds."""
        with self._lock:
            if self.finalized_at and self.started_at:
                return max(0.0, round(self.finalized_at - self.started_at, 3))
            if self.started_at:
                return max(0.0, round(time.time() - self.started_at, 3))
            return 0.0

    @property
    def audio_duration_sec(self) -> float:
        """Total audio duration in seconds."""
        with self._lock:
            if self._total_audio_samples > 0:
                return round(self._total_audio_samples / 16000.0, 3)
            return self.duration

    @property
    def speech_metadata(self) -> dict:
        """Speech and silence metadata for this session."""
        with self._lock:
            return {
                "speech_segment_count": self._speech_segment_count,
                "pause_count": self._pause_count,
                "total_silence_sec": round(self._total_silence_sec, 2),
                "has_had_speech": self._has_had_speech,
                "audio_duration_sec": self.audio_duration_sec,
            }

    @property
    def transcript(self) -> str:
        """Raw transcript produced by ASR for this session."""
        with self._lock:
            return self.raw_transcript

    @property
    def final_text(self) -> str:
        """Final output text (polished, corrected, or raw fallback)."""
        with self._lock:
            return self.polished_text or self.corrected_transcript or self.raw_transcript

    @property
    def is_cancelled(self) -> bool:
        """Return True if the session was cancelled."""
        with self._lock:
            return self._state == SessionState.CANCELLED

    @property
    def is_injected(self) -> bool:
        """Return True if text from this session has already been successfully injected."""
        with self._lock:
            return getattr(self, "_injected_successfully", False)

    @property
    def is_injectable(self) -> bool:
        """Return True if session is in a valid state to inject final text."""
        with self._lock:
            return (
                not getattr(self, "_injected_successfully", False)
                and self._state not in (SessionState.CANCELLED, SessionState.DONE, SessionState.ERROR)
                and bool(self.final_text and self.final_text.strip())
            )

    def mark_injected(self, success: bool = True) -> bool:
        """Atomically mark this session as successfully injected (preventing duplicate injection)."""
        with self._lock:
            if getattr(self, "_injected_successfully", False):
                return False
            self._injected_successfully = bool(success)
            return True

    @property
    def has_audio(self) -> bool:
        """Return True if session has captured audio in memory or on disk."""
        with self._lock:
            return bool(self.audio_path or len(self._audio_chunks) > 0)

    @property
    def context_snapshot(self):
        """Session-scoped ContextSnapshot for this dictation session."""
        with self._lock:
            return self._context_snapshot

    @property
    def lookback_context(self) -> str:
        """Bounded, sanitized cursor lookback text for this session."""
        with self._lock:
            return self._lookback_context

    @lookback_context.setter
    def lookback_context(self, value: str) -> None:
        self.set_lookback_context(value)

    def set_lookback_context(self, raw_lookback: Optional[str]) -> str:
        """Update this session's cursor lookback, applying sensitivity & length bounds."""
        with self._lock:
            self._context_snapshot = self._context_snapshot.with_cursor_text(raw_lookback)
            self._lookback_context = self._context_snapshot.bounded_cursor_text
            return self._lookback_context

    def refresh_context_snapshot(
        self,
        updated_context: Optional[dict] = None,
        raw_lookback: Optional[str] = None,
        style: Optional[str] = None,
    ):
        """Rebuild the session-scoped ContextSnapshot when context/style/lookback updates."""
        from context_snapshot import build_context_snapshot

        with self._lock:
            if updated_context:
                orig_hwnd = self.target_hwnd
                self.context_info.update(updated_context)
                if orig_hwnd is not None:
                    self.context_info["target_hwnd"] = orig_hwnd
            if style is not None:
                self.style = style
            effective_lookback = (
                raw_lookback if raw_lookback is not None else self._lookback_context
            )
            self._context_snapshot = build_context_snapshot(
                session_id=self._session_id,
                context_info=self.context_info,
                raw_lookback=effective_lookback,
                user_style=self.style,
            )
            self._lookback_context = self._context_snapshot.bounded_cursor_text
            return self._context_snapshot

    @property
    def mode(self) -> SessionMode:
        return self._mode

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def events(self) -> list[LifecycleLogEntry]:
        with self._lock:
            return list(self._events)

    @property
    def speech_segment_count(self) -> int:
        with self._lock:
            return self._speech_segment_count

    @property
    def pause_count(self) -> int:
        with self._lock:
            return self._pause_count

    @property
    def is_active_capture(self) -> bool:
        """Return True if the session is currently capturing audio (RECORDING or PAUSED)."""
        with self._lock:
            return self._state in (SessionState.RECORDING, SessionState.PAUSED)

    @property
    def is_terminal(self) -> bool:
        """Return True if the session has finished (DONE, ERROR, or CANCELLED)."""
        with self._lock:
            return self._state in (SessionState.DONE, SessionState.ERROR, SessionState.CANCELLED)

    # ------------------------------------------------------------------
    # Privacy-safe lifecycle logging
    # ------------------------------------------------------------------

    def _log_event(self, event: SessionEvent, **metadata) -> None:
        """Record and emit a privacy-safe structured lifecycle log."""
        safe_meta = {
            k: v for k, v in metadata.items()
            if k not in (
                "raw_text",
                "raw_transcript",
                "corrected_transcript",
                "polished_text",
                "text",
                "audio_data",
                "transcript",
                "final_text",
                "lookback_context",
                "bounded_cursor_text",
                "pre_text",
            )
        }
        entry = LifecycleLogEntry(
            session_id=self._session_id,
            event=event,
            state=self._state,
            timestamp=time.time(),
            metadata=safe_meta,
        )
        self._events.append(entry)

        meta_str = " ".join(f"{k}={v}" for k, v in safe_meta.items())
        if meta_str:
            logging.info(
                f"[Session:{self._session_id[:8]}] {event.value} "
                f"(state={self._state.value}, mode={self._mode.value}) {meta_str}"
            )
        else:
            logging.info(
                f"[Session:{self._session_id[:8]}] {event.value} "
                f"(state={self._state.value}, mode={self._mode.value})"
            )

    def _transition_to(self, new_state: SessionState) -> None:
        """Validate and transition to `new_state` under lock."""
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise RuntimeError(
                f"Invalid session state transition for session {self._session_id[:8]}: "
                f"{self._state.value} -> {new_state.value}"
            )
        self._state = new_state

    # ------------------------------------------------------------------
    # Audio buffer management & memory safety
    # ------------------------------------------------------------------

    def append_audio_chunk(self, chunk: np.ndarray) -> None:
        """Append captured audio frames to this session without duplicate copies."""
        if chunk is None or len(chunk) == 0:
            return
        with self._lock:
            if not self.is_active_capture:
                return
            # Zero unnecessary copies: chunk is already an isolated numpy block from sounddevice
            self._audio_chunks.append(chunk)
            self._total_audio_samples += len(chunk)

    def get_concatenated_audio(self) -> Optional[np.ndarray]:
        """Return all audio frames accumulated across the entire session."""
        with self._lock:
            if not self._audio_chunks:
                return None
            return np.concatenate(self._audio_chunks, axis=0)

    def release_audio_buffers(self) -> None:
        """Release in-memory audio chunks to free RAM immediately after transcription."""
        with self._lock:
            self._audio_chunks.clear()

    def cleanup_audio(self) -> None:
        """Clean up both in-memory audio buffers and temporary WAV file on disk."""
        with self._lock:
            self._audio_chunks.clear()
            if self.audio_path:
                try:
                    if os.path.exists(self.audio_path):
                        os.remove(self.audio_path)
                except OSError:
                    pass
                self.audio_path = None

    # ------------------------------------------------------------------
    # State machine transitions & lifecycle methods
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Transition IDLE -> RECORDING and emit SESSION_STARTED."""
        with self._lock:
            if self._state != SessionState.IDLE:
                return False
            self._transition_to(SessionState.RECORDING)
            self.started_at = time.time()
            self._log_event(SessionEvent.SESSION_STARTED)
            return True

    def on_speech_detected(self) -> bool:
        """Called by VAD when voice activity begins or resumes after silence.

        - First speech in session -> emits SPEECH_STARTED
        - Speech after PAUSED -> transitions PAUSED -> RECORDING and emits SPEECH_RESUMED
        """
        with self._lock:
            if self._state == SessionState.RECORDING:
                if not self._has_had_speech:
                    self._has_had_speech = True
                    self._speech_segment_count += 1
                    self._log_event(
                        SessionEvent.SPEECH_STARTED,
                        segment=self._speech_segment_count,
                    )
                    return True
                return False

            if self._state == SessionState.PAUSED:
                self._transition_to(SessionState.RECORDING)
                self._speech_segment_count += 1
                self._log_event(
                    SessionEvent.SPEECH_RESUMED,
                    segment=self._speech_segment_count,
                    pause_count=self._pause_count,
                )
                return True

            return False

    def on_silence_detected(self, silence_duration: float = 0.0) -> bool:
        """Called by VAD when silence is detected after speech.

        CRITICAL: This transitions RECORDING -> PAUSED (speech-activity state only).
        It NEVER finalizes or terminates the session.
        """
        with self._lock:
            if self._state != SessionState.RECORDING:
                return False
            self._transition_to(SessionState.PAUSED)
            self._pause_count += 1
            if silence_duration > 0:
                self._total_silence_sec += silence_duration
            self._log_event(
                SessionEvent.SPEECH_PAUSED,
                pause_count=self._pause_count,
                silence_sec=round(silence_duration, 2),
            )
            return True

    def finalize(self, audio_path: Optional[str] = None) -> bool:
        """Explicitly finalize audio capture for the session.

        Idempotent: calling `finalize()` more than once returns False and
        prevents duplicate processing.
        """
        with self._lock:
            if self._finalized_once or self._state not in (
                SessionState.RECORDING,
                SessionState.PAUSED,
            ):
                logging.debug(
                    f"[Session:{self._session_id[:8]}] Duplicate or invalid finalize() ignored "
                    f"(state={self._state.value}, already_finalized={self._finalized_once})."
                )
                return False

            self._finalized_once = True
            self._transition_to(SessionState.FINALIZING)
            self.finalized_at = time.time()
            if audio_path is not None:
                self.audio_path = audio_path

            duration_s = (
                round(self.finalized_at - self.started_at, 2)
                if self.started_at
                else 0.0
            )
            self._log_event(
                SessionEvent.SESSION_FINALIZED,
                duration_sec=duration_s,
                speech_segments=self._speech_segment_count,
                pauses=self._pause_count,
                has_audio=bool(self.audio_path or self._audio_chunks),
            )
            return True

    def begin_transcription(self) -> bool:
        """Transition FINALIZING -> TRANSCRIBING (guarded to run at most once)."""
        with self._lock:
            if self._transcribed_once or self._state != SessionState.FINALIZING:
                logging.debug(
                    f"[Session:{self._session_id[:8]}] Duplicate or invalid begin_transcription() ignored "
                    f"(state={self._state.value}, already_transcribed={self._transcribed_once})."
                )
                return False
            self._transcribed_once = True
            self._transition_to(SessionState.TRANSCRIBING)
            self.transcription_started_at = time.time()
            self._log_event(SessionEvent.SESSION_TRANSCRIPTION_STARTED)
            return True

    def complete_transcription(self, raw_text: str) -> None:
        """Record transcription completion without logging sensitive text content.

        Preserves `self.raw_transcript` verbatim for recovery/debugging and
        computes `self.corrected_transcript` with natural spoken corrections resolved.
        """
        with self._lock:
            if self._state != SessionState.TRANSCRIBING:
                raise RuntimeError(
                    f"complete_transcription called in state {self._state.value}"
                )
            self.transcription_completed_at = time.time()
            self.raw_transcript = raw_text or ""
            from spoken_corrections import resolve_spoken_corrections
            self.corrected_transcript = resolve_spoken_corrections(self.raw_transcript)
            char_len = len(self.raw_transcript)
            word_count = len(self.raw_transcript.split()) if self.raw_transcript.strip() else 0
            self._log_event(
                SessionEvent.SESSION_TRANSCRIPTION_COMPLETED,
                has_speech=bool(self.raw_transcript.strip()),
                char_len=char_len,
                word_count=word_count,
                corrections_applied=(self.corrected_transcript != self.raw_transcript),
            )

    def begin_polishing(self) -> bool:
        """Transition TRANSCRIBING -> POLISHING (guarded to run at most once)."""
        with self._lock:
            if self._polished_once or self._state != SessionState.TRANSCRIBING:
                logging.debug(
                    f"[Session:{self._session_id[:8]}] Duplicate or invalid begin_polishing() ignored "
                    f"(state={self._state.value}, already_polished={self._polished_once})."
                )
                return False
            self._polished_once = True
            self._transition_to(SessionState.POLISHING)
            self.polish_started_at = time.time()
            self._log_event(SessionEvent.SESSION_POLISH_STARTED, style=self.style)
            return True

    def complete_polishing(
        self,
        polished_text: str,
        provider: Optional[str] = None,
        is_fallback: bool = False,
    ) -> None:
        """Record polish completion without logging sensitive text content."""
        with self._lock:
            if self._state != SessionState.POLISHING:
                raise RuntimeError(
                    f"complete_polishing called in state {self._state.value}"
                )
            self.polish_completed_at = time.time()
            self.polished_text = polished_text or ""
            self.provider_used = provider
            self.is_fallback = is_fallback
            char_len = len(self.polished_text)
            word_count = len(self.polished_text.split()) if self.polished_text.strip() else 0
            self._log_event(
                SessionEvent.SESSION_POLISH_COMPLETED,
                provider=provider or "none",
                fallback=is_fallback,
                char_len=char_len,
                word_count=word_count,
            )

    def begin_injection(self) -> bool:
        """Transition POLISHING -> INJECTING (guarded to run at most once)."""
        with self._lock:
            if self._injected_once or self._state != SessionState.POLISHING:
                logging.debug(
                    f"[Session:{self._session_id[:8]}] Duplicate or invalid begin_injection() ignored "
                    f"(state={self._state.value}, already_injected={self._injected_once})."
                )
                return False
            self._injected_once = True
            self._transition_to(SessionState.INJECTING)
            self.injection_started_at = time.time()
            self._log_event(SessionEvent.SESSION_INJECTION_STARTED)
            return True

    def complete_session(self, reason: str = "ok", **meta) -> bool:
        """Transition to DONE from FINALIZING, TRANSCRIBING, POLISHING, or INJECTING."""
        with self._lock:
            if self._state == SessionState.DONE:
                return False
            if self._state not in (
                SessionState.FINALIZING,
                SessionState.TRANSCRIBING,
                SessionState.POLISHING,
                SessionState.INJECTING,
            ):
                return False
            self._transition_to(SessionState.DONE)
            self.completed_at = time.time()
            if self.injection_started_at and not self.injection_completed_at:
                self.injection_completed_at = self.completed_at
            total_s = (
                round(self.completed_at - self.started_at, 2)
                if self.started_at
                else 0.0
            )
            self._log_event(
                SessionEvent.SESSION_COMPLETED,
                reason=reason,
                total_sec=total_s,
                **meta,
            )
            return True

    def cancel(self, reason: str = "user_cancelled") -> bool:
        """Cancel an active or finalizing session cleanly, releasing all audio resources."""
        with self._lock:
            if self.is_terminal:
                return False
            self._state = SessionState.CANCELLED
            self.completed_at = time.time()
            self.cleanup_audio()
            self._log_event(
                SessionEvent.SESSION_CANCELLED,
                reason=reason,
            )
            return True

    def fail_session(self, error_reason: str) -> bool:
        """Transition to ERROR from any non-terminal state."""
        with self._lock:
            if self.is_terminal:
                return False
            self.error = error_reason
            self._state = SessionState.ERROR
            self.completed_at = time.time()
            self.cleanup_audio()
            self._log_event(
                SessionEvent.SESSION_ERROR,
                reason=error_reason,
            )
            return True

    def get_metrics(self) -> SessionMetrics:
        """Calculate and return performance, duration, and memory metrics for this session."""
        with self._lock:
            t_tx = (
                round(self.transcription_completed_at - self.transcription_started_at, 3)
                if (self.transcription_started_at and self.transcription_completed_at)
                else 0.0
            )
            t_pl = (
                round(self.polish_completed_at - self.polish_started_at, 3)
                if (self.polish_started_at and self.polish_completed_at)
                else 0.0
            )
            t_inj = (
                round(self.injection_completed_at - self.injection_started_at, 3)
                if (self.injection_started_at and self.injection_completed_at)
                else 0.0
            )
            t_proc = round(t_tx + t_pl + t_inj, 3)

            # Estimate peak buffer memory allocated during session
            # 16-bit PCM @ 16kHz is 32,000 bytes/sec
            audio_bytes = self._total_audio_samples * 2
            peak_mb = round(audio_bytes / (1024 * 1024), 2)

            t_len = len(self.raw_transcript)
            t_words = len(self.raw_transcript.split()) if self.raw_transcript.strip() else 0
            f_len = len(self.final_text)
            f_words = len(self.final_text.split()) if self.final_text.strip() else 0

            return SessionMetrics(
                session_id=self._session_id,
                audio_duration_sec=self.audio_duration_sec,
                speech_duration_sec=max(0.0, round(self.audio_duration_sec - self._total_silence_sec, 3)),
                silence_duration_sec=round(self._total_silence_sec, 3),
                processing_time_sec=t_proc,
                transcription_time_sec=t_tx,
                polish_time_sec=t_pl,
                injection_time_sec=t_inj,
                peak_memory_bytes=audio_bytes,
                peak_memory_mb=peak_mb,
                transcript_length=t_len,
                transcript_word_count=t_words,
                final_text_length=f_len,
                final_text_word_count=f_words,
                pause_count=self._pause_count,
                speech_segment_count=self._speech_segment_count,
                provider_used=self.provider_used,
                is_fallback=self.is_fallback,
                cancelled=self.is_cancelled,
            )

    @property
    def metrics(self) -> SessionMetrics:
        return self.get_metrics()


class DictationSessionCoordinator:
    """High-level coordinator that executes a finalized DictationSession end-to-end.

    Sits ABOVE:
      - AudioRecorder (local microphone capture & VAD speech/silence activity)
      - AIBrain (faster-whisper local ASR + FreeLLMAPI -> Gemini -> Ollama -> raw fallback)
      - TextInjector (snippet expansion + keyboard/clipboard injection)
      - HistoryVault (SQLite history & telemetry)

    Enforces:
      - One session -> one transcription -> one polish -> one injection
      - Zero duplicate finalization or duplicate pipeline execution
      - Guaranteed audio cleanup and cancellation resilience
    """

    def __init__(
        self,
        recorder,
        brain,
        injector,
        vault=None,
        on_status_change: Optional[Callable[[str, Optional[str]], None]] = None,
        on_mode_change: Optional[Callable[[str, bool], None]] = None,
        on_history_entry: Optional[Callable[[str, str], None]] = None,
        on_telemetry_refresh: Optional[Callable[[], None]] = None,
    ):
        self.recorder = recorder
        self.brain = brain
        self.injector = injector
        self.vault = vault
        self.on_status_change = on_status_change
        self.on_mode_change = on_mode_change
        self.on_history_entry = on_history_entry
        self.on_telemetry_refresh = on_telemetry_refresh

        self._lock = threading.RLock()
        self._active_session: Optional[DictationSession] = None
        self._processed_session_ids: set[str] = set()
        self.last_injected_text: str = ""

    @property
    def active_session(self) -> Optional[DictationSession]:
        with self._lock:
            return self._active_session

    def _emit_status(self, status: str, hint: Optional[str] = None) -> None:
        if callable(self.on_status_change):
            try:
                self.on_status_change(status, hint)
            except Exception:
                pass

    def start_session(
        self,
        mode: SessionMode | str,
        context_info: Optional[dict] = None,
        style: str = "Normal",
        enable_vad_activity: bool = True,
    ) -> Optional[DictationSession]:
        """Create and start a new DictationSession, beginning local audio capture."""
        with self._lock:
            if self._active_session is not None and self._active_session.is_active_capture:
                logging.debug(
                    f"[Coordinator] Cannot start new session while session "
                    f"{self._active_session.session_id[:8]} is active."
                )
                return None

            session = DictationSession(
                mode=mode,
                context_info=context_info,
                style=style,
            )
            session.start()
            self._active_session = session

        # Wire VAD speech/silence activity callbacks so silence pauses activity state
        # without ever stopping the recorder or finalizing the session in continuous mode.
        def _on_speech():
            session.on_speech_detected()
            if session.mode == SessionMode.CONTINUOUS and session.is_active_capture:
                self._emit_status(
                    "recording",
                    "Continuous session: Listening... (Ctrl+Shift+A to finish)",
                )

        def _on_silence(silence_dur: float):
            if session.on_silence_detected(silence_dur):
                if session.mode == SessionMode.CONTINUOUS and session.is_active_capture:
                    self._emit_status(
                        "paused",
                        "Paused (thinking)... Speak to continue, or Ctrl+Shift+A to finish",
                    )

        try:
            start_kwargs = {}
            if enable_vad_activity:
                start_kwargs["on_speech_callback"] = _on_speech
                start_kwargs["on_silence_callback"] = _on_silence
                start_kwargs["chunk_callback"] = session.append_audio_chunk
            else:
                start_kwargs["chunk_callback"] = session.append_audio_chunk

            try:
                self.recorder.start(**start_kwargs)
            except TypeError:
                # Fallback if custom/mock recorder only accepts basic signature
                self.recorder.start()

            if not getattr(self.recorder, "is_recording", True):
                session.fail_session("Audio stream failed to initialize")
                with self._lock:
                    if self._active_session is session:
                        self._active_session = None
                return None
        except Exception as e:
            session.fail_session(f"Recorder start error: {e}")
            with self._lock:
                if self._active_session is session:
                    self._active_session = None
            raise

        return session

    def finalize_session(
        self,
        session: Optional[DictationSession] = None,
        updated_context: Optional[dict] = None,
    ) -> Optional[DictationSession]:
        """Explicitly stop audio capture and transition the session to FINALIZING.

        Idempotent: if the session was already finalized, returns None.
        """
        with self._lock:
            target = session or self._active_session
            if target is None:
                return None
            if target._finalized_once or not target.is_active_capture:
                return None

            # Stop recorder once for this session
            audio_path = None
            try:
                audio_path = self.recorder.stop()
            except Exception as e:
                logging.error(f"[Coordinator] Error stopping recorder: {e}")

            if updated_context:
                target.refresh_context_snapshot(updated_context=updated_context)

            ok = target.finalize(audio_path=audio_path)
            if self._active_session is target:
                self._active_session = None

            if not ok:
                return None
            return target

    def cancel_session(
        self,
        session: Optional[DictationSession] = None,
        reason: str = "cancelled",
    ) -> bool:
        """Cancel an ongoing or finalizing session immediately, aborting recording."""
        with self._lock:
            target = session or self._active_session
            if target is None:
                return False

            if hasattr(self.recorder, "cancel"):
                try:
                    self.recorder.cancel()
                except Exception:
                    pass
            elif hasattr(self.recorder, "stop"):
                try:
                    self.recorder.stop()
                except Exception:
                    pass

            ok = target.cancel(reason=reason)
            if self._active_session is target:
                self._active_session = None

            self._emit_status("ready", "Dictation cancelled.")
            return ok

    def process_session(
        self,
        session: DictationSession,
        cleanup_audio: bool = True,
    ) -> bool:
        """Execute transcription -> polish -> injection for a finalized DictationSession.

        Strictly idempotent: each stage is guarded by both `_processed_session_ids`
        and the session's internal state machine + per-stage one-shot flags.
        """
        if session is None:
            return False

        with self._lock:
            if session.is_cancelled:
                return False
            if session.session_id in self._processed_session_ids:
                logging.debug(
                    f"[Coordinator] Session {session.session_id[:8]} already processed; skipping duplicate call."
                )
                return False
            if session.state != SessionState.FINALIZING:
                logging.debug(
                    f"[Coordinator] Session {session.session_id[:8]} not in FINALIZING state "
                    f"(state={session.state.value}); skipping."
                )
                return False
            self._processed_session_ids.add(session.session_id)

        audio_path = session.audio_path
        snapshot = session.refresh_context_snapshot()
        context = session.context_info
        in_memory_audio: Optional[np.ndarray] = None

        try:
            # Handle disk failure: if audio_path is None but in-memory chunks exist, use memory
            if not audio_path:
                in_memory_audio = session.get_concatenated_audio()
                if in_memory_audio is None or len(in_memory_audio) < 4800: # < 0.3s
                    session.complete_session(reason="no_audio")
                    self._emit_status("ready")
                    return False

            # Check for cancellation before transcription
            if session.is_cancelled:
                return False

            # 1. TRANSCRIBING (strictly once)
            if not session.begin_transcription():
                return False

            self._emit_status("transcribing", "Transcribing speech locally...")

            logging.info(
                f"[GUI] Active context: {snapshot.app_name} "
                f"(category={snapshot.app_category.value}, coding_mode={snapshot.coding_mode})"
            )

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

            # Check for cancellation after transcription
            if session.is_cancelled:
                return False

            session.complete_transcription(raw_text)

            # Free in-memory audio buffers immediately to bound RAM
            session.release_audio_buffers()
            in_memory_audio = None

            if not raw_text or not raw_text.strip():
                session.complete_session(reason="no_speech_detected")
                self._emit_status("ready", "No speech detected.")
                return False

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
                    self._emit_status("ready", cmd_res.message)
                else:
                    self._emit_status("ready", cmd_res.message or "Voice command skipped")
                return True

            # Check dictionary meta-command ("add <word> to my dictionary")
            from ai_brain import detect_editing_command
            command, _remainder = detect_editing_command(raw_text)
            if command and command.startswith("dict_add_"):
                word_to_add = command[len("dict_add_"):]
                self.brain._add_to_dictionary(word_to_add)
                session.complete_session(reason="dictionary_command")
                self._emit_status("ready", f"Learned: '{word_to_add}' added to memory!")
                return True

            # Check for cancellation before polishing
            if session.is_cancelled:
                return False

            # 2. POLISHING (strictly once)
            # Priority: FreeLLMAPI -> Gemini -> Local LLM -> Raw transcript
            if not session.begin_polishing():
                return False

            self._emit_status("processing", "Polishing transcription...")
            snapshot = session.refresh_context_snapshot()
            pre_text = snapshot.bounded_cursor_text

            try:
                try:
                    pipeline_res = self.brain.polish_with_provider_fallbacks(
                        raw_text=raw_text,
                        style=session.style,
                        context_info=context,
                        pre_text=pre_text,
                        context_snapshot=snapshot,
                    )
                except TypeError:
                    pipeline_res = self.brain.polish_with_provider_fallbacks(
                        raw_text=raw_text,
                        style=session.style,
                        context_info=context,
                        pre_text=pre_text,
                    )
                polished_text = pipeline_res.text or raw_text
                provider_used = pipeline_res.provider
                is_fallback = pipeline_res.is_fallback
            except Exception as llm_err:
                # Resilience: Never lose the user's words even on unexpected LLM/provider error
                logging.warning(
                    f"[Coordinator] LLM polishing failed ({llm_err}); falling back to spoken-corrected transcript."
                )
                polished_text = session.corrected_transcript or raw_text
                provider_used = "raw_fallback"
                is_fallback = True

            # Check for cancellation after polishing
            if session.is_cancelled:
                return False

            session.complete_polishing(
                polished_text=polished_text,
                provider=provider_used,
                is_fallback=is_fallback,
            )

            if callable(self.on_mode_change):
                try:
                    self.on_mode_change(provider_used, is_fallback)
                except Exception:
                    pass

            polished_expanded = self.injector.expand_snippets(polished_text)
            normalized_polished = polished_expanded.strip().replace("\r\n", "\n")

            # Terminal Safety Guard (via ContextSnapshot category or app_hint fallback)
            app_hint = context.get("app_hint", "") if context else ""
            if snapshot.single_line_output or any(
                term in app_hint.lower()
                for term in ["terminal", "cmd", "powershell", "bash", "wsl"]
            ):
                normalized_polished = normalized_polished.replace("\n", " ").strip()

            session.polished_text = normalized_polished

            # 3. INJECTING (strictly once)
            # Before injection verify:
            # 1. session exists
            # 2. session has valid final text
            # 3. session is not already injected
            # 4. session is in an injectable state
            if not session.is_injectable or not session.begin_injection():
                logging.debug(
                    f"[Coordinator] Session {session.session_id[:8]} cannot inject: "
                    f"is_injectable={session.is_injectable}, state={session.state.value}."
                )
                return False

            self._emit_status("typing", "Typing polished text...")
            target_hwnd = context.get("target_hwnd") or context.get("hwnd")
            inject_res = self.injector.inject(normalized_polished, target_hwnd=target_hwnd)

            if inject_res.success:
                session.mark_injected(True)
                self.last_injected_text = inject_res.injected_text
                GLOBAL_INSERTION_HISTORY.record_insertion(
                    session_id=session.session_id,
                    text=normalized_polished,
                    target_hwnd=target_hwnd,
                )
            else:
                logging.warning(f"[GUI] Text injection skipped or failed: {inject_res.error}")
                self._emit_status("warning", f"Injection: {inject_res.error}")

            # 4. Log to HistoryVault & update UI
            if self.vault is not None:
                try:
                    ts = self.vault.add_entry(normalized_polished, raw_text)
                    if callable(self.on_history_entry):
                        self.on_history_entry(ts, normalized_polished)
                except Exception as vault_err:
                    logging.error(f"[Coordinator] Vault logging failed: {vault_err}")

            if callable(self.on_telemetry_refresh):
                try:
                    self.on_telemetry_refresh()
                except Exception:
                    pass

            session.complete_session(
                reason="ok",
                injected=bool(inject_res.success),
                method=getattr(inject_res, "method", "unknown"),
            )
            self._emit_status("ready")
            return bool(inject_res.success)

        except Exception as e:
            logging.error(f"[Coordinator] Session {session.session_id[:8]} error: {e}")
            session.fail_session(str(e))
            raise
        finally:
            if cleanup_audio:
                session.cleanup_audio()

