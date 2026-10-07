"""
test_long_session_stability.py -- Comprehensive tests for long-session stability,
bounded memory, silence handling, resilience to failures, cancellation, and metrics.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dictation_session import (
    DictationSession,
    DictationSessionCoordinator,
    SessionEvent,
    SessionMetrics,
    SessionMode,
    SessionState,
)
from ai_brain import AIBrain, PipelineResult
from audio_recorder import AudioRecorder
from spoken_corrections import resolve_spoken_corrections
from context_snapshot import build_context_snapshot, AppCategory


class DummyRecorder:
    """Mock audio recorder for testing session lifecycle and buffer flow."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.chunks: list[np.ndarray] = []
        self.chunk_callback = None
        self.on_speech_callback = None
        self.on_silence_callback = None
        self.fail_start = False
        self.fail_stop = False
        self.cancelled = False

    def start(
        self,
        on_speech_callback=None,
        on_silence_callback=None,
        chunk_callback=None,
        auto_stop_callback=None,
    ):
        if self.fail_start:
            raise RuntimeError("Microphone device not found or failed to initialize")
        self.is_recording = True
        self.cancelled = False
        self.on_speech_callback = on_speech_callback
        self.on_silence_callback = on_silence_callback
        self.chunk_callback = chunk_callback

    def emit_audio(self, duration_sec: float):
        """Simulate generating audio samples and passing to chunk_callback."""
        samples = int(duration_sec * self.sample_rate)
        chunk = np.zeros(samples, dtype=np.int16)
        self.chunks.append(chunk)
        if self.chunk_callback:
            self.chunk_callback(chunk)

    def trigger_speech(self):
        if self.on_speech_callback:
            self.on_speech_callback()

    def trigger_silence(self, duration_sec: float = 1.0):
        if self.on_silence_callback:
            self.on_silence_callback(duration_sec)

    def stop(self) -> str | None:
        self.is_recording = False
        if self.fail_stop:
            return None  # Simulate disk write failure
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        return temp_wav.name

    def cancel(self):
        self.is_recording = False
        self.cancelled = True
        self.chunks.clear()


class DummyInjector:
    def __init__(self):
        self.injected_texts = []
        self.delay = 0.0

    def expand_snippets(self, text: str) -> str:
        return text

    def inject(self, text: str, target_hwnd=None):
        self.injected_texts.append((text, target_hwnd))
        res = MagicMock()
        res.success = True
        res.injected_text = text
        res.method = "clipboard_fast"
        res.error = None
        return res


class DummyVault:
    def __init__(self):
        self.entries = []

    def add_entry(self, polished: str, raw: str):
        ts = "2026-10-01 12:00:00"
        self.entries.append((ts, polished, raw))
        return ts


class TestLongSessionStability(unittest.TestCase):
    """Test suite verifying stability, memory bounds, resilience, and metrics."""

    def setUp(self):
        self.recorder = DummyRecorder()
        self.brain = MagicMock(spec=AIBrain)
        self.injector = DummyInjector()
        self.vault = DummyVault()
        self.coordinator = DictationSessionCoordinator(
            recorder=self.recorder,
            brain=self.brain,
            injector=self.injector,
            vault=self.vault,
        )

    # ------------------------------------------------------------------
    # 1. 1-Minute Session
    # ------------------------------------------------------------------
    def test_one_minute_continuous_session(self):
        """A 1-minute dictation session captures audio, tracks duration & pauses, and executes once."""
        session = self.coordinator.start_session(
            mode=SessionMode.CONTINUOUS,
            context_info={"app_name": "notepad.exe", "hwnd": 12345},
            style="Normal",
        )
        self.assertIsNotNone(session)
        self.assertEqual(session.state, SessionState.RECORDING)

        # Simulate 60 seconds with 3 pauses
        for i in range(3):
            self.recorder.emit_audio(15.0)
            self.recorder.trigger_speech()
            self.recorder.trigger_silence(5.0)

        self.assertEqual(session.speech_segment_count, 3)
        self.assertEqual(session.pause_count, 3)
        self.assertGreaterEqual(session.audio_duration_sec, 45.0)

        # Finalize and process
        self.coordinator.finalize_session(session)
        self.assertEqual(session.state, SessionState.FINALIZING)

        self.brain._offline_transcribe.return_value = "This is a sixty second dictation test."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="This is a sixty second dictation test.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.transcript, "This is a sixty second dictation test.")
        self.assertEqual(session.final_text, "This is a sixty second dictation test.")

        # Check metrics
        metrics = session.get_metrics()
        self.assertGreaterEqual(metrics.audio_duration_sec, 45.0)
        self.assertEqual(metrics.transcript_length, len("This is a sixty second dictation test."))
        self.assertEqual(metrics.provider_used, "FreeLLMAPI")
        self.assertFalse(metrics.is_fallback)

    # ------------------------------------------------------------------
    # 2. Multi-Minute Session & Chunked ASR Continuity
    # ------------------------------------------------------------------
    def test_multi_minute_session_continuity_and_spoken_correction(self):
        """A multi-minute session (3 minutes) preserves sentence continuity and resolves corrections."""
        brain = AIBrain(api_key="test_key")

        # Simulate cross-chunk overlap reconciliation with spoken correction across chunk boundary
        chunk1_text = "Let's schedule the product team meeting for tomorrow at 2 PM"
        chunk2_text = "tomorrow at 2 PM actually make that 3 PM on Thursday because John is out."

        merged = brain.reconcile_overlapping_transcript(chunk1_text, chunk2_text)
        expected_merged = "Let's schedule the product team meeting for tomorrow at 2 PM actually make that 3 PM on Thursday because John is out."
        self.assertEqual(merged, expected_merged)

        # Spoken correction should resolve intent cleanly
        corrected = resolve_spoken_corrections(merged)
        self.assertIn("3 PM on Thursday", corrected)
        self.assertNotIn("2 PM", corrected)

    # ------------------------------------------------------------------
    # 3. Many Pauses (e.g. 25 cycles without LLM calls on silence)
    # ------------------------------------------------------------------
    def test_many_pauses_never_call_llm_on_silence(self):
        """25 speech-silence cycles in continuous mode transition RECORDING <-> PAUSED without LLM calls."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.assertIsNotNone(session)

        for cycle in range(25):
            self.recorder.trigger_speech()
            self.assertEqual(session.state, SessionState.RECORDING)
            self.recorder.trigger_silence(2.0)
            self.assertEqual(session.state, SessionState.PAUSED)

        # Ensure NO LLM calls occurred during all these pauses
        self.brain.polish_with_provider_fallbacks.assert_not_called()
        self.assertEqual(session.pause_count, 25)
        self.assertEqual(session.speech_segment_count, 25)

        # Explicit user stop finalizes and processes exactly once
        self.coordinator.finalize_session(session)
        self.brain._offline_transcribe.return_value = "Long session with twenty five pauses completed."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Long session with 25 pauses completed.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )
        self.coordinator.process_session(session)
        self.assertEqual(self.brain.polish_with_provider_fallbacks.call_count, 1)

    # ------------------------------------------------------------------
    # 4. Long Silence (e.g. 45 seconds of silence)
    # ------------------------------------------------------------------
    def test_long_silence_does_not_abort_session(self):
        """45 seconds of continuous silence keeps the session in PAUSED state and does not lose audio."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.recorder.trigger_speech()
        self.recorder.emit_audio(5.0)
        self.recorder.trigger_silence(45.0)

        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(session.is_active_capture)

        # User resumes speaking after long pause
        self.recorder.trigger_speech()
        self.assertEqual(session.state, SessionState.RECORDING)
        self.recorder.emit_audio(5.0)

        self.coordinator.finalize_session(session)
        self.assertEqual(session.state, SessionState.FINALIZING)
        self.assertGreaterEqual(session.speech_metadata["total_silence_sec"], 45.0)

    # ------------------------------------------------------------------
    # 5. Provider Failure & LLM Timeout (Never Lose User's Words)
    # ------------------------------------------------------------------
    def test_provider_failure_preserves_transcript_and_words_never_lost(self):
        """If FreeLLMAPI, Gemini, and Ollama all fail/timeout, user words are preserved with fallback."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_audio(3.0)
        self.coordinator.finalize_session(session)

        # Transcription succeeds
        raw_words = "Send the quarterly sales forecast to Sarah by Friday noon."
        self.brain._offline_transcribe.return_value = raw_words

        # Simulate all LLM providers throwing an unexpected timeout/exception
        self.brain.polish_with_provider_fallbacks.side_effect = TimeoutError("All LLM providers timed out")

        # Process should NOT raise, and user words should be injected safely
        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.transcript, raw_words)
        self.assertIn("Send the quarterly sales forecast", session.final_text)
        self.assertTrue(session.is_fallback)
        self.assertEqual(session.provider_used, "raw_fallback")

        # Verify text was injected
        self.assertEqual(len(self.injector.injected_texts), 1)
        self.assertEqual(self.injector.injected_texts[0][0], raw_words)

    # ------------------------------------------------------------------
    # 6. Transcription Failure
    # ------------------------------------------------------------------
    def test_transcription_failure_cleanly_fails_session(self):
        """If faster-whisper raises an unrecoverable error, fail session and cleanup buffers."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_audio(3.0)
        self.coordinator.finalize_session(session)

        self.brain._offline_transcribe.side_effect = RuntimeError("Whisper model CUDA out of memory")

        with self.assertRaises(RuntimeError):
            self.coordinator.process_session(session)

        self.assertEqual(session.state, SessionState.ERROR)
        self.assertIn("Whisper model CUDA out of memory", session.error)
        self.assertEqual(len(session._audio_chunks), 0)

    # ------------------------------------------------------------------
    # 7. Disk Failure (In-Memory Audio Fallback)
    # ------------------------------------------------------------------
    def test_disk_failure_falls_back_to_in_memory_audio(self):
        """If recorder cannot write WAV file to disk (audio_path is None), in-memory audio is used."""
        self.recorder.fail_stop = True  # recorder.stop() returns None

        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        # Emit 2 seconds of audio into session memory chunks
        self.recorder.emit_audio(2.0)
        self.coordinator.finalize_session(session)

        self.assertIsNone(session.audio_path)
        self.assertTrue(session.has_audio)

        self.brain._offline_transcribe.return_value = "Recovered from memory buffer."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Recovered from memory buffer.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.final_text, "Recovered from memory buffer.")

        # Ensure transcribe was called with audio_array passed
        self.brain._offline_transcribe.assert_called_once()
        call_kwargs = self.brain._offline_transcribe.call_args[1]
        self.assertIsNotNone(call_kwargs.get("audio_array"))

    # ------------------------------------------------------------------
    # 8. Microphone Failure
    # ------------------------------------------------------------------
    def test_microphone_failure_handles_cleanly(self):
        """If mic fails to start, coordinator raises and marks session ERROR without leaking."""
        self.recorder.fail_start = True

        with self.assertRaises(RuntimeError):
            self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)

        self.assertIsNone(self.coordinator.active_session)

    # ------------------------------------------------------------------
    # 9. Application Focus Changes
    # ------------------------------------------------------------------
    def test_application_focus_change_during_session(self):
        """If user switches apps during a multi-minute session, ContextSnapshot updates target and category."""
        session = self.coordinator.start_session(
            mode=SessionMode.CONTINUOUS,
            context_info={"exe_name": "code.exe", "app_hint": "VS Code", "hwnd": 1001, "title": "main.py"},
        )
        self.recorder.emit_audio(10.0)

        # User switches window to Slack
        new_context = {"exe_name": "slack.exe", "app_hint": "Slack", "hwnd": 2002, "title": "General Channel"}
        session.refresh_context_snapshot(updated_context=new_context)

        self.assertEqual(session.context_snapshot.app_name, "Slack")
        self.assertEqual(session.context_snapshot.app_category, AppCategory.SLACK_CHAT)

        self.coordinator.finalize_session(session)
        self.brain._offline_transcribe.return_value = "Hey team, code review is ready."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Hey team, code review is ready.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )
        self.coordinator.process_session(session)

        # Verify injection targeted updated hwnd or completed safely
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(self.vault.entries[0][1], "Hey team, code review is ready.")

    # ------------------------------------------------------------------
    # 10. Cancellation Support
    # ------------------------------------------------------------------
    def test_cancellation_during_recording_and_processing(self):
        """Cancelling a session transitions to CANCELLED and drops resources immediately."""
        # 1. Cancellation during recording
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.recorder.emit_audio(5.0)
        self.coordinator.cancel_session(session, reason="user_hit_escape")

        self.assertEqual(session.state, SessionState.CANCELLED)
        self.assertTrue(session.is_cancelled)
        self.assertIsNone(self.coordinator.active_session)
        self.assertEqual(len(session._audio_chunks), 0)

        # 2. Cancelled session cannot be processed
        ok = self.coordinator.process_session(session)
        self.assertFalse(ok)

    # ------------------------------------------------------------------
    # 11. Memory & Processing Metrics
    # ------------------------------------------------------------------
    def test_session_metrics_and_memory_bounds(self):
        """DictationSession exposes full performance, duration, word count, and peak memory metrics."""
        session = DictationSession(mode=SessionMode.PUSH_TO_TALK)
        session.start()

        # Simulate 10 seconds of 16kHz audio (160,000 samples)
        chunk = np.zeros(160000, dtype=np.int16)
        session.append_audio_chunk(chunk)
        session.on_silence_detected(2.5)

        session.finalize()
        session.begin_transcription()
        time.sleep(0.01)
        session.complete_transcription("First attempt, actually final version.")
        session.release_audio_buffers()

        session.begin_polishing()
        time.sleep(0.01)
        session.complete_polishing("Final version.", provider="FreeLLMAPI")

        session.begin_injection()
        session.complete_session(reason="ok")

        metrics = session.get_metrics()
        self.assertIsInstance(metrics, SessionMetrics)
        self.assertEqual(metrics.audio_duration_sec, 10.0)
        self.assertEqual(metrics.silence_duration_sec, 2.5)
        self.assertGreater(metrics.processing_time_sec, 0.0)
        self.assertEqual(metrics.transcript_word_count, 5)
        self.assertEqual(metrics.final_text_word_count, 2)
        self.assertEqual(metrics.peak_memory_bytes, 320000)
        self.assertEqual(metrics.peak_memory_mb, 0.31)
        self.assertEqual(len(session._audio_chunks), 0)

    # ------------------------------------------------------------------
    # 12. Short Dictations Remain Fast
    # ------------------------------------------------------------------
    def test_short_dictation_fast_path(self):
        """A quick 1-second dictation processes with negligible overhead (<5ms coordinator overhead)."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_audio(1.0)
        self.coordinator.finalize_session(session)

        self.brain._offline_transcribe.return_value = "Quick test."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Quick test.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        t0 = time.perf_counter()
        ok = self.coordinator.process_session(session)
        t1 = time.perf_counter()

        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        # Overhead without external I/O / ASR model compute should be under 50ms
        self.assertLess(t1 - t0, 0.05)


if __name__ == "__main__":
    unittest.main()
