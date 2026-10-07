"""
test_ui_state_and_race_conditions.py -- Tests verifying UI state reflection,
'PAUSED != ENDED' semantics, race condition prevention, and session state as source of truth.
"""

import os
import sys
import tempfile
import threading
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
    SessionMode,
    SessionState,
)
from ai_brain import AIBrain, PipelineResult


class MockRecorder:
    def __init__(self):
        self.is_recording = False
        self.chunk_callback = None
        self.on_speech_callback = None
        self.on_silence_callback = None
        self.chunks = []
        self.start_calls = 0
        self.stop_calls = 0
        self.current_rms = 450.0

    def start(self, on_speech_callback=None, on_silence_callback=None, chunk_callback=None, auto_stop_callback=None):
        self.is_recording = True
        self.start_calls += 1
        self.on_speech_callback = on_speech_callback
        self.on_silence_callback = on_silence_callback
        self.chunk_callback = chunk_callback

    def stop(self) -> str | None:
        self.is_recording = False
        self.stop_calls += 1
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        return tmp.name

    def emit_chunk(self, duration_sec: float = 0.5):
        samples = int(duration_sec * 16000)
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


class MockInjector:
    def __init__(self):
        self.injected = []
        self.delay = 0.0

    def expand_snippets(self, text: str) -> str:
        return text

    def inject(self, text: str, target_hwnd=None):
        self.injected.append((text, target_hwnd))
        res = MagicMock()
        res.success = True
        res.injected_text = text
        res.method = "keyboard"
        res.error = None
        return res


class TestUIStateAndRaceConditions(unittest.TestCase):
    """Test suite for UI state semantics and race-condition immunity."""

    def setUp(self):
        self.recorder = MockRecorder()
        self.brain = MagicMock(spec=AIBrain)
        self.injector = MockInjector()
        self.status_history = []

        def on_status_change(status: str, hint: str | None = None):
            self.status_history.append((status, hint))

        self.coordinator = DictationSessionCoordinator(
            recorder=self.recorder,
            brain=self.brain,
            injector=self.injector,
            on_status_change=on_status_change,
        )

    # ------------------------------------------------------------------
    # 1. State Semantics: START -> RECORDING -> SILENCE -> PAUSED -> SPEECH -> RECORDING -> STOP -> DONE
    # ------------------------------------------------------------------
    def test_continuous_mode_lifecycle_one_single_session(self):
        """Continuous mode transitions through RECORDING <-> PAUSED without ending the session."""
        # 1. START
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.assertIsNotNone(session)
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertEqual(session.mode, SessionMode.CONTINUOUS)
        session_id_1 = session.session_id

        # 2. SPEECH
        self.recorder.emit_chunk(2.0)
        self.recorder.trigger_speech()
        self.assertEqual(session.state, SessionState.RECORDING)

        # 3. SILENCE / THINKING -> PAUSED (NOT ended)
        self.recorder.trigger_silence(4.0)
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(session.is_active_capture)
        self.assertFalse(session.is_terminal)

        # Verify last emitted status is 'paused' with non-intrusive message
        last_status, last_hint = self.status_history[-1]
        self.assertEqual(last_status, "paused")
        self.assertIn("Paused", last_hint)

        # 4. USER SPEAKS AGAIN -> RECORDING (Resumed seamlessly)
        self.recorder.trigger_speech()
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertEqual(session.session_id, session_id_1)  # EXACT SAME SESSION

        last_status, last_hint = self.status_history[-1]
        self.assertEqual(last_status, "recording")

        # 5. USER EXPLICITLY STOPS -> FINALIZING -> PROCESSING
        finalized_session = self.coordinator.finalize_session(session)
        self.assertIs(finalized_session, session)
        self.assertEqual(session.state, SessionState.FINALIZING)

        self.brain._offline_transcribe.return_value = "Continuous flow text with thought pauses."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Continuous flow text with thought pauses.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.speech_segment_count, 2)
        self.assertEqual(session.pause_count, 1)

    # ------------------------------------------------------------------
    # 2. Double-Start & Rapid Hotkey Mashing Prevention
    # ------------------------------------------------------------------
    def test_double_start_is_prevented(self):
        """Calling start_session while a session is already capturing returns None."""
        s1 = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.assertIsNotNone(s1)
        self.assertEqual(self.recorder.start_calls, 1)

        # Second start attempt while s1 is active
        s2 = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.assertIsNone(s2)
        self.assertEqual(self.recorder.start_calls, 1)

    def test_rapid_start_and_finalize_concurrency(self):
        """Rapid concurrent start and finalize calls do not create zombie sessions."""
        results = []

        def worker_start():
            res = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
            if res:
                results.append(res)

        threads = [threading.Thread(target=worker_start) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly ONE session must have succeeded in starting
        self.assertEqual(len(results), 1)
        active = self.coordinator.active_session
        self.assertIsNotNone(active)

        # Finalize once
        f1 = self.coordinator.finalize_session(active)
        self.assertIsNotNone(f1)
        self.assertEqual(self.recorder.stop_calls, 1)

        # Immediate duplicate finalize must be safely ignored
        f2 = self.coordinator.finalize_session(active)
        self.assertIsNone(f2)
        self.assertEqual(self.recorder.stop_calls, 1)

    # ------------------------------------------------------------------
    # 3. Processing Hotkey Collision Immunity
    # ------------------------------------------------------------------
    def test_hotkey_during_processing_is_safely_ignored(self):
        """When coordinator or GUI is processing a session, duplicate process_session calls are ignored."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(session)

        self.brain._offline_transcribe.return_value = "Hello world."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Hello world.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        # First process call succeeds
        ok1 = self.coordinator.process_session(session)
        self.assertTrue(ok1)

        # Second process call for same session is rejected idempotently
        ok2 = self.coordinator.process_session(session)
        self.assertFalse(ok2)

    # ------------------------------------------------------------------
    # 4. Push-to-Talk vs Continuous Mode Separation
    # ------------------------------------------------------------------
    def test_push_to_talk_and_continuous_mode_separation(self):
        """Push-to-talk sessions record until released; continuous sessions record across pauses."""
        # 1. Push to talk
        ptt_session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.assertEqual(ptt_session.mode, SessionMode.PUSH_TO_TALK)
        self.coordinator.finalize_session(ptt_session)
        self.assertEqual(ptt_session.state, SessionState.FINALIZING)

        self.brain._offline_transcribe.return_value = "Push to talk."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Push to talk.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )
        self.coordinator.process_session(ptt_session)
        self.assertEqual(ptt_session.state, SessionState.DONE)

        # 2. Continuous session starts cleanly after ptt is finished
        cont_session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.assertEqual(cont_session.mode, SessionMode.CONTINUOUS)
        self.assertNotEqual(cont_session.session_id, ptt_session.session_id)
        self.coordinator.finalize_session(cont_session)

    # ------------------------------------------------------------------
    # 5. Duplicate Injection & Concurrency Scenarios
    # ------------------------------------------------------------------
    def test_duplicate_injection_prevention_same_session(self):
        """A single session is injected exactly once, never twice (preventing 'text text' bugs)."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(session)

        self.brain._offline_transcribe.return_value = "Deploy the Kubernetes cluster."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Deploy the Kubernetes cluster.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )

        # Concurrently trigger process_session from 5 worker threads
        results = []
        def worker():
            res = self.coordinator.process_session(session)
            results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 thread must have succeeded in processing/injecting
        self.assertEqual(results.count(True), 1)
        self.assertEqual(len(self.injector.injected), 1)
        self.assertEqual(self.injector.injected[0][0], "Deploy the Kubernetes cluster.")
        self.assertTrue(session.is_injected)

    def test_provider_retry_does_not_duplicate_injection(self):
        """When LLM provider falls back (FreeLLMAPI -> Gemini -> Ollama), exactly one injection happens."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(session)

        self.brain._offline_transcribe.return_value = "Refactor database migrations."
        # Simulate FreeLLMAPI failure, Gemini fallback in brain
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Refactor database migrations.",
            provider="gemini",
            is_fallback=True,
            previous_providers=["freellmapi"],
        )

        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(len(self.injector.injected), 1)
        self.assertEqual(self.injector.injected[0][0], "Refactor database migrations.")
        self.assertTrue(session.is_injected)

    def test_session_cancellation_aborts_injection(self):
        """Cancelled sessions are immediately aborted and never inject text."""
        session = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(session)

        # Cancel the session before processing
        session.cancel(reason="user_esc_pressed")
        self.assertTrue(session.is_cancelled)

        ok = self.coordinator.process_session(session)
        self.assertFalse(ok)
        self.assertEqual(len(self.injector.injected), 0)
        self.assertFalse(session.is_injected)

    def test_two_independent_sessions_never_share_state(self):
        """Consecutive sessions have unique immutable session_ids and independent buffers."""
        # Session 1
        s1 = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(s1)

        self.brain._offline_transcribe.return_value = "First session text."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="First session text.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )
        self.coordinator.process_session(s1)

        # Session 2
        s2 = self.coordinator.start_session(mode=SessionMode.PUSH_TO_TALK)
        self.recorder.emit_chunk(1.0)
        self.coordinator.finalize_session(s2)

        self.brain._offline_transcribe.return_value = "Second session text."
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            text="Second session text.",
            provider="FreeLLMAPI",
            is_fallback=False,
        )
        self.coordinator.process_session(s2)

        self.assertNotEqual(s1.session_id, s2.session_id)
        self.assertEqual(len(self.injector.injected), 2)
        self.assertEqual(self.injector.injected[0][0], "First session text.")
        self.assertEqual(self.injector.injected[1][0], "Second session text.")


if __name__ == "__main__":
    unittest.main()

