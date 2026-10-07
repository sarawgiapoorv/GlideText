"""
tests/test_dictation_session.py
Comprehensive test suite for the DictationSession architecture in GlideText.

Verifies:
  A. speech -> 5 sec silence -> speech -> explicit stop => ONE session
  B. speech -> 10 sec silence -> speech -> explicit stop => ONE session
  C. speech -> explicit stop => ONE session
  D. continuous mode -> silence -> speech -> explicit stop => ONE session
  E. push-to-talk -> speech -> release => ONE independent session
  F. double-finalization => only one finalization
  G. double-injection => only one injection
  Plus:
  - VAD speech/silence activity transitions (RECORDING <-> PAUSED) without auto-stopping
  - Unique immutable session_id
  - Privacy-safe lifecycle logging (never logs raw or polished text content)
  - Audio preservation across silence/thinking intervals
  - GUI continuous toggle & _on_vad_auto_stop behavior
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from audio_recorder import AudioRecorder, VAD_MIN_SPEECH_DURATION, VAD_SILENCE_DURATION
from ai_brain import PipelineResult, ProviderAttempt
from text_injector import InjectionResult
from dictation_session import (
    DictationSession,
    DictationSessionCoordinator,
    SessionEvent,
    SessionMode,
    SessionState,
)


class MockRecorder:
    """Simulated AudioRecorder that supports VAD activity callbacks and audio frame accumulation."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self._auto_stop_callback = None
        self._on_speech_callback = None
        self._on_silence_callback = None
        self._chunk_callback = None
        self.stop_calls = 0
        self.start_calls = 0
        self._frames: list[np.ndarray] = []
        self._temp_files: list[str] = []

    def start(
        self,
        auto_stop_callback=None,
        on_speech_callback=None,
        on_silence_callback=None,
        chunk_callback=None,
    ):
        self.start_calls += 1
        self.is_recording = True
        self._auto_stop_callback = auto_stop_callback
        self._on_speech_callback = on_speech_callback
        self._on_silence_callback = on_silence_callback
        self._chunk_callback = chunk_callback
        self._frames = []

    def simulate_speech(self, duration_sec: float = 1.0):
        """Simulate user speaking for `duration_sec`."""
        if not self.is_recording:
            raise RuntimeError("Recorder is not recording!")
        samples = int(self.sample_rate * duration_sec)
        chunk = np.full((samples, 1), 1500, dtype=np.int16)
        self._frames.append(chunk)
        if self._chunk_callback:
            self._chunk_callback(chunk)
        if self._on_speech_callback:
            self._on_speech_callback()

    def simulate_silence(self, duration_sec: float = 5.0):
        """Simulate user pausing/thinking in silence for `duration_sec`."""
        if not self.is_recording:
            raise RuntimeError("Recorder is not recording!")
        samples = int(self.sample_rate * duration_sec)
        chunk = np.zeros((samples, 1), dtype=np.int16)
        self._frames.append(chunk)
        if self._chunk_callback:
            self._chunk_callback(chunk)
        if duration_sec >= VAD_SILENCE_DURATION:
            if self._on_silence_callback:
                self._on_silence_callback(duration_sec)
            if self._auto_stop_callback:
                self._auto_stop_callback(None)

    def stop(self) -> str | None:
        if not self.is_recording:
            return None
        self.stop_calls += 1
        self.is_recording = False
        if not self._frames:
            return None
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="test_session_")
        os.close(fd)
        self._temp_files.append(path)
        return path

    def cleanup(self):
        for p in self._temp_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


class TestDictationSessionArchitecture(unittest.TestCase):
    def setUp(self):
        self.recorder = MockRecorder()
        self.brain = MagicMock()
        self.brain._offline_transcribe = MagicMock(
            return_value="I have been thinking about redesigning auth and moving storage to Redis actually PostgreSQL"
        )
        self.brain.polish_with_provider_fallbacks = MagicMock(
            return_value=PipelineResult(
                success=True,
                text="I've been thinking about redesigning auth and moving storage to PostgreSQL.",
                raw_transcript="I have been thinking about redesigning auth and moving storage to Redis actually PostgreSQL",
                provider="freellmapi",
                fallback_used=False,
                attempts=[ProviderAttempt(provider="freellmapi", success=True, model="auto")],
            )
        )

        self.injector = MagicMock()
        self.injector.expand_snippets = MagicMock(side_effect=lambda t: t)
        self.injector.inject = MagicMock(
            side_effect=lambda t, target_hwnd=None: InjectionResult(
                success=True, method="keyboard", injected_text=t
            )
        )

        self.vault = MagicMock()
        self.vault.add_entry = MagicMock(return_value="2026-10-01 12:00:00")

        self.status_history: list[tuple[str, str | None]] = []
        self.coordinator = DictationSessionCoordinator(
            recorder=self.recorder,
            brain=self.brain,
            injector=self.injector,
            vault=self.vault,
            on_status_change=lambda s, h=None: self.status_history.append((s, h)),
        )

    def tearDown(self):
        self.recorder.cleanup()

    # ==================================================================
    # Scenario A: speech -> 5 sec silence -> speech -> explicit stop
    # Expected: ONE session
    # ==================================================================
    def test_scenario_a_speech_5s_silence_speech_explicit_stop(self):
        """A: speech -> 5 sec silence -> speech -> explicit stop produces ONE session."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.assertIsNotNone(session)
        session_id = session.session_id
        self.assertEqual(session.state, SessionState.RECORDING)

        # 1. User speaks
        self.recorder.simulate_speech(2.0)
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertTrue(self.recorder.is_recording)

        # 2. User pauses for 5 seconds while thinking
        self.recorder.simulate_silence(5.0)
        # Session must NOT finalize; it transitions to PAUSED while recorder stays open
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(self.recorder.is_recording)
        self.assertEqual(self.coordinator.active_session, session)
        self.brain._offline_transcribe.assert_not_called()
        self.injector.inject.assert_not_called()

        # 3. User resumes speaking in the SAME session
        self.recorder.simulate_speech(2.5)
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.speech_segment_count, 2)
        self.assertEqual(session.pause_count, 1)

        # Verify audio was preserved across the 5s silence
        concat_audio = session.get_concatenated_audio()
        self.assertIsNotNone(concat_audio)
        expected_samples = int(self.recorder.sample_rate * (2.0 + 5.0 + 2.5))
        self.assertEqual(len(concat_audio), expected_samples)

        # 4. User explicitly stops the session
        finalized = self.coordinator.finalize_session()
        self.assertIs(finalized, session)
        self.assertEqual(session.state, SessionState.FINALIZING)
        self.assertFalse(self.recorder.is_recording)

        # 5. Process the entire session once
        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.session_id, session_id)

        # Exactly ONE transcription, ONE polish, ONE injection
        self.brain._offline_transcribe.assert_called_once()
        self.brain.polish_with_provider_fallbacks.assert_called_once()
        self.injector.inject.assert_called_once()
        self.vault.add_entry.assert_called_once()

    # ==================================================================
    # Scenario B: speech -> 10 sec silence -> speech -> explicit stop
    # Expected: ONE session
    # ==================================================================
    def test_scenario_b_speech_10s_silence_speech_explicit_stop(self):
        """B: speech -> 10 sec silence -> speech -> explicit stop produces ONE session."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        session_id = session.session_id

        # Speech 1 -> 10s silence -> Speech 2 -> 6s silence -> Speech 3 -> explicit stop
        self.recorder.simulate_speech(1.5)
        self.recorder.simulate_silence(10.0)
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(self.recorder.is_recording)

        self.recorder.simulate_speech(2.0)
        self.assertEqual(session.state, SessionState.RECORDING)

        self.recorder.simulate_silence(6.0)
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(self.recorder.is_recording)

        self.recorder.simulate_speech(1.5)
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertEqual(session.speech_segment_count, 3)
        self.assertEqual(session.pause_count, 2)

        # Explicit user stop
        finalized = self.coordinator.finalize_session()
        self.assertIs(finalized, session)
        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.session_id, session_id)

        # Strictly ONE session processed
        self.brain._offline_transcribe.assert_called_once()
        self.brain.polish_with_provider_fallbacks.assert_called_once()
        self.injector.inject.assert_called_once()

    # ==================================================================
    # Scenario C: speech -> explicit stop
    # Expected: ONE session
    # ==================================================================
    def test_scenario_c_speech_explicit_stop(self):
        """C: speech -> explicit stop (no pauses) produces ONE session."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        session_id = session.session_id

        self.recorder.simulate_speech(3.0)
        self.assertEqual(session.state, SessionState.RECORDING)
        self.assertEqual(session.pause_count, 0)

        finalized = self.coordinator.finalize_session()
        self.assertIs(finalized, session)
        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.session_id, session_id)

        self.brain._offline_transcribe.assert_called_once()
        self.brain.polish_with_provider_fallbacks.assert_called_once()
        self.injector.inject.assert_called_once()

    # ==================================================================
    # Scenario D: continuous mode -> silence -> speech -> explicit stop
    # Expected: ONE session
    # ==================================================================
    def test_scenario_d_continuous_mode_initial_silence_then_speech_then_stop(self):
        """D: continuous mode -> initial silence -> speech -> explicit stop produces ONE session."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        session_id = session.session_id

        # User starts continuous mode, thinks silently for 8 seconds before speaking
        self.recorder.simulate_silence(8.0)
        self.assertTrue(self.recorder.is_recording)
        self.assertFalse(session.is_terminal)

        # User speaks
        self.recorder.simulate_speech(2.0)
        self.assertEqual(session.state, SessionState.RECORDING)

        # Explicit stop
        finalized = self.coordinator.finalize_session()
        self.assertIs(finalized, session)
        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)
        self.assertEqual(session.session_id, session_id)

        self.brain._offline_transcribe.assert_called_once()
        self.brain.polish_with_provider_fallbacks.assert_called_once()
        self.injector.inject.assert_called_once()

    # ==================================================================
    # Scenario E: push-to-talk -> speech -> release
    # Expected: ONE independent session
    # ==================================================================
    def test_scenario_e_push_to_talk_speech_release(self):
        """E: push-to-talk press -> speech -> release produces ONE independent session."""
        session = self.coordinator.start_session(
            mode=SessionMode.PUSH_TO_TALK,
            enable_vad_activity=False,
        )
        self.assertIsNotNone(session)
        self.assertEqual(session.mode, SessionMode.PUSH_TO_TALK)
        self.assertEqual(session.state, SessionState.RECORDING)

        self.recorder.simulate_speech(1.8)

        # Release key -> finalize and process immediately
        finalized = self.coordinator.finalize_session()
        self.assertIs(finalized, session)
        self.assertEqual(session.state, SessionState.FINALIZING)

        ok = self.coordinator.process_session(session)
        self.assertTrue(ok)
        self.assertEqual(session.state, SessionState.DONE)

        self.brain._offline_transcribe.assert_called_once()
        self.brain.polish_with_provider_fallbacks.assert_called_once()
        self.injector.inject.assert_called_once()

    # ==================================================================
    # Scenario F: double-finalization
    # Expected: only one finalization
    # ==================================================================
    def test_scenario_f_double_finalization_prevented(self):
        """F: calling finalize() or finalize_session() multiple times only finalizes once."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.recorder.simulate_speech(2.0)

        # First finalization succeeds
        first = self.coordinator.finalize_session(session)
        self.assertIs(first, session)
        self.assertEqual(session.state, SessionState.FINALIZING)
        self.assertEqual(self.recorder.stop_calls, 1)

        # Second finalization via coordinator or direct session.finalize() must be rejected
        second = self.coordinator.finalize_session(session)
        self.assertIsNone(second)
        direct_second = session.finalize(audio_path="another_file.wav")
        self.assertFalse(direct_second)
        self.assertEqual(self.recorder.stop_calls, 1)

        # Only one SESSION_FINALIZED event in lifecycle log
        finalized_events = [
            e for e in session.events if e.event == SessionEvent.SESSION_FINALIZED
        ]
        self.assertEqual(len(finalized_events), 1)

    # ==================================================================
    # Scenario G: double-injection (and duplicate transcribe/polish)
    # Expected: only one injection
    # ==================================================================
    def test_scenario_g_double_injection_and_duplicate_processing_prevented(self):
        """G: attempting to process or inject a session twice only executes once."""
        session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
        self.recorder.simulate_speech(2.0)
        self.coordinator.finalize_session(session)

        # First process_session call succeeds
        first_ok = self.coordinator.process_session(session)
        self.assertTrue(first_ok)
        self.assertEqual(self.injector.inject.call_count, 1)
        self.assertEqual(self.brain._offline_transcribe.call_count, 1)
        self.assertEqual(self.brain.polish_with_provider_fallbacks.call_count, 1)

        # Second process_session call on the same session is a no-op
        second_ok = self.coordinator.process_session(session)
        self.assertFalse(second_ok)
        self.assertEqual(self.injector.inject.call_count, 1)
        self.assertEqual(self.brain._offline_transcribe.call_count, 1)
        self.assertEqual(self.brain.polish_with_provider_fallbacks.call_count, 1)

        # Direct call to session.begin_injection() / begin_transcription() / begin_polishing() also returns False
        self.assertFalse(session.begin_transcription())
        self.assertFalse(session.begin_polishing())
        self.assertFalse(session.begin_injection())

        injection_events = [
            e for e in session.events if e.event == SessionEvent.SESSION_INJECTION_STARTED
        ]
        self.assertEqual(len(injection_events), 1)

    # ==================================================================
    # Additional verification: AudioRecorder VAD does not stop stream
    # ==================================================================
    def test_audio_recorder_vad_does_not_stop_recording_on_silence(self):
        """AudioRecorder._process_vad_frame_state must fire silence callback without stopping recording."""
        rec = AudioRecorder()
        rec._is_recording = True
        rec._vad_enabled = True

        speech_events = []
        silence_events = []
        auto_stop_events = []

        rec._on_speech_callback = lambda: speech_events.append(True)
        rec._on_silence_callback = lambda dur: silence_events.append(dur)
        rec._auto_stop_callback = lambda path: auto_stop_events.append(path)

        t0 = 1000.0
        # 1. Speech starts and continues for 1.0s (> VAD_MIN_SPEECH_DURATION)
        rec._process_vad_frame_state(True, t0)
        rec._process_vad_frame_state(True, t0 + 0.5)
        rec._process_vad_frame_state(True, t0 + 1.0)
        self.assertEqual(len(speech_events), 1)
        self.assertTrue(rec.is_recording)

        # 2. Silence for 2.5s (> VAD_SILENCE_DURATION)
        rec._process_vad_frame_state(False, t0 + 1.0 + 2.5)
        self.assertEqual(len(silence_events), 1)
        self.assertEqual(len(auto_stop_events), 1)
        self.assertIsNone(auto_stop_events[0])
        # CRITICAL: Recorder is STILL recording!
        self.assertTrue(rec.is_recording)

        # 3. Continued silence for 10s does not re-trigger pause repeatedly
        rec._process_vad_frame_state(False, t0 + 1.0 + 10.0)
        self.assertEqual(len(silence_events), 1)
        self.assertTrue(rec.is_recording)

        # 4. Speech resumes after 10s silence
        rec._process_vad_frame_state(True, t0 + 12.0)
        self.assertEqual(len(speech_events), 2)
        self.assertFalse(rec._is_in_silence_pause)
        self.assertTrue(rec.is_recording)

    # ==================================================================
    # Privacy & Lifecycle Event Logging Verification
    # ==================================================================
    def test_lifecycle_events_and_privacy(self):
        """Verify all lifecycle events are emitted in order and never leak dictated text."""
        secret_raw = "my ultra secret password is correct horse battery staple"
        secret_polished = "My ultra-secret password is correct horse battery staple."
        self.brain._offline_transcribe.return_value = secret_raw
        self.brain.polish_with_provider_fallbacks.return_value = PipelineResult(
            success=True,
            text=secret_polished,
            raw_transcript=secret_raw,
            provider="freellmapi",
            fallback_used=False,
        )

        captured_logs: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(self.format(record))

        handler = ListHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            session = self.coordinator.start_session(mode=SessionMode.CONTINUOUS)
            self.recorder.simulate_speech(1.0)
            self.recorder.simulate_silence(5.0)
            self.recorder.simulate_speech(1.0)
            self.coordinator.finalize_session(session)
            self.coordinator.process_session(session)
        finally:
            root_logger.removeHandler(handler)

        event_types = [e.event for e in session.events]
        expected_sequence = [
            SessionEvent.SESSION_STARTED,
            SessionEvent.SPEECH_STARTED,
            SessionEvent.SPEECH_PAUSED,
            SessionEvent.SPEECH_RESUMED,
            SessionEvent.SESSION_FINALIZED,
            SessionEvent.SESSION_TRANSCRIPTION_STARTED,
            SessionEvent.SESSION_TRANSCRIPTION_COMPLETED,
            SessionEvent.SESSION_POLISH_STARTED,
            SessionEvent.SESSION_POLISH_COMPLETED,
            SessionEvent.SESSION_INJECTION_STARTED,
            SessionEvent.SESSION_COMPLETED,
        ]
        self.assertEqual(event_types, expected_sequence)

        # Verify session_id is immutable
        with self.assertRaises(AttributeError):
            session.session_id = "new_id"  # type: ignore

        # Verify sensitive text is never present in any session lifecycle log line
        session_logs = [line for line in captured_logs if f"[Session:{session.session_id[:8]}]" in line]
        self.assertTrue(len(session_logs) >= len(expected_sequence))
        for line in session_logs:
            self.assertNotIn("correct horse battery staple", line)
            self.assertNotIn(secret_raw, line)
            self.assertNotIn(secret_polished, line)

    # ==================================================================
    # GUI-level _on_vad_auto_stop & _run_session_pipeline verification
    # ==================================================================
    def test_gui_on_vad_auto_stop_pauses_without_finalizing(self):
        """GlideTextApp._on_vad_auto_stop must transition session to PAUSED without ending continuous mode."""
        import queue
        import threading
        import gui_app

        dummy_app = MagicMock(spec=gui_app.GlideTextApp)
        dummy_app.is_continuous_mode = True
        dummy_app._pipeline_queue = queue.Queue()
        dummy_app._set_status = MagicMock()

        session = DictationSession(mode=SessionMode.CONTINUOUS)
        session.start()
        session.on_speech_detected()
        dummy_app._active_session = session

        # Invoke the actual GlideTextApp._on_vad_auto_stop method
        gui_app.GlideTextApp._on_vad_auto_stop(dummy_app, None)

        # Must remain in continuous mode, session must be PAUSED (not finalized), and pipeline queue empty
        self.assertTrue(dummy_app.is_continuous_mode)
        self.assertEqual(session.state, SessionState.PAUSED)
        self.assertTrue(dummy_app._pipeline_queue.empty())
        dummy_app._set_status.assert_called_once()
        self.assertEqual(dummy_app._set_status.call_args[0][0], "paused")


if __name__ == "__main__":
    unittest.main()
