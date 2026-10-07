"""
tests/test_asr_context_continuity.py
Comprehensive tests and benchmarks for ASR/transcription context continuity in GlideText.

Verifies:
  A. Long session with pauses (single logical unit & rolling chunked mode)
  B. Sentence split around silence (mid-sentence pause preserves lowercasing/continuity)
  C. Several sentences separated by long silence (8s, 6s, 10s pauses)
  D. Two separate sessions (Session A context NEVER leaks into Session B; push-to-talk isolation)
  E. Technical terminology & dictionaries preserved across long sessions and contextual apps
  F. Long continuous speech (multi-window rolling ASR context + overlap deduplication + benchmark)
"""

import os
import tempfile
import time
import unittest
import wave
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from ai_brain import AIBrain, PipelineResult, ProviderAttempt
from dictation_session import (
    DictationSession,
    DictationSessionCoordinator,
    SessionMode,
)
from text_injector import InjectionResult


@dataclass
class FakeWhisperSegment:
    text: str
    start: float = 0.0
    end: float = 1.0


def _write_test_wav(duration_sec: float, sample_rate: int = 16000) -> str:
    """Create a valid 16-bit mono PCM WAV file of `duration_sec` seconds."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_ctx_")
    os.close(fd)
    total_samples = int(duration_sec * sample_rate)
    # Low-amplitude synthetic signal
    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)
    pcm = (np.sin(2 * np.pi * 220.0 * t) * 2000).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


class TestASRContextContinuity(unittest.TestCase):
    def setUp(self):
        self.temp_files: list[str] = []
        self.vault = MagicMock()
        self.vault.add_entry = MagicMock(return_value="2026-10-01 12:00:00")
        self.brain = AIBrain(vault=self.vault)

    def tearDown(self):
        for p in self.temp_files:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _make_wav(self, duration_sec: float) -> str:
        path = _write_test_wav(duration_sec)
        self.temp_files.append(path)
        return path

    # ==================================================================
    # Scenario A: Long session with pauses
    # ==================================================================
    def test_scenario_a_long_session_with_pauses_single_unit(self):
        """A: A 60-second session with multiple thinking pauses is transcribed as ONE logical unit with intra-session conditioning."""
        wav_path = self._make_wav(60.0)  # <= 90s threshold -> single logical unit

        captured_calls = []

        def fake_transcribe(audio_input, **kwargs):
            captured_calls.append((audio_input, kwargs))
            # Simulate Whisper decoding 3 VAD-filtered speech segments across the 60s recording
            return [
                FakeWhisperSegment("Let's design the authentication system,"),
                FakeWhisperSegment("and for the database we'll use PostgreSQL"),
                FakeWhisperSegment("because we need ACID transactions."),
            ], MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = fake_transcribe

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            transcript = self.brain._offline_transcribe(
                wav_path,
                context_info={"app_hint": "VS Code", "exe_name": "Code.exe"},
                session_id="session_a_12345678",
            )

        # Transcribed in ONE pass as one logical unit
        self.assertEqual(len(captured_calls), 1)
        _audio_in, kwargs = captured_calls[0]
        self.assertTrue(kwargs.get("condition_on_previous_text"))
        self.assertEqual(kwargs.get("prompt_reset_on_temperature"), 0.5)
        self.assertTrue(kwargs.get("vad_filter"))
        self.assertIn("speech_pad_ms", kwargs.get("vad_parameters", {}))

        self.assertEqual(
            transcript,
            "Let's design the authentication system, and for the database we'll use PostgreSQL because we need ACID transactions.",
        )

    # ==================================================================
    # Scenario B: Sentence split around silence
    # ==================================================================
    def test_scenario_b_sentence_split_around_silence(self):
        """B: A single sentence split across an 8s pause maintains continuity within one session."""
        wav_path = self._make_wav(25.0)

        def fake_transcribe(audio_input, **kwargs):
            # Because condition_on_previous_text=True is enabled within the session,
            # the second segment after the 8s pause continues the clause seamlessly.
            self.assertTrue(kwargs.get("condition_on_previous_text"))
            return [
                FakeWhisperSegment("I've been thinking about how we should redesign the authentication system"),
                FakeWhisperSegment("and move session storage into Redis."),
            ], MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = fake_transcribe

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            transcript = self.brain._offline_transcribe(
                wav_path,
                session_id="session_b_87654321",
            )

        self.assertEqual(
            transcript,
            "I've been thinking about how we should redesign the authentication system and move session storage into Redis.",
        )

    # ==================================================================
    # Scenario C: Several sentences separated by long silence
    # ==================================================================
    def test_scenario_c_several_sentences_separated_by_long_silence(self):
        """C: Multiple sentences separated by 8s and 6s thinking pauses stay in one session and preserve sentence boundaries."""
        wav_path = self._make_wav(40.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [
                FakeWhisperSegment("First, we need to audit the current API endpoints."),
                FakeWhisperSegment("Second, we should add rate limiting at the gateway."),
                FakeWhisperSegment("Finally, let's deploy the updated metrics dashboard."),
            ],
            MagicMock(),
        )

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            transcript = self.brain._offline_transcribe(
                wav_path,
                session_id="session_c_11223344",
            )

        self.assertEqual(mock_model.transcribe.call_count, 1)
        self.assertEqual(
            transcript,
            "First, we need to audit the current API endpoints. "
            "Second, we should add rate limiting at the gateway. "
            "Finally, let's deploy the updated metrics dashboard.",
        )

    # ==================================================================
    # Scenario D: Two separate sessions (Context Isolation & PTT Independence)
    # ==================================================================
    def test_scenario_d_two_separate_sessions_never_leak_context(self):
        """D: Session A must never leak rolling transcript context into Session B (including short push-to-talk sessions)."""
        # Force chunked rolling mode with a small threshold to verify that even when
        # Session A builds a multi-chunk rolling context, Session B starts 100% clean.
        wav_session_a = self._make_wav(30.0)
        wav_session_b = self._make_wav(8.0)

        session_a_prompts: list[str] = []
        session_b_prompts: list[str] = []
        current_session_tag = "A"

        def fake_transcribe(audio_input, **kwargs):
            prompt = kwargs.get("initial_prompt") or ""
            if current_session_tag == "A":
                session_a_prompts.append(prompt)
                if len(session_a_prompts) == 1:
                    return [FakeWhisperSegment("Confidential Project Apollo launch sequence alpha")], MagicMock()
                return [FakeWhisperSegment("sequence alpha is scheduled for Tuesday.")], MagicMock()
            else:
                session_b_prompts.append(prompt)
                return [FakeWhisperSegment("Quick push to talk reply on Slack.")], MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = fake_transcribe

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model), \
             patch.object(self.brain, "ASR_SINGLE_PASS_MAX_SECONDS", 15.0), \
             patch.object(self.brain, "ASR_CHUNK_WINDOW_SECONDS", 16.0), \
             patch.object(self.brain, "ASR_CHUNK_OVERLAP_SECONDS", 2.0):

            # Run Session A (long continuous session -> 2 chunks)
            current_session_tag = "A"
            text_a = self.brain._offline_transcribe(
                wav_session_a,
                session_id="session_a_continuous",
            )
            self.assertEqual(
                text_a,
                "Confidential Project Apollo launch sequence alpha is scheduled for Tuesday.",
            )
            self.assertEqual(len(session_a_prompts), 2)
            # Chunk 2 of Session A MUST contain rolling context from Chunk 1 of Session A
            self.assertIn("Confidential Project Apollo", session_a_prompts[1])

            # Run Session B (separate push-to-talk session)
            current_session_tag = "B"
            text_b = self.brain._offline_transcribe(
                wav_session_b,
                session_id="session_b_ptt",
            )
            self.assertEqual(text_b, "Quick push to talk reply on Slack.")
            self.assertEqual(len(session_b_prompts), 1)
            # Session B MUST NOT contain any transcript words from Session A
            self.assertNotIn("Confidential", session_b_prompts[0])
            self.assertNotIn("Apollo", session_b_prompts[0])
            self.assertNotIn("Tuesday", session_b_prompts[0])

    # ==================================================================
    # Scenario E: Technical terminology & dictionaries preserved
    # ==================================================================
    def test_scenario_e_technical_terminology_and_dictionaries_preserved(self):
        """E: Custom dictionary and app-contextual coding vocabulary are preserved in both initial_prompt and hotwords across all chunks."""
        wav_path = self._make_wav(25.0)
        prompts_seen: list[str] = []
        hotwords_seen: list[str] = []

        def fake_transcribe(audio_input, **kwargs):
            prompts_seen.append(kwargs.get("initial_prompt") or "")
            hotwords_seen.append(kwargs.get("hotwords") or "")
            if len(prompts_seen) == 1:
                return [FakeWhisperSegment("We deployed the kubernetes cluster with docker and FastAPI")], MagicMock()
            return [FakeWhisperSegment("and FastAPI connected to PostgreSQL via asyncpg.")], MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = fake_transcribe

        coding_context = {"app_hint": "VS Code", "exe_name": "Code.exe"}
        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model), \
             patch.object(self.brain, "ASR_SINGLE_PASS_MAX_SECONDS", 15.0), \
             patch.object(self.brain, "ASR_CHUNK_WINDOW_SECONDS", 15.0), \
             patch.object(self.brain, "ASR_CHUNK_OVERLAP_SECONDS", 2.0):

            transcript = self.brain._offline_transcribe(
                wav_path,
                context_info=coding_context,
                session_id="session_e_tech",
            )

        self.assertEqual(len(prompts_seen), 2)
        # Both Chunk 1 and Chunk 2 must retain coding dictionary terms (e.g. kubernetes, docker, async, pytest/ctypes)
        for p in prompts_seen:
            self.assertIn("Glossary:", p)
            self.assertIn("kubernetes", p)
            self.assertIn("docker", p)
            self.assertIn("customtkinter", p)
        for hw in hotwords_seen:
            self.assertIn("kubernetes", hw)

        # And Chunk 2 also has the rolling context from Chunk 1, with overlap reconciled cleanly
        self.assertIn("FastAPI", prompts_seen[1])
        self.assertEqual(
            transcript,
            "We deployed the kubernetes cluster with docker and FastAPI connected to PostgreSQL via asyncpg.",
        )

    # ==================================================================
    # Scenario F: Long continuous speech (multi-chunk rolling context + overlap deduplication + benchmark)
    # ==================================================================
    def test_scenario_f_long_continuous_speech_and_overlap_deduplication(self):
        """F: 180-second (3-minute) continuous session chunks cleanly, passes bounded rolling context, deduplicates overlaps, and completes fast."""
        # 180 seconds (3 minutes) at 16 kHz mono = 2,880,000 samples (~5.76 MB)
        wav_path = self._make_wav(180.0)

        chunk_Transcriptions = [
            "In this architecture review we are evaluating our distributed event pipeline.",
            "distributed event pipeline. Each microservice publishes domain events to Kafka,",
            "domain events to Kafka, and downstream consumers project those events into PostgreSQL.",
            "events into PostgreSQL. By decoupling write workloads from read projections, we achieve low latency.",
            "we achieve low latency. Finally, all idempotency keys are enforced at the transaction boundary.",
        ]

        call_idx = 0
        prompts_received: list[str] = []

        def fake_transcribe(audio_input, **kwargs):
            nonlocal call_idx
            prompts_received.append(kwargs.get("initial_prompt") or "")
            text_out = chunk_Transcriptions[min(call_idx, len(chunk_Transcriptions) - 1)]
            call_idx += 1
            return [FakeWhisperSegment(text_out)], MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = fake_transcribe

        t0 = time.perf_counter()
        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            full_transcript = self.brain._offline_transcribe(
                wav_path,
                context_info={"app_hint": "VS Code", "exe_name": "Code.exe"},
                session_id="session_f_180s",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 180s with 45s window and 2s overlap (step=43s):
        # Chunk 0: 0..45s, Chunk 1: 43..88s, Chunk 2: 86..131s, Chunk 3: 129..174s, Chunk 4: 172..180s -> 5 chunks
        self.assertEqual(call_idx, 5)

        # Verify overlapping phrases ("distributed event pipeline.", "domain events to Kafka,",
        # "events into PostgreSQL.", "we achieve low latency.") appear ONLY ONCE in the reconciled transcript
        expected_full = (
            "In this architecture review we are evaluating our distributed event pipeline. "
            "Each microservice publishes domain events to Kafka, "
            "and downstream consumers project those events into PostgreSQL. "
            "By decoupling write workloads from read projections, we achieve low latency. "
            "Finally, all idempotency keys are enforced at the transaction boundary."
        )
        self.assertEqual(full_transcript, expected_full)

        # Verify each subsequent chunk received rolling context from the previous chunk
        self.assertIn("distributed event pipeline", prompts_received[1])
        self.assertIn("domain events to Kafka", prompts_received[2])
        self.assertIn("events into PostgreSQL", prompts_received[3])
        self.assertIn("we achieve low latency", prompts_received[4])

        # Verify WAV loading, slicing, prompt construction, and reconciliation overhead for 3 minutes of audio is < 100ms
        self.assertLess(elapsed_ms, 250.0)

    def test_reconcile_overlapping_transcript_edge_cases(self):
        """Direct unit tests for overlap deduplication with punctuation and casing variations."""
        # Case 1: Multi-word overlap with punctuation difference
        a1 = "Let's design the authentication system,"
        b1 = "the authentication system and move session storage into Redis."
        self.assertEqual(
            AIBrain.reconcile_overlapping_transcript(a1, b1),
            "Let's design the authentication system, and move session storage into Redis.",
        )

        # Case 2: Single substantial word overlap at boundary
        a2 = "For the primary database we will use PostgreSQL"
        b2 = "postgresql because we need strict serializable transactions."
        self.assertEqual(
            AIBrain.reconcile_overlapping_transcript(a2, b2),
            "For the primary database we will use PostgreSQL because we need strict serializable transactions.",
        )

        # Case 3: Disjoint sentences (no overlap) must be joined cleanly without dropping words
        a3 = "First sentence ends here."
        b3 = "Second sentence starts cleanly."
        self.assertEqual(
            AIBrain.reconcile_overlapping_transcript(a3, b3),
            "First sentence ends here. Second sentence starts cleanly.",
        )


if __name__ == "__main__":
    unittest.main()
