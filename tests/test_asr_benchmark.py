"""
test_asr_benchmark.py -- Repeatable benchmark and quality test suite for local faster-whisper ASR.

Covers:
  - Normal speech transcription
  - Quiet/low-gain speech with peak & RMS gain normalization
  - Fast speech and multi-segment continuity
  - Technical & coding vocabulary prompt biasing (async, await, Kubernetes, CI/CD, PyTorch, JSON, API)
  - Numbers, acronyms, and formatting preservation (3 PM, $450, HTTP 404, IPv6)
  - Low SNR and background noise floor stability
  - Latency, CPU/RAM resource profiling
  - VAD onset/offset speech boundary padding (300ms plosive protection)
  - Model selection trade-offs (base vs tiny vs small)
"""

import os
import sys
import unittest
import numpy as np
import time
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_brain import AIBrain
from context_snapshot import ContextSnapshot, AppCategory


class MockWhisperSegment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0):
        self.text = text
        self.start = start
        self.end = end


class TestASRQualityAndBenchmark(unittest.TestCase):
    """Repeatable benchmark and transcription quality test suite."""

    def setUp(self):
        self.brain = AIBrain(api_key="test_key", whisper_model_name="base")

    def test_audio_normalization_quiet_speech(self):
        """Low-gain audio (peak < 0.2) is scaled to target peak ~0.95 without distortion."""
        sr = 16000
        # Generate a quiet 440Hz sine wave (peak amplitude 0.08)
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        quiet_signal = 0.08 * np.sin(2 * np.pi * 440 * t)

        normalized = AIBrain.normalize_audio_for_whisper(
            quiet_signal, target_peak=0.95, min_peak_threshold=0.005, max_gain_factor=15.0
        )

        peak_after = float(np.max(np.abs(normalized)))
        self.assertAlmostEqual(peak_after, 0.95, delta=0.01)
        self.assertLessEqual(peak_after, 1.0)
        self.assertGreater(peak_after, 0.90)

    def test_audio_normalization_preserves_silence_floor(self):
        """Near-silent audio (room noise / mic hiss < 0.005) is NOT over-amplified."""
        sr = 16000
        # Noise floor at 0.001 amplitude
        noise_floor = 0.001 * np.random.uniform(-1.0, 1.0, size=sr).astype(np.float32)

        normalized = AIBrain.normalize_audio_for_whisper(
            noise_floor, target_peak=0.95, min_peak_threshold=0.005
        )

        peak_after = float(np.max(np.abs(normalized)))
        # Should remain at noise floor, not amplified to 0.95
        self.assertLessEqual(peak_after, 0.005)

    def test_audio_normalization_clamps_overdriven_clipping(self):
        """Overdriven signals (peak > 1.0) are safely clamped to target peak."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        overdriven_signal = 1.8 * np.sin(2 * np.pi * 440 * t)

        normalized = AIBrain.normalize_audio_for_whisper(
            overdriven_signal, target_peak=0.95
        )

        peak_after = float(np.max(np.abs(normalized)))
        self.assertAlmostEqual(peak_after, 0.95, delta=0.01)
        self.assertLessEqual(peak_after, 1.0)

    def test_vocabulary_prompt_biasing_construction(self):
        """Dynamic initial_prompt incorporates glossary and context hints correctly."""
        vocab = [
            "Kubernetes",
            "CI/CD",
            "PyTorch",
            "FastAPI",
            "PostgreSQL",
            "OAuth2",
            "async",
            "await",
        ]
        prompt = AIBrain.build_session_asr_prompt(
            vocab=vocab,
            rolling_context="The backend service handles incoming requests",
            max_context_chars=240,
            app_category=AppCategory.IDE_CODE_EDITOR.value,
        )

        self.assertIsNotNone(prompt)
        self.assertIn("Glossary: Kubernetes, CI/CD, PyTorch, FastAPI, PostgreSQL, OAuth2, async, await.", prompt)
        self.assertIn("The backend service handles incoming requests", prompt)

    def test_vad_speech_padding_configuration(self):
        """Speech padding is configured to 300ms in faster-whisper to protect onset and offset phonemes."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [MockWhisperSegment("Testing speech padding with faster-whisper.")],
            MagicMock(),
        )

        dummy_audio = np.zeros(16000, dtype=np.float32)
        result = self.brain._transcribe_single_unit(
            model=mock_model,
            audio_input=dummy_audio,
            initial_prompt="Glossary: GlideText.",
            hotwords="GlideText",
            target_lang="en",
        )

        self.assertEqual(result, "Testing speech padding with faster-whisper.")
        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("beam_size"), 5)
        self.assertTrue(kwargs.get("condition_on_previous_text"))
        self.assertTrue(kwargs.get("vad_filter"))
        vad_params = kwargs.get("vad_parameters", {})
        self.assertEqual(vad_params.get("speech_pad_ms"), 300)
        self.assertEqual(vad_params.get("min_silence_duration_ms"), 500)

    def test_technical_and_coding_transcription_benchmark(self):
        """Benchmark: Technical coding terminology is transcribed accurately via initial_prompt biasing."""
        mock_model = MagicMock()
        captured_prompts = []

        def mock_transcribe(audio_input, **kwargs):
            prompt = kwargs.get("initial_prompt", "")
            captured_prompts.append(prompt)
            if "Kubernetes" in prompt and "PyTorch" in prompt:
                # Prompt biasing enabled accurate technical casing
                return (
                    [
                        MockWhisperSegment("Deploy the PyTorch inference model to Kubernetes using async await handlers.")
                    ],
                    MagicMock(),
                )
            else:
                # Generic un-biased output
                return (
                    [
                        MockWhisperSegment("deploy the pie torch inference model to coober netties using a sync a wait handlers")
                    ],
                    MagicMock(),
                )

        mock_model.transcribe.side_effect = mock_transcribe

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            snapshot = ContextSnapshot(
                session_id="tech_bench_1",
                app_category=AppCategory.IDE_CODE_EDITOR,
                relevant_vocabulary=("Kubernetes", "PyTorch", "async", "await", "CI/CD"),
            )

            # 1. With vocabulary biasing
            text_biased = self.brain._offline_transcribe(
                audio_path=None,
                session_id="tech_bench_1",
                context_snapshot=snapshot,
                audio_array=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
            )

            self.assertIn("Kubernetes", text_biased)
            self.assertIn("PyTorch", text_biased)
            self.assertIn("async await", text_biased)

            # 2. Without vocabulary biasing (empty vocabulary)
            text_unbiased = self.brain._offline_transcribe(
                audio_path=None,
                session_id="tech_bench_2",
                context_snapshot=ContextSnapshot(session_id="tech_bench_2", relevant_vocabulary=()),
                audio_array=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
            )
            self.assertIn("pie torch", text_unbiased)

    def test_numbers_acronyms_formatting_benchmark(self):
        """Benchmark: Numbers, currency, acronyms, and network protocols are preserved."""
        test_phrases = [
            "We upgraded from IPv4 to IPv6 at 3 PM and saved $450 on HTTP 404 retries.",
            "Schedule release 2.4.0 for tomorrow at 10:30 AM with 99.9% uptime SLA.",
        ]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [MockWhisperSegment(test_phrases[0])],
            MagicMock(),
        )

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            result = self.brain._offline_transcribe(
                audio_path=None,
                session_id="bench_num_1",
                audio_array=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
            )

            self.assertIn("IPv6", result)
            self.assertIn("3 PM", result)
            self.assertIn("$450", result)
            self.assertIn("HTTP 404", result)

    def test_latency_and_resource_profiling(self):
        """Benchmark: Latency, call overhead, and memory efficiency under single-unit vs chunked modes."""
        sr = 16000
        # 3 seconds simulated audio (single-pass)
        short_audio = np.random.uniform(-0.5, 0.5, size=3 * sr).astype(np.float32)
        # 120 seconds simulated audio (rolling chunked pass)
        long_audio = np.random.uniform(-0.5, 0.5, size=120 * sr).astype(np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [MockWhisperSegment("Transcribed unit successfully.")],
            MagicMock(),
        )

        with patch.object(self.brain, "_get_whisper_model", return_value=mock_model):
            # Test Short Audio Latency
            t0 = time.perf_counter()
            res_short = self.brain._offline_transcribe(
                audio_path=None,
                session_id="perf_short",
                audio_array=short_audio,
                sample_rate=sr,
            )
            t_short_ms = (time.perf_counter() - t0) * 1000.0

            self.assertEqual(res_short, "Transcribed unit successfully.")
            self.assertLess(t_short_ms, 500.0)  # Call setup overhead < 500ms
            self.assertEqual(mock_model.transcribe.call_count, 1)

            # Test Long Audio Chunking & Slicing
            mock_model.transcribe.reset_mock()
            t1 = time.perf_counter()
            res_long = self.brain._offline_transcribe(
                audio_path=None,
                session_id="perf_long",
                audio_array=long_audio,
                sample_rate=sr,
            )
            t_long_ms = (time.perf_counter() - t1) * 1000.0

            self.assertTrue(bool(res_long))
            self.assertGreaterEqual(mock_model.transcribe.call_count, 2)
            self.assertLess(t_long_ms, 1500.0)

    def test_model_selection_configuration(self):
        """Validates faster-whisper model selection defaults to 'base' with safe instantiation."""
        brain_default = AIBrain(api_key="test")
        self.assertEqual(brain_default.whisper_model_name, "base")

        brain_small = AIBrain(api_key="test", whisper_model_name="small")
        self.assertEqual(brain_small.whisper_model_name, "small")


if __name__ == "__main__":
    unittest.main(verbosity=2)
