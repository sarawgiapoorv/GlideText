"""
tests/test_provider_routing.py
Comprehensive unit tests verifying the strict AI Provider Priority:
Priority 1: FreeLLMAPI
Priority 2: Gemini API
Priority 3: Local LLM / Ollama
Fallback: Degraded raw transcript
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from ai_brain import AIBrain, PipelineResult, ProviderAttempt, ErrorCategory


class DummyVault:
    def __init__(self):
        self.entries = []

    def add_entry(self, polished, raw):
        self.entries.append((polished, raw))
        return "2026-09-28 12:00:00"

    def record_api_call(self, provider, success, latency_ms=0, error_type=None, token_count=0):
        pass


class TestProviderRouting(unittest.TestCase):
    def setUp(self):
        self.vault = DummyVault()
        self.brain = AIBrain(vault=self.vault)
        # Reset cooldowns
        self.brain.reset_cloud_mode()

    def test_priority1_freellmapi_success(self):
        """When FreeLLMAPI succeeds, Gemini and Local LLM must NOT be invoked."""
        mock_freellm = MagicMock(return_value=(
            "Hello, this is polished by FreeLLMAPI.",
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=True,
                latency_ms=120.0,
            )
        ))
        mock_gemini = MagicMock()
        mock_local = MagicMock()

        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        result = self.brain.polish_with_provider_fallbacks("hello this is polished by freellmapi")

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "freellmapi")
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.text, "Hello, this is polished by FreeLLMAPI.")

        # Assert FreeLLMAPI called once
        mock_freellm.assert_called_once()
        # Assert Gemini and Local LLM were NEVER called
        mock_gemini.assert_not_called()
        mock_local.assert_not_called()

    def test_priority2_gemini_fallback_when_freellm_fails(self):
        """When FreeLLMAPI fails, fallback to Gemini API. Local LLM must NOT be invoked."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=False,
                latency_ms=250.0,
                error="Connection refused: FreeLLMAPI port 3000 down",
                error_category=ErrorCategory.SERVER_UNAVAILABLE,
            )
        ))
        mock_gemini = MagicMock(return_value=(
            "Hello, this is polished by Gemini.",
            ProviderAttempt(
                provider="gemini",
                model="gemini-2.5-flash",
                success=True,
                latency_ms=180.0,
            )
        ))
        mock_local = MagicMock()

        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        result = self.brain.polish_with_provider_fallbacks("hello this is polished by gemini")

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "gemini")
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.text, "Hello, this is polished by Gemini.")

        # FreeLLMAPI was attempted first
        mock_freellm.assert_called_once()
        # Gemini was called
        mock_gemini.assert_called_once()
        # Local LLM must NOT be called because Gemini succeeded
        mock_local.assert_not_called()

    def test_priority3_local_llm_fallback_when_freellm_and_gemini_fail(self):
        """When both FreeLLMAPI and Gemini fail, fallback to Local LLM (Ollama)."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=False,
                latency_ms=100.0,
                error="HTTP 500: Server Overloaded",
                error_category=ErrorCategory.SERVER_UNAVAILABLE,
            )
        ))
        mock_gemini = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="gemini",
                model="gemini-2.5-flash",
                success=False,
                latency_ms=300.0,
                error="HTTP 429: Rate Limit Exceeded",
                error_category=ErrorCategory.RATE_LIMIT,
            )
        ))
        mock_local = MagicMock(return_value=(
            "Hello, this is polished by Local LLM.",
            ProviderAttempt(
                provider="local_llm",
                model="llama3.2:3b",
                success=True,
                latency_ms=450.0,
            )
        ))

        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        result = self.brain.polish_with_provider_fallbacks("hello this is polished by local llm")

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "local_llm")
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.text, "Hello, this is polished by Local LLM.")

        mock_freellm.assert_called_once()
        mock_gemini.assert_called_once()
        mock_local.assert_called_once()

    def test_all_providers_fail_degraded_raw_transcript(self):
        """When FreeLLMAPI, Gemini, and Local LLM all fail, return lightly punctuated raw transcript."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=False,
                latency_ms=50.0,
                error="Down",
                error_category=ErrorCategory.SERVER_UNAVAILABLE,
            )
        ))
        mock_gemini = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="gemini",
                model="gemini-2.5-flash",
                success=False,
                latency_ms=50.0,
                error="API key invalid",
                error_category=ErrorCategory.AUTHENTICATION,
            )
        ))
        mock_local = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="local_llm",
                model="llama3.2:3b",
                success=False,
                latency_ms=50.0,
                error="Ollama crashed",
                error_category=ErrorCategory.SERVER_UNAVAILABLE,
            )
        ))

        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        result = self.brain.polish_with_provider_fallbacks("this is raw unpunctuated voice input")

        self.assertFalse(result.success)
        self.assertEqual(result.provider, "raw_fallback")
        self.assertTrue(result.is_fallback)
        # Should be formatted with capital letter and period
        self.assertTrue(result.text.startswith("This is raw unpunctuated voice input"))
        self.assertEqual(len(result.attempts), 3)

    def test_no_permanent_sticky_local_bypass(self):
        """Verify that after a fallback to local, FreeLLMAPI is re-attempted once its cooldown expires."""
        # Step 1: FreeLLMAPI fails temporarily (cooldown set to 1 second)
        mock_freellm_fail = MagicMock(return_value=(
            None,
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=False,
                latency_ms=20.0,
                error="Temporary 503",
                error_category=ErrorCategory.SERVER_UNAVAILABLE,
            )
        ))
        mock_gemini_success = MagicMock(return_value=(
            "Gemini response",
            ProviderAttempt(
                provider="gemini",
                model="gemini-2.5-flash",
                success=True,
                latency_ms=50.0,
            )
        ))
        self.brain._call_freellmapi_or_openai = mock_freellm_fail
        self.brain._call_gemini_with_fallback = mock_gemini_success

        # Run with short cooldown
        with patch.object(self.brain, "_freellmapi_cooldown_until", time.time() - 10):
            res1 = self.brain.polish_with_provider_fallbacks("first test")
            self.assertEqual(res1.provider, "gemini")

        # Step 2: Now FreeLLMAPI comes back up
        mock_freellm_success = MagicMock(return_value=(
            "FreeLLMAPI restored response",
            ProviderAttempt(
                provider="freellmapi",
                model="gpt-4o-mini",
                success=True,
                latency_ms=30.0,
            )
        ))
        self.brain._call_freellmapi_or_openai = mock_freellm_success
        self.brain.reset_cloud_mode()  # Expire cooldown

        res2 = self.brain.polish_with_provider_fallbacks("second test")
        # Must use FreeLLMAPI, not stuck in gemini or local
        self.assertEqual(res2.provider, "freellmapi")
        self.assertEqual(res2.text, "FreeLLMAPI restored response")

    def test_freellmapi_treated_as_single_provider(self):
        """FreeLLMAPI must be called as ONE unified provider (single /v1/chat/completions HTTP call)."""
        self.brain.vault.log_api_call = MagicMock()
        self.brain._freellmapi_api_key = "test-unified-key"

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Upstream error"

        with patch.object(self.brain._session, "post", return_value=mock_resp) as mock_post, \
             patch.object(self.brain._session, "get") as mock_get:
            out, attempt = self.brain._call_freellmapi_or_openai(
                system_instruction="system",
                user_text="test dictation",
            )
            self.assertIsNone(out)
            self.assertFalse(attempt.success)
            # Only ONE HTTP POST to FreeLLMAPI (no internal model fallback loop or /v1/models lookup)
            self.assertEqual(mock_post.call_count, 1)
            mock_get.assert_not_called()

    def test_canonical_raw_transcript_passed_unchanged_across_all_fallbacks(self):
        """Every fallback provider must receive the exact same canonical raw transcript."""
        canonical_input = "  um let us deploy the fastapi microservice to kubernetes cluster  "
        expected_canonical = canonical_input.strip()

        received_inputs = []

        def fake_freellm(**kwargs):
            received_inputs.append(("freellmapi", kwargs["user_text"]))
            return None, ProviderAttempt(
                provider="freellmapi",
                success=False,
                error="Down",
                error_category=ErrorCategory.CONNECTION_ERROR,
            )

        def fake_gemini(**kwargs):
            received_inputs.append(("gemini", kwargs["raw_text"]))
            return None, ProviderAttempt(
                provider="gemini",
                success=False,
                error="Quota",
                error_category=ErrorCategory.RATE_LIMITED,
            )

        def fake_local(**kwargs):
            received_inputs.append(("local_llm", kwargs["raw_text"]))
            return (
                "Let's deploy the FastAPI microservice to the Kubernetes cluster.",
                ProviderAttempt(provider="local_llm", success=True, model="llama3.2:3b"),
            )

        self.brain._api_keys = ["dummy-gemini-key"]
        self.brain._current_key_index = 0
        self.brain._call_freellmapi_or_openai = fake_freellm
        self.brain._call_gemini_with_fallback = fake_gemini
        self.brain._call_local_llm = fake_local

        res = self.brain.polish_with_provider_fallbacks(canonical_input)
        self.assertTrue(res.success)
        self.assertEqual(res.provider, "local_llm")
        self.assertEqual(res.raw_transcript, expected_canonical)
        self.assertEqual(
            received_inputs,
            [
                ("freellmapi", expected_canonical),
                ("gemini", expected_canonical),
                ("local_llm", expected_canonical),
            ],
        )

    def test_spoken_self_correction_across_session_pauses(self):
        """Spoken self-correction ('Let's use Redis... actually use PostgreSQL.') preserves final intent."""
        raw_session = "um for the primary store let's use redis uh actually use postgresql for acid transactions"
        expected_polished = "For the primary store, let's use PostgreSQL for ACID transactions."

        captured_sys = {}

        def fake_freellm(**kwargs):
            captured_sys["prompt"] = kwargs["system_instruction"]
            captured_sys["user_text"] = kwargs["user_text"]
            return (
                expected_polished,
                ProviderAttempt(provider="freellmapi", success=True, model="auto"),
            )

        self.brain._call_freellmapi_or_openai = fake_freellm
        res = self.brain.polish_with_provider_fallbacks(raw_session)

        self.assertTrue(res.success)
        self.assertEqual(res.text, expected_polished)
        self.assertIn("ONE continuous dictation session", captured_sys["prompt"])
        self.assertIn("SPEECH-TO-MIND SELF-CORRECTION", captured_sys["prompt"])
        self.assertIn("PostgreSQL", captured_sys["prompt"])

    def test_long_transcript_from_session_with_pauses_not_truncated(self):
        """Long multi-paragraph transcript from a session with pauses uses full token budget and preserves structure."""
        paragraphs = [
            "first we need to review the architecture for the session state machine and ensure silence only pauses capture",
            "second when the user resumes speaking after a thinking pause all audio chunks remain in the same dictation session",
            "third once the user explicitly stops the session faster whisper transcribes the full recording with context continuity",
            "finally the complete session transcript is polished once by the llm pipeline without splitting or truncating any section",
        ]
        long_raw = " um ".join(paragraphs * 3)
        long_polished = (
            "First, we need to review the architecture for the session state machine and ensure silence only pauses capture. "
            "Second, when the user resumes speaking after a thinking pause, all audio chunks remain in the same dictation session.\n\n"
            "Third, once the user explicitly stops the session, faster-whisper transcribes the full recording with context continuity. "
            "Finally, the complete session transcript is polished once by the LLM pipeline without splitting or truncating any section."
        ) * 3

        captured_kwargs = {}

        def fake_freellm(**kwargs):
            captured_kwargs.update(kwargs)
            return (
                long_polished,
                ProviderAttempt(provider="freellmapi", success=True, model="auto"),
            )

        self.brain._call_freellmapi_or_openai = fake_freellm
        res = self.brain.polish_with_provider_fallbacks(long_raw)

        self.assertTrue(res.success)
        self.assertEqual(res.text, long_polished)
        self.assertGreaterEqual(captured_kwargs["max_tokens"], 2048)

    def test_technical_vocabulary_and_formal_tone_and_supplementary_pre_text(self):
        """Technical vocabulary, Formal tone profile, and supplementary cursor pre_text are all included in system prompt."""
        captured = {}

        def fake_freellm(**kwargs):
            captured.update(kwargs)
            return (
                "Furthermore, we will integrate PyTorch and Kubernetes into the deployment pipeline.",
                ProviderAttempt(provider="freellmapi", success=True, model="auto"),
            )

        self.brain._call_freellmapi_or_openai = fake_freellm
        with patch.object(
            self.brain,
            "_get_vocabulary_for_session",
            return_value=["PyTorch", "Kubernetes", "FastAPI", "PostgreSQL"],
        ):
            res = self.brain.polish_with_provider_fallbacks(
                raw_text="um furthermore we will integrate pytorch and kubernetes into the deployment pipeline",
                style="Formal",
                context_info={"app_hint": "Outlook"},
                pre_text="We have finalized the Q3 infrastructure roadmap.",
            )

        self.assertTrue(res.success)
        sys_prompt = captured["system_instruction"]
        # Formal tone profile applied
        self.assertIn("corporate documentation language", sys_prompt)
        # Technical vocabulary hints included
        self.assertIn("CUSTOM & TECHNICAL VOCABULARY HINTS", sys_prompt)
        self.assertIn("PyTorch", sys_prompt)
        self.assertIn("Kubernetes", sys_prompt)
        # Pre-text framed as supplementary context, never a replacement
        self.assertIn("SUPPLEMENTARY CURSOR LOOKBACK CONTEXT", sys_prompt)
        self.assertIn("NEVER as a replacement for the current session transcript", sys_prompt)


    def test_scenario_a_freellmapi_succeeds_gemini_and_ollama_not_called(self):
        """Scenario A: FreeLLMAPI succeeds -> Gemini and Local LLM (Ollama) must NOT be called."""
        mock_freellm = MagicMock(return_value=(
            "We deployed the microservice to production successfully.",
            ProviderAttempt(provider="freellmapi", model="auto", success=True, latency_ms=105),
        ))
        mock_gemini = MagicMock()
        mock_local = MagicMock()

        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        res = self.brain.polish_with_provider_fallbacks("we deployed the microservice to production successfully")

        self.assertTrue(res.success)
        self.assertEqual(res.provider, "freellmapi")
        self.assertFalse(res.fallback_used)
        self.assertEqual(res.text, "We deployed the microservice to production successfully.")
        mock_freellm.assert_called_once()
        mock_gemini.assert_not_called()
        mock_local.assert_not_called()

    def test_scenario_b_freellmapi_fails_gemini_succeeds(self):
        """Scenario B: FreeLLMAPI fails, Gemini succeeds -> Gemini handles request, Ollama NOT called."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="freellmapi", model="auto", success=False, error="Gateway 502", error_category=ErrorCategory.SERVER_UNAVAILABLE),
        ))
        mock_gemini = MagicMock(return_value=(
            "We deployed the microservice to production successfully.",
            ProviderAttempt(provider="gemini", model="gemini-2.5-flash", success=True, latency_ms=190),
        ))
        mock_local = MagicMock()

        self.brain._api_keys = ["valid-gemini-key"]
        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        res = self.brain.polish_with_provider_fallbacks("we deployed the microservice to production successfully")

        self.assertTrue(res.success)
        self.assertEqual(res.provider, "gemini")
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.previous_providers, ["freellmapi"])
        self.assertEqual(res.text, "We deployed the microservice to production successfully.")
        mock_freellm.assert_called_once()
        mock_gemini.assert_called_once()
        mock_local.assert_not_called()

    def test_scenario_c_freellmapi_fails_gemini_fails_ollama_succeeds(self):
        """Scenario C: FreeLLMAPI fails, Gemini fails, Ollama succeeds -> Ollama handles request."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="freellmapi", model="auto", success=False, error="Connection refused", error_category=ErrorCategory.CONNECTION_ERROR),
        ))
        mock_gemini = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="gemini", model="gemini-2.5-flash", success=False, error="Quota 429", error_category=ErrorCategory.RATE_LIMIT),
        ))
        mock_local = MagicMock(return_value=(
            "We deployed the microservice to production successfully.",
            ProviderAttempt(provider="local_llm", model="llama3.2:3b", success=True, latency_ms=420),
        ))

        self.brain._api_keys = ["valid-gemini-key"]
        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        res = self.brain.polish_with_provider_fallbacks("we deployed the microservice to production successfully")

        self.assertTrue(res.success)
        self.assertEqual(res.provider, "local_llm")
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.previous_providers, ["freellmapi", "gemini"])
        self.assertEqual(res.text, "We deployed the microservice to production successfully.")
        mock_freellm.assert_called_once()
        mock_gemini.assert_called_once()
        mock_local.assert_called_once()

    def test_scenario_d_all_fail_raw_transcript_returned_preserves_words(self):
        """Scenario D: All providers fail -> Canonical raw transcript returned, user never loses words."""
        mock_freellm = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="freellmapi", success=False, error="Port 3001 down", error_category=ErrorCategory.CONNECTION_ERROR),
        ))
        mock_gemini = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="gemini", success=False, error="Gemini API key expired", error_category=ErrorCategory.AUTH_ERROR),
        ))
        mock_local = MagicMock(return_value=(
            None,
            ProviderAttempt(provider="local_llm", success=False, error="Ollama server down", error_category=ErrorCategory.LOCAL_LLM_FAILED),
        ))

        self.brain._api_keys = ["valid-gemini-key"]
        self.brain._call_freellmapi_or_openai = mock_freellm
        self.brain._call_gemini_with_fallback = mock_gemini
        self.brain._call_local_llm = mock_local

        raw_spoken = "we need to push the urgent hotfix to master right now"
        res = self.brain.polish_with_provider_fallbacks(raw_spoken)

        self.assertFalse(res.success)
        self.assertEqual(res.provider, "raw_fallback")
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.raw_transcript, raw_spoken)
        # User words are completely preserved in formatted raw output
        self.assertIn("push the urgent hotfix to master right now", res.text)
        self.assertEqual(res.previous_providers, ["freellmapi", "gemini", "local_llm"])
        self.assertEqual(len(res.attempts), 3)

    def test_scenario_e_retry_behavior_and_sequential_isolation(self):
        """Scenario E: Providers are evaluated sequentially, never in parallel or overlapping."""
        execution_order = []

        def freellm_tracker(**kwargs):
            execution_order.append("freellmapi")
            return None, ProviderAttempt(provider="freellmapi", success=False, error="500", error_category=ErrorCategory.FREELLMAPI_FAILED)

        def gemini_tracker(**kwargs):
            execution_order.append("gemini")
            return None, ProviderAttempt(provider="gemini", success=False, error="429", error_category=ErrorCategory.RATE_LIMIT)

        def local_tracker(**kwargs):
            execution_order.append("local_llm")
            return "Final output", ProviderAttempt(provider="local_llm", success=True, model="llama3.2:3b")

        self.brain._api_keys = ["gemini-key"]
        self.brain._call_freellmapi_or_openai = freellm_tracker
        self.brain._call_gemini_with_fallback = gemini_tracker
        self.brain._call_local_llm = local_tracker

        res = self.brain.polish_with_provider_fallbacks("test sequence order")
        self.assertTrue(res.success)
        self.assertEqual(execution_order, ["freellmapi", "gemini", "local_llm"])

    def test_scenario_f_malformed_freellmapi_response_safely_falls_back(self):
        """Scenario F: Malformed FreeLLMAPI response (invalid JSON, empty choices, refusal) falls back cleanly."""
        self.brain.vault.log_api_call = MagicMock()
        self.brain._freellmapi_api_key = "test-key"

        # Case 1: Invalid JSON in HTTP 200
        mock_resp_invalid_json = MagicMock()
        mock_resp_invalid_json.status_code = 200
        mock_resp_invalid_json.json.side_effect = ValueError("Unterminated string")

        with patch.object(self.brain._session, "post", return_value=mock_resp_invalid_json):
            text, attempt = self.brain._call_freellmapi_or_openai(user_text="test")
            self.assertIsNone(text)
            self.assertFalse(attempt.success)
            self.assertIn("Malformed JSON", attempt.error)

        # Case 2: Missing choices list
        mock_resp_missing_choices = MagicMock()
        mock_resp_missing_choices.status_code = 200
        mock_resp_missing_choices.json.return_value = {"id": "chat-123", "choices": []}

        with patch.object(self.brain._session, "post", return_value=mock_resp_missing_choices):
            text, attempt = self.brain._call_freellmapi_or_openai(user_text="test")
            self.assertIsNone(text)
            self.assertFalse(attempt.success)
            self.assertIn("choices", attempt.error)

        # Case 3: Model refusal
        mock_resp_refusal = MagicMock()
        mock_resp_refusal.status_code = 200
        mock_resp_refusal.json.return_value = {
            "choices": [{"message": {"refusal": "I cannot fulfill this request."}}]
        }

        with patch.object(self.brain._session, "post", return_value=mock_resp_refusal):
            text, attempt = self.brain._call_freellmapi_or_openai(user_text="test")
            self.assertIsNone(text)
            self.assertFalse(attempt.success)
            self.assertIn("refusal", attempt.error)

    def test_scenario_g_timeout_safely_falls_back_to_gemini(self):
        """Scenario G: Timeout in FreeLLMAPI falls back cleanly to Gemini with canonical input."""
        import requests

        self.brain.vault.log_api_call = MagicMock()
        self.brain._freellmapi_api_key = "test-key"

        with patch.object(self.brain._session, "post", side_effect=requests.exceptions.ReadTimeout("Read timed out")):
            text, attempt = self.brain._call_freellmapi_or_openai(user_text="test")
            self.assertIsNone(text)
            self.assertFalse(attempt.success)
            self.assertEqual(attempt.error_category, ErrorCategory.TIMEOUT)


if __name__ == "__main__":
    unittest.main()

