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


if __name__ == "__main__":
    unittest.main()
