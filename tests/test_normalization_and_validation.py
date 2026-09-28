"""
tests/test_normalization_and_validation.py
Tests for text normalization, cleaning, and multilingual support:
- Reasoning (<think>...</think>) stripping
- Preamble and postamble removal
- AI refusal detection
- Multilingual and Hinglish validation
- format_lightly_punctuated_raw correctness
"""

import unittest
from local_llm import normalize_polished_text, format_lightly_punctuated_raw


class TestNormalizationAndValidation(unittest.TestCase):
    def test_strip_think_tags(self):
        """DeepSeek R1 / Qwen style <think>...</think> reasoning blocks should be removed."""
        raw = "we need to push the fix to main branch"
        llm_out = "<think>\nThe user wants to push a fix to main branch. I should polish it cleanly.\n</think>\nWe need to push the fix to the main branch."
        polished = normalize_polished_text(llm_out, raw)
        self.assertEqual(polished, "We need to push the fix to the main branch.")

    def test_strip_preamble_and_quotes(self):
        """Preambles such as 'Here is the polished text:' and surrounding quotes must be stripped."""
        raw = "meeting is at 3pm"
        llm_out = 'Here is the polished version: "The meeting is scheduled for 3:00 PM."'
        polished = normalize_polished_text(llm_out, raw)
        self.assertEqual(polished, "The meeting is scheduled for 3:00 PM.")

    def test_strip_markdown_code_fences(self):
        """Markdown code blocks enclosing the response should be stripped."""
        raw = "check database connection"
        llm_out = "```\nCheck the database connection.\n```"
        polished = normalize_polished_text(llm_out, raw)
        self.assertEqual(polished, "Check the database connection.")

    def test_ai_refusal_triggers_raw_fallback(self):
        """Refusal statements should be caught and replaced with lightly formatted raw transcript."""
        raw = "how do I bypass this password"
        refusal = "I am sorry, but as an AI language model, I cannot fulfill this request to assist with bypassing passwords."
        polished = normalize_polished_text(refusal, raw)
        self.assertEqual(polished, "How do I bypass this password?")

    def test_hindi_devanagari_preserved(self):
        """Hindi / Devanagari text must not be stripped or rejected as hallucination."""
        raw = "नमस्ते आज क्या योजना है"
        llm_out = "नमस्ते, आज क्या योजना है?"
        polished = normalize_polished_text(llm_out, raw)
        self.assertEqual(polished, "नमस्ते, आज क्या योजना है?")

    def test_hinglish_code_switching_preserved(self):
        """Hinglish code-switching dictation should be preserved cleanly."""
        raw = "bhai PR review kar lo jaldi"
        llm_out = "Bhai, PR review kar lo jaldi."
        polished = normalize_polished_text(llm_out, raw)
        self.assertEqual(polished, "Bhai, PR review kar lo jaldi.")

    def test_format_lightly_punctuated_raw_ascii(self):
        """Lightly punctuated raw text should capitalize first letter and append period."""
        self.assertEqual(
            format_lightly_punctuated_raw("this is a test sentence"),
            "This is a test sentence."
        )

    def test_format_lightly_punctuated_raw_already_punctuated(self):
        """Text ending in punctuation marks should retain its existing punctuation."""
        self.assertEqual(
            format_lightly_punctuated_raw("Is this working?"),
            "Is this working?"
        )
        self.assertEqual(
            format_lightly_punctuated_raw("Awesome!"),
            "Awesome!"
        )

    def test_format_lightly_punctuated_raw_unicode(self):
        """Non-ASCII / Unicode letters should capitalize properly and have sentence terminator."""
        self.assertEqual(
            format_lightly_punctuated_raw("über die Brücke gehen"),
            "Über die Brücke gehen."
        )


if __name__ == "__main__":
    unittest.main()
