"""
tests/test_spoken_corrections.py
Comprehensive unit tests for natural spoken self-corrections in GlideText:
- simple correction
- number correction
- date/time correction
- name correction
- technology / coding terminology correction
- sentence / longer correction
- multiple corrections (chained and multi-sentence)
- "actually" / "sorry" / "instead" / "rather" used normally
- ambiguous correction preserved without invention
- DictationSession & AIBrain integration (raw_transcript preserved internally, privacy-safe logs)
"""

import unittest
from unittest.mock import MagicMock

from spoken_corrections import (
    resolve_spoken_corrections,
    SUPPORTED_CORRECTION_PATTERNS,
)
from dictation_session import DictationSession, SessionMode
from ai_brain import AIBrain


class TestSpokenCorrections(unittest.TestCase):
    def test_simple_correction(self):
        """Simple self-corrections with 'no actually', 'rather', 'instead', 'correction'."""
        self.assertEqual(
            resolve_spoken_corrections("Send the invoice to Mark, no actually send it to Sarah."),
            "Send the invoice to Sarah.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Deploy to staging, correction: deploy to production."),
            "Deploy to production.",
        )

    def test_number_correction(self):
        """Numeric and hyphenated/quantified noun corrections."""
        self.assertEqual(
            resolve_spoken_corrections("Create a three-column table—no, make it four columns."),
            "Create a four-column table.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Order 5 boxes, change that to 10."),
            "Order 10 boxes.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Set the timeout to 30 seconds, wait, make that 60 seconds."),
            "Set the timeout to 60 seconds.",
        )

    def test_date_and_time_correction(self):
        """Date and clock-time spoken corrections."""
        self.assertEqual(
            resolve_spoken_corrections("Let's meet at five PM, actually make that six thirty."),
            "Let's meet at six thirty.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Send it tomorrow—sorry, Thursday."),
            "Send it Thursday.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Schedule the review for Monday, or rather Wednesday."),
            "Schedule the review for Wednesday.",
        )

    def test_name_correction(self):
        """Person name corrections both at clause end and mid-sentence."""
        self.assertEqual(
            resolve_spoken_corrections("Email John—I mean David—about the contract."),
            "Email David about the contract.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Assign the ticket to Priya, sorry, Rahul."),
            "Assign the ticket to Rahul.",
        )

    def test_technology_and_coding_terminology_correction(self):
        """Technology and programming terminology corrections keep coding terms intact."""
        self.assertEqual(
            resolve_spoken_corrections("Use Redis—actually use PostgreSQL for persistence."),
            "Use PostgreSQL for persistence.",
        )
        self.assertEqual(
            resolve_spoken_corrections("I want to use React—actually, let's use Vue."),
            "Let's use Vue.",
        )
        self.assertEqual(
            resolve_spoken_corrections("Install the package with npm, instead use pnpm."),
            "Install the package with pnpm.",
        )

    def test_sentence_and_longer_correction(self):
        """Full-clause and multi-word corrections using 'scratch that' and 'forget that'."""
        self.assertEqual(
            resolve_spoken_corrections(
                "We should deploy on Friday afternoon, scratch that, let's wait until Monday morning after QA finishes."
            ),
            "Let's wait until Monday morning after QA finishes.",
        )
        self.assertEqual(
            resolve_spoken_corrections(
                "Send the summary to the entire engineering list—forget that, just send it to the backend leads."
            ),
            "Just send it to the backend leads.",
        )

    def test_multiple_corrections_in_session(self):
        """Multiple chained corrections in one sentence and across multiple sentences in a session."""
        chained = "Let's meet on Monday, sorry Tuesday, actually make that Wednesday."
        self.assertEqual(
            resolve_spoken_corrections(chained),
            "Let's meet on Wednesday.",
        )

        multi_sentence = (
            "Let's meet at five PM, actually make that six thirty. "
            "Use Redis—actually use PostgreSQL for persistence."
        )
        self.assertEqual(
            resolve_spoken_corrections(multi_sentence),
            "Let's meet at six thirty. Use PostgreSQL for persistence.",
        )

    def test_actually_and_sorry_used_normally_preserved(self):
        """Normal grammatical uses of 'actually', 'sorry', 'instead', and 'rather' must NOT be deleted."""
        normal_sentences = [
            "I actually really like the new architecture.",
            "She is actually the lead engineer on that service.",
            "I am sorry for the delay in responding to your message.",
            "Sorry to bother you during the sprint planning meeting.",
            "We used PostgreSQL instead of Redis for the primary store.",
            "I would rather deploy tomorrow morning.",
        ]
        for sentence in normal_sentences:
            self.assertEqual(
                resolve_spoken_corrections(sentence),
                sentence,
                msg=f"Normal sentence was unexpectedly modified: {sentence}",
            )

    def test_ambiguous_correction_preserved_verbatim(self):
        """When a phrase with 'actually' or 'sorry' is ambiguous commentary, preserve user's wording."""
        ambiguous_sentences = [
            "I was looking at the logs, actually I'm not sure what caused the timeout.",
            "We talked about the roadmap, sorry I forgot to attach the spreadsheet.",
            "Let's check the metrics, actually it might be a network issue.",
        ]
        for sentence in ambiguous_sentences:
            self.assertEqual(
                resolve_spoken_corrections(sentence),
                sentence,
                msg=f"Ambiguous sentence should be preserved verbatim: {sentence}",
            )

    def test_session_preserves_raw_transcript_and_privacy_safe_logs(self):
        """DictationSession and AIBrain preserve raw_transcript internally while resolving spoken corrections."""
        raw_spoken = "Use Redis—actually use PostgreSQL for persistence."
        session = DictationSession(mode=SessionMode.CONTINUOUS)
        session.start()
        session.finalize(audio_path="dummy.wav")
        session.begin_transcription()
        session.complete_transcription(raw_spoken)

        # Raw transcript is preserved verbatim for recovery/debugging
        self.assertEqual(session.raw_transcript, raw_spoken)
        # Corrected transcript reflects the resolved intent
        self.assertEqual(session.corrected_transcript, "Use PostgreSQL for persistence.")

        # Verify no lifecycle event exposes raw or corrected transcript text
        for ev in session.events:
            for val in ev.metadata.values():
                self.assertNotIn("Redis", str(val))
                self.assertNotIn("PostgreSQL", str(val))

        # Verify AIBrain fallback pipeline preserves original raw_transcript while outputting corrected text
        vault = MagicMock()
        brain = AIBrain(vault=vault)
        brain._call_freellmapi_or_openai = MagicMock(return_value=(None, MagicMock(success=False, status_code=500, error="down", error_category=None)))
        brain._api_keys = []
        brain._call_local_llm = MagicMock(return_value=(None, MagicMock(success=False, error="down", error_category=None)))

        res = brain.polish_with_provider_fallbacks(raw_spoken)
        self.assertEqual(res.raw_transcript, raw_spoken)
        self.assertEqual(res.text, "Use PostgreSQL for persistence.")
        self.assertTrue(len(SUPPORTED_CORRECTION_PATTERNS) >= 4)


if __name__ == "__main__":
    unittest.main()
