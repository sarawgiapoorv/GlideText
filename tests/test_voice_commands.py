"""
test_voice_commands.py -- Comprehensive tests for GlideText's VoiceCommand layer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voice_commands import (
    InsertionHistory,
    InsertionRecord,
    VoiceCommand,
    VoiceCommandExecutor,
    VoiceCommandResult,
    VoiceCommandType,
    parse_voice_command,
)
from dictation_session import (
    DictationSession,
    DictationSessionCoordinator,
    SessionMode,
    SessionState,
)
from ai_brain import AIBrain, PipelineResult


class TestVoiceCommandParser(unittest.TestCase):
    """Test strict parsing of voice commands and false-positive immunity."""

    def test_scratch_that_variants(self):
        """'scratch that', 'please scratch that', 'scratch that.' recognized accurately."""
        cmd1 = parse_voice_command("scratch that")
        self.assertIsNotNone(cmd1)
        self.assertEqual(cmd1.command_type, VoiceCommandType.SCRATCH_THAT)

        cmd2 = parse_voice_command("Please scratch that.")
        self.assertIsNotNone(cmd2)
        self.assertEqual(cmd2.command_type, VoiceCommandType.SCRATCH_THAT)

        cmd3 = parse_voice_command('"scratch that"')
        self.assertIsNotNone(cmd3)
        self.assertEqual(cmd3.command_type, VoiceCommandType.SCRATCH_THAT)

    def test_delete_that_and_undo_variants(self):
        """'delete that', 'undo last dictation', 'clear last dictation' recognized."""
        cmd_del = parse_voice_command("delete that")
        self.assertIsNotNone(cmd_del)
        self.assertEqual(cmd_del.command_type, VoiceCommandType.DELETE_THAT)

        cmd_undo = parse_voice_command("undo last dictation")
        self.assertIsNotNone(cmd_undo)
        self.assertEqual(cmd_undo.command_type, VoiceCommandType.UNDO_LAST_DICTATION)

        cmd_clear = parse_voice_command("clear last dictation")
        self.assertIsNotNone(cmd_clear)
        self.assertEqual(cmd_clear.command_type, VoiceCommandType.CLEAR_LAST_DICTATION)

    def test_sentence_and_paragraph_deletion_commands(self):
        """'delete previous sentence' and 'delete previous paragraph' recognized."""
        cmd_s = parse_voice_command("delete previous sentence")
        self.assertIsNotNone(cmd_s)
        self.assertEqual(cmd_s.command_type, VoiceCommandType.DELETE_PREVIOUS_SENTENCE)

        cmd_p = parse_voice_command("delete previous paragraph")
        self.assertIsNotNone(cmd_p)
        self.assertEqual(cmd_p.command_type, VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH)

    def test_false_positive_immunity_embedded_sentences(self):
        """Phrases embedded in ordinary sentences must NEVER be executed as voice commands."""
        # 1. Narrative sentences with "scratch that"
        self.assertIsNone(parse_voice_command("I scratched that idea from the document."))
        self.assertIsNone(parse_voice_command("We can scratch that proposal for now."))
        self.assertIsNone(parse_voice_command("He said scratch that during the meeting."))

        # 2. Sentences with "delete that"
        self.assertIsNone(parse_voice_command("Don't delete that file from the server."))
        self.assertIsNone(parse_voice_command("Please delete that duplicate row in table two."))
        self.assertIsNone(parse_voice_command("I want to delete that line of code."))

        # 3. Sentences with "undo last dictation"
        self.assertIsNone(parse_voice_command("How do we undo last dictation bugs in Python?"))
        self.assertIsNone(parse_voice_command("The button will undo last dictation when clicked."))

        # 4. Long dictations
        self.assertIsNone(parse_voice_command("Let's schedule the meeting for 3 PM instead of 2 PM."))


class TestVoiceCommandExecutor(unittest.TestCase):
    """Test safe execution against InsertionHistory."""

    def setUp(self):
        self.history = InsertionHistory()
        self.executor = VoiceCommandExecutor(self.history)

    def test_no_previous_insertion_does_nothing_destructive(self):
        """If insertion history is empty, executor does nothing destructive and returns clean status."""
        cmd = VoiceCommand(command_type=VoiceCommandType.SCRATCH_THAT, raw_spoken_text="scratch that")

        with patch.object(self.executor, "_send_backspaces") as mock_backspace:
            res = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertFalse(res.executed)
            self.assertEqual(res.message, "No previous dictation to undo")
            mock_backspace.assert_not_called()

    def test_scratch_that_removes_most_recent_insertion(self):
        """'scratch that' removes exactly the characters from the last insertion."""
        self.history.record_insertion(session_id="s1", text="Hello world", target_hwnd=1001)

        cmd = VoiceCommand(command_type=VoiceCommandType.SCRATCH_THAT, raw_spoken_text="scratch that")

        with patch.object(self.executor, "_send_backspaces", return_value=True) as mock_backspace:
            res = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertTrue(res.success)
            self.assertTrue(res.executed)
            self.assertEqual(res.deleted_text, "Hello world")
            mock_backspace.assert_called_once_with(len("Hello world"))

        # History is now empty
        self.assertIsNone(self.history.peek())
        # Recovery stack has the deleted record
        recovery = self.history.get_recovery_records()
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0].text, "Hello world")

    def test_delete_previous_sentence(self):
        """'delete previous sentence' deletes only the last sentence of a multi-sentence insertion."""
        text = "First sentence was completed. Second sentence had a mistake."
        self.history.record_insertion(session_id="s1", text=text, target_hwnd=1001)

        cmd = VoiceCommand(
            command_type=VoiceCommandType.DELETE_PREVIOUS_SENTENCE,
            raw_spoken_text="delete previous sentence",
        )

        with patch.object(self.executor, "_send_backspaces", return_value=True) as mock_backspace:
            res = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertTrue(res.success)
            self.assertTrue(res.executed)
            self.assertEqual(res.deleted_text, "Second sentence had a mistake.")
            mock_backspace.assert_called_once()

        # Remaining insertion in history should be the first sentence
        remaining = self.history.peek()
        self.assertIsNotNone(remaining)
        self.assertEqual(remaining.text, "First sentence was completed.")

    def test_delete_previous_paragraph(self):
        """'delete previous paragraph' deletes the last paragraph in a multi-paragraph insertion."""
        text = "Paragraph one content.\n\nParagraph two content to remove."
        self.history.record_insertion(session_id="s1", text=text, target_hwnd=1001)

        cmd = VoiceCommand(
            command_type=VoiceCommandType.DELETE_PREVIOUS_PARAGRAPH,
            raw_spoken_text="delete previous paragraph",
        )

        with patch.object(self.executor, "_send_backspaces", return_value=True) as mock_backspace:
            res = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertTrue(res.success)
            self.assertTrue(res.executed)
            self.assertEqual(res.deleted_text, "Paragraph two content to remove.")
            mock_backspace.assert_called_once()

        # Remaining insertion is paragraph one
        self.assertEqual(self.history.peek().text, "Paragraph one content.")

    def test_target_window_mismatch_prevents_destructive_action(self):
        """If active window changed between insertion and command, undo is safely skipped."""
        self.history.record_insertion(session_id="s1", text="Secret notes", target_hwnd=1001)

        cmd = VoiceCommand(command_type=VoiceCommandType.SCRATCH_THAT, raw_spoken_text="scratch that")

        with patch.object(self.executor, "_send_backspaces") as mock_backspace:
            # Command arrives while user is in window 2002
            res = self.executor.execute_command(cmd, target_hwnd=2002)
            self.assertFalse(res.executed)
            self.assertIn("Active window changed", res.message)
            mock_backspace.assert_not_called()

        # Insertion is NOT lost or deleted
        self.assertEqual(self.history.peek().text, "Secret notes")

    def test_multiple_sequential_undos(self):
        """Multiple sequential 'scratch that' commands undo prior insertions in reverse order."""
        self.history.record_insertion(session_id="s1", text="Insertion A", target_hwnd=1001)
        self.history.record_insertion(session_id="s2", text="Insertion B", target_hwnd=1001)
        self.history.record_insertion(session_id="s3", text="Insertion C", target_hwnd=1001)

        cmd = VoiceCommand(command_type=VoiceCommandType.SCRATCH_THAT, raw_spoken_text="scratch that")

        with patch.object(self.executor, "_send_backspaces", return_value=True):
            res1 = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertEqual(res1.deleted_text, "Insertion C")

            res2 = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertEqual(res2.deleted_text, "Insertion B")

            res3 = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertEqual(res3.deleted_text, "Insertion A")

            res4 = self.executor.execute_command(cmd, target_hwnd=1001)
            self.assertFalse(res4.executed)
            self.assertEqual(res4.message, "No previous dictation to undo")


class TestVoiceCommandCoordinatorIntegration(unittest.TestCase):
    """Test voice command flow inside DictationSessionCoordinator."""

    def test_voice_command_skips_llm_polish_and_completes_session(self):
        """When user speaks 'scratch that', coordinator executes voice command and skips LLM polishing."""
        from voice_commands import GLOBAL_INSERTION_HISTORY, GLOBAL_VOICE_COMMAND_EXECUTOR

        GLOBAL_INSERTION_HISTORY.clear()
        GLOBAL_INSERTION_HISTORY.record_insertion(session_id="prev", text="To be deleted", target_hwnd=123)

        recorder = MagicMock()
        recorder.stop.return_value = "dummy.wav"
        brain = MagicMock(spec=AIBrain)
        brain._offline_transcribe.return_value = "scratch that"
        injector = MagicMock()

        coordinator = DictationSessionCoordinator(recorder=recorder, brain=brain, injector=injector)

        session = DictationSession(mode=SessionMode.PUSH_TO_TALK, context_info={"hwnd": 123})
        session.start()
        session.finalize(audio_path="dummy.wav")

        with patch.object(GLOBAL_VOICE_COMMAND_EXECUTOR, "_send_backspaces", return_value=True) as mock_backspace:
            ok = coordinator.process_session(session)
            self.assertTrue(ok)
            self.assertEqual(session.state, SessionState.DONE)
            mock_backspace.assert_called_once_with(len("To be deleted"))

        # LLM polishing must NOT have been called for the voice command
        brain.polish_with_provider_fallbacks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
