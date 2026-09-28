"""
test_clean.py -- Unit tests for LocalLLMEngine model output cleaner.
"""

import sys
import os

# Add parent directory to path so local_llm can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_llm import LocalLLMEngine


def test_clean_model_output():
    tests = [
        # (raw_model_output, raw_speech_input, expected_cleaned_output)
        ("Here is the polished text: Hello world.", "Hello world.", "Hello world."),
        ("Sure! The weather is nice.", "The weather is nice.", "The weather is nice."),
        ("1.5 million people attended the event.", "1.5 million people attended the event.", "1.5 million people attended the event."),
        ("First, we need to inspect the code.", "First, we need to inspect the code.", "First, we need to inspect the code."),
        ("Let me know if you need anything else.", "Let me know if you need anything else.", "Let me know if you need anything else."),
        ("You can press Right Alt to start recording.", "You can press Right Alt to start recording.", "You can press Right Alt to start recording."),
        ("As an AI language model, I cannot fulfill this request.", "Tell me a joke about computers", "Tell me a joke about computers."),
    ]

    passed = 0
    failed = 0

    for idx, (raw_out, raw_in, expected) in enumerate(tests, 1):
        cleaned = LocalLLMEngine._clean_model_output(raw_out, raw_text=raw_in)
        if cleaned == expected:
            print(f"[PASS] Case {idx}: {repr(raw_out)} -> {repr(cleaned)}")
            passed += 1
        else:
            print(f"[FAIL] Case {idx}: expected {repr(expected)}, got {repr(cleaned)}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} test cases.")
    assert failed == 0, f"{failed} test cases failed!"


if __name__ == "__main__":
    test_clean_model_output()
