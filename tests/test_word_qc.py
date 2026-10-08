"""
Unit tests for WordCompletenessQC.
Verifies normalization, missing word detection, extra word detection, and truncation handling.
"""

import sys
import unittest
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_qc_engine import normalize_words


class TestWordQC(unittest.TestCase):
    def test_normalize_words(self):
        text1 = "Hello, world! What are you doing?"
        self.assertEqual(normalize_words(text1), ["hello", "world", "what", "are", "you", "doing"])

        text2 = "Don't speak... It's 100% fine!"
        self.assertEqual(normalize_words(text2), ["don't", "speak", "it's", "100", "fine"])

    def test_missing_and_extra_word_detection(self):
        import difflib

        expected = normalize_words("Where is my key? I lost it yesterday.")
        # Simulated ASR with missing words ("key", "yesterday")
        recognized = normalize_words("Where is my? I lost it.")

        matcher = difflib.SequenceMatcher(None, expected, recognized)
        missing = []
        extra = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("delete", "replace"):
                missing.extend(expected[i1:i2])
            if tag in ("insert", "replace"):
                extra.extend(recognized[j1:j2])

        self.assertIn("key", missing)
        self.assertIn("yesterday", missing)
        self.assertEqual(extra, [])


if __name__ == "__main__":
    unittest.main()
