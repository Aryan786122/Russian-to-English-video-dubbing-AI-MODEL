import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_engine import (
    ImmutableCue,
    SRTIntegrityError,
    clean_subtitle_text,
    parse_subtitles_deterministic,
    verify_srt_integrity,
)


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,500
Hello, world!

2
00:00:04,200 --> 00:00:06,800
This is cue number two.

3
00:00:07,100 --> 00:00:10,000
Third cue with <i>formatting</i> tags.
"""


class TestSRTEngine(unittest.TestCase):
    def test_clean_subtitle_text(self):
        self.assertEqual(clean_subtitle_text("<i>Hello</i> <b>world!</b>"), "Hello world!")
        self.assertEqual(clean_subtitle_text("Line 1\nLine 2"), "Line 1 Line 2")
        self.assertEqual(clean_subtitle_text("{\\an8}Top of screen"), "Top of screen")

    def test_parse_and_verify_valid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(SAMPLE_SRT)
            srt_path = Path(f.name)

        try:
            cues = parse_subtitles_deterministic(srt_path)
            self.assertEqual(len(cues), 3)

            self.assertEqual(cues[0].cue_id, 1)
            self.assertEqual(cues[0].start_ms, 1000)
            self.assertEqual(cues[0].end_ms, 3500)
            self.assertEqual(cues[0].duration_ms, 2500)
            self.assertEqual(cues[0].text, "Hello, world!")
            self.assertEqual(cues[0].wav_filename, "cue_000001.wav")

            self.assertEqual(cues[2].text, "Third cue with formatting tags.")

            # Verification against a 12-second video
            verify_srt_integrity(cues, source_duration_sec=12.0)
        finally:
            srt_path.unlink(missing_ok=True)

    def test_verify_srt_integrity_aborts_on_violations(self):
        # 1. Negative timestamp
        bad_cues = [
            ImmutableCue.from_raw(cue_id=1, start_ms=-100, end_ms=2000, text="Bad start")
        ]
        with self.assertRaises(SRTIntegrityError):
            verify_srt_integrity(bad_cues, source_duration_sec=10.0)

        # 2. Start >= End
        bad_cues_2 = [
            ImmutableCue.from_raw(cue_id=1, start_ms=2500, end_ms=2000, text="Inverted")
        ]
        with self.assertRaises(SRTIntegrityError):
            verify_srt_integrity(bad_cues_2, source_duration_sec=10.0)

        # 3. Exceeds duration
        bad_cues_3 = [
            ImmutableCue.from_raw(cue_id=1, start_ms=1000, end_ms=15000, text="Way too long")
        ]
        with self.assertRaises(SRTIntegrityError):
            verify_srt_integrity(bad_cues_3, source_duration_sec=10.0, tolerance_sec=0.1)

        # 4. Out of order IDs
        bad_cues_4 = [
            ImmutableCue.from_raw(cue_id=2, start_ms=1000, end_ms=2000, text="Second first"),
            ImmutableCue.from_raw(cue_id=1, start_ms=3000, end_ms=4000, text="First second"),
        ]
        with self.assertRaises(SRTIntegrityError):
            verify_srt_integrity(bad_cues_4, source_duration_sec=10.0)


if __name__ == "__main__":
    unittest.main()
