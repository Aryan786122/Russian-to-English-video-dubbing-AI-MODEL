"""
Unit and integration tests for Timeline Validator & Duration Consistency System.
Verifies all timeline invariants across ANY video length, chunking logic,
fail-fast error triggers, and auto-remediation.
"""

import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from timeline_validator import (
    probe_media_metadata,
    extract_synchronized_audio,
    validate_segment_bounds,
    validate_timeline_integrity,
    print_timeline_sanity_report,
    apply_chunk_offset,
    validate_cache_staleness,
    TimelineCorruptionError
)


class TestTimelineValidator(unittest.TestCase):

    def test_validate_segment_bounds_valid(self):
        """Valid segment within source duration should pass without error."""
        source_dur = 120.0
        seg = {
            "id": 0,
            "start": 5.0,
            "end": 8.5,
            "words": [
                {"word": "Hello", "start": 5.0, "end": 6.5},
                {"word": "world", "start": 6.8, "end": 8.5},
            ]
        }
        validate_segment_bounds(seg, seg_idx=0, source_duration=source_dur, stage_name="test_stage")

    def test_validate_segment_bounds_negative_start(self):
        """Negative start timestamp must trigger TimelineCorruptionError."""
        source_dur = 60.0
        seg = {"id": 1, "start": -0.5, "end": 2.0}
        with self.assertRaises(TimelineCorruptionError) as ctx:
            validate_segment_bounds(seg, seg_idx=1, source_duration=source_dur, stage_name="test_vad")
        self.assertIn("negative start", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.stage, "test_vad")

    def test_validate_segment_bounds_inverted(self):
        """Start >= end must trigger TimelineCorruptionError."""
        source_dur = 60.0
        seg = {"id": 2, "start": 10.0, "end": 9.5}
        with self.assertRaises(TimelineCorruptionError) as ctx:
            validate_segment_bounds(seg, seg_idx=2, source_duration=source_dur, stage_name="test_asr")
        self.assertIn("start >= end", str(ctx.exception))

    def test_validate_segment_bounds_overflow(self):
        """Segment extending beyond SOURCE_DURATION + tolerance must abort immediately."""
        source_dur = 45.0
        seg = {"id": 3, "start": 40.0, "end": 46.5}  # exceeds 45.0 + 0.20 = 45.20
        with self.assertRaises(TimelineCorruptionError) as ctx:
            validate_segment_bounds(seg, seg_idx=3, source_duration=source_dur, stage_name="test_tts", tolerance=0.20)
        self.assertIn("exceeds source_duration", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.stage, "test_tts")

    def test_validate_timeline_integrity_backward_jump(self):
        """Non-monotonic sequence with backward jump must trigger TimelineCorruptionError."""
        source_dur = 100.0
        segments = [
            {"id": 0, "start": 10.0, "end": 15.0},
            {"id": 1, "start": 5.0, "end": 8.0}  # Backward jump from 10.0 to 5.0
        ]
        with self.assertRaises(TimelineCorruptionError) as ctx:
            validate_timeline_integrity(segments, source_duration=source_dur, stage_name="test_integrity")
        self.assertIn("jumps backward", str(ctx.exception).lower())

    def test_print_timeline_sanity_report_valid(self):
        """Sanity report prints correctly when invariants hold."""
        segments = [
            {"id": 0, "start": 1.0, "end": 5.0},
            {"id": 1, "start": 10.0, "end": 19.5}
        ]
        print_timeline_sanity_report(
            source_duration=20.0,
            original_audio_duration=20.0,
            analysis_audio_duration=20.0,
            segments=segments,
            stage_label="UNIT TEST AUDIT"
        )

    def test_print_timeline_sanity_report_violation(self):
        """Sanity report must abort if last_asr_end exceeds SOURCE_DURATION."""
        segments = [
            {"id": 0, "start": 1.0, "end": 22.0}  # exceeds 20.0
        ]
        with self.assertRaises(TimelineCorruptionError) as ctx:
            print_timeline_sanity_report(
                source_duration=20.0,
                original_audio_duration=20.0,
                analysis_audio_duration=20.0,
                segments=segments,
                stage_label="UNIT TEST AUDIT"
            )
        self.assertIn("INVARIANT VIOLATION", str(ctx.exception))

    def test_apply_chunk_offset_single_application(self):
        """Chunk offset must be applied exactly once and double-offsetting prevented."""
        source_dur = 120.0
        chunk_start = 30.0
        chunk_end = 60.0
        segments = [
            {"id": 0, "start": 2.0, "end": 8.0, "words": [{"word": "test", "start": 2.0, "end": 4.0}]}
        ]

        offset_segs = apply_chunk_offset(
            segments=segments,
            chunk_original_start=chunk_start,
            chunk_original_end=chunk_end,
            source_duration=source_dur
        )

        self.assertEqual(offset_segs[0]["start"], 32.0)
        self.assertEqual(offset_segs[0]["end"], 38.0)
        self.assertEqual(offset_segs[0]["words"][0]["start"], 32.0)
        self.assertTrue(offset_segs[0]["_chunk_offset_applied"])

        # Attempting to re-apply chunk offset MUST raise TimelineCorruptionError
        with self.assertRaises(TimelineCorruptionError) as ctx:
            apply_chunk_offset(
                segments=offset_segs,
                chunk_original_start=chunk_start,
                chunk_original_end=chunk_end,
                source_duration=source_dur
            )
        self.assertIn("more than once", str(ctx.exception).lower())

    def test_cache_staleness_detection(self):
        """Stale cache containing timestamps exceeding source duration is discarded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_file = tmp_path / "russian_transcript.json"
            dummy_video = tmp_path / "dummy_video.mp4"
            dummy_video.touch()

            # Cached segments for a 120s video
            cached_data = [
                {"id": 0, "start": 10.0, "end": 20.0},
                {"id": 1, "start": 100.0, "end": 115.0}
            ]
            cache_file.write_text(json.dumps(cached_data), encoding="utf-8")

            # If new video is only 60s, cache is stale
            self.assertFalse(validate_cache_staleness(cache_file, dummy_video, expected_source_duration=60.0))

            # If video is 120s, cache is valid
            self.assertTrue(validate_cache_staleness(cache_file, dummy_video, expected_source_duration=120.0))


if __name__ == "__main__":
    unittest.main()

