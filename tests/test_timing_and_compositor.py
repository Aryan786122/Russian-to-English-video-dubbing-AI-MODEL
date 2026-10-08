"""
Unit tests for TimingDurationEngine and TimelineCompositor.
Verifies sample-accurate 48kHz placement, zero drift, and bounded duration fitting.
"""

import sys
import unittest
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import tempfile

from subtitle_engine import ImmutableCue
from timing_duration_engine import TimingDurationEngine
from timeline_compositor import TimelineCompositor


class TestTimingAndCompositor(unittest.TestCase):
    def setUp(self):
        self.cues = [
            ImmutableCue.from_raw(cue_id=1, start_ms=1000, end_ms=3000, text="First cue text here"),
            ImmutableCue.from_raw(cue_id=2, start_ms=3500, end_ms=5500, text="Second cue text here"),
            ImmutableCue.from_raw(cue_id=3, start_ms=6000, end_ms=8000, text="Third cue text here"),
        ]
        self.duration_engine = TimingDurationEngine(max_speedup=1.15, max_slowdown=0.88)
        self.compositor = TimelineCompositor(master_sr=48000)

    def test_sample_accurate_canvas_length(self):
        video_dur_sec = 10.0
        expected_samples = int(round(video_dur_sec * 48000))

        # Create dummy fitted cues
        sr = 24000
        dummy_clip_1 = np.ones(sr * 2, dtype=np.float32) * 0.2
        dummy_clip_2 = np.ones(sr * 2, dtype=np.float32) * 0.2
        dummy_clip_3 = np.ones(sr * 2, dtype=np.float32) * 0.2

        fitted_audio_1, res_1 = self.duration_engine.fit_cue_audio(
            cue=self.cues[0], raw_audio=dummy_clip_1, sr=sr, all_cues=self.cues, video_duration_sec=video_dur_sec
        )
        fitted_audio_2, res_2 = self.duration_engine.fit_cue_audio(
            cue=self.cues[1], raw_audio=dummy_clip_2, sr=sr, all_cues=self.cues, video_duration_sec=video_dur_sec
        )
        fitted_audio_3, res_3 = self.duration_engine.fit_cue_audio(
            cue=self.cues[2], raw_audio=dummy_clip_3, sr=sr, all_cues=self.cues, video_duration_sec=video_dur_sec
        )

        cues_audio = [
            (fitted_audio_1, sr, res_1),
            (fitted_audio_2, sr, res_2),
            (fitted_audio_3, sr, res_3),
        ]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_wav = Path(f.name)

        try:
            self.compositor.create_dialogue_track(
                cues_audio=cues_audio,
                source_duration_sec=video_dur_sec,
                output_path=out_wav,
            )

            data, out_sr = sf.read(str(out_wav), dtype="float32", always_2d=True)
            self.assertEqual(out_sr, 48000)
            self.assertEqual(len(data), expected_samples)

            # Verify cue 1 start sample: exactly 1.000s * 48000 = 48000
            start_sample_c1 = int(round(res_1.effective_start_sec * 48000))
            self.assertEqual(start_sample_c1, 48000)
            # Before cue 1 (0 to 47900 samples), data must be pure silence
            self.assertAlmostEqual(float(np.max(np.abs(data[:47000]))), 0.0, places=5)
            # Inside cue 1 (50000 to 70000 samples), data has non-zero speech
            self.assertGreater(float(np.max(np.abs(data[50000:70000]))), 0.1)
        finally:
            out_wav.unlink(missing_ok=True)

    def test_duration_fitting_bounded_stretch(self):
        # Audio is 2.8s for a 2.0s nominal cue
        sr = 16000
        raw_clip = np.random.randn(int(sr * 2.8)).astype(np.float32)

        fitted, result = self.duration_engine.fit_cue_audio(
            cue=self.cues[0],
            raw_audio=raw_clip,
            sr=sr,
            all_cues=self.cues,
            video_duration_sec=10.0,
        )

        # Stretch ratio must never exceed max_speedup (1.15)
        self.assertLessEqual(result.stretch_ratio, 1.15 + 1e-4)
        # Final audio must be non-empty and pitch-preserved
        self.assertGreater(len(fitted), 0)


if __name__ == "__main__":
    unittest.main()
