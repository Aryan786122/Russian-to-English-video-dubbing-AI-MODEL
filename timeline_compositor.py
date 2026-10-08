"""
Absolute Sample-Accurate Timeline Compositor.

Specification:
- Creates final dialogue canvas at 48 kHz.
- Total length in samples strictly equals: round(source_video_duration_sec * 48000).
- Every cue is placed at its exact absolute sample:
    start_sample = round(start_seconds * 48000)
- NEVER calculates placement using accumulated duration or previous clip ends.
- Zero cumulative drift.
- Supports overlapping speakers seamlessly with multi-track layering.
- Applies soft headroom limiting to prevent clipping during overlapping lines.
- Exports audio/english_dialogue.wav.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from timing_duration_engine import CueDurationFittingResult


class TimelineCompositor:
    def __init__(self, master_sr: int = 48000):
        self.master_sr = master_sr

    def create_dialogue_track(
        self,
        cues_audio: List[Tuple[np.ndarray, int, CueDurationFittingResult]],
        source_duration_sec: float,
        output_path: str | Path,
    ) -> Path:
        """
        Renders all fitted cue audios into a sample-accurate 48 kHz stereo master canvas.
        Each cue is anchored to its absolute timestamp.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_samples = int(round(source_duration_sec * self.master_sr))
        print(f"[TimelineCompositor] Initializing 48kHz canvas: {total_samples} samples ({source_duration_sec:.3f}s)...", flush=True)

        # 2-channel stereo float32 canvas
        canvas = np.zeros((total_samples, 2), dtype=np.float32)
        overlap_detector = np.zeros(total_samples, dtype=np.int32)

        for audio_data, sr, fit_result in cues_audio:
            if len(audio_data) == 0:
                continue

            # Resample to 48kHz if not already
            if sr != self.master_sr:
                g = math.gcd(sr, self.master_sr)
                audio_48k = resample_poly(audio_data, self.master_sr // g, sr // g, axis=0).astype(np.float32)
            else:
                audio_48k = audio_data.astype(np.float32)

            # Ensure stereo shape (N, 2)
            if audio_48k.ndim == 1:
                stereo_clip = np.column_stack([audio_48k, audio_48k])
            elif audio_48k.shape[1] == 1:
                stereo_clip = np.repeat(audio_48k, 2, axis=1)
            else:
                stereo_clip = audio_48k[:, :2]

            # Calculate ABSOLUTE sample placement: round(effective_start_sec * 48000)
            start_sample = int(round(fit_result.effective_start_sec * self.master_sr))
            start_sample = max(0, start_sample)
            clip_len = len(stereo_clip)

            if start_sample >= total_samples:
                print(f"[TimelineCompositor] Warning: Cue {fit_result.cue_id} start sample exceeds canvas. Skipping.", flush=True)
                continue

            end_sample = min(total_samples, start_sample + clip_len)
            insert_len = end_sample - start_sample

            # Place directly on canvas (independent anchoring)
            canvas[start_sample:end_sample] += stereo_clip[:insert_len]
            overlap_detector[start_sample:end_sample] += 1

        # Check for overlapping regions and manage gain
        max_overlaps = int(np.max(overlap_detector)) if len(overlap_detector) > 0 else 0
        if max_overlaps > 1:
            print(f"[TimelineCompositor] Overlapping dialogue detected (max concurrency: {max_overlaps} speakers). Applying headroom management.", flush=True)

        # Transparent soft limiting to prevent digital clipping
        canvas = self._soft_limit_master(canvas, knee=0.88, ceiling=0.98)

        # Write 24-bit WAV file
        sf.write(str(output_path), canvas, self.master_sr, subtype="PCM_24")
        print(f"[TimelineCompositor] Rendered sample-accurate dialogue track to {output_path}", flush=True)
        return output_path

    def _soft_limit_master(self, x: np.ndarray, knee: float = 0.88, ceiling: float = 0.98) -> np.ndarray:
        peak = np.max(np.abs(x))
        if peak <= knee:
            return x

        a = np.abs(x)
        over = a > knee
        if np.any(over):
            span = ceiling - knee
            x[over] = np.sign(x[over]) * (knee + span * np.tanh((a[over] - knee) / span))

        final_peak = np.max(np.abs(x))
        if final_peak > 0.99:
            x = x * (0.98 / final_peak)
        return x
