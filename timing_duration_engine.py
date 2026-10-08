"""
Pause Preservation & Timing/Duration Fitting Engine.

Specification:
- SRT is authoritative for start, end, and nominal duration.
- NEVER solves duration by cutting audio, chopping final words, or dropping phrases.
- Priority:
  1. Preserve every word
  2. Natural voice
  3. Natural timing
  4. Cue boundary precision
- Borrows safe adjacent silence before/after when free space permits (no dialogue collisions).
- Safe high-quality pitch-preserving time-scale modification strictly bounded to +-10-15%.
- Preserves natural pauses based on SRT punctuation and source performance.
- Flags cues for human review if they cannot be naturally fitted without exceeding bounds.
- Stores qc/duration_qc.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from performance_analyzer import CuePerformanceProfile
from subtitle_engine import ImmutableCue


@dataclass
class CueDurationFittingResult:
    cue_id: int
    nominal_start_sec: float
    nominal_end_sec: float
    nominal_duration_sec: float
    raw_duration_sec: float
    final_duration_sec: float
    borrowed_before_sec: float
    borrowed_after_sec: float
    stretch_ratio: float
    effective_start_sec: float
    status: str              # "PERFECT", "BORROWED_SILENCE", "STRETCHED_SAFE", "FLAGGED_FOR_REVIEW"
    flag_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "nominal_start_sec": round(self.nominal_start_sec, 3),
            "nominal_end_sec": round(self.nominal_end_sec, 3),
            "nominal_duration_sec": round(self.nominal_duration_sec, 3),
            "raw_duration_sec": round(self.raw_duration_sec, 3),
            "final_duration_sec": round(self.final_duration_sec, 3),
            "borrowed_before_sec": round(self.borrowed_before_sec, 3),
            "borrowed_after_sec": round(self.borrowed_after_sec, 3),
            "stretch_ratio": round(self.stretch_ratio, 3),
            "effective_start_sec": round(self.effective_start_sec, 3),
            "status": self.status,
            "flag_reason": self.flag_reason,
        }


class TimingDurationEngine:
    def __init__(
        self,
        max_speedup: float = 1.15,
        max_slowdown: float = 0.88,
        max_borrow_before_sec: float = 0.35,
        max_borrow_after_sec: float = 0.60,
    ):
        self.max_speedup = max_speedup
        self.max_slowdown = max_slowdown
        self.max_borrow_before_sec = max_borrow_before_sec
        self.max_borrow_after_sec = max_borrow_after_sec

    def fit_cue_audio(
        self,
        cue: ImmutableCue,
        raw_audio: np.ndarray,
        sr: int,
        all_cues: List[ImmutableCue],
        video_duration_sec: float,
        performance_profile: Optional[CuePerformanceProfile] = None,
    ) -> Tuple[np.ndarray, CueDurationFittingResult]:
        """
        Iterative duration fitting:
        1. Natural measurement
        2. Safe silence borrowing from adjacent non-colliding intervals
        3. High-quality pitch-preserving time-scale modification (within +-15%)
        4. Never chops words. Flags for review if impossible.
        """
        nominal_dur = cue.duration_sec
        raw_dur = len(raw_audio) / sr

        # Find available adjacent silence without collision
        idx = cue.cue_id - 1
        prev_end = all_cues[idx - 1].end_sec if idx > 0 else 0.0
        next_start = all_cues[idx + 1].start_sec if idx + 1 < len(all_cues) else video_duration_sec

        free_before = max(0.0, cue.start_sec - prev_end)
        free_after = max(0.0, next_start - cue.end_sec)

        borrow_before = 0.0
        borrow_after = 0.0
        stretch_ratio = 1.0
        status = "PERFECT"
        flag_reason = ""
        effective_start = cue.start_sec
        final_audio = raw_audio.copy()

        # Case 1: Naturally fits within nominal cue (+- 8%)
        if 0.90 * nominal_dur <= raw_dur <= nominal_dur + 0.05:
            final_audio = raw_audio
            final_dur = len(final_audio) / sr
            status = "PERFECT"

        # Case 2: Generated speech is longer than nominal cue
        elif raw_dur > nominal_dur:
            excess_sec = raw_dur - nominal_dur

            # 2a: Try borrowing safe adjacent silence first
            avail_borrow_before = min(free_before * 0.45, self.max_borrow_before_sec)
            avail_borrow_after = min(free_after * 0.65, self.max_borrow_after_sec)
            total_borrow_avail = avail_borrow_before + avail_borrow_after

            if total_borrow_avail >= excess_sec:
                # Can fit entirely by borrowing silence without any time-stretch!
                borrow_before = min(avail_borrow_before, excess_sec * 0.4)
                borrow_after = excess_sec - borrow_before
                effective_start = cue.start_sec - borrow_before
                status = "BORROWED_SILENCE"
            else:
                # Borrow maximum safe silence, then apply high-quality time-stretch to remainder
                borrow_before = avail_borrow_before
                borrow_after = avail_borrow_after
                effective_start = cue.start_sec - borrow_before

                budget_dur = nominal_dur + borrow_before + borrow_after
                needed_speedup = raw_dur / budget_dur

                if needed_speedup <= self.max_speedup:
                    # Safe stretch within limit
                    stretch_ratio = needed_speedup
                    final_audio = self._pitch_preserving_stretch(raw_audio, rate=stretch_ratio)
                    status = "STRETCHED_SAFE"
                else:
                    # Cannot fit even with max speedup and silence borrowing without sounding unnatural
                    # Cap strictly at max_speedup to PRESERVE NATURAL VOICE and NEVER DROP WORDS
                    stretch_ratio = self.max_speedup
                    final_audio = self._pitch_preserving_stretch(raw_audio, rate=stretch_ratio)
                    status = "FLAGGED_FOR_REVIEW"
                    flag_reason = (
                        f"Speech duration ({raw_dur:.2f}s) exceeded nominal ({nominal_dur:.2f}s) "
                        f"beyond safe {self.max_speedup}x speedup limit. Preserved all words."
                    )

        # Case 3: Generated speech is shorter than nominal cue
        else:
            under_ratio = raw_dur / max(0.2, nominal_dur)
            # If extremely short compared to long cue, stretch slightly within safe slowdown
            if under_ratio < 0.70 and nominal_dur > 1.2:
                stretch_ratio = max(self.max_slowdown, under_ratio)
                final_audio = self._pitch_preserving_stretch(raw_audio, rate=stretch_ratio)
                status = "STRETCHED_SAFE"
            else:
                final_audio = raw_audio
                status = "PERFECT"

        final_dur = len(final_audio) / sr

        result = CueDurationFittingResult(
            cue_id=cue.cue_id,
            nominal_start_sec=cue.start_sec,
            nominal_end_sec=cue.end_sec,
            nominal_duration_sec=nominal_dur,
            raw_duration_sec=raw_dur,
            final_duration_sec=final_dur,
            borrowed_before_sec=borrow_before,
            borrowed_after_sec=borrow_after,
            stretch_ratio=stretch_ratio,
            effective_start_sec=effective_start,
            status=status,
            flag_reason=flag_reason,
        )
        return final_audio, result

    def _pitch_preserving_stretch(self, x: np.ndarray, rate: float) -> np.ndarray:
        """High-quality pitch-preserving time-scale modification via librosa/phase-vocoder."""
        if abs(rate - 1.0) < 0.005:
            return x
        # librosa expects float32
        x_f32 = x.astype(np.float32)
        try:
            stretched = librosa.effects.time_stretch(x_f32, rate=rate)
            return stretched.astype(np.float32)
        except Exception as e:
            print(f"[TimingDurationEngine] Time-stretch warning ({e}), falling back to linear resample", flush=True)
            from scipy.signal import resample
            n_out = int(round(len(x) / rate))
            return resample(x, n_out).astype(np.float32)

    def export_duration_qc(self, results: List[CueDurationFittingResult], output_dir: str | Path) -> Path:
        out_dir = Path(output_dir).resolve()
        qc_dir = out_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        report_path = qc_dir / "duration_qc.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        print(f"[TimingDurationEngine] Exported duration QC report to {report_path}", flush=True)
        return report_path
