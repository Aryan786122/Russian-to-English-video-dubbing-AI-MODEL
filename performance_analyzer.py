"""
Source Performance & Delivery Analyzer.

Specification:
- Inspects original source audio for every subtitle cue.
- Estimates:
  * Speech onset & offset within cue bounds
  * Internal pause structure (durations & locations)
  * Pitch / F0 statistics & contour (using librosa.pyin)
  * RMS energy & loudness contour
  * Speaking rate & speech density
  * Expressive delivery character (excited, neutral, quiet, angry, amused, whisper)
  * Non-verbal detection (laughter, gasp, breath)
- NEVER modifies subtitle text.
- Stores analysis/cue_performance.json to guide English TTS delivery & pacing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from subtitle_engine import ImmutableCue


@dataclass
class CuePerformanceProfile:
    cue_id: int
    start_sec: float
    end_sec: float
    speech_onset_sec: float
    speech_offset_sec: float
    internal_pauses_ms: List[Dict[str, Any]]
    speech_density: float
    mean_f0_hz: float
    f0_std: float
    f0_contour: str          # "rising", "falling", "dynamic", "flat"
    mean_rms: float
    peak_rms: float
    energy_level: str        # "high", "moderate", "low"
    speaking_rate_cps: float # chars or syllables per second
    delivery_character: str  # "excited", "angry", "quiet", "whisper", "amused", "neutral"
    has_laugh: bool = False
    has_gasp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "speech_onset_sec": round(self.speech_onset_sec, 3),
            "speech_offset_sec": round(self.speech_offset_sec, 3),
            "internal_pauses_ms": self.internal_pauses_ms,
            "speech_density": round(self.speech_density, 3),
            "mean_f0_hz": round(self.mean_f0_hz, 1),
            "f0_std": round(self.f0_std, 1),
            "f0_contour": self.f0_contour,
            "mean_rms": round(self.mean_rms, 4),
            "peak_rms": round(self.peak_rms, 4),
            "energy_level": self.energy_level,
            "speaking_rate_cps": round(self.speaking_rate_cps, 2),
            "delivery_character": self.delivery_character,
            "has_laugh": self.has_laugh,
            "has_gasp": self.has_gasp,
        }


class PerformanceAnalyzer:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def analyze_all_cues(
        self,
        audio_path: str | Path,
        cues: List[ImmutableCue],
        output_dir: str | Path,
        vocals_path: Optional[str | Path] = None,
    ) -> List[CuePerformanceProfile]:
        """
        Analyzes performance dynamics of each cue from source audio (preferably vocals stem).
        """
        output_dir = Path(output_dir).resolve()
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        source_to_use = Path(vocals_path if (vocals_path and Path(vocals_path).exists()) else audio_path).resolve()
        print(f"[PerformanceAnalyzer] Analyzing performance dynamics from {source_to_use.name}...", flush=True)

        audio_data, sr = sf.read(str(source_to_use), dtype="float32", always_2d=True)
        audio_mono = np.mean(audio_data, axis=1)

        # Resample to target_sr if needed
        if sr != self.target_sr:
            g = math.gcd(sr, self.target_sr)
            audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        total_samples = len(audio_mono)
        profiles: List[CuePerformanceProfile] = []

        for cue in cues:
            start_samp = int(round(cue.start_sec * sr))
            end_samp = int(round(cue.end_sec * sr))
            if start_samp >= total_samples or end_samp <= start_samp:
                # Fallback neutral profile
                profiles.append(self._default_profile(cue))
                continue

            clip = audio_mono[start_samp:end_samp]
            profile = self._analyze_single_clip(cue, clip, sr)
            profiles.append(profile)

        # Export cue_performance.json
        out_file = analysis_dir / "cue_performance.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2)

        print(f"[PerformanceAnalyzer] Analyzed {len(profiles)} cue performances. Saved to {out_file}", flush=True)
        return profiles

    def _analyze_single_clip(self, cue: ImmutableCue, clip: np.ndarray, sr: int) -> CuePerformanceProfile:
        dur = len(clip) / sr
        if dur < 0.20 or np.max(np.abs(clip)) < 0.005:
            return self._default_profile(cue)

        # Energy & RMS
        frame_len = int(sr * 0.03)  # 30ms
        hop_len = int(sr * 0.01)    # 10ms
        rms = librosa.feature.rms(y=clip, frame_length=frame_len, hop_length=hop_len)[0]
        mean_rms = float(np.mean(rms))
        peak_rms = float(np.max(rms)) if len(rms) > 0 else 0.0

        # Onset / offset detection inside cue
        noise_floor = float(np.percentile(rms, 15))
        thresh = max(noise_floor * 2.0, 0.015)
        active_idx = np.where(rms > thresh)[0]

        if len(active_idx) > 0:
            speech_onset = cue.start_sec + (active_idx[0] * hop_len / sr)
            speech_offset = cue.start_sec + (active_idx[-1] * hop_len / sr)
            speech_density = len(active_idx) / len(rms)
        else:
            speech_onset = cue.start_sec
            speech_offset = cue.end_sec
            speech_density = 0.5

        # Detect internal pauses (gaps > 120ms)
        internal_pauses = []
        if len(active_idx) > 1:
            diffs = np.diff(active_idx)
            for i, d in enumerate(diffs):
                gap_frames = d - 1
                gap_ms = int(gap_frames * hop_len * 1000 / sr)
                if gap_ms >= 120:
                    gap_time_sec = cue.start_sec + (active_idx[i] * hop_len / sr)
                    internal_pauses.append({
                        "offset_sec": round(gap_time_sec, 3),
                        "pause_ms": gap_ms
                    })

        # Pitch (F0) estimation using pyin (bounded for human speech 65Hz - 500Hz)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                clip,
                fmin=librosa.note_to_hz("C2"), # ~65Hz
                fmax=librosa.note_to_hz("C6"), # ~1046Hz
                sr=sr,
                frame_length=frame_len * 2,
                hop_length=hop_len,
            )
            voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
            if len(voiced_f0) > 3:
                mean_f0 = float(np.mean(voiced_f0))
                f0_std = float(np.std(voiced_f0))
                # Contour: compare first third vs last third of voiced speech
                n_v = len(voiced_f0)
                first_part = np.mean(voiced_f0[: n_v // 3])
                last_part = np.mean(voiced_f0[-n_v // 3 :])
                f0_diff = last_part - first_part
                if f0_diff > 25.0:
                    f0_contour = "rising"
                elif f0_diff < -25.0:
                    f0_contour = "falling"
                elif f0_std > 35.0:
                    f0_contour = "dynamic"
                else:
                    f0_contour = "flat"
            else:
                mean_f0 = 150.0
                f0_std = 10.0
                f0_contour = "flat"
        except Exception:
            mean_f0 = 150.0
            f0_std = 10.0
            f0_contour = "flat"

        # Energy level
        if mean_rms > 0.08:
            energy_level = "high"
        elif mean_rms < 0.02:
            energy_level = "low"
        else:
            energy_level = "moderate"

        # Speaking rate
        active_sec = max(0.3, (speech_offset - speech_onset))
        speaking_rate_cps = len(cue.text.strip()) / active_sec

        # Character / Emotion classification
        delivery = "neutral"
        if energy_level == "low" and mean_rms < 0.015:
            delivery = "whisper"
        elif energy_level == "low":
            delivery = "quiet"
        elif energy_level == "high" and f0_std > 40.0:
            delivery = "excited"
        elif energy_level == "high":
            delivery = "angry"
        elif f0_std > 30.0 and len(internal_pauses) >= 2:
            delivery = "amused"

        # Reaction flags
        has_laugh = (delivery == "amused") or ("ha" in cue.text.lower())
        has_gasp = (speech_onset - cue.start_sec > 0.20) and (energy_level == "high")

        return CuePerformanceProfile(
            cue_id=cue.cue_id,
            start_sec=cue.start_sec,
            end_sec=cue.end_sec,
            speech_onset_sec=speech_onset,
            speech_offset_sec=speech_offset,
            internal_pauses_ms=internal_pauses,
            speech_density=speech_density,
            mean_f0_hz=mean_f0,
            f0_std=f0_std,
            f0_contour=f0_contour,
            mean_rms=mean_rms,
            peak_rms=peak_rms,
            energy_level=energy_level,
            speaking_rate_cps=speaking_rate_cps,
            delivery_character=delivery,
            has_laugh=has_laugh,
            has_gasp=has_gasp,
        )

    def _default_profile(self, cue: ImmutableCue) -> CuePerformanceProfile:
        return CuePerformanceProfile(
            cue_id=cue.cue_id,
            start_sec=cue.start_sec,
            end_sec=cue.end_sec,
            speech_onset_sec=cue.start_sec,
            speech_offset_sec=cue.end_sec,
            internal_pauses_ms=[],
            speech_density=0.8,
            mean_f0_hz=150.0,
            f0_std=15.0,
            f0_contour="flat",
            mean_rms=0.04,
            peak_rms=0.08,
            energy_level="moderate",
            speaking_rate_cps=14.0,
            delivery_character="neutral",
            has_laugh=False,
            has_gasp=False,
        )
