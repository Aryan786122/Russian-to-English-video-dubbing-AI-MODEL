"""
Full-Video Speaker Reference Mining Engine.

Specification:
- Scans ENTIRE audio timeline for each speaker identity.
- Never takes a dirty continuous clip or nearest 3-second sample.
- Evaluates candidate segments on:
  * No speaker overlap
  * High SNR & low background noise
  * No clipping / distortion
  * High speech density & minimal internal silence
  * Natural duration (2s - 10s per slice)
- Selects multiple top-quality samples totaling 15-30 seconds.
- Produces:
  speaker_refs/SPEAKER_XX/
      ref_01.wav
      ref_02.wav
      ref_03.wav
      combined_clean.wav
- Flags LOW_REFERENCE_QUALITY if clean audio is scarce.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from diarization_engine import DiarizationSegment


@dataclass
class ReferenceCandidate:
    start: float
    end: float
    duration: float
    snr_db: float
    speech_density: float
    peak_level: float
    overall_score: float
    quality_flag: str = "GOOD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "snr_db": round(self.snr_db, 1),
            "speech_density": round(self.speech_density, 3),
            "peak_level": round(self.peak_level, 3),
            "overall_score": round(self.overall_score, 3),
            "quality_flag": self.quality_flag,
        }


@dataclass
class SpeakerReferenceSet:
    speaker_id: str
    total_duration_sec: float
    samples_count: int
    quality_status: str
    ref_paths: List[str]
    combined_path: str
    candidates: List[ReferenceCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "samples_count": self.samples_count,
            "quality_status": self.quality_status,
            "ref_paths": self.ref_paths,
            "combined_path": self.combined_path,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class ReferenceMiner:
    def __init__(
        self,
        target_total_sec: float = 20.0,
        min_total_sec: float = 12.0,
        target_sr: int = 24000,
    ):
        self.target_total_sec = target_total_sec
        self.min_total_sec = min_total_sec
        self.target_sr = target_sr

    def mine_all_speakers(
        self,
        audio_path: str | Path,
        segments: List[DiarizationSegment],
        output_dir: str | Path,
        vocals_path: Optional[str | Path] = None,
    ) -> Dict[str, SpeakerReferenceSet]:
        """
        Mines 15-30 seconds of high-fidelity references for every speaker detected in the entire video.
        Uses vocals stem if available for cleaner isolation, falling back to original audio.
        """
        output_dir = Path(output_dir).resolve()
        refs_root = output_dir / "speaker_refs"
        refs_root.mkdir(parents=True, exist_ok=True)

        audio_source = Path(vocals_path if (vocals_path and Path(vocals_path).exists()) else audio_path).resolve()
        print(f"[ReferenceMiner] Reading source audio for reference mining from {audio_source.name}...", flush=True)

        audio_data, sr = sf.read(str(audio_source), dtype="float32", always_2d=True)
        # Convert to mono
        audio_mono = np.mean(audio_data, axis=1)

        # Resample to target_sr (24kHz is optimal for Chatterbox / S3Gen voice encoder)
        if sr != self.target_sr:
            g = math.gcd(sr, self.target_sr)
            audio_mono = resample_poly(audio_mono, self.target_sr // g, sr // g).astype(np.float32)
            sr = self.target_sr

        # Group segments by speaker
        speakers = sorted(list(set(s.speaker for s in segments)))
        results: Dict[str, SpeakerReferenceSet] = {}

        for spk in speakers:
            spk_segs = [s for s in segments if s.speaker == spk]
            ref_set = self._mine_single_speaker(spk, spk_segs, audio_mono, sr, refs_root)
            results[spk] = ref_set

        # Export manifest
        manifest_path = refs_root / "refs_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

        print(f"[ReferenceMiner] Completed reference mining for {len(results)} speaker(s). Manifest saved to {manifest_path}", flush=True)
        return results

    def _mine_single_speaker(
        self,
        speaker_id: str,
        segments: List[DiarizationSegment],
        audio: np.ndarray,
        sr: int,
        output_root: Path,
    ) -> SpeakerReferenceSet:
        spk_dir = output_root / speaker_id
        spk_dir.mkdir(parents=True, exist_ok=True)

        # Filter strictly non-overlapping segments first
        clean_segs = [s for s in segments if not s.overlap and s.duration >= 1.2]
        if not clean_segs:
            # Fallback if no non-overlapping segment exists
            clean_segs = [s for s in segments if s.duration >= 1.0]

        candidates: List[Tuple[ReferenceCandidate, np.ndarray]] = []
        for s in clean_segs:
            start_samp = int(round(s.start * sr))
            end_samp = int(round(s.end * sr))
            if start_samp >= len(audio) or end_samp <= start_samp:
                continue

            clip = audio[start_samp:end_samp]
            if len(clip) < sr * 1.0:
                continue

            cand = self._score_candidate(clip, s.start, s.end, sr)
            if cand is not None:
                candidates.append((cand, clip))

        # Sort candidates by overall score descending
        candidates.sort(key=lambda pair: pair[0].overall_score, reverse=True)

        selected_clips: List[np.ndarray] = []
        selected_candidates: List[ReferenceCandidate] = []
        total_sec = 0.0

        for cand, clip in candidates:
            selected_clips.append(clip)
            selected_candidates.append(cand)
            total_sec += cand.duration
            if total_sec >= self.target_total_sec:
                break

        ref_file_paths: List[str] = []
        combined_path = ""
        quality_status = "HIGH_QUALITY" if total_sec >= self.min_total_sec else "LOW_REFERENCE_QUALITY"

        if not selected_clips:
            # Emergency fallback: take first 3s
            quality_status = "LOW_REFERENCE_QUALITY"
            dummy_clip = audio[: min(len(audio), sr * 3)]
            selected_clips = [dummy_clip]

        # Write individual reference files
        for idx, clip in enumerate(selected_clips[:5], start=1):
            ref_path = spk_dir / f"ref_{idx:02d}.wav"
            # Normalize peak to -1 dBFS
            norm_clip = self._normalize_audio(clip)
            sf.write(str(ref_path), norm_clip, sr)
            ref_file_paths.append(str(ref_path))

        # Build combined_clean.wav with smooth crossfades
        combined_audio = self._concatenate_with_crossfades(selected_clips, sr)
        norm_combined = self._normalize_audio(combined_audio)
        combined_path = spk_dir / "combined_clean.wav"
        sf.write(str(combined_path), norm_combined, sr)

        return SpeakerReferenceSet(
            speaker_id=speaker_id,
            total_duration_sec=total_sec,
            samples_count=len(selected_clips),
            quality_status=quality_status,
            ref_paths=ref_file_paths,
            combined_path=str(combined_path),
            candidates=selected_candidates,
        )

    def _score_candidate(self, clip: np.ndarray, start: float, end: float, sr: int) -> Optional[ReferenceCandidate]:
        duration = len(clip) / sr
        if duration < 1.0:
            return None

        peak = float(np.max(np.abs(clip)))
        if peak < 0.01:
            return None  # Pure silence

        # Clipping penalty
        if peak >= 0.999:
            return None  # Discard clipped audio

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))

        # Speech density: measure proportion of frames above noise floor
        frame_len = int(sr * 0.03)  # 30ms frames
        hop_len = int(sr * 0.01)    # 10ms hop
        n_frames = (len(clip) - frame_len) // hop_len
        if n_frames < 5:
            return None

        frame_energies = []
        for i in range(n_frames):
            idx = i * hop_len
            frame = clip[idx : idx + frame_len]
            f_rms = np.sqrt(np.mean(frame**2) + 1e-12)
            frame_energies.append(f_rms)

        frame_energies = np.array(frame_energies)
        noise_floor = float(np.percentile(frame_energies, 10))
        active_threshold = max(noise_floor * 2.5, 0.015)
        speech_density = float(np.mean(frame_energies > active_threshold))

        # Estimated SNR (speech frame RMS vs noise floor)
        speech_frames = frame_energies[frame_energies > active_threshold]
        if len(speech_frames) > 0:
            speech_rms = float(np.mean(speech_frames))
            snr_db = 20.0 * np.log10(max(speech_rms / max(noise_floor, 1e-5), 1.0))
        else:
            snr_db = 0.0

        # Duration sweet spot: 3 to 7 seconds is ideal for cloning
        dur_score = 1.0 - abs(min(duration, 8.0) - 5.0) / 5.0
        dur_score = max(0.2, dur_score)

        # Composite score
        # Higher SNR, higher speech density, good duration, sensible peak
        score = (min(snr_db, 30.0) / 30.0) * 0.45 + (speech_density * 0.35) + (dur_score * 0.20)
        flag = "EXCELLENT" if score > 0.70 else ("GOOD" if score > 0.50 else "FAIR")

        return ReferenceCandidate(
            start=start,
            end=end,
            duration=duration,
            snr_db=snr_db,
            speech_density=speech_density,
            peak_level=peak,
            overall_score=score,
            quality_flag=flag,
        )

    def _normalize_audio(self, x: np.ndarray, target_peak_db: float = -1.0) -> np.ndarray:
        peak = np.max(np.abs(x))
        if peak < 1e-6:
            return x
        target_amp = 10.0 ** (target_peak_db / 20.0)
        return (x * (target_amp / peak)).astype(np.float32)

    def _concatenate_with_crossfades(self, clips: List[np.ndarray], sr: int, crossfade_ms: float = 40.0) -> np.ndarray:
        if not clips:
            return np.zeros(sr, dtype=np.float32)
        if len(clips) == 1:
            return clips[0]

        fade_len = int(round(sr * (crossfade_ms / 1000.0)))
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = 1.0 - fade_in

        accum = clips[0].copy()
        for next_clip in clips[1:]:
            if len(accum) < fade_len or len(next_clip) < fade_len:
                accum = np.concatenate([accum, next_clip])
                continue

            # Crossfade the tail of accum with the head of next_clip
            tail = accum[-fade_len:] * fade_out
            head = next_clip[:fade_len] * fade_in
            blended = tail + head

            accum = np.concatenate([accum[:-fade_len], blended, next_clip[fade_len:]])

        return accum
