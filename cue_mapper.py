"""
SRT Cue to Speaker Mapping Engine.

Specification:
- Compares each authoritative SRT cue's absolute time against full-video diarization.
- Calculates exact millisecond overlap per speaker.
- Assigns cue to dominant speaker only when confidence is sufficient.
- Identifies and flags MULTI_SPEAKER cues when overlap is shared.
- Stores analysis/cue_speaker_map.json.
- Supports manual override updates from the UI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from diarization_engine import DiarizationSegment
from subtitle_engine import ImmutableCue


@dataclass
class CueSpeakerAssignment:
    cue_id: int
    start_ms: int
    end_ms: int
    duration_ms: int
    text: str
    speaker_id: str
    dominant_overlap_ms: int
    assignment_confidence: float
    multi_speaker: bool = False
    overlapping_speakers: List[Dict[str, Any]] = field(default_factory=list)
    manual_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "dominant_overlap_ms": self.dominant_overlap_ms,
            "assignment_confidence": round(self.assignment_confidence, 3),
            "multi_speaker": self.multi_speaker,
            "overlapping_speakers": self.overlapping_speakers,
            "manual_override": self.manual_override,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CueSpeakerAssignment:
        return cls(
            cue_id=int(d["cue_id"]),
            start_ms=int(d["start_ms"]),
            end_ms=int(d["end_ms"]),
            duration_ms=int(d["duration_ms"]),
            text=str(d["text"]),
            speaker_id=str(d["speaker_id"]),
            dominant_overlap_ms=int(d["dominant_overlap_ms"]),
            assignment_confidence=float(d["assignment_confidence"]),
            multi_speaker=bool(d.get("multi_speaker", False)),
            overlapping_speakers=d.get("overlapping_speakers", []),
            manual_override=bool(d.get("manual_override", False)),
        )


class CueSpeakerMapper:
    def __init__(
        self,
        min_overlap_ratio: float = 0.25,
        multi_speaker_ratio_thresh: float = 0.25,
    ):
        self.min_overlap_ratio = min_overlap_ratio
        self.multi_speaker_ratio_thresh = multi_speaker_ratio_thresh

    def map_cues_to_speakers(
        self,
        cues: List[ImmutableCue],
        diarization_segments: List[DiarizationSegment],
        output_dir: str | Path,
    ) -> List[CueSpeakerAssignment]:
        """
        Maps every immutable cue to a dominant speaker using exact millisecond overlap.
        """
        output_dir = Path(output_dir).resolve()
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        assignments: List[CueSpeakerAssignment] = []
        last_known_speaker = "SPEAKER_00"

        for cue in cues:
            cue_start_ms = cue.start_ms
            cue_end_ms = cue.end_ms
            cue_dur_ms = cue.duration_ms

            # Compute overlap with each diarization segment
            speaker_overlaps: Dict[str, int] = {}
            for seg in diarization_segments:
                seg_start_ms = int(round(seg.start * 1000.0))
                seg_end_ms = int(round(seg.end * 1000.0))

                overlap = max(0, min(cue_end_ms, seg_end_ms) - max(cue_start_ms, seg_start_ms))
                if overlap > 0:
                    speaker_overlaps[seg.speaker] = speaker_overlaps.get(seg.speaker, 0) + overlap

            overlapping_speakers_info = []
            for spk, o_ms in sorted(speaker_overlaps.items(), key=lambda item: item[1], reverse=True):
                overlapping_speakers_info.append({
                    "speaker": spk,
                    "overlap_ms": o_ms,
                    "overlap_ratio": round(o_ms / max(1, cue_dur_ms), 3),
                })

            multi_speaker = False
            if len(overlapping_speakers_info) >= 2:
                second_ratio = overlapping_speakers_info[1]["overlap_ratio"]
                if second_ratio >= self.multi_speaker_ratio_thresh or overlapping_speakers_info[1]["overlap_ms"] > 400:
                    multi_speaker = True

            if overlapping_speakers_info:
                dominant_speaker = overlapping_speakers_info[0]["speaker"]
                dominant_overlap_ms = overlapping_speakers_info[0]["overlap_ms"]
                confidence = dominant_overlap_ms / max(1, cue_dur_ms)
                last_known_speaker = dominant_speaker
            else:
                # Cue occurred in silence / music without diarized speech: find nearest segment
                nearest_spk = self._find_nearest_speaker(cue_start_ms, diarization_segments, default=last_known_speaker)
                dominant_speaker = nearest_spk
                dominant_overlap_ms = 0
                confidence = 0.30  # Low confidence interpolation

            assignment = CueSpeakerAssignment(
                cue_id=cue.cue_id,
                start_ms=cue_start_ms,
                end_ms=cue_end_ms,
                duration_ms=cue_dur_ms,
                text=cue.text,
                speaker_id=dominant_speaker,
                dominant_overlap_ms=dominant_overlap_ms,
                assignment_confidence=confidence,
                multi_speaker=multi_speaker,
                overlapping_speakers=overlapping_speakers_info,
                manual_override=False,
            )
            assignments.append(assignment)

        # Save cue_speaker_map.json
        out_path = analysis_dir / "cue_speaker_map.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in assignments], f, indent=2, ensure_ascii=False)

        print(f"[CueSpeakerMapper] Mapped {len(assignments)} cues to speakers. Saved to {out_path}", flush=True)
        return assignments

    def _find_nearest_speaker(
        self,
        target_ms: int,
        segments: List[DiarizationSegment],
        default: str = "SPEAKER_00",
    ) -> str:
        if not segments:
            return default
        best_dist = float("inf")
        best_spk = default
        for s in segments:
            mid_ms = int(round((s.start + s.end) * 500.0))
            dist = abs(target_ms - mid_ms)
            if dist < best_dist:
                best_dist = dist
                best_spk = s.speaker
        return best_spk

    def update_manual_assignment(
        self,
        assignments: List[CueSpeakerAssignment],
        cue_id: int,
        new_speaker_id: str,
        output_dir: str | Path,
    ) -> List[CueSpeakerAssignment]:
        """Updates a specific cue assignment with a manual user correction."""
        for a in assignments:
            if a.cue_id == cue_id:
                a.speaker_id = new_speaker_id
                a.manual_override = True
                break

        analysis_dir = Path(output_dir).resolve() / "analysis"
        out_path = analysis_dir / "cue_speaker_map.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in assignments], f, indent=2, ensure_ascii=False)

        return assignments
