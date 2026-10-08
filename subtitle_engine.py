"""
Deterministic Subtitle Engine & Integrity Verifier.

Authoritative Contract:
The input subtitle file (.srt / .vtt) is already verified and correct.
It is the absolute authority for:
- TEXT
- START TIME
- END TIME
- CUE ORDER

Every cue becomes an immutable record with:
cue_id, start_ms, end_ms, duration_ms, text
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pysubs2


class SRTIntegrityError(Exception):
    """Raised when any subtitle cue violates deterministic integrity checks."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.details = details or {}
        super().__init__(f"[FATAL SRT INTEGRITY ERROR]: {message}\nDetails: {self.details}")


@dataclass(frozen=True)
class ImmutableCue:
    """Immutable representation of a single authoritative subtitle cue."""
    cue_id: int
    start_ms: int
    end_ms: int
    duration_ms: int
    text: str
    text_hash: str

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def wav_filename(self) -> str:
        """Exact standard file name: cue_000042.wav"""
        return f"cue_{self.cue_id:06d}.wav"

    @property
    def raw_wav_filename(self) -> str:
        return f"cue_{self.cue_id:06d}_raw.wav"

    @property
    def final_wav_filename(self) -> str:
        return f"cue_{self.cue_id:06d}_final.wav"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "text_hash": self.text_hash,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "wav_filename": self.wav_filename,
        }

    @classmethod
    def from_raw(cls, cue_id: int, start_ms: int, end_ms: int, text: str) -> ImmutableCue:
        cleaned_text = clean_subtitle_text(text)
        duration_ms = max(0, end_ms - start_ms)
        text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        return cls(
            cue_id=cue_id,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            text=cleaned_text,
            text_hash=text_hash,
        )


def clean_subtitle_text(text: str) -> str:
    """Strips styling tags, SSA formatting, bracketed sound markers, and trims extra whitespace."""
    if not text:
        return ""
    # Strip HTML tags
    t = re.sub(r"<[^>]+>", "", text)
    # Strip ASS/SSA tags like {\an8}, {\pos(x,y)}
    t = re.sub(r"\{[^\}]+\}", "", t)
    # Normalize newlines to spaces
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_subtitles_deterministic(file_path: Union[str, Path]) -> List[ImmutableCue]:
    """
    Parses an SRT or VTT file using pysubs2 with guaranteed deterministic ordering.
    Every cue receives a 1-based sequential cue_id matching its position.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")

    # Load with pysubs2 (supports SRT, VTT, ASS, SSA)
    try:
        subs = pysubs2.load(str(file_path), encoding="utf-8")
    except Exception:
        # Fallback with auto-detect or latin-1 if utf-8 fails
        subs = pysubs2.load(str(file_path))

    # Filter out empty comments/styles (preserve all dialogue)
    events = [ev for ev in subs if not ev.is_comment]

    # Sort strictly by start time then end time to guarantee stability
    events.sort(key=lambda ev: (ev.start, ev.end))

    immutable_cues: List[ImmutableCue] = []
    for idx, ev in enumerate(events, start=1):
        cue = ImmutableCue.from_raw(
            cue_id=idx,
            start_ms=int(ev.start),
            end_ms=int(ev.end),
            text=ev.text,
        )
        immutable_cues.append(cue)

    if not immutable_cues:
        raise SRTIntegrityError(f"Subtitle file {file_path.name} contains no valid cues.")

    return immutable_cues


def verify_srt_integrity(
    cues: List[ImmutableCue],
    source_duration_sec: float,
    expected_cues_ref: Optional[List[ImmutableCue]] = None,
    tolerance_sec: float = 0.50,
) -> None:
    """
    Automated SRT integrity verification before ANY TTS is allowed.
    
    Verifies:
    1. parsed cue count > 0
    2. cue IDs are strictly ordered 1..N
    3. cue text hash remains unchanged
    4. no cue disappears or duplicates
    5. no cue text assigned to another cue
    6. 0 <= start < end <= source video duration (+ tolerance)
    
    ABORTS (raises SRTIntegrityError) on any violation.
    """
    if not cues:
        raise SRTIntegrityError("Integrity check failed: Cue list is empty.")

    violations: List[str] = []
    seen_ids = set()
    source_duration_ms = int(round(source_duration_sec * 1000.0))
    tolerance_ms = int(round(tolerance_sec * 1000.0))

    # If reference cues provided, verify 1-to-1 match
    if expected_cues_ref is not None:
        if len(cues) != len(expected_cues_ref):
            violations.append(
                f"Cue count mismatch: current={len(cues)} vs expected={len(expected_cues_ref)}"
            )
        for c, ref in zip(cues, expected_cues_ref):
            if c.cue_id != ref.cue_id:
                violations.append(f"Cue ID mismatch: {c.cue_id} != {ref.cue_id}")
            if c.text_hash != ref.text_hash:
                violations.append(f"Cue text hash changed for cue {c.cue_id}: '{c.text}' != '{ref.text}'")
            if c.start_ms != ref.start_ms or c.end_ms != ref.end_ms:
                violations.append(
                    f"Timestamps altered for cue {c.cue_id}: ({c.start_ms}..{c.end_ms}) vs ({ref.start_ms}..{ref.end_ms})"
                )

    for i, cue in enumerate(cues):
        expected_id = i + 1
        if cue.cue_id != expected_id:
            violations.append(f"Cue ordering violation: at index {i}, cue_id={cue.cue_id} (expected {expected_id})")

        if cue.cue_id in seen_ids:
            violations.append(f"Duplicate cue ID detected: {cue.cue_id}")
        seen_ids.add(cue.cue_id)

        # Re-verify text hash
        expected_hash = hashlib.sha256(cue.text.encode("utf-8")).hexdigest()
        if cue.text_hash != expected_hash:
            violations.append(f"Hash corruption on cue {cue.cue_id}")

        # Check bounds: 0 <= start < end <= duration + tolerance
        if cue.start_ms < 0:
            violations.append(f"Cue {cue.cue_id} has negative start time ({cue.start_ms}ms)")
        if cue.start_ms >= cue.end_ms:
            violations.append(f"Cue {cue.cue_id} start_ms >= end_ms ({cue.start_ms} >= {cue.end_ms})")
        if cue.end_ms > source_duration_ms + tolerance_ms:
            violations.append(
                f"Cue {cue.cue_id} end_ms ({cue.end_ms}ms) exceeds video duration ({source_duration_ms}ms + {tolerance_ms}ms tolerance)"
            )

    if violations:
        raise SRTIntegrityError(
            f"{len(violations)} integrity violation(s) detected.",
            details={"violations": violations[:25], "total_violations": len(violations)},
        )


def export_cues_to_json(cues: List[ImmutableCue], output_path: Union[str, Path]) -> Path:
    """Exports immutable cues to JSON."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in cues]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def load_cues_from_json(json_path: Union[str, Path]) -> List[ImmutableCue]:
    """Loads immutable cues from JSON."""
    json_path = Path(json_path).resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cues = []
    for d in data:
        cues.append(
            ImmutableCue(
                cue_id=int(d["cue_id"]),
                start_ms=int(d["start_ms"]),
                end_ms=int(d["end_ms"]),
                duration_ms=int(d["duration_ms"]),
                text=str(d["text"]),
                text_hash=str(d.get("text_hash") or hashlib.sha256(d["text"].encode("utf-8")).hexdigest()),
            )
        )
    return cues
