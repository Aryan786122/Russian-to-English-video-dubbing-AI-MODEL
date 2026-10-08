from pathlib import Path
import math
from typing import List, Dict, Any, Optional

from timeline_validator import validate_timeline_integrity, TimelineCorruptionError


def seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT time format: HH:MM:SS,mmm"""
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """Convert float seconds to VTT time format: HH:MM:SS.mmm"""
    return seconds_to_srt_time(seconds).replace(",", ".")


def generate_srt(
    segments: list[dict],
    output_path: str | Path,
    source_duration: Optional[float] = None,
    tolerance: float = 0.20
) -> Path:
    """
    Generate standard SRT subtitle file from dialogue segments.
    Each segment should have 'start', 'end', and 'english' (or 'text').
    Validates timeline bounds against source_duration if provided.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_duration is not None and source_duration > 0:
        validate_timeline_integrity(
            segments=segments,
            source_duration=source_duration,
            stage_name="generate_srt",
            tolerance=tolerance,
            allow_empty=True
        )

    lines = []
    sub_index = 1
    for seg in segments:
        text = (seg.get("english") or seg.get("text") or "").strip()
        if not text:
            continue
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])

        start_str = seconds_to_srt_time(start_sec)
        end_str = seconds_to_srt_time(end_sec)

        speaker = seg.get("speaker_id")
        speaker_tag = f"[{speaker}] " if speaker else ""

        lines.append(f"{sub_index}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{speaker_tag}{text}")
        lines.append("")  # Blank line separator
        sub_index += 1

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def generate_vtt(
    segments: list[dict],
    output_path: str | Path,
    source_duration: Optional[float] = None,
    tolerance: float = 0.20
) -> Path:
    """
    Generate WebVTT subtitle file.
    Validates timeline bounds against source_duration if provided.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_duration is not None and source_duration > 0:
        validate_timeline_integrity(
            segments=segments,
            source_duration=source_duration,
            stage_name="generate_vtt",
            tolerance=tolerance,
            allow_empty=True
        )

    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        text = (seg.get("english") or seg.get("text") or "").strip()
        if not text:
            continue
        start_str = seconds_to_vtt_time(float(seg["start"]))
        end_str = seconds_to_vtt_time(float(seg["end"]))

        speaker = seg.get("speaker_id")
        speaker_tag = f"[{speaker}] " if speaker else ""

        lines.append(f"{i}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{speaker_tag}{text}")
        lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return output_path
