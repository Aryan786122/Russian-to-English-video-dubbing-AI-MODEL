import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from timeline_validator import TimelineCorruptionError


@dataclass
class SubtitleCue:
    """Represents a single subtitle cue/segment."""
    id: int
    start: float
    end: float
    text: str
    duration: float = field(init=False)
    speaker_id: Union[str, None] = None
    confidence: Union[float, None] = None
    flagged: bool = False
    flag_reason: str = ""

    def __post_init__(self):
        self.duration = self.end - self.start


def _parse_time(time_str: str) -> float:
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def parse_srt(path: Union[str, Path]) -> list[SubtitleCue]:
    """Parse a standard SRT format file into a list of SubtitleCues."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        content = f.read()

    # Split by double newline to get individual blocks
    blocks = re.split(r'\r?\n\r?\n', content.strip())
    cues = []
    
    time_re = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        try:
            # Index line is optional in some exported SRTs
            if time_re.search(lines[0]):
                cue_id = len(cues) + 1
                time_line, text_lines = lines[0], lines[1:]
            else:
                try:
                    cue_id = int(lines[0])
                except ValueError:
                    cue_id = len(cues) + 1
                time_line, text_lines = lines[1], lines[2:]

            time_match = time_re.search(time_line)
            if not time_match or not text_lines:
                continue

            start_time = _parse_time(time_match.group(1))
            end_time = _parse_time(time_match.group(2))

            text = " ".join(text_lines)
            text = _strip_html(text).strip()
            if not text:
                continue

            cues.append(SubtitleCue(
                id=cue_id,
                start=start_time,
                end=end_time,
                text=text
            ))
        except Exception:
            continue

    return cues


def parse_vtt(path: Union[str, Path]) -> list[SubtitleCue]:
    """Parse a WebVTT format file into a list of SubtitleCues."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines()]
    if not lines or not lines[0].startswith("WEBVTT"):
        raise ValueError("Not a valid WebVTT file")

    cues = []
    cue_id_counter = 1
    
    i = 1
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
            
        # VTT can have an optional ID before the timestamp
        time_match = re.search(r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s*-->\s*(\d{2}:)?\d{2}:\d{2}\.\d{3}", lines[i])
        
        cue_id = cue_id_counter
        
        if not time_match and i + 1 < len(lines):
            # Maybe the current line is an ID
            next_time_match = re.search(r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s*-->\s*(\d{2}:)?\d{2}:\d{2}\.\d{3}", lines[i+1])
            if next_time_match:
                time_match = next_time_match
                i += 1 # Skip ID line
        
        if time_match:
            time_str = time_match.group(0)
            parts = [p.strip() for p in time_str.split("-->")]
            if len(parts) == 2:
                try:
                    start_time = _parse_time(parts[0])
                    end_time = _parse_time(parts[1])
                    
                    text_lines = []
                    i += 1
                    while i < len(lines) and lines[i] and not re.search(r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s*-->\s*(\d{2}:)?\d{2}:\d{2}\.\d{3}", lines[i]):
                        text_lines.append(lines[i])
                        i += 1
                        
                    text = " ".join(text_lines)
                    text = _strip_html(text)
                    
                    cues.append(SubtitleCue(
                        id=cue_id,
                        start=start_time,
                        end=end_time,
                        text=text
                    ))
                    cue_id_counter += 1
                    continue
                except Exception:
                    pass
        i += 1

    return cues


def parse_subtitle_file(path: Union[str, Path]) -> list[SubtitleCue]:
    """Auto-detect format and parse subtitle file (.srt or .vtt)."""
    path = Path(path)
    if path.suffix.lower() == ".srt":
        return parse_srt(path)
    elif path.suffix.lower() == ".vtt":
        return parse_vtt(path)
    else:
        raise ValueError(f"Unsupported subtitle format: {path.suffix}")


def validate_subtitle_timeline(cues: list[SubtitleCue], source_duration: float, tolerance: float = 0.20) -> list[SubtitleCue]:
    """
    Validate subtitle timestamps against the source duration.
    Flags invalid cues and raises TimelineCorruptionError if any cue exceeds bounds.
    """
    has_errors = False
    
    for cue in cues:
        if cue.start < 0 or cue.end < 0:
            cue.flagged = True
            cue.flag_reason = "Negative timestamp"
            has_errors = True
        elif cue.start >= cue.end:
            cue.flagged = True
            cue.flag_reason = "Start time >= end time"
            has_errors = True
        elif cue.end > source_duration + tolerance:
            cue.flagged = True
            cue.flag_reason = f"End time ({cue.end}) exceeds source duration ({source_duration}) + tolerance ({tolerance})"
            has_errors = True
            
    if has_errors:
        bad = [
            {"id": c.id, "start": c.start, "end": c.end, "reason": c.flag_reason}
            for c in cues if c.flagged
        ]
        raise TimelineCorruptionError(
            stage="subtitle_validation",
            message=f"{len(bad)} subtitle cue(s) violate 0 <= start < end <= SOURCE_DURATION ({source_duration:.3f}s) + {tolerance}s",
            details={"bad_cues": bad[:20]},
        )
        
    return cues


def cue_from_dict(d: dict) -> "SubtitleCue":
    """Rebuild a SubtitleCue from asdict() output (duration is derived, not passed)."""
    cue = SubtitleCue(id=int(d["id"]), start=float(d["start"]), end=float(d["end"]), text=str(d.get("text", "")))
    cue.speaker_id = d.get("speaker_id")
    cue.confidence = d.get("confidence")
    cue.flagged = bool(d.get("flagged", False))
    cue.flag_reason = d.get("flag_reason") or ""
    return cue


def _format_time_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    # Handle rounding edge case where millis rounds up to 1000
    if millis == 1000:
        secs += 1
        millis = 0
    if secs == 60:
        minutes += 1
        secs = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cues_to_srt(cues: list[SubtitleCue], output_path: Union[str, Path]) -> Path:
    """Write a list of SubtitleCues to an SRT format file."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            start_str = _format_time_srt(cue.start)
            end_str = _format_time_srt(cue.end)
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{cue.text}\n\n")
    return output_path


def get_subtitle_hash(path: Union[str, Path]) -> str:
    """Calculate SHA256 hash of the subtitle file for caching."""
    path = Path(path)
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
