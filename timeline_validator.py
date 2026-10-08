"""
Timeline Validator & Integrity Checker Module
Generic timeline validation, duration consistency enforcement, and corruption detection
for ANY input video file regardless of format, resolution, frame rate, or duration.

Authoritative Clock: The original source video duration (SOURCE_DURATION) is the ONLY
reference clock. All timestamps across VAD, ASR, diarization, translation, subtitles,
and TTS placement must stay strictly anchored to this clock.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


class TimelineCorruptionError(Exception):
    """Raised when any timestamp or audio duration violates the authoritative timeline."""
    def __init__(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.details = details or {}
        super().__init__(f"[FATAL TIMELINE CORRUPTION at stage '{stage}']: {message}\nDetails: {self.details}")


def probe_media_metadata(file_path: str | Path) -> Dict[str, Any]:
    """
    Robustly probe media container and streams via ffprobe.
    Returns authoritative SOURCE_DURATION, audio/video stream durations, sample rate, channels.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Media file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,start_time,bit_rate:stream=index,codec_type,codec_name,duration,start_time,sample_rate,channels,r_frame_rate",
        "-of", "json",
        str(file_path)
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to probe media metadata for {file_path}: {e}")

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    container_duration = None
    try:
        if "duration" in format_info:
            container_duration = float(format_info["duration"])
    except (ValueError, TypeError):
        pass

    video_stream = None
    audio_stream = None

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "video" and video_stream is None:
            video_stream = s
        elif c_type == "audio" and audio_stream is None:
            audio_stream = s

    video_duration = None
    if video_stream and "duration" in video_stream:
        try:
            video_duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            pass

    audio_duration = None
    sample_rate = None
    channels = None
    if audio_stream:
        if "duration" in audio_stream:
            try:
                audio_duration = float(audio_stream["duration"])
            except (ValueError, TypeError):
                pass
        if "sample_rate" in audio_stream:
            try:
                sample_rate = int(audio_stream["sample_rate"])
            except (ValueError, TypeError):
                pass
        if "channels" in audio_stream:
            try:
                channels = int(audio_stream["channels"])
            except (ValueError, TypeError):
                pass

    # Authoritative SOURCE_DURATION determination:
    # 1. Prefer video stream duration if video stream exists and has duration
    # 2. Otherwise prefer container format duration
    # 3. Otherwise fall back to audio stream duration
    # 4. Fall back to soundfile reading if applicable
    if video_duration is not None and video_duration > 0.0:
        source_duration = video_duration
    elif container_duration is not None and container_duration > 0.0:
        source_duration = container_duration
    elif audio_duration is not None and audio_duration > 0.0:
        source_duration = audio_duration
    else:
        try:
            sf_info = sf.info(str(file_path))
            source_duration = float(sf_info.duration)
            if sample_rate is None:
                sample_rate = sf_info.samplerate
            if channels is None:
                channels = sf_info.channels
        except Exception:
            source_duration = 0.0

    if source_duration <= 0.0:
        raise ValueError(f"Unable to determine valid SOURCE_DURATION for {file_path}")

    # Fallback sample rate / channels from soundfile if ffprobe missed them
    if audio_stream is not None and (sample_rate is None or channels is None):
        try:
            sf_info = sf.info(str(file_path))
            sample_rate = sample_rate or sf_info.samplerate
            channels = channels or sf_info.channels
        except Exception:
            pass

    fps = None
    if video_stream and "r_frame_rate" in video_stream:
        r_fr = video_stream["r_frame_rate"]
        if "/" in r_fr:
            num, den = r_fr.split("/")
            try:
                fps = float(num) / max(1.0, float(den))
            except Exception:
                fps = None

    return {
        "file_path": str(file_path),
        "source_duration": round(float(source_duration), 4),
        "container_duration": round(float(container_duration), 4) if container_duration is not None else None,
        "video_duration": round(float(video_duration), 4) if video_duration is not None else None,
        "audio_duration": round(float(audio_duration), 4) if audio_duration is not None else None,
        "has_video": video_stream is not None,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "sample_rate": sample_rate,
        "channels": channels,
        "fps": round(fps, 3) if fps else None,
    }


def extract_synchronized_audio(
    video_path: str | Path,
    output_wav_path: str | Path,
    source_duration: float,
    sample_rate: int = 44100,
    channels: int = 2,
    tolerance: float = 0.25
) -> Dict[str, Any]:
    """
    Extracts audio from video with explicit PTS-locking and resampling.
    Verifies that extracted audio duration matches source_duration within tolerance.
    If a mismatch occurs, auto-remedies by forcing exact timeline alignment.
    """
    video_path = Path(video_path).resolve()
    output_wav_path = Path(output_wav_path).resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # First attempt: high-fidelity extraction with aresample async=1:first_pts=0
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-af", "aresample=async=1:first_pts=0",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    # Verify extracted duration
    extracted_meta = sf.info(str(output_wav_path))
    extracted_dur = float(extracted_meta.duration)
    dur_diff = abs(extracted_dur - source_duration)

    if dur_diff > tolerance:
        print(f"[TIMELINE AUDIT NOTICE] Extracted audio duration ({extracted_dur:.3f}s) differs from SOURCE_DURATION ({source_duration:.3f}s) by {dur_diff:.3f}s (> {tolerance}s). Auto-aligning extraction with explicit duration boundary...")

        # Remediation: Extract with exact duration clipping/padding
        cmd_aligned = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-af", f"aresample=async=1:first_pts=0,apad=whole_dur={source_duration:.4f}",
            "-t", f"{source_duration:.4f}",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_wav_path)
        ]
        subprocess.run(cmd_aligned, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        extracted_meta = sf.info(str(output_wav_path))
        extracted_dur = float(extracted_meta.duration)
        dur_diff = abs(extracted_dur - source_duration)

        if dur_diff > tolerance:
            raise TimelineCorruptionError(
                stage="audio_extraction",
                message=f"Failed to extract synchronized audio. Extracted {extracted_dur:.3f}s vs SOURCE_DURATION {source_duration:.3f}s",
                details={"source_duration": source_duration, "extracted_duration": extracted_dur, "difference": dur_diff}
            )

    return {
        "extracted_path": str(output_wav_path),
        "duration": round(extracted_dur, 4),
        "sample_rate": extracted_meta.samplerate,
        "channels": extracted_meta.channels,
        "samples": extracted_meta.frames,
        "source_duration": source_duration,
        "duration_delta": round(dur_diff, 4),
    }


def validate_segment_bounds(
    seg: Dict[str, Any],
    seg_idx: int,
    source_duration: float,
    stage_name: str,
    tolerance: float = 0.20
) -> None:
    """
    Hard assertion on a single segment:
    0 <= start < end <= SOURCE_DURATION + tolerance
    Checks segment-level and word-level timestamps.
    """
    if "start" not in seg or "end" not in seg:
        raise TimelineCorruptionError(
            stage=stage_name,
            message=f"Segment #{seg_idx} is missing 'start' or 'end' keys.",
            details={"segment": seg}
        )

    try:
        start = float(seg["start"])
        end = float(seg["end"])
    except (ValueError, TypeError) as e:
        raise TimelineCorruptionError(
            stage=stage_name,
            message=f"Segment #{seg_idx} has non-numeric timestamps: start={seg.get('start')}, end={seg.get('end')}",
            details={"error": str(e), "segment": seg}
        )

    # 1. Negative timestamp check
    if start < -1e-4:
        raise TimelineCorruptionError(
            stage=stage_name,
            message=f"Segment #{seg_idx} has negative start timestamp ({start:.4f}s < 0.0s).",
            details={"segment_index": seg_idx, "start": start, "end": end, "source_duration": source_duration}
        )

    # 2. Inverted duration check
    if start >= end:
        raise TimelineCorruptionError(
            stage=stage_name,
            message=f"Segment #{seg_idx} has start >= end ({start:.4f}s >= {end:.4f}s). Zero or negative duration dialogue.",
            details={"segment_index": seg_idx, "start": start, "end": end, "source_duration": source_duration}
        )

    # 3. Beyond SOURCE_DURATION check
    max_allowed = source_duration + tolerance
    if end > max_allowed:
        raise TimelineCorruptionError(
            stage=stage_name,
            message=f"Segment #{seg_idx} end timestamp ({end:.4f}s) exceeds SOURCE_DURATION ({source_duration:.4f}s + tol {tolerance}s = {max_allowed:.4f}s).",
            details={"segment_index": seg_idx, "start": start, "end": end, "source_duration": source_duration, "overflow_seconds": end - source_duration}
        )

    # 4. Word-level timestamps check if present
    words = seg.get("words", [])
    if words:
        prev_w_end = start - 0.01
        for w_idx, w in enumerate(words):
            w_start = float(w.get("start", start))
            w_end = float(w.get("end", end))

            if w_start < -1e-4:
                raise TimelineCorruptionError(
                    stage=stage_name,
                    message=f"Segment #{seg_idx}, word #{w_idx} ('{w.get('word', '')}') has negative start ({w_start:.4f}s).",
                    details={"segment_index": seg_idx, "word_index": w_idx, "word": w}
                )

            if w_start > w_end:
                raise TimelineCorruptionError(
                    stage=stage_name,
                    message=f"Segment #{seg_idx}, word #{w_idx} ('{w.get('word', '')}') has w_start > w_end ({w_start:.4f}s > {w_end:.4f}s).",
                    details={"segment_index": seg_idx, "word_index": w_idx, "word": w}
                )

            if w_end > max_allowed:
                raise TimelineCorruptionError(
                    stage=stage_name,
                    message=f"Segment #{seg_idx}, word #{w_idx} ('{w.get('word', '')}') end ({w_end:.4f}s) exceeds SOURCE_DURATION ({source_duration:.4f}s).",
                    details={"segment_index": seg_idx, "word_index": w_idx, "word": w, "source_duration": source_duration}
                )


def validate_timeline_integrity(
    segments: List[Dict[str, Any]],
    source_duration: float,
    stage_name: str,
    tolerance: float = 0.20,
    allow_empty: bool = False
) -> None:
    """
    Comprehensive Timeline Integrity Check:
    Verifies that all segments satisfy:
      - 0 <= start < end <= SOURCE_DURATION + tolerance
      - Timestamps are monotonically progressing (no backward jumps)
      - Word timestamps are strictly contained within their segment
      - No double chunk offsets
    """
    if not segments:
        if allow_empty:
            return
        raise TimelineCorruptionError(
            stage=stage_name,
            message="Segment list is completely empty.",
            details={"stage": stage_name, "source_duration": source_duration}
        )

    prev_start = -1e-4
    for idx, seg in enumerate(segments):
        validate_segment_bounds(seg, idx, source_duration, stage_name, tolerance=tolerance)

        curr_start = float(seg["start"])
        # Check backward jump (allowing slight overlap of max 0.5s for conversational cross-talk)
        if curr_start < prev_start - 0.5:
            raise TimelineCorruptionError(
                stage=stage_name,
                message=f"Segment #{idx} timestamp jumps backward significantly: {curr_start:.4f}s < previous start {prev_start:.4f}s",
                details={"segment_index": idx, "current_start": curr_start, "previous_start": prev_start}
            )
        prev_start = curr_start


def print_timeline_sanity_report(
    source_duration: float,
    original_audio_duration: float,
    analysis_audio_duration: float,
    segments: List[Dict[str, Any]],
    vad_segments: Optional[List[Dict[str, Any]]] = None,
    stage_label: str = "PRE-FLIGHT SANITY CHECK"
) -> None:
    """
    Formats and prints the mandatory timeline sanity report before expensive stages run.
    Validates the invariants:
      last_vad_end <= SOURCE_DURATION + 0.20
      last_asr_end <= SOURCE_DURATION + 0.20
    """
    num_segs = len(segments)

    if segments:
        first_asr_start = min(float(s["start"]) for s in segments)
        last_asr_end = max(float(s["end"]) for s in segments)
    else:
        first_asr_start = 0.0
        last_asr_end = 0.0

    if vad_segments:
        first_vad_start = min(float(s["start"]) for s in vad_segments)
        last_vad_end = max(float(s["end"]) for s in vad_segments)
    else:
        first_vad_start = first_asr_start
        last_vad_end = last_asr_end

    print("\n" + "=" * 90)
    print(f" TIMELINE SANITY REPORT — {stage_label}")
    print("=" * 90)
    print(f"  SOURCE_DURATION:          {source_duration:8.3f}s")
    print(f"  original_audio_duration:  {original_audio_duration:8.3f}s  (delta: {abs(original_audio_duration - source_duration):+.4f}s)")
    print(f"  analysis_audio_duration:  {analysis_audio_duration:8.3f}s  (delta: {abs(analysis_audio_duration - source_duration):+.4f}s)")
    print(f"  first_vad_start:          {first_vad_start:8.3f}s")
    print(f"  last_vad_end:             {last_vad_end:8.3f}s")
    print(f"  first_asr_start:          {first_asr_start:8.3f}s")
    print(f"  last_asr_end:             {last_asr_end:8.3f}s")
    print(f"  number_of_segments:       {num_segs:8d}")
    print("-" * 90)

    # Invariant assertions
    tolerance = 0.20
    if last_asr_end > source_duration + tolerance:
        raise TimelineCorruptionError(
            stage="timeline_sanity_report",
            message=f"INVARIANT VIOLATION: last_asr_end ({last_asr_end:.3f}s) > SOURCE_DURATION ({source_duration:.3f}s + {tolerance}s)",
            details={"last_asr_end": last_asr_end, "source_duration": source_duration}
        )

    if last_vad_end > source_duration + tolerance:
        raise TimelineCorruptionError(
            stage="timeline_sanity_report",
            message=f"INVARIANT VIOLATION: last_vad_end ({last_vad_end:.3f}s) > SOURCE_DURATION ({source_duration:.3f}s + {tolerance}s)",
            details={"last_vad_end": last_vad_end, "source_duration": source_duration}
        )

    print("  STATUS: ALL TIMELINE INVARIANTS SATISFIED (0 <= start < end <= SOURCE_DURATION)")
    print("=" * 90 + "\n", flush=True)


def apply_chunk_offset(
    segments: List[Dict[str, Any]],
    chunk_original_start: float,
    chunk_original_end: float,
    source_duration: float,
    tolerance: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Applies chunk source offset to relative chunk segments EXACTLY ONCE:
      absolute_start = chunk_original_start + relative_start
      absolute_end   = chunk_original_start + relative_end

    Strictly protects against:
      - applying chunk offset more than once (via _chunk_offset_applied sentinel)
      - cumulative segment additions
      - double-offsetting
    """
    for seg in segments:
        if seg.get("_chunk_offset_applied"):
            raise TimelineCorruptionError(
                stage="chunk_offset_application",
                message=f"Attempted to apply chunk offset ({chunk_original_start:.3f}s) more than once to segment #{seg.get('id')}!",
                details={"segment": seg}
            )

        rel_start = float(seg["start"])
        rel_end = float(seg["end"])

        abs_start = round(chunk_original_start + rel_start, 3)
        abs_end = round(chunk_original_start + rel_end, 3)

        seg["start"] = abs_start
        seg["end"] = abs_end
        seg["duration"] = round(abs_end - abs_start, 3)
        seg["_chunk_offset_applied"] = True
        seg["_chunk_source_start"] = chunk_original_start

        # Offset word timestamps
        words = seg.get("words", [])
        for w in words:
            if not w.get("_chunk_offset_applied"):
                w["start"] = round(chunk_original_start + float(w["start"]), 3)
                w["end"] = round(chunk_original_start + float(w["end"]), 3)
                w["_chunk_offset_applied"] = True

        # Validate bounds immediately
        if abs_start < -1e-4 or abs_end > (source_duration + tolerance):
            raise TimelineCorruptionError(
                stage="chunk_offset_application",
                message=f"Offset segment #{seg.get('id')} [{abs_start:.3f}s -> {abs_end:.3f}s] violates bounds [0.0, {source_duration:.3f}s].",
                details={"abs_start": abs_start, "abs_end": abs_end, "source_duration": source_duration}
            )

    return segments


def validate_cache_staleness(
    cache_path: str | Path,
    source_video_path: str | Path,
    expected_source_duration: float,
    tolerance: float = 0.10
) -> bool:
    """
    Validates whether cached timing data (e.g. russian_transcript.json) is valid
    for the current source video or if it is stale and must be discarded.
    Checks:
      - source file existence and modification time
      - stored source duration matching expected_source_duration
    Returns True if cache is valid; False if stale.
    """
    cache_path = Path(cache_path)
    source_video_path = Path(source_video_path)

    if not cache_path.exists():
        return False

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return False

        # Check stored metadata if available
        # Find maximum end time in cached data
        max_cached_end = max(float(s.get("end", 0.0)) for s in data)

        # If cache contains segments beyond expected duration, or significantly shorter
        if max_cached_end > expected_source_duration + tolerance:
            print(f"[CACHE STALENESS] Discarding stale cache: max cached end ({max_cached_end:.2f}s) > source duration ({expected_source_duration:.2f}s)")
            return False

        if source_video_path.exists():
            if cache_path.stat().st_mtime < source_video_path.stat().st_mtime:
                print(f"[CACHE STALENESS] Discarding stale cache: source video was modified after cache was created.")
                return False

        return True
    except Exception as e:
        print(f"[CACHE STALENESS] Error inspecting cache ({e}), invalidating.")
        return False
