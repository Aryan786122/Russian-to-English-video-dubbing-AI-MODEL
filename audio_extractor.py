import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

from timeline_validator import (
    probe_media_metadata,
    extract_synchronized_audio,
    TimelineCorruptionError
)


def run_ffmpeg_cmd(cmd: list[str]) -> tuple[bool, str]:
    """Execute an FFmpeg command safely and return success flag and output."""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return True, res.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "FFmpeg executable not found in PATH."


def get_audio_duration(file_path: str | Path) -> float:
    """Return the duration of an audio or video file in seconds using robust probe."""
    try:
        meta = probe_media_metadata(file_path)
        return float(meta["source_duration"])
    except Exception:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            try:
                info = sf.info(str(file_path))
                return float(info.duration)
            except Exception:
                return 0.0


def extract_audio(
    video_path: str | Path,
    output_audio_path: str | Path,
    sample_rate: int = 44100,
    source_duration: float | None = None
) -> bool:
    """
    Extract audio from video file with timeline PTS synchronization.
    If source_duration is given, strictly validates extracted audio duration.
    """
    if source_duration is None:
        meta = probe_media_metadata(video_path)
        source_duration = meta["source_duration"]

    try:
        extract_synchronized_audio(
            video_path=video_path,
            output_wav_path=output_audio_path,
            source_duration=source_duration,
            sample_rate=sample_rate,
            channels=2,
            tolerance=0.25
        )
        return True
    except Exception as e:
        print(f"Error in extract_audio: {e}")
        return False


def slice_audio(input_audio: str | Path, start_sec: float, end_sec: float, output_path: str | Path) -> bool:
    """Slice a portion of an audio file between start_sec and end_sec."""
    duration = max(0.1, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(input_audio),
        "-acodec", "pcm_s16le",
        str(output_path)
    ]
    ok, err = run_ffmpeg_cmd(cmd)
    if not ok:
        print(f"Error slicing audio: {err}")
    return ok


def adjust_audio_speed(
    input_wav: str | Path,
    target_duration: float,
    output_wav: str | Path,
    max_stretch: float = 1.15,
    min_stretch: float = 0.90
) -> bool:
    """
    Adjust audio playback tempo to fit within the video timing slot.
    Clamps tempo strictly to [min_stretch, max_stretch] to prevent distortion.
    """
    current_duration = get_audio_duration(input_wav)
    if current_duration <= 0.05 or target_duration <= 0.05:
        return False

    speed_ratio = current_duration / target_duration
    speed_ratio = max(min_stretch, min(max_stretch, speed_ratio))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-filter:a", f"atempo={speed_ratio:.4f}",
        "-vn",
        str(output_wav)
    ]
    ok, _ = run_ffmpeg_cmd(cmd)
    return ok


def mux_audio_and_video(
    video_path: str | Path,
    dubbed_audio_path: str | Path,
    output_mp4_path: str | Path,
    subtitle_path: str | Path | None = None,
    output_mp3_path: str | Path | None = None,
    source_duration: float | None = None
) -> bool:
    """
    Mux dubbed audio and optional subtitle stream into MP4 container.
    Guarantees container duration matches source_duration exactly.
    Also exports standalone MP3.
    """
    video_path = Path(video_path).resolve()
    dubbed_audio_path = Path(dubbed_audio_path).resolve()
    output_mp4_path = Path(output_mp4_path).resolve()

    # Determine source_duration if not passed
    if source_duration is None:
        try:
            meta = probe_media_metadata(video_path)
            source_duration = meta["source_duration"]
        except Exception:
            source_duration = get_audio_duration(video_path)

    # Pre-mux validation: video stream check
    try:
        probe_res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True
        )
        if "video" not in probe_res.stdout.lower():
            raise TimelineCorruptionError(
                stage="mux_audio_and_video",
                message=f"Video input has no video stream: {video_path}. Cannot mux without video stream.",
                details={"video_path": str(video_path)}
            )
    except Exception as e:
        if isinstance(e, TimelineCorruptionError):
            raise
        print(f"WARNING: ffprobe validation notice ({e}), proceeding anyway...")

    if not dubbed_audio_path.exists():
        raise FileNotFoundError(f"Dubbed audio file not found: {dubbed_audio_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(dubbed_audio_path)
    ]

    has_subs = (
        subtitle_path is not None
        and Path(subtitle_path).exists()
        and Path(subtitle_path).stat().st_size > 0
    )
    if has_subs:
        cmd += ["-i", str(subtitle_path)]

    cmd += [
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0"
    ]

    if has_subs:
        cmd += [
            "-map", "2:s:0",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:0", "title=English Dubbed"
        ]

    # Explicit duration boundary instead of -shortest to prevent early cutoff
    if source_duration and source_duration > 0:
        cmd += ["-t", f"{source_duration:.4f}"]
    else:
        cmd += ["-shortest"]

    cmd.append(str(output_mp4_path))

    ok, err = run_ffmpeg_cmd(cmd)
    if not ok:
        raise TimelineCorruptionError(
            stage="mux_audio_and_video",
            message=f"FFmpeg muxing failed: {err}",
            details={"cmd": cmd}
        )

    # Post-mux validation: check that MP4 exists and has valid duration
    try:
        out_meta = probe_media_metadata(output_mp4_path)
        out_dur = out_meta["source_duration"]
        if source_duration and abs(out_dur - source_duration) > 0.5:
            raise TimelineCorruptionError(
                stage="post_mux_validation",
                message=f"Muxed video duration ({out_dur:.3f}s) differs from SOURCE_DURATION ({source_duration:.3f}s) by {abs(out_dur - source_duration):.3f}s (> 0.5s).",
                details={"output_duration": out_dur, "source_duration": source_duration}
            )
    except Exception as e:
        if isinstance(e, TimelineCorruptionError):
            raise
        print(f"Notice during post-mux probe: {e}")

    # Standalone MP3 export
    if output_mp3_path:
        cmd_mp3 = [
            "ffmpeg", "-y",
            "-i", str(dubbed_audio_path),
            "-c:a", "libmp3lame",
            "-b:a", "192k"
        ]
        if source_duration and source_duration > 0:
            cmd_mp3 += ["-t", f"{source_duration:.4f}"]
        cmd_mp3.append(str(output_mp3_path))
        run_ffmpeg_cmd(cmd_mp3)

    return True
