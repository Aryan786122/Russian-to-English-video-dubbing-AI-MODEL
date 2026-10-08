"""
UpgradedDubbingPipeline — Multi-Speaker Russian → English Voice-Cloned Dubbing.

Pipeline stages:
  1. Extract audio from video
  2. Russian transcription (Whisper large-v3, task=transcribe, language=ru)
  3. Context-aware English translation (MarianMT + polish)
  4. Demucs stem separation (vocals / background)
  5. Speaker diarization (ECAPA-TDNN + hierarchical cosine clustering)
  6. Per-speaker voice reference extraction
  7. Per-speaker F5-TTS voice cloning with absolute timestamp canvas placement
  8. Transient-preserving audio mix (100% original outside speech, Demucs bg during speech)
  9. Speaker-labeled SRT subtitle generation
 10. FFprobe-validated video muxing
"""

from pathlib import Path
import math
import time
import json
import shutil
import subprocess
import numpy as np
import soundfile as sf
import torch

import config
from timeline_validator import (
    probe_media_metadata,
    validate_timeline_integrity,
    print_timeline_sanity_report,
    validate_cache_staleness,
    TimelineCorruptionError
)
from audio_extractor import (
    extract_audio,
    slice_audio,
    get_audio_duration,
    adjust_audio_speed,
    mux_audio_and_video
)
from audio_mixer import separate_vocals_and_background, preserve_and_mix_dub
from transcriber import RussianDubbingTranscriber, format_debug_table, export_russian_transcripts
from subtitles import generate_srt, generate_vtt
from tts_engine import VoiceCloner
from diarizer import SpeakerDiarizer


class UpgradedDubbingPipeline:
    def __init__(self, whisper_model: str = None, tts_backend: str = "f5-tts"):
        if whisper_model in ["large-v3", None] and Path(config.DEFAULT_WHISPER_MODEL).exists():
            self.whisper_model = config.DEFAULT_WHISPER_MODEL
        else:
            self.whisper_model = whisper_model or config.DEFAULT_WHISPER_MODEL
        self.tts_backend = tts_backend
        self.transcriber = RussianDubbingTranscriber(model_size=self.whisper_model)
        self.voice_cloner = VoiceCloner(backend=tts_backend)
        self.diarizer = SpeakerDiarizer()

    def transcribe_and_translate(
        self,
        video_path: str | Path,
        progress_cb=None,
        debug_mode: bool = True
    ) -> list[dict]:
        """
        Runs ONLY Russian transcription + English translation.
        For debug preview before dubbing.
        """
        video_path = Path(video_path).resolve()
        meta = probe_media_metadata(video_path)
        source_duration = float(meta["source_duration"])

        session_id = f"{video_path.stem}_{int(time.time())}"
        session_temp = config.TEMP_DIR / session_id
        session_temp.mkdir(parents=True, exist_ok=True)

        source_wav = session_temp / "source.wav"

        if progress_cb:
            progress_cb(0.1, "Extracting audio from video...")
        extract_audio(video_path, source_wav, sample_rate=44100, source_duration=source_duration)

        if progress_cb:
            progress_cb(0.3, "Transcribing Russian speech using Whisper Large-V3...")

        segments = self.transcriber.process_audio(
            audio_path=source_wav,
            export_dir=session_temp,
            debug_mode=debug_mode,
            source_duration=source_duration
        )

        # Copy debug exports to output directory
        for fname in ["russian_transcript.json", "russian_transcript.srt", "transcript_debug.txt"]:
            src = session_temp / fname
            if src.exists():
                suffix_map = {
                    "russian_transcript.json": "_russian.json",
                    "russian_transcript.srt": "_russian.srt",
                    "transcript_debug.txt": "_debug_table.txt"
                }
                shutil.copyfile(src, config.OUTPUT_DIR / f"{video_path.stem}{suffix_map[fname]}")

        if progress_cb:
            progress_cb(1.0, f"Extracted & translated {len(segments)} dialogue segments.")

        return segments

    def run(
        self,
        video_path: str | Path,
        ref_start: float = 0.0,
        ref_end: float = 0.0,
        custom_ref_audio: str | Path | None = None,
        custom_segments: list[dict] | None = None,
        enable_stem_separation: bool = True,
        debug_mode: bool = True,
        progress_cb=None
    ) -> dict:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        session_id = f"{video_path.stem}_{int(time.time())}"
        session_temp = config.TEMP_DIR / session_id
        session_temp.mkdir(parents=True, exist_ok=True)

        source_wav = session_temp / "source.wav"
        vocals_wav = session_temp / "vocals.wav"
        background_wav = session_temp / "background.wav"
        master_dub_wav = session_temp / "master_dub.wav"
        refs_dir = session_temp / "speaker_refs"
        refs_dir.mkdir(parents=True, exist_ok=True)

        output_mp4 = config.OUTPUT_DIR / f"{video_path.stem}_dubbed.mp4"
        output_mp3 = config.OUTPUT_DIR / f"{video_path.stem}_dubbed.mp3"
        output_srt = config.OUTPUT_DIR / f"{video_path.stem}_dubbed.srt"

        # ── 1. PROBE AUTHORITATIVE TIMELINE & AUDIO EXTRACTION ──────────
        meta = probe_media_metadata(video_path)
        source_duration = float(meta["source_duration"])
        print("\n" + "=" * 90)
        print(f" TIMELINE INGESTION PROBE — {video_path.name}")
        print("=" * 90)
        print(f"  SOURCE_DURATION (Authoritative): {source_duration:8.4f}s")
        print(f"  Container Duration:              {meta.get('container_duration')}s")
        print(f"  Video Stream Duration:           {meta.get('video_duration')}s  (Codec: {meta.get('video_codec')}, FPS: {meta.get('fps')})")
        print(f"  Audio Stream Duration:           {meta.get('audio_duration')}s  (Codec: {meta.get('audio_codec')}, SR: {meta.get('sample_rate')}, Ch: {meta.get('channels')})")
        print("=" * 90 + "\n", flush=True)

        if progress_cb:
            progress_cb(0.05, "Extracting audio from video...")
        extract_audio(video_path, source_wav, sample_rate=44100, source_duration=source_duration)
        orig_info = sf.info(str(source_wav))
        orig_audio_duration = float(orig_info.duration)
        print(f"Extracted original audio duration: {orig_audio_duration:.4f}s (delta: {abs(orig_audio_duration - source_duration):+.4f}s)")

        total_duration = source_duration

        # ── 2. RUSSIAN TRANSCRIPTION (Whisper Large-V3) ─────────────────
        segments = custom_segments
        if segments is not None:
            # Check for stale cache against SOURCE_DURATION
            for seg in segments:
                if float(seg.get("end", 0.0)) > source_duration + 0.20:
                    print(f"[CACHE STALENESS] Custom/cached segments exceed SOURCE_DURATION ({source_duration:.2f}s). Discarding stale cache!")
                    segments = None
                    break

        if segments is None:
            if progress_cb:
                progress_cb(0.15, "Transcribing Russian using Whisper Large-V3...")
            segments = self.transcriber.process_audio(
                audio_path=source_wav,
                export_dir=session_temp,
                debug_mode=debug_mode,
                source_duration=source_duration
            )

        if not segments:
            raise ValueError("No speech segments detected in video audio.")

        # Filter out zero/near-zero duration segments (Whisper artifacts or bad cache entries)
        MIN_SEG_DUR = 0.05
        pre_count = len(segments)
        segments = [s for s in segments if float(s["end"]) - float(s["start"]) >= MIN_SEG_DUR]
        if len(segments) < pre_count:
            print(f"  Filtered out {pre_count - len(segments)} zero/near-zero duration segment(s).")
            for i, seg in enumerate(segments):
                seg["id"] = i

        # Copy debug transcript files
        for fname in ["russian_transcript.json", "russian_transcript.srt", "transcript_debug.txt"]:
            src = session_temp / fname
            if src.exists():
                suffix_map = {
                    "russian_transcript.json": "_russian.json",
                    "russian_transcript.srt": "_russian.srt",
                    "transcript_debug.txt": "_debug_table.txt"
                }
                shutil.copyfile(src, config.OUTPUT_DIR / f"{video_path.stem}{suffix_map[fname]}")

        # ── 3. DEMUCS STEM SEPARATION ───────────────────────────────────
        if enable_stem_separation:
            if progress_cb:
                progress_cb(0.30, "Separating vocals from background (Demucs)...")
            separate_vocals_and_background(source_wav, vocals_wav, background_wav)
        else:
            shutil.copyfile(source_wav, vocals_wav)
            shutil.copyfile(source_wav, background_wav)

        diarize_source = vocals_wav if vocals_wav.exists() else source_wav
        analysis_info = sf.info(str(diarize_source))
        analysis_audio_duration = float(analysis_info.duration)

        # ── 4. SPEAKER DIARIZATION ──────────────────────────────────────
        if progress_cb:
            progress_cb(0.40, "Detecting speakers via ECAPA-TDNN diarization...")

        try:
            diarized_segments, cluster_meta = self.diarizer.diarize_segments(
                audio_path=diarize_source,
                segments=segments,
                source_duration=source_duration
            )
            segments = diarized_segments

            # Unique speakers found
            speakers_found = sorted(set(s.get("speaker_id", "SPEAKER_00") for s in segments))
            print(f"Diarization complete: {len(speakers_found)} speakers detected: {speakers_found}")

        except Exception as e:
            if isinstance(e, TimelineCorruptionError):
                raise
            print(f"WARNING: Diarization failed ({e}), falling back to single speaker.")
            for seg in segments:
                seg["speaker_id"] = "SPEAKER_00"
            speakers_found = ["SPEAKER_00"]

        # ── 4.5. TIMELINE INTEGRITY AUDIT BEFORE EXPENSIVE STAGES ────────
        print_timeline_sanity_report(
            source_duration=source_duration,
            original_audio_duration=orig_audio_duration,
            analysis_audio_duration=analysis_audio_duration,
            segments=segments,
            stage_label="PRE-TTS SANITY AUDIT"
        )

        validate_timeline_integrity(
            segments=segments,
            source_duration=source_duration,
            stage_name="pre_tts_synthesis",
            tolerance=0.20
        )

        # ── 5. PER-SPEAKER VOICE REFERENCES ─────────────────────────────
        if progress_cb:
            progress_cb(0.48, f"Building voice references for {len(speakers_found)} speakers...")

        speaker_refs = {}   # speaker_id -> Path to ref WAV
        ref_texts = {}      # speaker_id -> Russian ref text for F5-TTS conditioning

        try:
            speaker_profiles = self.diarizer.build_speaker_voice_references(
                audio_path=diarize_source,
                segments=segments,
                output_dir=refs_dir,
                target_ref_duration=12.0,
                min_ref_duration=2.5,
                whisper_transcriber=None  # Use vetted segment transcripts directly to avoid hallucinations
            )

            for spk_id, profile in speaker_profiles.items():
                ref_path_str = profile.get("ref_audio_path") or profile.get("ref_wav_path", "")
                ref_path = Path(ref_path_str)
                if ref_path.exists():
                    speaker_refs[spk_id] = ref_path
                    ref_texts[spk_id] = profile.get("ref_text") or profile.get("ref_transcript", "")
                    dur = get_audio_duration(ref_path)
                    print(f"  [{spk_id}] ref: {dur:.1f}s | transcript: \"{ref_texts[spk_id][:60]}...\"")

        except Exception as e:
            print(f"WARNING: Voice reference extraction failed ({e}), using fallback.")

        # Fallback: if any speaker lacks a reference, use a generic one
        if not speaker_refs:
            # Build a single reference from the best segment
            ref_audio_file = session_temp / "speaker_ref.wav"
            ref_source = vocals_wav if vocals_wav.exists() else source_wav

            if custom_ref_audio and Path(custom_ref_audio).exists():
                ref_audio_file = Path(custom_ref_audio)
            elif ref_end > ref_start and (ref_end - ref_start) >= 2.0:
                slice_audio(ref_source, ref_start, ref_end, ref_audio_file)
            else:
                candidates = [s for s in segments if 4.0 <= (s["end"] - s["start"]) <= 15.0]
                best_seg = candidates[0] if candidates else segments[0]
                slice_audio(ref_source, best_seg["start"], best_seg["end"], ref_audio_file)
                ref_texts["SPEAKER_00"] = best_seg.get("russian_text", "")

            for spk in speakers_found:
                if spk not in speaker_refs:
                    speaker_refs[spk] = ref_audio_file
                    ref_texts.setdefault(spk, "")

        # Ensure every speaker has a ref (fallback to SPEAKER_00's)
        fallback_ref = next(iter(speaker_refs.values()))
        fallback_text = next(iter(ref_texts.values()), "")
        for spk in speakers_found:
            speaker_refs.setdefault(spk, fallback_ref)
            ref_texts.setdefault(spk, fallback_text)

        # ── FREE VRAM BEFORE TTS ─────────────────────────────────────────
        # Unload Whisper, translation, and diarizer models so F5-TTS has
        # 100% of GPU VRAM available and runs fast without paging or thrashing.
        self.transcriber.unload_models()
        self.diarizer.unload_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("GPU VRAM cleared for F5-TTS synthesis.")

        # ── 6. PER-SPEAKER VOICE CLONING + CANVAS PLACEMENT ────────────
        if progress_cb:
            progress_cb(0.50, "Synthesizing per-speaker voice clones...")

        # Load original audio for sample rate & authoritative canvas sizing
        orig_audio, sr = sf.read(str(source_wav), dtype="float32", always_2d=True)
        total_samples = int(round(source_duration * sr))
        voice_canvas = np.zeros((total_samples, 2), dtype=np.float32)

        synthesized_segments = []
        num_segments = len(segments)

        for idx, seg in enumerate(segments):
            if progress_cb:
                pct = 0.50 + (0.30 * (idx / max(1, num_segments)))
                progress_cb(pct, f"Voice cloning: segment {idx + 1}/{num_segments}...")

            text = (seg.get("english_text") or seg.get("english") or
                    seg.get("text", "")).strip()
            if not text:
                continue

            s_start = float(seg["start"])
            s_end = float(seg["end"])
            budget = max(0.2, s_end - s_start)
            spk_id = seg.get("speaker_id", "SPEAKER_00")

            # Check bounds against SOURCE_DURATION
            if s_start < -1e-4 or s_end > (source_duration + 0.20):
                raise TimelineCorruptionError(
                    stage="tts_synthesis_bounds_check",
                    message=f"Segment #{idx} [{s_start:.3f}s -> {s_end:.3f}s] exceeds SOURCE_DURATION ({source_duration:.3f}s)",
                    details={"segment": seg, "source_duration": source_duration}
                )

            # Get this speaker's reference
            if spk_id not in speaker_refs:
                spk_id = "SPEAKER_00"
            ref_wav = speaker_refs.get(spk_id, fallback_ref)
            ref_text = ref_texts.get(spk_id, "")

            seg_raw_wav = session_temp / f"seg_{idx}_{spk_id}_raw.wav"
            seg_fitted_wav = session_temp / f"seg_{idx}_{spk_id}_fitted.wav"

            # Synthesize with speaker-specific voice clone
            ok = self.voice_cloner.clone_and_speak(
                text=text,
                ref_audio_path=ref_wav,
                output_path=seg_raw_wav,
                ref_text=ref_text
            )

            if not ok or not seg_raw_wav.exists():
                print(f"  WARNING: TTS failed for segment {idx} ({spk_id}), skipping.")
                continue

            raw_meta = sf.info(str(seg_raw_wav))
            raw_dur = float(raw_meta.duration)

            # Duration budgeting & pacing: adapt tempo so speech naturally fills the visible speaking duration
            speed_ratio = raw_dur / budget
            if speed_ratio > 1.05 or speed_ratio < 0.92:
                adjust_audio_speed(
                    seg_raw_wav, target_duration=budget,
                    output_wav=seg_fitted_wav,
                    max_stretch=1.15, min_stretch=0.92
                )
                use_wav = seg_fitted_wav if seg_fitted_wav.exists() else seg_raw_wav
            else:
                use_wav = seg_raw_wav

            # Read and resample if needed
            clip_data, clip_sr = sf.read(str(use_wav), dtype="float32", always_2d=True)
            if clip_sr != sr:
                from scipy.signal import resample_poly
                divisor = math.gcd(clip_sr, sr)
                clip_data = resample_poly(clip_data, sr // divisor, clip_sr // divisor, axis=0)

            # HARD GUARD: clip MUST NOT exceed budget samples
            max_samples = int(round(budget * sr))
            if len(clip_data) > max_samples:
                fade_len = min(int(sr * 0.010), max_samples)
                clip_data = clip_data[:max_samples]
                if fade_len > 0:
                    clip_data[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, None]

            # Ensure stereo
            if clip_data.ndim == 1:
                clip_data = np.column_stack([clip_data, clip_data])
            elif clip_data.shape[1] == 1:
                clip_data = np.repeat(clip_data, 2, axis=1)

            actual_clip_dur = len(clip_data) / sr

            # Micro-fades (5ms) to prevent clicks
            edge = min(len(clip_data) // 4, max(1, int(sr * 0.005)))
            clip_data[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
            clip_data[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None]

            # Place at ABSOLUTE timestamp on silent canvas
            start_idx = int(round(s_start * sr))
            if start_idx >= total_samples:
                raise TimelineCorruptionError(
                    stage="tts_canvas_placement",
                    message=f"Segment #{idx} start_idx ({start_idx}) >= total_samples ({total_samples}) on canvas of duration {source_duration:.3f}s",
                    details={"segment": seg, "source_duration": source_duration}
                )
            end_idx = min(start_idx + len(clip_data), total_samples)
            chunk_len = end_idx - start_idx
            voice_canvas[start_idx:end_idx] += clip_data[:chunk_len]

            synthesized_segments.append({
                "id": seg.get("id", idx),
                "start": s_start,
                "end": s_end,
                "render_end": min(source_duration, s_start + actual_clip_dur),
                "wav_path": str(use_wav),
                "english": text,
                "russian": seg.get("russian_text", ""),
                "speaker_id": spk_id
            })

        # Validate synthesized segments bounds
        validate_timeline_integrity(
            segments=synthesized_segments,
            source_duration=source_duration,
            stage_name="synthesized_segments",
            tolerance=0.20
        )

        # Save voice-only canvas for debug
        voice_solo_wav = config.OUTPUT_DIR / f"{video_path.stem}_voice_solo.wav"
        sf.write(str(voice_solo_wav), voice_canvas, sr)

        # ── 7. TRANSIENT-PRESERVING AUDIO MIX ───────────────────────────
        if progress_cb:
            progress_cb(0.85, "Mixing: preserving BGM, SFX & Foley...")

        preserved_sfx_wav = config.OUTPUT_DIR / f"{video_path.stem}_preserved_sfx_bgm.wav"

        speech_intervals = [(float(s["start"]), float(s["end"])) for s in segments]
        preserve_and_mix_dub(
            original_wav_path=source_wav,
            background_wav_path=background_wav,
            voice_segments=synthesized_segments,
            total_duration=source_duration,
            output_master_wav_path=master_dub_wav,
            output_preserved_sfx_path=preserved_sfx_wav,
            output_voice_solo_path=None,  # already saved above
            crossfade_ms=30.0,
            speech_intervals=speech_intervals
        )

        # ── 8. SPEAKER-LABELED SRT SUBTITLES ────────────────────────────
        if progress_cb:
            progress_cb(0.92, "Generating speaker-labeled English subtitles...")

        generate_srt(
            segments=synthesized_segments,
            output_path=output_srt,
            source_duration=source_duration,
            tolerance=0.20
        )

        # ── 9. MUX FINAL VIDEO ──────────────────────────────────────────
        if progress_cb:
            progress_cb(0.96, "Muxing final video with dubbed audio...")

        mux_audio_and_video(
            video_path=video_path,
            dubbed_audio_path=master_dub_wav,
            output_mp4_path=output_mp4,
            subtitle_path=output_srt,
            output_mp3_path=output_mp3,
            source_duration=source_duration
        )

        if progress_cb:
            progress_cb(1.0, "Dubbing completed!")

        return {
            "output_mp4": str(output_mp4),
            "output_mp3": str(output_mp3),
            "output_srt": str(output_srt),
            "debug_table": format_debug_table(segments),
            "russian_json": str(config.OUTPUT_DIR / f"{video_path.stem}_russian.json"),
            "russian_srt": str(config.OUTPUT_DIR / f"{video_path.stem}_russian.srt"),
            "segments": synthesized_segments,
            "total_duration": total_duration,
            "speakers_detected": speakers_found if 'speakers_found' in dir() else ["SPEAKER_00"],
        }


def _to_srt_time(sec: float) -> str:
    """Convert seconds to SRT timestamp format."""
    h, rem = divmod(sec, 3600)
    m, s_val = divmod(rem, 60)
    ms = int((s_val - int(s_val)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s_val):02d},{ms:03d}"
