"""
SubtitleDubbingPipeline — subtitle-driven, multi-speaker, voice-cloned dubbing.

Inputs : ORIGINAL VIDEO + FINAL ENGLISH SUBTITLES (.srt / .vtt)
Output : English-dubbed MP4 where each line is spoken by the cloned voice of
         the person who said it, placed exactly at the subtitle timestamp.

The subtitle file is AUTHORITATIVE for both text and timing. Nothing here
retranscribes or retranslates the dialogue. (Whisper is only used to
(a) transcribe each speaker's *reference clip* so F5-TTS gets matching
conditioning text, and (b) QC the generated English clips.)

Stage order (each stage cached, GPU models loaded one at a time):
    open_project      -> ffprobe SOURCE_DURATION, parse + validate subtitles
    prepare_audio     -> source.wav (48 kHz) + Demucs vocals/background stems
    diarize           -> ECAPA-TDNN speaker clustering on the vocals stem
    speaker map       -> subtitle cue -> speaker (editable, saved as JSON)
    build_refs        -> clean <=11 s reference per speaker + ref transcript
    preview_voices    -> short English test line per speaker
    render(start,end) -> TTS -> QC -> sample-accurate canvas -> mix -> mux
"""

from __future__ import annotations

import csv
import difflib
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

# CTranslate2 (faster-whisper) needs torch's CUDA DLLs on Windows
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    try:
        os.add_dll_directory(str(_torch_lib))
    except Exception:
        pass
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import config
from timeline_validator import probe_media_metadata, TimelineCorruptionError
from audio_extractor import extract_audio, mux_audio_and_video
from audio_mixer import separate_vocals_and_background
from subtitle_parser import (
    SubtitleCue,
    parse_subtitle_file,
    validate_subtitle_timeline,
    cues_to_srt,
    cue_from_dict,
)
from cache_manager import compute_file_hash
from tts_engine import VoiceCloner

ProgressCB = Optional[Callable[[float, str], None]]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
SR = 48000                      # master timeline sample rate
MIN_CUE_DURATION = 0.05
MAX_SPEEDUP = 1.15              # never compress speech harder than this
MAX_SLOWDOWN = 1.10             # may stretch short lines up to ~0.9x speed
OVERFLOW_GRACE = 0.40           # s a line may run past its cue end (if free)
ENGLISH_CPS = 14.0              # natural English chars/sec for estimates
REF_MAX_SEC = 11.0              # F5-TTS clips refs at ~12 s
REF_TARGET_SEC = 10.0
SUPPRESS_PAD = 0.06             # s of extra original-voice suppression per side
SUPPRESS_RAMP = 0.04            # s crossfade original <-> background
QC_TEXT_PASS = 0.70
QC_SPK_PASS = 0.45
DIAR_LOW_CONF = 0.25
AMBIGUOUS_FLAGS = {"mixed_speakers_in_segment", "interpolated_speaker"}


def _log(msg: str):
    print(msg, flush=True)


def _p(cb: ProgressCB, pct: float, msg: str):
    _log(f"[{pct*100:5.1f}%] {msg}")
    if cb:
        try:
            cb(pct, msg)
        except Exception:
            pass


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_words(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w for w in text.split() if w]


def natural_duration(text: str) -> float:
    """Rough natural English speaking time for ``text``."""
    t = text.strip()
    words = len(t.split())
    pauses = len(re.findall(r"[,;:]", t)) * 0.12 + len(re.findall(r"[.!?…]+(?=\s|$)", t)) * 0.20
    min_dur = max(0.60, min(1.2, words * 0.45))
    return max(min_dur, len(t) / ENGLISH_CPS + pauses)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def _to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x


def _crossfade_join(clip1: np.ndarray, clip2: np.ndarray, crossfade_ms: float = 40.0, sr: int = SR) -> np.ndarray:
    """Concatenate two mono float32 clips with an equal-power linear crossfade to eliminate clicks."""
    n_fade = int(sr * (crossfade_ms / 1000.0))
    if len(clip1) <= n_fade or len(clip2) <= n_fade:
        return np.concatenate([clip1, clip2])
    out = np.zeros(len(clip1) + len(clip2) - n_fade, dtype=np.float32)
    out[:len(clip1)] = clip1
    fade_in = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    fade_out = 1.0 - fade_in
    out[len(clip1) - n_fade:len(clip1)] = (clip1[-n_fade:] * fade_out) + (clip2[:n_fade] * fade_in)
    out[len(clip1):] = clip2[n_fade:]
    return out


def _read_slice(path: Path, start: float, dur: float, target_sr: int = SR) -> np.ndarray:
    """Read [start, start+dur) seconds of a WAV as stereo float32 at target_sr (zero padded)."""
    info = sf.info(str(path))
    sr = info.samplerate
    s0 = max(0, int(round(start * sr)))
    n = int(round(dur * sr))
    data, _ = sf.read(str(path), start=s0, frames=n, dtype="float32", always_2d=True)
    data = _to_stereo(data)
    if len(data) < n:
        data = np.vstack([data, np.zeros((n - len(data), 2), dtype=np.float32)])
    data = _resample(data, sr, target_sr)
    n_out = int(round(dur * target_sr))
    if len(data) < n_out:
        data = np.vstack([data, np.zeros((n_out - len(data), 2), dtype=np.float32)])
    return data[:n_out]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12)) if x.size else 0.0


def _soft_limit(x: np.ndarray, knee: float = 0.89, ceiling: float = 0.98) -> np.ndarray:
    a = np.abs(x)
    over = a > knee
    if np.any(over):
        span = ceiling - knee
        x[over] = np.sign(x[over]) * (knee + span * np.tanh((a[over] - knee) / span))
    return x


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------
@dataclass
class DubProject:
    video_path: str
    subtitle_path: str
    source_duration: float
    meta: Dict[str, Any]
    video_hash: str
    subtitle_hash: str
    cues: List[Dict[str, Any]] = field(default_factory=list)

    # -- directories ----------------------------------------------------
    @property
    def stem(self) -> str:
        return Path(self.video_path).stem

    @property
    def media_dir(self) -> Path:          # depends on video only
        d = config.TEMP_DIR / "dub" / f"{self.stem}_{self.video_hash[:10]}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def work_dir(self) -> Path:           # depends on video + subtitles
        d = self.media_dir / f"subs_{self.subtitle_hash[:10]}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def out_dir(self) -> Path:
        d = config.OUTPUT_DIR / self.stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def source_wav(self) -> Path: return self.media_dir / "source_48k.wav"
    @property
    def vocals_wav(self) -> Path: return self.media_dir / "vocals_48k.wav"
    @property
    def background_wav(self) -> Path: return self.media_dir / "background_48k.wav"
    @property
    def diarization_json(self) -> Path: return self.work_dir / "diarization.json"
    @property
    def speaker_map_json(self) -> Path: return self.work_dir / "speaker_map.json"
    @property
    def refs_dir(self) -> Path: return self.work_dir / "speaker_refs"
    @property
    def refs_json(self) -> Path: return self.refs_dir / "refs.json"
    @property
    def tts_dir(self) -> Path:
        d = self.work_dir / "tts_segments"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cue_objects(self) -> List[SubtitleCue]:
        return [cue_from_dict(c) for c in self.cues]


# ---------------------------------------------------------------------------
# Lazy Whisper helper (ref transcription + QC only)
# ---------------------------------------------------------------------------
class _Whisper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def __enter__(self):
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = "int8_float16" if device == "cuda" else "int8"
        _log(f"Loading Whisper for ref-transcription/QC ({ctype})...")
        self.model = WhisperModel(self.model_path, device=device, compute_type=ctype)
        return self

    def transcribe(self, wav: Path, language: Optional[str] = None) -> Tuple[str, str]:
        d = self.detail(wav, language=language)
        return d["text"], d["language"]

    def detail(self, audio: Path | np.ndarray, language: Optional[str] = None, words: bool = False) -> Dict[str, Any]:
        """Transcribe and return text, language, quality stats and (optionally) word timings."""
        if not isinstance(audio, np.ndarray):
            from diarizer import load_mono_16k  # ffmpeg-based decode (avoids PyAV API mismatch)
            audio = load_mono_16k(audio)
        audio = audio.astype(np.float32)
        segs, info = self.model.transcribe(
            audio, language=language, beam_size=5, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0, word_timestamps=words,
        )
        segs = list(segs)
        out_words = []
        if words:
            for s in segs:
                for w in (s.words or []):
                    out_words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        return {
            "text": " ".join(s.text.strip() for s in segs).strip(),
            "language": getattr(info, "language", language or ""),
            "no_speech": max([s.no_speech_prob for s in segs], default=1.0),
            "logprob": float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0,
            "words": out_words,
        }

    def __exit__(self, *exc):
        del self.model
        self.model = None
        _free_gpu()


_HALLUCINATIONS = (
    "субтитр", "dimatorzok", "elderman", "elder_man", "продолжение следует", "спасибо за просмотр",
    "подписывайтесь", "редактор", "amara", "thank you for watching", "subtitles by",
    "vitellus", "diolchum", "orbitus", "dacus", "moris", "lorem ipsum", "sous-titres", "untertitel",
)


def _is_hallucination(text: str) -> bool:
    t = text.lower().strip()
    if not t or len(t) < 3:
        return True
    if any(h in t for h in _HALLUCINATIONS):
        return True
    words = [w.strip(".,!?:;\"'") for w in t.split()]
    if len(words) >= 4:
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2] and len(words[i]) >= 2:
                return True
    return False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class SubtitleDubbingPipeline:
    def __init__(self, tts_backend: str = "f5-tts", whisper_model: Optional[str] = None):
        self.tts_backend = tts_backend
        self.whisper_model = whisper_model or config.DEFAULT_WHISPER_MODEL
        self.voice_cloner = VoiceCloner(backend=tts_backend)
        self._diarizer = None

    # ------------------------------------------------------------------
    def _get_diarizer(self):
        if self._diarizer is None:
            from diarizer import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer()
        return self._diarizer

    def _release_diarizer(self):
        if self._diarizer is not None:
            try:
                self._diarizer.unload_model()
            except Exception:
                pass
        _free_gpu()

    # ==================================================================
    # 1. OPEN / VALIDATE
    # ==================================================================
    def open_project(self, video_path: str | Path, subtitle_path: str | Path) -> DubProject:
        video_path = Path(video_path).resolve()
        subtitle_path = Path(subtitle_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not subtitle_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

        meta = probe_media_metadata(video_path)
        if not meta.get("has_video"):
            raise ValueError(f"'{video_path.name}' has no video stream. Upload the ORIGINAL source video.")
        if not meta.get("has_audio"):
            raise ValueError(f"'{video_path.name}' has no audio stream.")
        source_duration = float(meta["source_duration"])

        cues = parse_subtitle_file(subtitle_path)
        if not cues:
            raise ValueError(f"No subtitle cues could be parsed from {subtitle_path.name}")
        cues = sorted(cues, key=lambda c: (c.start, c.end))
        cues = [c for c in cues if c.duration >= MIN_CUE_DURATION and c.text.strip()]
        if len({c.id for c in cues}) != len(cues):
            for i, c in enumerate(cues, 1):
                c.id = i
        validate_subtitle_timeline(cues, source_duration, tolerance=0.20)
        # clamp tiny container-rounding overshoot
        for c in cues:
            if c.end > source_duration:
                c.end = source_duration
                c.duration = c.end - c.start

        proj = DubProject(
            video_path=str(video_path),
            subtitle_path=str(subtitle_path),
            source_duration=source_duration,
            meta=meta,
            video_hash=compute_file_hash(video_path),
            subtitle_hash=compute_file_hash(subtitle_path),
            cues=[asdict(c) for c in cues],
        )

        _log("=" * 90)
        _log(f" SOURCE VALIDATION — {video_path.name}")
        _log(f"  SOURCE_DURATION : {source_duration:.3f}s")
        _log(f"  Video           : {meta.get('video_codec')} @ {meta.get('fps')} fps")
        _log(f"  Audio           : {meta.get('audio_codec')} {meta.get('sample_rate')} Hz x{meta.get('channels')}")
        _log(f"  Subtitle cues   : {len(cues)}  ({cues[0].start:.2f}s → {cues[-1].end:.2f}s)")
        _log(f"  Work dir        : {proj.work_dir}")
        _log("=" * 90)

        # If a speaker map already exists (user corrections), re-apply it
        if proj.speaker_map_json.exists():
            self._apply_saved_map(proj)
        return proj

    # ==================================================================
    # 2. AUDIO PREP (extraction + Demucs, cached per video)
    # ==================================================================
    def prepare_audio(self, proj: DubProject, progress: ProgressCB = None) -> None:
        src = proj.source_wav
        if not (src.exists() and abs(sf.info(str(src)).duration - proj.source_duration) < 0.3):
            _p(progress, 0.02, "Extracting original audio (48 kHz)...")
            ok = extract_audio(proj.video_path, src, sample_rate=SR, source_duration=proj.source_duration)
            if not ok or not src.exists():
                raise RuntimeError("Audio extraction failed.")
        else:
            _log(f"Cached source audio: {src.name}")

        if proj.vocals_wav.exists() and proj.background_wav.exists():
            _log("Cached Demucs stems found.")
            return

        _p(progress, 0.06, "Separating dialogue from music/SFX with Demucs (one-time per video)...")
        raw_voc = proj.media_dir / "vocals_raw.wav"
        raw_bg = proj.media_dir / "background_raw.wav"
        ok = separate_vocals_and_background(src, raw_voc, raw_bg)
        _free_gpu()
        if not ok:
            raise RuntimeError(
                "Demucs separation failed — refusing to continue, because the dub would contain the "
                "original voices. Check the console log, or supply an M&E track."
            )
        for raw, out in ((raw_voc, proj.vocals_wav), (raw_bg, proj.background_wav)):
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ar", str(SR), "-ac", "2",
                 "-c:a", "pcm_f32le", str(out)],
                check=True,
            )
            raw.unlink(missing_ok=True)
        _log("Demucs stems ready (48 kHz).")

    # ==================================================================
    # 3. DIARIZATION + SPEAKER MAP
    # ==================================================================
    def diarize(
        self,
        proj: DubProject,
        num_speakers: Optional[int] = None,
        force: bool = False,
        progress: ProgressCB = None,
    ) -> List[SubtitleCue]:
        cache_tag = f"vad2|n={num_speakers or 'auto'}"
        if proj.diarization_json.exists() and not force:
            data = json.loads(proj.diarization_json.read_text(encoding="utf-8"))
            if data.get("tag") == cache_tag:
                _log("Using cached diarization.")
                if not proj.speaker_map_json.exists():
                    self._map_from_diarization(proj, data["segments"])
                return self._apply_saved_map(proj)

        self.prepare_audio(proj, progress)
        _p(progress, 0.20, "Speaker diarization (ECAPA-TDNN on voiced regions of isolated vocals)...")
        segs = self._voiced_segments(proj)
        diar = self._get_diarizer()
        diarized, _ = diar.diarize_segments(
            audio_path=proj.vocals_wav,
            segments=segs,
            num_speakers=num_speakers if num_speakers and num_speakers > 0 else None,
            source_duration=proj.source_duration,
        )
        self._release_diarizer()

        voiced_by_id = {s["id"]: s.get("words", []) for s in segs}
        slim = [
            {
                "id": s["id"], "start": s["start"], "end": s["end"],
                "speaker_id": s.get("speaker_id"),
                "confidence": float(s.get("speaker_confidence", 0.0)),
                "flags": list(s.get("speaker_flags", [])),
                "voiced": [[round(w["start"], 3), round(w["end"], 3)] for w in voiced_by_id.get(s["id"], [])],
            }
            for s in diarized
        ]
        proj.diarization_json.write_text(json.dumps({"tag": cache_tag, "segments": slim}, indent=2), encoding="utf-8")
        self._map_from_diarization(proj, slim)
        # New diarization invalidates refs
        if proj.refs_json.exists():
            proj.refs_json.unlink()
        return proj.cue_objects()

    def _voiced_segments(self, proj: DubProject) -> List[dict]:
        """Energy VAD on the isolated vocals inside each subtitle window.

        Returns diarizer-ready segments whose ``words`` are voiced islands, so
        ECAPA embeds actual speech instead of silence / residual noise.
        """
        from diarizer import load_mono_16k
        sr = 16000
        voc = load_mono_16k(proj.vocals_wav)
        hop = int(0.02 * sr)
        out = []
        n_empty = 0
        for c in proj.cues:
            a = max(0, int((c["start"] - 0.10) * sr))
            b = min(len(voc), int((c["end"] + 0.10) * sr))
            seg = {"id": c["id"], "start": c["start"], "end": c["end"], "text": c["text"], "words": []}
            x = voc[a:b]
            if len(x) >= hop * 3:
                nfr = len(x) // hop
                fr = x[: nfr * hop].reshape(nfr, hop)
                db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
                thr = max(-48.0, float(np.max(db)) - 28.0)
                on = db > thr
                islands, i = [], 0
                while i < nfr:
                    if on[i]:
                        j = i
                        while j < nfr and on[j]:
                            j += 1
                        islands.append([i, j])
                        i = j
                    else:
                        i += 1
                merged = []
                for s0, e0 in islands:                   # bridge gaps < 250 ms
                    if merged and s0 - merged[-1][1] <= 12:
                        merged[-1][1] = e0
                    else:
                        merged.append([s0, e0])
                for s0, e0 in merged:
                    if e0 - s0 >= 12:                    # keep islands >= 240 ms
                        st = max(c["start"], (a + s0 * hop) / sr)
                        en = min(c["end"], (a + e0 * hop) / sr)
                        if en - st >= 0.2:
                            seg["words"].append({"start": st, "end": en})
            if not seg["words"]:
                n_empty += 1
            out.append(seg)
        voiced = sum(w["end"] - w["start"] for s in out for w in s["words"])
        _log(f"VAD: {voiced:.0f}s of voiced speech inside subtitle windows; {n_empty} cue(s) with no clear voice.")
        return out

    def _map_from_diarization(self, proj: DubProject, segments: List[dict]) -> None:
        by_id = {s["id"]: s for s in segments}
        for c in proj.cues:
            s = by_id.get(c["id"])
            if not s:
                c.update(speaker_id=None, confidence=0.0, flagged=True, flag_reason="no_speaker_detected")
                continue
            flags = set(s.get("flags", [])) & AMBIGUOUS_FLAGS
            reason = ",".join(sorted(flags))
            if s["confidence"] < DIAR_LOW_CONF:
                reason = (reason + "," if reason else "") + "low_confidence"
            c.update(
                speaker_id=s["speaker_id"],
                confidence=round(float(s["confidence"]), 3),
                flagged=bool(reason),
                flag_reason=reason,
            )
        self.save_speaker_map(proj)
        spk = sorted({c["speaker_id"] for c in proj.cues if c["speaker_id"]})
        n_flag = sum(1 for c in proj.cues if c["flagged"])
        _log(f"Speaker map: {len(spk)} speakers {spk}; {n_flag} cue(s) flagged for review.")

    def save_speaker_map(self, proj: DubProject) -> Path:
        rows = [
            {
                "subtitle_id": c["id"], "start": round(c["start"], 3), "end": round(c["end"], 3),
                "text": c["text"], "speaker": c.get("speaker_id"),
                "confidence": c.get("confidence"), "flagged": bool(c.get("flagged")),
                "flag_reason": c.get("flag_reason") or "",
            }
            for c in proj.cues
        ]
        proj.speaker_map_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return proj.speaker_map_json

    def _apply_saved_map(self, proj: DubProject) -> List[SubtitleCue]:
        try:
            rows = json.loads(proj.speaker_map_json.read_text(encoding="utf-8"))
        except Exception:
            return proj.cue_objects()
        by_id = {r["subtitle_id"]: r for r in rows}
        for c in proj.cues:
            r = by_id.get(c["id"])
            if r:
                c.update(
                    speaker_id=r.get("speaker"), confidence=r.get("confidence"),
                    flagged=bool(r.get("flagged")), flag_reason=r.get("flag_reason") or "",
                )
        return proj.cue_objects()

    def set_speaker(self, proj: DubProject, cue_id: int, speaker_id: Optional[str]) -> None:
        for c in proj.cues:
            if c["id"] == cue_id:
                c["speaker_id"] = speaker_id or None
                c["flagged"] = False
                c["flag_reason"] = "manual" if speaker_id else "no_speaker"
                c["confidence"] = 1.0 if speaker_id else 0.0
        self.save_speaker_map(proj)

    def merge_speakers(self, proj: DubProject, old_id: str, new_id: str) -> int:
        n = 0
        for c in proj.cues:
            if c.get("speaker_id") == old_id:
                c["speaker_id"] = new_id
                c["flag_reason"] = "manual_merge"
                n += 1
        self.save_speaker_map(proj)
        return n

    def speaker_stats(self, proj: DubProject) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for c in proj.cues:
            spk = c.get("speaker_id") or "UNASSIGNED"
            s = out.setdefault(spk, {"cues": 0, "seconds": 0.0, "conf_sum": 0.0, "flagged": 0})
            s["cues"] += 1
            s["seconds"] += c["end"] - c["start"]
            s["conf_sum"] += float(c.get("confidence") or 0.0)
            s["flagged"] += 1 if c.get("flagged") else 0
        for s in out.values():
            s["avg_conf"] = s["conf_sum"] / max(1, s["cues"])
        return out

    def extract_cue_audio(self, proj: DubProject, cue_id: int, pad: float = 0.25) -> Optional[Path]:
        """Slice the ORIGINAL audio for one cue (for review playback)."""
        cue = next((c for c in proj.cues if c["id"] == cue_id), None)
        if not cue or not proj.source_wav.exists():
            return None
        st = max(0.0, cue["start"] - pad)
        en = min(proj.source_duration, cue["end"] + pad)
        out = proj.work_dir / "review" / f"cue_{cue_id:05d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), _read_slice(proj.source_wav, st, en - st), SR)
        return out

    # ==================================================================
    # 4. VOICE REFERENCES
    # ==================================================================
    def _map_hash(self, proj: DubProject) -> str:
        return _sha(*[(c["id"], c.get("speaker_id")) for c in proj.cues])

    def build_refs(self, proj: DubProject, force: bool = False, progress: ProgressCB = None) -> Dict[str, Dict[str, Any]]:
        map_hash = self._map_hash(proj)
        if proj.refs_json.exists() and not force:
            data = json.loads(proj.refs_json.read_text(encoding="utf-8"))
            profs = data.get("profiles", {})
            if (data.get("map_hash") == map_hash
                and profs
                and all(Path(p["ref_audio_path"]).exists() for p in profs.values())
                and not any(_is_hallucination(p.get("ref_text", "")) for p in profs.values())):
                _log("Using cached speaker references.")
                return profs

        self.prepare_audio(proj, progress)
        _p(progress, 0.32, "Building clean per-speaker voice references...")
        proj.refs_dir.mkdir(parents=True, exist_ok=True)
        voiced: Dict[int, List[List[float]]] = {}
        if proj.diarization_json.exists():
            for s in json.loads(proj.diarization_json.read_text(encoding="utf-8")).get("segments", []):
                voiced[s["id"]] = s.get("voiced", [])

        # Collect candidate segments per speaker
        spk_cands: Dict[str, List[Dict[str, Any]]] = {}
        for c in proj.cues:
            spk = c.get("speaker_id")
            if not spk:
                continue
            if spk not in spk_cands:
                spk_cands[spk] = []

            reason = c.get("flag_reason") or ""
            flags = [f for f in reason.split(",") if f in AMBIGUOUS_FLAGS]
            conf = float(c.get("confidence") or 0.5)

            isl = voiced.get(c["id"]) or []
            if voiced and not isl:
                continue  # no detectable speech in this cue
            st = isl[0][0] if isl else c["start"]
            en = isl[-1][1] if isl else c["end"]
            dur = en - st
            if dur < 0.8:
                continue

            # Score candidate: sweet spot is 3.5s - 7.5s, high confidence, no ambiguous flags
            score = 0.0
            if not flags:
                score += 50.0
            score += conf * 20.0
            if 3.5 <= dur <= 7.5:
                score += 100.0 - abs(dur - 5.5) * 5.0
            elif dur > 7.5:
                score += 70.0 - (dur - 7.5) * 2.0
            else:  # 0.8 <= dur < 3.5
                score += dur * 15.0

            spk_cands[spk].append({
                "cue_id": c["id"],
                "start": st,
                "end": en,
                "dur": dur,
                "confidence": conf,
                "flags": flags,
                "score": score,
            })

        if not spk_cands:
            raise RuntimeError("No cues have a speaker assigned — run speaker detection first.")

        for spk in spk_cands:
            spk_cands[spk].sort(key=lambda x: x["score"], reverse=True)

        voc_audio, voc_sr = sf.read(str(proj.vocals_wav), dtype="float32", always_2d=True)
        voc_mono = voc_audio.mean(axis=1)
        del voc_audio

        profiles: Dict[str, Dict[str, Any]] = {}
        _p(progress, 0.35, "Evaluating clean reference utterances with Whisper...")

        with _Whisper(self.whisper_model) as w:
            for spk_idx, (spk, cands) in enumerate(spk_cands.items()):
                _p(progress, 0.35 + 0.08 * spk_idx / max(1, len(spk_cands)), f"Finding clean voice reference for {spk}...")
                best_single = None
                valid_shorts = []

                # Evaluate top candidates (up to 5) for this speaker
                for cand in cands[:5]:
                    st_s = int(round(cand["start"] * voc_sr))
                    en_s = int(round(cand["end"] * voc_sr))
                    sub = voc_mono[st_s:en_s]
                    if len(sub) < int(0.6 * voc_sr):
                        continue

                    try:
                        det = w.detail(sub, words=True)
                    except Exception as e:
                        _log(f"Whisper detail error on candidate cue #{cand['cue_id']}: {e}")
                        continue

                    # Filter hallucinations, high silence probability, or empty text
                    if not det["text"].strip() or _is_hallucination(det["text"]):
                        continue
                    if det["no_speech"] > 0.40 or len(det.get("words", [])) < 2:
                        continue

                    # Check word boundaries
                    words = det["words"]
                    w0, w1 = words[0]["start"], words[-1]["end"]
                    w_span = w1 - w0
                    if w_span < 0.5:
                        continue

                    # Trim to exact words span with 40ms safety pad
                    t0 = max(0, int(round((w0 - 0.04) * voc_sr)))
                    t1 = min(len(sub), int(round((w1 + 0.04) * voc_sr)))
                    trimmed = sub[t0:t1]
                    dur_trimmed = len(trimmed) / voc_sr

                    cand_info = {
                        "cue_id": cand["cue_id"],
                        "audio": trimmed,
                        "duration": dur_trimmed,
                        "text": det["text"].strip(),
                        "language": det["language"],
                        "start_time": cand["start"] + w0,
                        "end_time": cand["start"] + w1,
                    }

                    # Single clean utterance between 2.5s and 8.5s is ideal
                    if 2.5 <= dur_trimmed <= 8.5:
                        best_single = cand_info
                        break
                    elif dur_trimmed > 8.5:
                        cand_info["audio"] = trimmed[: int(8.0 * voc_sr)]
                        cand_info["duration"] = 8.0
                        best_single = cand_info
                        break
                    else:
                        valid_shorts.append(cand_info)

                # Determine final audio and text
                if best_single is not None:
                    final_audio = best_single["audio"]
                    final_text = best_single["text"]
                    final_lang = best_single["language"]
                    used_regions = [{
                        "subtitle_id": best_single["cue_id"],
                        "start": round(best_single["start_time"], 3),
                        "end": round(best_single["end_time"], 3),
                    }]
                elif len(valid_shorts) >= 2:
                    # Combine top 2 clean short utterances with 40ms crossfade
                    c1, c2 = valid_shorts[0], valid_shorts[1]
                    final_audio = _crossfade_join(c1["audio"], c2["audio"], crossfade_ms=40.0, sr=voc_sr)
                    final_text = f"{c1['text']} {c2['text']}"
                    final_lang = c1["language"]
                    used_regions = [
                        {"subtitle_id": c1["cue_id"], "start": round(c1["start_time"], 3), "end": round(c1["end_time"], 3)},
                        {"subtitle_id": c2["cue_id"], "start": round(c2["start_time"], 3), "end": round(c2["end_time"], 3)},
                    ]
                elif len(valid_shorts) == 1:
                    c1 = valid_shorts[0]
                    final_audio = c1["audio"]
                    final_text = c1["text"]
                    final_lang = c1["language"]
                    used_regions = [{"subtitle_id": c1["cue_id"], "start": round(c1["start_time"], 3), "end": round(c1["end_time"], 3)}]
                else:
                    # Raw candidate fallback
                    top = cands[0]
                    st_s = int(round(top["start"] * voc_sr))
                    en_s = int(round(top["end"] * voc_sr))
                    final_audio = voc_mono[st_s:en_s][: int(7.0 * voc_sr)]
                    try:
                        det = w.detail(final_audio, words=False)
                        final_text = det["text"].strip()
                        final_lang = det["language"]
                    except Exception:
                        final_text = ""
                        final_lang = ""
                    used_regions = [{"subtitle_id": top["cue_id"], "start": top["start"], "end": top["end"]}]

                # Gentle loudness normalisation: peak to -1.4 dBFS (0.85)
                pk = float(np.max(np.abs(final_audio)))
                if pk > 1e-4:
                    final_audio = final_audio * (0.85 / pk)

                # Ensure reference text ends with trailing space for F5-TTS token boundary
                clean_ref_text = final_text.strip()
                if clean_ref_text and not clean_ref_text.endswith(" "):
                    clean_ref_text += " "

                out_wav = proj.refs_dir / f"{spk}.wav"
                sf.write(str(out_wav), final_audio, voc_sr)
                dur_s = round(len(final_audio) / voc_sr, 2)

                profiles[spk] = {
                    "speaker_id": spk,
                    "ref_audio_path": str(out_wav),
                    "ref_text": clean_ref_text,
                    "ref_language": final_lang,
                    "ref_hash": _sha(_file_sha(out_wav), clean_ref_text),
                    "duration_sec": dur_s,
                    "low_confidence": (dur_s < 2.0 or not clean_ref_text),
                    "source_regions": used_regions,
                }
                _log(f"  [{spk}] Clean ref built: {dur_s:.1f}s ({final_lang}) \"{clean_ref_text[:70]}\"")

        proj.refs_json.write_text(
            json.dumps({"map_hash": map_hash, "profiles": profiles}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return profiles

    # ==================================================================
    # 5. VOICE PREVIEW
    # ==================================================================
    def preview_voices(
        self,
        proj: DubProject,
        text: str = "Hello, this is a test of the voice cloning system.",
        progress: ProgressCB = None,
    ) -> Dict[str, Dict[str, str]]:
        profiles = self.build_refs(proj, progress=progress)
        out_dir = proj.work_dir / "voice_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for i, (spk, prof) in enumerate(sorted(profiles.items())):
            _p(progress, 0.5 + 0.45 * i / max(1, len(profiles)), f"Cloning test line for {spk}...")
            out = out_dir / f"{spk}_{_sha(text, prof['ref_hash'])[:8]}.wav"
            if not out.exists():
                ok = self.voice_cloner.clone_and_speak(
                    text=text, ref_audio_path=prof["ref_audio_path"], output_path=out,
                    ref_text=prof["ref_text"], target_duration=None, seed=1234,
                )
                if not ok:
                    continue
            result[spk] = {"reference": prof["ref_audio_path"], "clone": str(out)}
        self.voice_cloner.unload_model()
        _free_gpu()
        return result

    # ==================================================================
    # 6. TTS
    # ==================================================================
    def _plan_cue(self, cue: Dict[str, Any], next_start: float, source_duration: float) -> Dict[str, Any]:
        window = cue["end"] - cue["start"]
        allowed_end = min(source_duration, max(cue["end"], min(next_start, cue["end"] + OVERFLOW_GRACE)))
        max_window = max(window, allowed_end - cue["start"])
        natural = natural_duration(cue["text"])
        flag = ""
        if natural <= max_window:
            target = None
        else:
            target = max(max_window, natural / MAX_SPEEDUP)
            if target > max_window + 0.05:
                flag = "too_long_for_window"
        return {
            "window": window, "max_window": max_window, "natural": natural,
            "target": round(target, 3) if target is not None else None,
            "target_duration": round(target, 3) if target is not None else None,
            "required_speedup": natural / max_window if max_window > 0 else 99.0,
            "flag": flag,
        }

    def _synth_one(self, cue, plan, prof, seed: int) -> Tuple[Optional[Path], float]:
        target_dur = plan.get("target_duration") if "target_duration" in plan else plan.get("target")
        key = _sha(cue["text"], cue["speaker_id"], prof["ref_hash"], str(target_dur), self.voice_cloner.nfe_step, seed)
        out = self._tts_path(cue, key)
        if not out.exists():
            ok = self.voice_cloner.clone_and_speak(
                text=cue["text"], ref_audio_path=prof["ref_audio_path"], output_path=out,
                ref_text=prof["ref_text"], target_duration=target_dur, seed=seed,
            )
            if not ok or not out.exists():
                return None, 0.0
        return out, float(sf.info(str(out)).duration)

    def _tts_path(self, cue, key: str) -> Path:
        return self._current_tts_dir / f"cue_{cue['id']:05d}_{cue['speaker_id']}_{key[:10]}.wav"

    def synthesize(
        self,
        proj: DubProject,
        cues: List[Dict[str, Any]],
        profiles: Dict[str, Dict[str, Any]],
        progress: ProgressCB = None,
        p0: float = 0.45,
        p1: float = 0.72,
    ) -> List[Dict[str, Any]]:
        self._current_tts_dir = proj.tts_dir
        starts = sorted(c["start"] for c in proj.cues)
        results = []
        for i, cue in enumerate(cues):
            _p(progress, p0 + (p1 - p0) * i / max(1, len(cues)),
               f"TTS {i+1}/{len(cues)} [{cue.get('speaker_id') or '?'}] {cue['text'][:40]}")
            nxt = next((s for s in starts if s > cue["start"] + 1e-3), proj.source_duration)
            plan = self._plan_cue(cue, nxt, proj.source_duration)
            rec = {
                "id": cue["id"], "start": cue["start"], "end": cue["end"], "text": cue["text"],
                "speaker_id": cue.get("speaker_id"), "speaker_confidence": cue.get("confidence"),
                "plan": plan, "wav_path": None, "generated_duration": 0.0, "seed": 1000 + cue["id"],
                "notes": [],
            }
            if plan["flag"]:
                rec["notes"].append(f"{plan['flag']} (needs {plan['required_speedup']:.2f}x)")
            if cue.get("flagged") and cue.get("flag_reason"):
                rec["notes"].append(f"speaker:{cue['flag_reason']}")
            spk = cue.get("speaker_id")
            if not spk or spk not in profiles:
                rec["notes"].append("NO_VOICE: speaker unassigned or has no reference — line left silent")
                results.append(rec)
                continue
            wav, dur = self._synth_one(cue, plan, profiles[spk], rec["seed"])
            if wav is None:
                rec["notes"].append("TTS_FAILED")
            rec["wav_path"] = str(wav) if wav else None
            rec["generated_duration"] = round(dur, 3)
            results.append(rec)
        self.voice_cloner.unload_model()
        _free_gpu()
        return results

    # ==================================================================
    # 7. QC (ASR text match + duration + speaker similarity), one retry
    # ==================================================================
    def _speaker_similarity(self, items: List[Tuple[Path, Path]]) -> List[Optional[float]]:
        try:
            from diarizer import load_mono_16k, l2_normalize  # noqa: F401
            diar = self._get_diarizer()
            emb_cache: Dict[str, np.ndarray] = {}

            def emb(path: Path) -> np.ndarray:
                k = str(path)
                if k not in emb_cache:
                    audio = load_mono_16k(path)
                    win = [{"start": 0.0, "end": len(audio) / 16000.0}]
                    emb_cache[k] = diar._extract_window_embeddings(audio, win)[0]
                return emb_cache[k]

            sims = []
            for gen, ref in items:
                try:
                    sims.append(round(float(emb(gen) @ emb(ref)), 3))
                except Exception:
                    sims.append(None)
            return sims
        finally:
            self._release_diarizer()

    def quality_check(
        self,
        proj: DubProject,
        recs: List[Dict[str, Any]],
        profiles: Dict[str, Dict[str, Any]],
        retry: bool = True,
        progress: ProgressCB = None,
    ) -> List[Dict[str, Any]]:
        todo = [r for r in recs if r.get("wav_path")]
        if not todo:
            return recs

        def run_asr(items):
            with _Whisper(self.whisper_model) as w:
                for j, r in enumerate(items):
                    _p(progress, 0.73 + 0.07 * j / max(1, len(items)), f"QC ASR {j+1}/{len(items)}")
                    text, _ = w.transcribe(Path(r["wav_path"]), language="en")
                    exp, got = _norm_words(r["text"]), _norm_words(text)
                    r["asr_text"] = text
                    r["asr_similarity"] = round(difflib.SequenceMatcher(None, exp, got).ratio(), 3) if exp else 1.0

        run_asr(todo)
        if retry:
            thr = lambda r: (0.5 if len(_norm_words(r["text"])) <= 2 else QC_TEXT_PASS)
            bad = [r for r in todo if r["asr_similarity"] < thr(r)]
            if bad:
                _p(progress, 0.80, f"QC: regenerating {len(bad)} clip(s) that failed text check...")
                self._current_tts_dir = proj.tts_dir
                cue_by_id = {c["id"]: c for c in proj.cues}
                retried = []
                for r in bad:
                    cue = cue_by_id[r["id"]]
                    wav, dur = self._synth_one(cue, r["plan"], profiles[r["speaker_id"]], r["seed"] + 7919)
                    if wav:
                        retried.append((r, {"wav_path": str(wav), "generated_duration": round(dur, 3), "text": r["text"]}))
                self.voice_cloner.unload_model()
                _free_gpu()
                if retried:
                    run_asr([alt for _, alt in retried])
                    for r, alt in retried:
                        if alt["asr_similarity"] > r["asr_similarity"]:
                            r.update(wav_path=alt["wav_path"], generated_duration=alt["generated_duration"],
                                     asr_text=alt["asr_text"], asr_similarity=alt["asr_similarity"])
                            r["notes"].append("regenerated_after_qc")

        sims = self._speaker_similarity([(Path(r["wav_path"]), Path(profiles[r["speaker_id"]]["ref_audio_path"])) for r in todo])
        for r, s in zip(todo, sims):
            r["speaker_similarity"] = s
            thr_t = 0.5 if len(_norm_words(r["text"])) <= 2 else QC_TEXT_PASS
            text_ok = r.get("asr_similarity", 0) >= thr_t
            dur_ok = r["generated_duration"] <= r["plan"]["max_window"] + 0.10
            spk_ok = s is None or s >= QC_SPK_PASS
            r["qc_text"] = "PASS" if text_ok else "FAIL"
            r["qc_duration"] = "PASS" if dur_ok else "OVERRUN"
            r["qc_speaker"] = "PASS" if spk_ok else "LOW"
            r["tts_qc"] = "PASS" if (text_ok and dur_ok and spk_ok) else "REVIEW"
        return recs

    # ==================================================================
    # 8. MIX (sample-accurate canvas + dialogue suppression)
    # ==================================================================
    def mix(
        self,
        proj: DubProject,
        recs: List[Dict[str, Any]],
        range_cues: List[Dict[str, Any]],
        start: float,
        end: float,
        out_dir: Path,
        me_track: Optional[Path] = None,
        progress: ProgressCB = None,
    ) -> Dict[str, Path]:
        _p(progress, 0.86, "Mixing: original soundtrack − original dialogue + English dub...")
        dur = end - start
        n = int(round(dur * SR))
        orig = _read_slice(proj.source_wav, start, dur)
        if me_track:
            bg = _read_slice(Path(me_track), start, dur)
            mode = "M&E track"
        else:
            bg = _read_slice(proj.background_wav, start, dur)
            mode = "Demucs background"
        vocals = _read_slice(proj.vocals_wav, start, dur).mean(axis=1)

        # -- English dialogue canvas ------------------------------------
        canvas = np.zeros((n, 2), dtype=np.float32)
        rendered_end: Dict[int, float] = {}
        from scipy.signal import butter, sosfilt
        hp = butter(2, 70, btype="highpass", fs=SR, output="sos")
        for r in recs:
            if not r.get("wav_path"):
                continue
            clip, csr = sf.read(r["wav_path"], dtype="float32", always_2d=True)
            clip = _to_stereo(_resample(clip, csr, SR))
            clip = sosfilt(hp, clip, axis=0).astype(np.float32)

            # loudness: match original speaker level in this cue window
            a = max(0, int(round((r["start"] - start) * SR)))
            b = min(n, int(round((r["end"] - start) * SR)))
            orig_rms = _rms(vocals[a:b]) if b > a else 0.0
            clip_rms = _rms(clip)
            target = orig_rms if orig_rms > 10 ** (-40 / 20) else 10 ** (-22 / 20)
            target = float(np.clip(target, 10 ** (-32 / 20), 10 ** (-12 / 20)))
            gain = float(np.clip(target / max(clip_rms, 1e-6), 10 ** (-12 / 20), 10 ** (12 / 20)))
            clip *= gain
            r["gain_db"] = round(20 * math.log10(gain), 1)

            edge = min(len(clip) // 4, int(SR * 0.008))
            if edge > 0:
                clip[:edge] *= np.linspace(0, 1, edge, dtype=np.float32)[:, None]
                clip[-edge:] *= np.linspace(1, 0, edge, dtype=np.float32)[:, None]

            s_idx = int(round((r["start"] - start) * SR))   # absolute anchor, never cumulative
            off = max(0, -s_idx)
            s_idx = max(0, s_idx)
            e_idx = min(n, s_idx + len(clip) - off)
            if e_idx > s_idx:
                canvas[s_idx:e_idx] += clip[off: off + (e_idx - s_idx)]
                rendered_end[r["id"]] = start + e_idx / SR

        # -- suppression mask: only where dialogue happens ---------------
        mask = np.zeros(n, dtype=np.float32)
        ramp = max(1, int(SUPPRESS_RAMP * SR))
        for c in range_cues:
            st = c["start"] - SUPPRESS_PAD
            en = max(c["end"], rendered_end.get(c["id"], c["end"])) + SUPPRESS_PAD
            a = int(round((st - start) * SR))
            b = int(round((en - start) * SR))
            a0, b0 = max(0, a), min(n, b)
            if b0 <= a0:
                continue
            mask[a0:b0] = 1.0
            ra0 = max(0, a0 - ramp)
            if a0 > ra0:
                mask[ra0:a0] = np.maximum(mask[ra0:a0], np.linspace(0, 1, a0 - ra0, dtype=np.float32))
            rb1 = min(n, b0 + ramp)
            if rb1 > b0:
                mask[b0:rb1] = np.maximum(mask[b0:rb1], np.linspace(1, 0, rb1 - b0, dtype=np.float32))

        m2 = mask[:, None]
        base = orig * (1.0 - m2) + bg * m2
        del orig, bg, vocals

        voice_only = out_dir / "dub_voice_only.wav"
        bg_only = out_dir / "background_only.wav"
        final_mix = out_dir / "final_mix.wav"
        sf.write(str(voice_only), canvas, SR, subtype="FLOAT")
        sf.write(str(bg_only), base, SR, subtype="FLOAT")
        base += canvas
        del canvas
        _soft_limit(base)
        sf.write(str(final_mix), base, SR, subtype="PCM_24")
        _log(f"Mix done ({mode}): {dur:.2f}s @ {SR} Hz — {final_mix}")
        return {"voice_only": voice_only, "background_only": bg_only, "final_mix": final_mix, "mode": mode}

    # ==================================================================
    # 9. RENDER (preview or full)
    # ==================================================================
    def render(
        self,
        proj: DubProject,
        start: float = 0.0,
        end: Optional[float] = None,
        me_track: Optional[str | Path] = None,
        run_qc: bool = True,
        progress: ProgressCB = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        end = proj.source_duration if end is None else min(float(end), proj.source_duration)
        start = max(0.0, float(start))
        if end - start < 1.0:
            raise ValueError("Render range must be at least 1 second.")
        is_full = start <= 0.01 and end >= proj.source_duration - 0.01

        tag = "full" if is_full else f"preview_{int(start)}-{int(end)}s"
        out_dir = proj.out_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        self.prepare_audio(proj, progress)
        if not any(c.get("speaker_id") for c in proj.cues):
            self.diarize(proj, progress=progress)
        profiles = self.build_refs(proj, progress=progress)

        range_cues = [c for c in proj.cues if start <= c["start"] < end]
        _log(f"Render {tag}: {len(range_cues)} cue(s) in {start:.2f}s → {end:.2f}s")
        recs = self.synthesize(proj, range_cues, profiles, progress)
        if run_qc:
            recs = self.quality_check(proj, recs, profiles, progress=progress)

        mixed = self.mix(proj, recs, range_cues, start, end, out_dir,
                         me_track=Path(me_track) if me_track else None, progress=progress)

        # -- video + subtitles --------------------------------------------
        _p(progress, 0.93, "Muxing with original video...")
        if is_full:
            video_in = Path(proj.video_path)
            sub_out = out_dir / Path(proj.subtitle_path).name       # user file, unchanged
            shutil.copy2(proj.subtitle_path, sub_out)
            out_mp4 = out_dir / "english_dubbed_final.mp4"
            master = out_dir / "master_dub.wav"
        else:
            video_in = out_dir / "_video_clip.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", proj.video_path,
                 "-t", f"{end - start:.3f}", "-map", "0:v:0", "-an", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(video_in)],
                check=True,
            )
            shifted = []
            for c in range_cues:
                sc = cue_from_dict(c)
                sc.start, sc.end = c["start"] - start, min(c["end"], end) - start
                shifted.append(sc)
            sub_out = cues_to_srt(shifted, out_dir / "preview_subtitles.srt")
            out_mp4 = out_dir / "preview_dubbed.mp4"
            master = out_dir / "master_dub.wav"
        shutil.copy2(mixed["final_mix"], master)
        mux_audio_and_video(
            video_path=video_in, dubbed_audio_path=master, output_mp4_path=out_mp4,
            subtitle_path=sub_out, output_mp3_path=None, source_duration=end - start,
        )
        if not is_full:
            video_in.unlink(missing_ok=True)

        # -- debug artefacts ------------------------------------------------
        shutil.copy2(proj.speaker_map_json, out_dir / "speaker_map.json")
        refs_out = out_dir / "speaker_refs"
        if refs_out.exists():
            shutil.rmtree(refs_out)
        shutil.copytree(proj.refs_dir, refs_out)
        if not is_full:
            tts_out = out_dir / "tts_segments"
            if tts_out.exists():
                shutil.rmtree(tts_out)
            tts_out.mkdir()
            for r in recs:
                if r.get("wav_path"):
                    shutil.copy2(r["wav_path"], tts_out / Path(r["wav_path"]).name)
        report = self.write_report(recs, out_dir / "dub_report.csv")
        (out_dir / "qc.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        _p(progress, 1.0, f"Done in {time.time() - t0:.0f}s → {out_mp4}")
        n_review = sum(1 for r in recs if r.get("tts_qc") == "REVIEW" or not r.get("wav_path"))
        return {
            "output_mp4": str(out_mp4), "master_dub": str(master), "subtitles": str(sub_out),
            "report_csv": str(report), "out_dir": str(out_dir),
            "voice_only": str(mixed["voice_only"]), "background_only": str(mixed["background_only"]),
            "mix_mode": mixed["mode"], "num_cues": len(range_cues),
            "num_voiced": sum(1 for r in recs if r.get("wav_path")),
            "num_review": n_review,
            "speakers": sorted({r["speaker_id"] for r in recs if r.get("speaker_id")}),
            "records": recs,
        }

    # ------------------------------------------------------------------
    def write_report(self, recs: List[Dict[str, Any]], path: Path) -> Path:
        fields = [
            "subtitle_id", "start", "end", "speaker", "text", "target_duration", "planned_duration",
            "generated_duration", "required_speedup", "voice_engine", "speaker_confidence",
            "asr_text", "asr_similarity", "speaker_similarity", "gain_db", "tts_qc", "notes",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in recs:
                p = r.get("plan", {})
                planned_str = f"{p['target']:.3f}" if p.get("target") is not None else "natural"
                gen_dur = r.get("generated_duration") or 0.0
                req_speed = p.get("required_speedup") or 1.0
                w.writerow({
                    "subtitle_id": r.get("id", ""),
                    "start": f"{(r.get('start') or 0.0):.3f}",
                    "end": f"{(r.get('end') or 0.0):.3f}",
                    "speaker": r.get("speaker_id") or "",
                    "text": r.get("text", ""),
                    "target_duration": f"{(p.get('window') or 0.0):.3f}",
                    "planned_duration": planned_str,
                    "generated_duration": f"{gen_dur:.3f}",
                    "required_speedup": f"{req_speed:.2f}",
                    "voice_engine": self.tts_backend,
                    "speaker_confidence": r.get("speaker_confidence") or "",
                    "asr_text": r.get("asr_text", ""),
                    "asr_similarity": r.get("asr_similarity", ""),
                    "speaker_similarity": r.get("speaker_similarity", ""),
                    "gain_db": r.get("gain_db", ""),
                    "tts_qc": r.get("tts_qc", "" if r.get("wav_path") else "NO_AUDIO"),
                    "notes": "; ".join(r.get("notes", [])),
                })
        return path
