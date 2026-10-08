import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

# Ensure CUDA DLLs from torch/lib are discoverable by CTranslate2 on Windows
torch_lib_dir = Path(torch.__file__).parent / "lib"
if torch_lib_dir.exists():
    try:
        os.add_dll_directory(str(torch_lib_dir))
    except Exception:
        pass
    os.environ["PATH"] = str(torch_lib_dir) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel
from translator import ContextualRussianTranslator
from timeline_validator import validate_timeline_integrity
import config



def load_or_convert_to_16k_mono(audio_path: str | Path, temp_dir: Path | None = None) -> np.ndarray:
    """
    Safely load or resample audio to 16kHz mono float32 NumPy array.
    Uses FFmpeg for fast, robust resampling, completely bypassing PyAV metadata_errors bugs.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check sample rate first
    info = sf.info(str(audio_path))
    if info.samplerate == 16000 and info.channels == 1:
        data, _ = sf.read(str(audio_path), dtype="float32")
        return data

    # Use FFmpeg to convert to clean 16kHz mono WAV in temp
    import subprocess
    target_temp = temp_dir or audio_path.parent
    converted_wav = target_temp / f"{audio_path.stem}_16k_mono.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(converted_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    data, _ = sf.read(str(converted_wav), dtype="float32")
    return data


def format_debug_table(segments: list[dict]) -> str:
    """
    Formats segments into a clean side-by-side debug comparison table:
    Timestamp | Russian transcript | English translation
    """
    lines = [
        "=" * 100,
        f"{'Timestamp':<20} | {'Russian Transcript':<38} | {'English Translation'}",
        "=" * 100
    ]
    for seg in segments:
        start_m, start_s = divmod(seg["start"], 60)
        end_m, end_s = divmod(seg["end"], 60)
        ts_str = f"{int(start_m):02d}:{start_s:05.2f} -> {int(end_m):02d}:{end_s:05.2f}"
        ru_text = seg.get("russian_text", "").strip()
        en_text = seg.get("english_text", "").strip()

        # Truncate for table display if too long
        ru_disp = ru_text if len(ru_text) <= 36 else ru_text[:33] + "..."
        lines.append(f"{ts_str:<20} | {ru_disp:<38} | {en_text}")

    lines.append("=" * 100)
    return "\n".join(lines)


def export_russian_transcripts(segments: list[dict], export_dir: Path, base_name: str = "russian_transcript"):
    """
    Export Russian transcript in JSON, SRT, and TXT formats for inspection and debugging.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / f"{base_name}.json"
    txt_path = export_dir / f"{base_name}.txt"
    srt_path = export_dir / f"{base_name}.srt"

    # 1. JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # 2. Plain Text
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s.get('russian_text', '')}\n")

    # 3. Russian SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(segments, 1):
            def to_srt_time(sec):
                h, rem = divmod(sec, 3600)
                m, s_val = divmod(rem, 60)
                ms = int((s_val - int(s_val)) * 1000)
                return f"{int(h):02d}:{int(m):02d}:{int(s_val):02d},{ms:03d}"

            start_tc = to_srt_time(s["start"])
            end_tc = to_srt_time(s["end"])
            f.write(f"{idx}\n{start_tc} --> {end_tc}\n{s.get('russian_text', '')}\n\n")

    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "srt": str(srt_path)
    }


class RussianDubbingTranscriber:
    """
    Accurate Two-Stage Transcriber and Translator:
    Stage 1: Russian transcription with Whisper Large-V3 on original source audio,
             task='transcribe', language='ru', word_timestamps=True, VAD enabled.
    Stage 2: Independent context-aware English neural translation with slang/profanity preservation.
    """
    def __init__(self, model_size: str = None, device: str = None, compute_type: str = None):
        if model_size in ["large-v3", None] and Path(config.DEFAULT_WHISPER_MODEL).exists():
            self.model_size = config.DEFAULT_WHISPER_MODEL
        else:
            self.model_size = model_size or config.DEFAULT_WHISPER_MODEL
        self.device = device or config.DEVICE
        self.compute_type = compute_type or config.COMPUTE_TYPE
        self._model = None
        self.translator = ContextualRussianTranslator(device=self.device)

    def load_model(self):
        if self._model is None:
            print(f"Loading faster-whisper Large-V3 model '{self.model_size}' on {self.device} ({self.compute_type})...", flush=True)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True if Path(self.model_size).exists() else False
            )
            print("Faster-whisper Large-V3 model loaded successfully.", flush=True)
        return self._model

    def unload_models(self):
        """Unload Whisper and MarianMT from GPU VRAM to leave maximum headroom for TTS."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, "translator") and self.translator is not None:
            if hasattr(self.translator, "_model") and self.translator._model is not None:
                del self.translator._model
                self.translator._model = None
            if hasattr(self.translator, "_tokenizer") and self.translator._tokenizer is not None:
                del self.translator._tokenizer
                self.translator._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Whisper & Translation models unloaded from VRAM.", flush=True)

    def process_audio(
        self,
        audio_path: str | Path,
        export_dir: Path | None = None,
        debug_mode: bool = True,
        source_duration: float | None = None
    ) -> list[dict]:
        """
        Executes complete verified workflow:
        1. Transcribe Russian speech from the original audio using Whisper Large-V3 (task='transcribe').
        2. Validate timeline bounds against authoritative SOURCE_DURATION.
        3. Export Russian transcript (JSON/SRT/TXT) for debugging and verification.
        4. Translate Russian to English using separate context-aware neural translation.
        5. Validate translated timeline integrity.
        6. Output side-by-side debug comparison table.
        """
        model = self.load_model()
        export_dir = Path(export_dir) if export_dir else Path(audio_path).parent

        # Load 16kHz mono audio array
        audio_data = load_or_convert_to_16k_mono(audio_path, temp_dir=export_dir)

        print("Step 1: Transcribing Russian speech using Whisper Large-V3 (task='transcribe', language='ru')...", flush=True)
        segments_gen, info = model.transcribe(
            audio_data,
            language="ru",
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.25,
                min_silence_duration_ms=300,
                speech_pad_ms=250
            ),
            temperature=[0.0, 0.2, 0.4]
        )

        russian_segments = []
        for i, seg in enumerate(segments_gen):
            ru_text = seg.text.strip()
            if not ru_text:
                continue

            # Automatic silence-gap splitting (>0.8s) using word timestamps:
            # Prevents merging words with large silence gaps into single bloated segments
            if seg.words and len(seg.words) > 1:
                current_sub_words = [seg.words[0]]
                for w in seg.words[1:]:
                    gap = w.start - current_sub_words[-1].end
                    if gap > 0.8:
                        sub_text = " ".join(sw.word.strip() for sw in current_sub_words)
                        if sub_text:
                            russian_segments.append({
                                "id": len(russian_segments),
                                "start": round(current_sub_words[0].start, 2),
                                "end": round(current_sub_words[-1].end, 2),
                                "duration": round(current_sub_words[-1].end - current_sub_words[0].start, 2),
                                "russian_text": sub_text,
                                "words": [{"word": sw.word.strip(), "start": round(sw.start, 2), "end": round(sw.end, 2), "probability": round(sw.probability, 3)} for sw in current_sub_words]
                            })
                        current_sub_words = [w]
                    else:
                        current_sub_words.append(w)
                if current_sub_words:
                    sub_text = " ".join(sw.word.strip() for sw in current_sub_words)
                    if sub_text:
                        russian_segments.append({
                            "id": len(russian_segments),
                            "start": round(current_sub_words[0].start, 2),
                            "end": round(current_sub_words[-1].end, 2),
                            "duration": round(current_sub_words[-1].end - current_sub_words[0].start, 2),
                            "russian_text": sub_text,
                            "words": [{"word": sw.word.strip(), "start": round(sw.start, 2), "end": round(sw.end, 2), "probability": round(sw.probability, 3)} for sw in current_sub_words]
                        })
            else:
                words_data = []
                if seg.words:
                    for w in seg.words:
                        words_data.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 3)
                        })
                russian_segments.append({
                    "id": len(russian_segments),
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "duration": round(seg.end - seg.start, 2),
                    "russian_text": ru_text,
                    "words": words_data
                })

        print(f"Extracted {len(russian_segments)} Russian dialogue segments with word timestamps.", flush=True)

        # Filter out zero-duration and near-zero-duration segments that Whisper
        # occasionally emits (especially on long audio). These carry no usable
        # speech and would crash the timeline validator (start >= end).
        MIN_SEGMENT_DURATION = 0.05  # 50ms — anything shorter is a Whisper artifact
        pre_filter_count = len(russian_segments)
        russian_segments = [
            s for s in russian_segments
            if float(s["end"]) - float(s["start"]) >= MIN_SEGMENT_DURATION
        ]
        dropped = pre_filter_count - len(russian_segments)
        if dropped > 0:
            print(f"  Filtered out {dropped} zero/near-zero duration Whisper artifact segment(s).", flush=True)
            # Re-index IDs after filtering
            for i, seg in enumerate(russian_segments):
                seg["id"] = i

        if not russian_segments:
            return []

        # Validate Russian transcription timeline bounds against authoritative SOURCE_DURATION
        if source_duration is not None and source_duration > 0:
            print(f"Validating transcription timeline against SOURCE_DURATION ({source_duration:.3f}s)...", flush=True)
            validate_timeline_integrity(
                segments=russian_segments,
                source_duration=source_duration,
                stage_name="whisper_transcription",
                tolerance=0.20
            )

        # Export Russian transcript files internally for debugging
        exported_paths = export_russian_transcripts(russian_segments, export_dir=export_dir)
        print(f"Exported internal Russian transcript: {exported_paths['json']}", flush=True)

        # Step 2: Separate Context-Aware English Translation (NO Whisper direct translation!)
        print("Step 2: Performing separate context-aware English translation (preserving slang & emotion)...", flush=True)
        final_segments = self.translator.translate_segments(russian_segments)

        # Validate translated segments timeline bounds against authoritative SOURCE_DURATION
        if source_duration is not None and source_duration > 0:
            print(f"Validating translation timeline against SOURCE_DURATION ({source_duration:.3f}s)...", flush=True)
            validate_timeline_integrity(
                segments=final_segments,
                source_duration=source_duration,
                stage_name="contextual_translation",
                tolerance=0.20
            )

        # Save debug comparison
        debug_table = format_debug_table(final_segments)
        debug_file = export_dir / "transcript_debug.txt"
        debug_file.write_text(debug_table, encoding="utf-8")

        if debug_mode:
            print("\n" + debug_table + "\n", flush=True)

        return final_segments
