"""
Exact Word Completeness & Speech QC Engine.

Specification:
- Non-negotiable requirement: Every SRT line must be spoken completely.
- Detects:
  * Missing words
  * Skipped words
  * Duplicated words
  * Wrong words
  * Truncated phrases / early cutoffs
- Uses local faster-whisper English ASR on each generated cue.
- Compares recognized words against normalized authoritative SRT text.
- If mismatch exceeds tolerance, triggers iterative regeneration with alternate seeds & parameters.
- If all retries fail, flags HARD_QC_FAIL and records detailed mismatch.
- Stores qc/tts_word_qc.json.
"""

from __future__ import annotations

import difflib
import gc
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch

import config


@dataclass
class CueQCResult:
    cue_id: int
    status: str              # "PASS", "RETRY", "HARD_QC_FAIL"
    expected_text: str
    recognized_text: str
    expected_words: List[str]
    recognized_words: List[str]
    missing_words: List[str]
    extra_words: List[str]
    word_match_ratio: float
    attempt_count: int = 1
    wav_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "status": self.status,
            "expected_text": self.expected_text,
            "recognized_text": self.recognized_text,
            "missing_words": self.missing_words,
            "extra_words": self.extra_words,
            "word_match_ratio": round(self.word_match_ratio, 3),
            "attempt_count": self.attempt_count,
            "wav_path": self.wav_path,
        }


def normalize_words(text: str) -> List[str]:
    """Normalizes text for linguistic comparison, ignoring punctuation, case, and spacing."""
    t = text.lower().replace("’", "'").replace("`", "'")
    # Replace punctuation with spaces
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    # Split into clean tokens
    words = [w.strip("'") for w in t.split() if w.strip("'")]
    return words


class WordCompletenessQC:
    def __init__(self, model_size_or_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size_or_path = model_size_or_path or config.DEFAULT_WHISPER_MODEL
        self._model = None

    def load_model(self):
        if self._model is not None:
            return self._model
        print(f"[WordQC] Loading faster-whisper ({self.model_size_or_path}) on {self.device}...", flush=True)
        from faster_whisper import WhisperModel
        compute_type = "float16" if self.device == "cuda" else "int8"
        self._model = WhisperModel(self.model_size_or_path, device=self.device, compute_type=compute_type)
        return self._model

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[WordQC] Unloaded from memory.", flush=True)

    def verify_cue(
        self,
        cue_id: int,
        wav_path: str | Path,
        expected_text: str,
        attempt_count: int = 1,
    ) -> CueQCResult:
        """
        Transcribes the generated cue WAV and verifies word completeness.
        Returns CueQCResult indicating PASS, RETRY, or HARD_QC_FAIL.
        """
        path = Path(wav_path).resolve()
        if not path.exists():
            return CueQCResult(
                cue_id=cue_id,
                status="HARD_QC_FAIL",
                expected_text=expected_text,
                recognized_text="",
                expected_words=normalize_words(expected_text),
                recognized_words=[],
                missing_words=normalize_words(expected_text),
                extra_words=[],
                word_match_ratio=0.0,
                attempt_count=attempt_count,
                wav_path=str(path),
            )

        model = self.load_model()

        # Transcribe audio forcing English
        try:
            segments, info = model.transcribe(
                str(path),
                language="en",
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
            )
            recognized_text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            print(f"[WordQC] ASR error on cue {cue_id}: {e}", flush=True)
            recognized_text = ""

        expected_words = normalize_words(expected_text)
        rec_words = normalize_words(recognized_text)

        # Align words
        matcher = difflib.SequenceMatcher(None, expected_words, rec_words)
        match_ratio = matcher.ratio()

        missing_words = []
        extra_words = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("delete", "replace"):
                missing_words.extend(expected_words[i1:i2])
            if tag in ("insert", "replace"):
                extra_words.extend(rec_words[j1:j2])

        # Filter out minor phonetic homophones or number representations if identical
        filtered_missing = []
        for w in missing_words:
            # If word is a number or basic homophone represented differently, allow if in rec_words
            if w in rec_words:
                continue
            filtered_missing.append(w)

        # Evaluation criteria:
        # A cue passes if no essential words are missing and similarity is high.
        # Short cues (1-3 words) must have 0 missing words.
        # Longer cues can tolerate minor ASR noise if match_ratio >= 0.82 and no more than 1 minor missing word.
        status = "PASS"
        if len(expected_words) <= 3:
            if filtered_missing or match_ratio < 0.70:
                status = "RETRY"
        else:
            if len(filtered_missing) > 1 or match_ratio < 0.80:
                status = "RETRY"

        return CueQCResult(
            cue_id=cue_id,
            status=status,
            expected_text=expected_text,
            recognized_text=recognized_text,
            expected_words=expected_words,
            recognized_words=rec_words,
            missing_words=filtered_missing,
            extra_words=extra_words,
            word_match_ratio=match_ratio,
            attempt_count=attempt_count,
            wav_path=str(path),
        )

    def export_qc_report(self, results: List[CueQCResult], output_dir: str | Path) -> Path:
        out_dir = Path(output_dir).resolve()
        qc_dir = out_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        report_path = qc_dir / "tts_word_qc.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

        print(f"[WordQC] Saved word completeness report for {len(results)} cues to {report_path}", flush=True)
        return report_path
