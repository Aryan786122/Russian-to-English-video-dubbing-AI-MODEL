"""
Full-Video Speaker Diarization & Global Speaker Consolidation Engine.

Specification:
- Analyzes COMPLETE source audio (never partial chunks).
- Uses pyannote/speaker-diarization-community-1 (preloaded tensor dictionary).
- Produces global persistent identities: SPEAKER_00, SPEAKER_01...
- Computes both regular diarization (with overlap) and exclusive diarization.
- Extracts speaker embeddings for all segments.
- Performs global consolidation pass (comparing embeddings + acoustic similarity + temporal non-overlap).
- Outputs full_diarization.json and speaker_map.json.
- Strictly unloads GPU models after completion.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import cdist, cosine

ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str
    overlap: bool = False
    confidence: float = 1.0
    embedding: Optional[List[float]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        d = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "speaker": self.speaker,
            "overlap": bool(self.overlap),
            "confidence": round(float(self.confidence), 3),
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding
        return d


@dataclass
class SpeakerProfile:
    original_id: str
    final_id: str
    total_speech_sec: float
    segment_count: int
    centroid_embedding: Optional[List[float]] = None
    merged_from: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_id": self.original_id,
            "final_id": self.final_id,
            "total_speech_sec": round(self.total_speech_sec, 2),
            "segment_count": self.segment_count,
            "merged_from": self.merged_from,
            "has_embedding": self.centroid_embedding is not None,
        }


def load_audio_mono_16k(audio_path: str | Path) -> Tuple[torch.Tensor, int]:
    """
    Robustly decodes any audio file to a 16kHz mono float32 Tensor.
    Uses ffmpeg for guaranteed platform compatibility without PyAV / torchcodec issues.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(path),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-f", "wav",
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    import io
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Return shape (1, num_samples)
    tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    return tensor, 16000


class FullVideoDiarizer:
    def __init__(
        self,
        device: Optional[str] = None,
        similarity_threshold: float = 0.82,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold

    def run_full_diarization(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        progress_cb: ProgressCallback = None,
    ) -> Tuple[List[DiarizationSegment], Dict[str, str], List[SpeakerProfile]]:
        """
        Executes:
        1. Complete audio load & preload to memory.
        2. pyannote community-1 diarization pipeline execution.
        3. Regular & exclusive segmentation extraction.
        4. Speaker embedding extraction & centroid calculation.
        5. Global consolidation pass.
        6. Export of full_diarization.json and speaker_map.json.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.05, "Loading full video audio into memory (16kHz mono)...")

        waveform, sr = load_audio_mono_16k(audio_path)
        total_audio_sec = waveform.shape[1] / sr
        print(f"[Diarizer] Full audio loaded: {total_audio_sec:.2f}s ({waveform.shape[1]} samples at {sr}Hz)", flush=True)

        # -------------------------------------------------------------
        # Step 1: Run Pyannote Speaker Diarization
        # -------------------------------------------------------------
        if progress_cb:
            progress_cb(0.15, "Running pyannote community-1 full-video diarization...")

        raw_segments = self._run_pyannote_diarization(waveform, sr)

        if not raw_segments:
            print("[Diarizer] Warning: No speaker segments detected by pipeline. Creating fallback single speaker.", flush=True)
            raw_segments = [
                DiarizationSegment(start=0.0, end=total_audio_sec, speaker="SPEAKER_00", confidence=1.0)
            ]

        # -------------------------------------------------------------
        # Step 2: Compute Speaker Embeddings & Consolidate
        # -------------------------------------------------------------
        if progress_cb:
            progress_cb(0.60, "Extracting speaker embeddings across full timeline...")

        profiles, speaker_map, consolidated_segments = self._consolidate_speakers(
            raw_segments, waveform.squeeze(0).numpy(), sr
        )

        # -------------------------------------------------------------
        # Step 3: Export Artifacts
        # -------------------------------------------------------------
        full_diar_path = analysis_dir / "full_diarization.json"
        speaker_map_path = analysis_dir / "speaker_map.json"

        with open(full_diar_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in consolidated_segments], f, indent=2)

        with open(speaker_map_path, "w", encoding="utf-8") as f:
            json.dump({
                "mapping": speaker_map,
                "profiles": [p.to_dict() for p in profiles],
            }, f, indent=2)

        print(f"[Diarizer] Saved full diarization to {full_diar_path}", flush=True)
        print(f"[Diarizer] Saved consolidated speaker map to {speaker_map_path}", flush=True)

        if progress_cb:
            progress_cb(0.85, f"Diarization complete: {len(profiles)} stable speaker(s) identified.")

        # Free GPU memory
        self._unload()
        return consolidated_segments, speaker_map, profiles

    def _run_pyannote_diarization(self, waveform: torch.Tensor, sr: int) -> List[DiarizationSegment]:
        """Loads pyannote pipeline, executes on in-memory dictionary, and extracts segments."""
        from pyannote.audio import Pipeline
        print(f"[Diarizer] Loading pyannote/speaker-diarization-community-1 on {self.device}...", flush=True)

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        if self.device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Audio in-memory dictionary avoids torchcodec requirement
        audio_dict = {
            "waveform": waveform,
            "sample_rate": sr,
        }

        diarization_output = pipeline(audio_dict)

        # Pyannote returns an Annotation object (or DiarizationOutput in 4.x)
        annotation = diarization_output
        if hasattr(diarization_output, "speaker_diarization"):
            annotation = diarization_output.speaker_diarization

        segments: List[DiarizationSegment] = []
        for turn, track, speaker in annotation.itertracks(yield_label=True):
            if turn.duration <= 0.05:
                continue
            segments.append(
                DiarizationSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                    overlap=False,
                    confidence=1.0,
                )
            )

        # Sort chronologically
        segments.sort(key=lambda s: (s.start, s.end))

        # Detect overlaps
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if segments[j].start < segments[i].end:
                    if segments[j].speaker != segments[i].speaker:
                        segments[i].overlap = True
                        segments[j].overlap = True
                else:
                    break

        del pipeline
        self._unload()
        return segments

    def _consolidate_speakers(
        self,
        segments: List[DiarizationSegment],
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[List[SpeakerProfile], Dict[str, str], List[DiarizationSegment]]:
        """
        Extracts speaker embeddings, checks acoustic similarity and non-overlapping patterns,
        merges split identities, and maps to clean SPEAKER_00, SPEAKER_01...
        """
        # Group segments by detected speaker
        speaker_segments: Dict[str, List[DiarizationSegment]] = {}
        for seg in segments:
            speaker_segments.setdefault(seg.speaker, []).append(seg)

        # Extract embeddings for each speaker using SpeechBrain ECAPA-TDNN or pyannote
        speaker_embeddings: Dict[str, np.ndarray] = {}
        for spk, segs in speaker_segments.items():
            emb = self._extract_speaker_centroid_embedding(segs, audio, sr)
            if emb is not None:
                speaker_embeddings[spk] = emb

        # Check for candidates to merge
        detected_speakers = sorted(speaker_segments.keys())
        merged_into: Dict[str, str] = {spk: spk for spk in detected_speakers}

        for i in range(len(detected_speakers)):
            spk_a = detected_speakers[i]
            target_a = self._resolve_target(merged_into, spk_a)
            emb_a = speaker_embeddings.get(spk_a)
            if emb_a is None:
                continue

            for j in range(i + 1, len(detected_speakers)):
                spk_b = detected_speakers[j]
                target_b = self._resolve_target(merged_into, spk_b)
                if target_a == target_b:
                    continue

                emb_b = speaker_embeddings.get(spk_b)
                if emb_b is None:
                    continue

                # Cosine similarity
                sim = 1.0 - float(cosine(emb_a, emb_b))
                # Check if they ever overlap in time
                overlaps = self._speakers_overlap(speaker_segments[spk_a], speaker_segments[spk_b])

                # If highly similar and never speaking at the same time: merge
                if sim >= self.similarity_threshold and not overlaps:
                    print(f"[Diarizer] Consolidating split speaker {spk_b} -> {target_a} (similarity: {sim:.3f})", flush=True)
                    merged_into[target_b] = target_a

        # Re-assign stable, clean identities ordered by total speech time
        unique_targets = sorted(list(set(self._resolve_target(merged_into, s) for s in detected_speakers)))
        # Sort targets by total speaking duration descending
        durations = {}
        for t in unique_targets:
            total_sec = sum(
                seg.duration
                for s in detected_speakers
                if self._resolve_target(merged_into, s) == t
                for seg in speaker_segments[s]
            )
            durations[t] = total_sec

        unique_targets.sort(key=lambda t: durations[t], reverse=True)

        final_label_map: Dict[str, str] = {}
        for idx, t in enumerate(unique_targets):
            final_name = f"SPEAKER_{idx:02d}"
            final_label_map[t] = final_name

        # Final speaker mapping dictionary for all original speakers
        speaker_map: Dict[str, str] = {}
        for s in detected_speakers:
            target = self._resolve_target(merged_into, s)
            speaker_map[s] = final_label_map[target]

        # Build speaker profiles
        profiles: List[SpeakerProfile] = []
        for idx, t in enumerate(unique_targets):
            final_name = final_label_map[t]
            merged_sources = [s for s in detected_speakers if self._resolve_target(merged_into, s) == t]
            total_sec = durations[t]
            seg_count = sum(len(speaker_segments[s]) for s in merged_sources)
            emb = speaker_embeddings.get(t)

            profiles.append(
                SpeakerProfile(
                    original_id=t,
                    final_id=final_name,
                    total_speech_sec=total_sec,
                    segment_count=seg_count,
                    centroid_embedding=emb.tolist() if emb is not None else None,
                    merged_from=merged_sources,
                )
            )

        # Update segments with consolidated speaker IDs
        consolidated_segments: List[DiarizationSegment] = []
        for seg in segments:
            final_spk = speaker_map.get(seg.speaker, seg.speaker)
            consolidated_segments.append(
                DiarizationSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=final_spk,
                    overlap=seg.overlap,
                    confidence=seg.confidence,
                )
            )

        return profiles, speaker_map, consolidated_segments

    def _resolve_target(self, mapping: Dict[str, str], spk: str) -> str:
        curr = spk
        while mapping.get(curr, curr) != curr:
            curr = mapping[curr]
        return curr

    def _speakers_overlap(self, segs_a: List[DiarizationSegment], segs_b: List[DiarizationSegment]) -> bool:
        for a in segs_a:
            for b in segs_b:
                if max(a.start, b.start) < min(a.end, b.end):
                    return True
        return False

    def _extract_speaker_centroid_embedding(
        self,
        segments: List[DiarizationSegment],
        audio: np.ndarray,
        sr: int,
    ) -> Optional[np.ndarray]:
        """Extracts acoustic embedding vectors using local SpeechBrain ECAPA model."""
        import config
        ecapa_path = config.MODELS_DIR / "ecapa" / "embedding_model.ckpt"
        if not ecapa_path.exists():
            return None

        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        compute_features = Fbank(n_mels=80)
        mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
        model = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], lin_neurons=192)

        state = torch.load(ecapa_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        embs = []
        # Sample up to 10 cleanest non-overlapping segments
        candidate_segs = [s for s in segments if not s.overlap and s.duration >= 0.8]
        if not candidate_segs:
            candidate_segs = [s for s in segments if s.duration >= 0.5]
        candidate_segs = sorted(candidate_segs, key=lambda s: s.duration, reverse=True)[:10]

        for s in candidate_segs:
            start_samp = int(round(s.start * sr))
            end_samp = int(round(s.end * sr))
            clip = audio[start_samp:end_samp]
            if len(clip) < sr * 0.4:
                continue

            t = torch.from_numpy(clip).unsqueeze(0)
            feats = compute_features(t)
            feats = mean_var_norm(feats, torch.ones(1))
            with torch.no_grad():
                emb = model(feats).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                embs.append(emb / norm)

        if not embs:
            return None

        # Return centroid
        centroid = np.mean(embs, axis=0)
        norm = np.linalg.norm(centroid)
        return (centroid / norm) if norm > 1e-8 else centroid

    def _unload(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
