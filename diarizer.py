"""
Speaker Diarization and Per-Speaker Voice Reference Extraction Module.
Uses SpeechBrain ECAPA-TDNN embeddings with hierarchical cosine clustering.
Provides clean per-speaker isolated reference samples for zero-shot voice cloning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# Ensure offline operation if cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import config
from timeline_validator import validate_timeline_integrity


def load_mono_16k(audio_path: str | Path) -> np.ndarray:
    """Loads any audio file and converts it to a 16kHz mono float32 numpy array."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use ffmpeg for guaranteed robust resampling to 16kHz mono
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-f", "wav",
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    import io
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return np.asarray(audio, dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit Euclidean norm along the last axis."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


class SpeakerDiarizer:
    """
    Offline ECAPA-TDNN Speaker Diarizer for multi-speaker Russian/English dubbing.
    Extracts 192-dim deep speaker embeddings, performs hierarchical clustering,
    assigns consistent speaker labels (SPEAKER_00, SPEAKER_01...), and builds
    clean 10-20s composite reference audio for each speaker.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        cluster_threshold: float = 0.32,
        window_sec: float = 2.5,
        hop_sec: float = 1.0,
    ):
        self.model_path = Path(model_path or (config.MODELS_DIR / "ecapa" / "embedding_model.ckpt")).resolve()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cluster_threshold = cluster_threshold
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self._model = None
        self._compute_features = None
        self._mean_var_norm = None

    def _load_model(self):
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ECAPA model checkpoint not found at {self.model_path}. "
                "Ensure embedding_model.ckpt is downloaded."
            )

        print(f"Loading ECAPA-TDNN speaker model on {self.device}...", flush=True)
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self._compute_features = Fbank(n_mels=80).to(self.device).eval()
        self._mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False).to(self.device).eval()

        model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192,
        )
        weights = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        self._model = model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        print("ECAPA-TDNN model loaded successfully.", flush=True)

    def unload_model(self):
        """Unload ECAPA-TDNN speaker model from GPU VRAM."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if self._compute_features is not None:
            del self._compute_features
            self._compute_features = None
        if self._mean_var_norm is not None:
            del self._mean_var_norm
            self._mean_var_norm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("ECAPA-TDNN model unloaded from VRAM.", flush=True)

    def _create_analysis_windows(
        self,
        segments: List[dict],
        total_audio_sec: float,
        audio_16k: np.ndarray,
        sr: int = 16000
    ) -> List[dict]:
        """
        Creates overlapping speech windows aligned with Whisper segments and words.
        """
        windows = []
        for seg_idx, seg in enumerate(segments):
            words = seg.get("words", [])
            intervals = [(float(w["start"]), float(w["end"])) for w in words if "start" in w and "end" in w]
            
            # Form contiguous speech islands within the segment
            islands = []
            source_intervals = intervals or [(float(seg["start"]), float(seg["end"]))]
            for st, en in sorted(source_intervals):
                st = max(0.0, min(total_audio_sec, st))
                en = max(0.0, min(total_audio_sec, en))
                if en <= st:
                    continue
                if islands and st - islands[-1][1] <= 0.4:
                    islands[-1][1] = max(islands[-1][1], en)
                else:
                    islands.append([st, en])

            # Slice islands into windows of duration window_sec with hop_sec
            for st, en in islands:
                dur = en - st
                if dur <= self.window_sec:
                    bounds = [(st, en)]
                else:
                    starts = list(np.arange(st, en - self.window_sec, self.hop_sec))
                    starts.append(en - self.window_sec)
                    bounds = [(float(a), float(a + self.window_sec)) for a in starts]

                for a, b in bounds:
                    clip = audio_16k[int(a * sr):int(b * sr)]
                    if len(clip) == 0:
                        continue
                    rms = float(np.sqrt(np.mean(clip ** 2) + 1e-12))
                    rms_dbfs = round(20 * math.log10(rms + 1e-12), 2)
                    clipped_frac = round(float(np.mean(np.abs(clip) >= 0.99)), 4)

                    windows.append({
                        "window_index": len(windows),
                        "segment_index": seg_idx,
                        "segment_id": seg.get("id", seg_idx),
                        "start": round(a, 3),
                        "end": round(b, 3),
                        "duration": round(b - a, 3),
                        "rms_dbfs": rms_dbfs,
                        "clipped_fraction": clipped_frac
                    })

        return windows

    def _extract_window_embeddings(
        self,
        audio_16k: np.ndarray,
        windows: List[dict],
        batch_size: int = 16,
        sr: int = 16000
    ) -> np.ndarray:
        """Extracts 192-dim L2-normalized embeddings for each window."""
        self._load_model()
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i : i + batch_size]
                clips = []
                for win in batch_windows:
                    st_sample = int(win["start"] * sr)
                    en_sample = int(win["end"] * sr)
                    clip = audio_16k[st_sample:en_sample]
                    # Min length for ECAPA frontend is 0.4s (6400 samples)
                    if len(clip) < 6400:
                        clip = np.pad(clip, (0, 6400 - len(clip)))
                    clips.append(clip)

                max_len = max(len(c) for c in clips)
                padded = np.zeros((len(clips), max_len), dtype=np.float32)
                lengths = []
                for row, c in enumerate(clips):
                    padded[row, :len(c)] = c
                    lengths.append(len(c) / max_len)

                samples = torch.from_numpy(padded).to(self.device)
                lengths_t = torch.tensor(lengths, dtype=torch.float32, device=self.device)

                features = self._mean_var_norm(self._compute_features(samples), lengths_t)
                emb = self._model(features, lengths_t).reshape(len(clips), -1)
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return l2_normalize(embeddings)

    def _cluster_windows(
        self,
        embeddings: np.ndarray,
        windows: List[dict],
        num_speakers: Optional[int] = None,
        cluster_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, dict]]:
        """
        Clusters window embeddings using average-linkage hierarchical clustering.
        Returns cluster assignments and cluster centroid metadata.
        """
        threshold = cluster_threshold or self.cluster_threshold

        # Filter clean windows for initial tree fitting (duration >= 1.0s, RMS > -42 dBFS)
        clean_mask = np.array([
            w["duration"] >= 1.0 and w["rms_dbfs"] > -42.0 and w["clipped_fraction"] < 0.05
            for w in windows
        ])
        if clean_mask.sum() < 2:
            clean_mask = np.array([w["duration"] >= 0.5 for w in windows])
        if not clean_mask.any():
            clean_mask[:] = True

        fit_indices = np.flatnonzero(clean_mask)

        if len(fit_indices) <= 1:
            labels = np.zeros(len(windows), dtype=int)
            centroids = [embeddings.mean(axis=0)]
        else:
            fit_embeddings = embeddings[fit_indices]
            distances = np.clip(pdist(fit_embeddings, metric="cosine"), 0.0, 2.0)
            tree = linkage(distances, method="average")

            if num_speakers and num_speakers >= 1:
                k_best = min(num_speakers, len(fit_indices))
            elif cluster_threshold is not None and cluster_threshold > 0.5:
                fit_labels = fcluster(tree, cluster_threshold, criterion="distance") - 1
                k_best = len(set(fit_labels))
            else:
                # Automatic optimal speaker count detection via silhouette score
                from sklearn.metrics import silhouette_score
                max_k = min(5, len(fit_indices) - 1)
                best_score = -1.0
                k_best = 2
                for candidate_k in range(2, max_k + 1):
                    cand_labels = fcluster(tree, candidate_k, criterion="maxclust") - 1
                    if len(set(cand_labels)) > 1:
                        score = silhouette_score(fit_embeddings, cand_labels, metric="cosine")
                        if score > best_score:
                            best_score = score
                            k_best = candidate_k
            fit_labels = fcluster(tree, k_best, criterion="maxclust") - 1

            # Chronological ordering: order speakers by first speech timestamp
            unique_labels = sorted(set(fit_labels))
            label_first_appearance = {
                lbl: min(windows[fit_indices[i]]["start"] for i, l in enumerate(fit_labels) if l == lbl)
                for lbl in unique_labels
            }
            ordered_labels = sorted(unique_labels, key=lambda l: label_first_appearance[l])
            label_map = {old: new for new, old in enumerate(ordered_labels)}
            fit_labels = np.array([label_map[l] for l in fit_labels])

            # Calculate robust centroids for each cluster
            centroids = []
            for group_idx in range(len(ordered_labels)):
                group_mask = fit_labels == group_idx
                group_vecs = fit_embeddings[group_mask]
                center = l2_normalize(group_vecs.mean(axis=0))
                # Refine centroid by pruning bottom 20% outliers
                if len(group_vecs) >= 4:
                    sims = group_vecs @ center
                    group_vecs = group_vecs[sims >= np.quantile(sims, 0.2)]
                    center = l2_normalize(group_vecs.mean(axis=0))
                centroids.append(center)

            centroids = np.stack(centroids)  # shape (K, 192)

            # Assign all windows (including noisy/short ones) to nearest centroid
            similarities = embeddings @ centroids.T  # shape (N, K)
            assigned_labels = similarities.argmax(axis=1)

            # Calculate confidence & margin
            for idx, win in enumerate(windows):
                best_lbl = int(assigned_labels[idx])
                best_sim = float(similarities[idx, best_lbl])
                second_sim = (
                    float(np.partition(similarities[idx], -2)[-2])
                    if len(centroids) > 1
                    else -1.0
                )
                margin = best_sim - second_sim
                score = 0.6 * np.clip((best_sim - 0.35) / 0.5, 0, 1) + 0.4 * np.clip(margin / 0.2, 0, 1)

                flags = []
                if win["duration"] < 1.0:
                    flags.append("short_window")
                    score *= 0.7
                if margin < 0.08:
                    flags.append("ambiguous_speaker")
                if best_sim < 0.45:
                    flags.append("outlier_embedding")

                win["speaker"] = f"SPEAKER_{best_lbl:02d}"
                win["confidence"] = round(float(score), 3)
                win["cosine_sim"] = round(best_sim, 3)
                win["similarity_margin"] = round(margin, 3)
                win["flags"] = flags

        return assigned_labels, {f"SPEAKER_{k:02d}": {"centroid": centroids[k]} for k in range(len(centroids))}

    def diarize_segments(
        self,
        audio_path: str | Path,
        segments: List[dict],
        num_speakers: Optional[int] = None,
        cluster_threshold: Optional[float] = None,
        source_duration: Optional[float] = None
    ) -> Tuple[List[dict], Dict[str, dict]]:
        """
        Diarizes transcribed segments and attaches speaker_id, speaker_confidence,
        and diagnostic flags to each segment.
        Validates timeline integrity against authoritative SOURCE_DURATION.
        """
        if not segments:
            return [], {}

        if source_duration is not None and source_duration > 0:
            validate_timeline_integrity(
                segments=segments,
                source_duration=source_duration,
                stage_name="pre_diarization",
                tolerance=0.20
            )

        audio_16k = load_mono_16k(audio_path)
        total_audio_sec = len(audio_16k) / 16000.0

        windows = self._create_analysis_windows(segments, total_audio_sec, audio_16k)
        if not windows:
            print("Warning: No speech windows created; assigning default SPEAKER_00.")
            for seg in segments:
                seg["speaker_id"] = "SPEAKER_00"
                seg["speaker_confidence"] = 1.0
                seg["speaker_flags"] = ["no_windows"]
            return segments, {"SPEAKER_00": {}}

        embeddings = self._extract_window_embeddings(audio_16k, windows)
        _, cluster_meta = self._cluster_windows(
            embeddings, windows, num_speakers=num_speakers, cluster_threshold=cluster_threshold
        )

        # Map window labels back to segments
        by_segment = defaultdict(list)
        for win in windows:
            by_segment[win["segment_index"]].append(win)

        for seg_idx, seg in enumerate(segments):
            seg_wins = by_segment.get(seg_idx, [])
            if not seg_wins:
                # Fallback to nearest window in time
                mid_time = (seg["start"] + seg["end"]) / 2.0
                closest_win = min(windows, key=lambda w: abs(((w["start"] + w["end"]) / 2.0) - mid_time))
                seg["speaker_id"] = closest_win["speaker"]
                seg["speaker_confidence"] = round(closest_win["confidence"] * 0.5, 3)
                seg["speaker_flags"] = ["interpolated_speaker"]
                continue

            # Word-level attribution if words exist
            words = seg.get("words", [])
            if words:
                for w in words:
                    w_mid = (w["start"] + w["end"]) / 2.0
                    chosen_win = min(
                        seg_wins,
                        key=lambda win: (
                            0 if win["start"] <= w_mid <= win["end"] else 1,
                            abs(((win["start"] + win["end"]) / 2.0) - w_mid),
                        )
                    )
                    w["speaker"] = chosen_win["speaker"]
                    w["speaker_confidence"] = chosen_win["confidence"]

            # Segment-level dominant speaker
            speaker_durations = defaultdict(float)
            for win in seg_wins:
                speaker_durations[win["speaker"]] += win["duration"]

            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
            total_dur = sum(speaker_durations.values())
            dominant_share = speaker_durations[dominant_speaker] / max(1e-6, total_dur)

            # Average confidence weighted by duration
            avg_conf = sum(
                w["confidence"] * w["duration"] for w in seg_wins if w["speaker"] == dominant_speaker
            ) / max(1e-6, speaker_durations[dominant_speaker])

            flags = set()
            for w in seg_wins:
                flags.update(w.get("flags", []))
            if len(speaker_durations) > 1 and dominant_share < 0.75:
                flags.add("mixed_speakers_in_segment")
            if (seg["end"] - seg["start"]) < 1.0:
                flags.add("short_segment")

            seg["speaker_id"] = dominant_speaker
            seg["speaker_confidence"] = round(float(avg_conf), 3)
            seg["speaker_flags"] = sorted(list(flags))

        if source_duration is not None and source_duration > 0:
            validate_timeline_integrity(
                segments=segments,
                source_duration=source_duration,
                stage_name="post_diarization",
                tolerance=0.20
            )

        return segments, cluster_meta

    def build_speaker_voice_references(
        self,
        audio_path: str | Path,
        segments: List[dict],
        output_dir: Path | str,
        target_ref_duration: float = 12.0,
        min_ref_duration: float = 2.5,
        whisper_transcriber=None
    ) -> Dict[str, dict]:
        """
        Builds a dedicated, clean reference audio WAV and exact reference transcript
        for EACH detected speaker.
        Prefers high-confidence, non-overlapping clean dialogue intervals.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_orig, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)

        speakers = sorted(set(s.get("speaker_id", "SPEAKER_00") for s in segments))
        speaker_profiles = {}

        print(f"\nBuilding per-speaker voice reference samples for: {', '.join(speakers)}...")

        for spk in speakers:
            spk_segments = [s for s in segments if s.get("speaker_id") == spk]

            # Candidate selection criteria:
            # 1. High speaker confidence (>= 0.60)
            # 2. No ambiguous_speaker or mixed_speakers flags
            # 3. Minimum duration >= 1.0s
            candidates = []
            for s in spk_segments:
                dur = s["end"] - s["start"]
                conf = s.get("speaker_confidence", 0.5)
                flags = s.get("speaker_flags", [])
                penalty = 0.0
                if "ambiguous_speaker" in flags:
                    penalty += 0.3
                if "mixed_speakers_in_segment" in flags:
                    penalty += 0.2
                if dur < 1.0:
                    penalty += 0.2

                score = conf - penalty + min(dur / 5.0, 0.4)
                candidates.append((score, dur, s))

            # Sort descending by quality score
            candidates.sort(key=lambda x: -x[0])

            # Gather audio intervals up to target_ref_duration
            collected_audio = []
            collected_transcripts = []
            accumulated_sec = 0.0
            used_segment_ids = []

            # Small 100ms silence padding between spliced intervals
            silence_pad = np.zeros((int(sr * 0.10), audio_orig.shape[1]), dtype=np.float32)

            for score, dur, seg in candidates:
                st_sample = max(0, int(round(seg["start"] * sr)))
                en_sample = min(len(audio_orig), int(round(seg["end"] * sr)))
                clip = audio_orig[st_sample:en_sample]

                if collected_audio:
                    collected_audio.append(silence_pad)
                collected_audio.append(clip)
                collected_transcripts.append(seg.get("russian_text", "").strip())
                used_segment_ids.append(seg.get("id"))
                accumulated_sec += dur

                if accumulated_sec >= target_ref_duration:
                    break

            # If still short, relax filters to get any available speech
            if accumulated_sec < min_ref_duration and spk_segments:
                for s in spk_segments:
                    if s.get("id") in used_segment_ids:
                        continue
                    st_sample = max(0, int(round(s["start"] * sr)))
                    en_sample = min(len(audio_orig), int(round(s["end"] * sr)))
                    clip = audio_orig[st_sample:en_sample]
                    if collected_audio:
                        collected_audio.append(silence_pad)
                    collected_audio.append(clip)
                    collected_transcripts.append(s.get("russian_text", "").strip())
                    used_segment_ids.append(s.get("id"))
                    accumulated_sec += (s["end"] - s["start"])
                    if accumulated_sec >= min_ref_duration:
                        break

            is_low_confidence = accumulated_sec < 3.0

            ref_wav_path = output_dir / f"{spk}_ref.wav"
            if collected_audio:
                final_ref_data = np.vstack(collected_audio)
                # Normalize peak to -1 dB
                pk = float(np.max(np.abs(final_ref_data)))
                if pk > 0.01:
                    final_ref_data = final_ref_data * (0.90 / pk)
                sf.write(str(ref_wav_path), final_ref_data, sr)
            else:
                # Blank fallback
                final_ref_data = np.zeros((sr * 2, 2), dtype=np.float32)
                sf.write(str(ref_wav_path), final_ref_data, sr)

            # Determine reference text from the actual speech segments collected
            valid_parts = []
            for t in collected_transcripts:
                t_clean = t.strip() if t else ""
                t_lower = t_clean.lower()
                if any(bad in t_lower for bad in ["субтитр", "dimatorzok", "elderman", "elder_man"]):
                    continue
                if t_clean:
                    valid_parts.append(t_clean)
            composite_text = " ".join(valid_parts)
            if not composite_text and collected_transcripts:
                composite_text = " ".join(t for t in collected_transcripts if t)

            profile = {
                "speaker_id": spk,
                "ref_audio_path": str(ref_wav_path),
                "ref_text": composite_text,
                "duration_sec": round(accumulated_sec, 2),
                "is_low_confidence": is_low_confidence,
                "segments_used": used_segment_ids,
                "num_segments": len(spk_segments),
            }
            speaker_profiles[spk] = profile

            status_str = f"OK ({accumulated_sec:.1f}s)" if not is_low_confidence else f"LOW CONFIDENCE ({accumulated_sec:.1f}s)"
            print(f"  [{spk}]: {status_str} -> {ref_wav_path.name}")
            if composite_text:
                preview = composite_text if len(composite_text) <= 60 else composite_text[:57] + "..."
                print(f"          Ref Text: \"{preview}\"")

        return speaker_profiles
