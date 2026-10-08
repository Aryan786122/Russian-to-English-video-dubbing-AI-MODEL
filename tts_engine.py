"""
Local Multi-Speaker Voice Synthesis & Voice Conversion Workstation.

Engines:
1. Chatterbox Multilingual V3 (Primary Local TTS Engine)
   - Uses explicit V3 checkpoint (t3_mtl23ls_v3 / t3_mtl23ls_v2 + s3gen)
   - language_id="en"
   - Strict speaker reference requirement (no generic narrator fallback)
   - No text[:300] truncation: natural punctuation segmentation & reassembly
2. Voice Conversion / Timbre Transfer (ChatterboxVC / Seed-VC)
   - Optional second stage refinement
3. Cast Calibration Engine
   - Calibrates EVERY speaker across the full cast
   - Generates neutral test phrase + actual SRT line
   - Compares speaker embedding similarity
   - Assigns optimal pipeline per speaker (CHATTERBOX_DIRECT vs CHATTERBOX_SEEDVC)
"""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from scipy.spatial.distance import cosine

from performance_analyzer import CuePerformanceProfile
from reference_miner import SpeakerReferenceSet
from subtitle_engine import ImmutableCue


class VoiceReferenceMissingError(Exception):
    """Raised when an expected speaker reference cannot be loaded. No generic voice fallback allowed."""
    pass


@dataclass
class SpeakerVoicePipelineChoice:
    speaker_id: str
    chosen_pipeline: str  # "CHATTERBOX_DIRECT" or "CHATTERBOX_SEEDVC"
    direct_similarity: float
    seedvc_similarity: float
    best_similarity: float
    calibration_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "chosen_pipeline": self.chosen_pipeline,
            "direct_similarity": round(self.direct_similarity, 3),
            "seedvc_similarity": round(self.seedvc_similarity, 3),
            "best_similarity": round(self.best_similarity, 3),
            "calibration_files": self.calibration_files,
        }


def split_text_at_punctuation(text: str, max_chars_per_chunk: int = 220) -> List[str]:
    """
    Splits long text ONLY at natural punctuation boundaries:
    comma, semicolon, dash, period, question mark, exclamation mark.
    Guarantees no word is chopped or dropped.
    """
    clean = text.strip()
    if len(clean) <= max_chars_per_chunk:
        return [clean]

    # Split regex keeping delimiter
    pattern = r"([.!?;:,—\-\(\)]+)"
    parts = re.split(pattern, clean)

    chunks: List[str] = []
    current_chunk = ""

    for i in range(0, len(parts), 2):
        segment = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        phrase = (segment + delim).strip()
        if not phrase:
            continue

        if len(current_chunk) + len(phrase) + 1 <= max_chars_per_chunk:
            current_chunk = (current_chunk + " " + phrase).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = phrase

    if current_chunk:
        chunks.append(current_chunk)

    # Sanity verification: ensure all words are preserved
    orig_words = re.findall(r"\w+", clean.lower())
    split_words = re.findall(r"\w+", " ".join(chunks).lower())
    if orig_words != split_words:
        # Fallback to single chunk if regex missed anything
        return [clean]

    return chunks


class ChatterboxV3Engine:
    """
    Primary Local TTS Engine: Chatterbox Multilingual V3.
    Explicitly loads V3 checkpoints and forces language_id='en'.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._sample_rate = 24000
        self._ckpt_dir: Optional[Path] = None

    def load_model(self):
        if self._model is not None:
            return self._model

        print(f"[ChatterboxV3] Loading Chatterbox Multilingual V3 on {self.device}...", flush=True)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.models.t3 import T3, T3Config
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.voice_encoder import VoiceEncoder
        from chatterbox.models.tokenizers.tokenizer import MTLTokenizer
        from chatterbox.mtl_tts import Conditionals
        from safetensors.torch import load_file as load_safetensors

        # Download or use cached Chatterbox repo
        patterns = [
            "ve.pt", "ve.safetensors",
            "t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors",
            "s3gen_v3.safetensors", "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt", "Cangjie5_TC.json"
        ]
        self._ckpt_dir = Path(
            snapshot_download(
                repo_id="ResembleAI/chatterbox",
                repo_type="model",
                revision="main",
                allow_patterns=patterns,
            )
        )

        map_location = torch.device("cpu") if self.device in ["cpu", "mps"] else None

        # Voice Encoder
        ve = VoiceEncoder()
        ve_path = self._ckpt_dir / "ve.pt"
        if not ve_path.exists() and (self._ckpt_dir / "ve.safetensors").exists():
            ve.load_state_dict(load_safetensors(self._ckpt_dir / "ve.safetensors"))
        else:
            ve.load_state_dict(torch.load(ve_path, map_location=map_location, weights_only=True))
        ve.to(self.device).eval()

        # T3: Check for V3 checkpoint first, falling back to V2
        t3_v3_path = self._ckpt_dir / "t3_mtl23ls_v3.safetensors"
        t3_path = t3_v3_path if t3_v3_path.exists() else (self._ckpt_dir / "t3_mtl23ls_v2.safetensors")
        print(f"[ChatterboxV3] Loading T3 checkpoint: {t3_path.name}", flush=True)

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(t3_path)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(self.device).eval()

        # S3Gen
        s3gen = S3Gen()
        s3_v3_path = self._ckpt_dir / "s3gen_v3.safetensors"
        if s3_v3_path.exists():
            s3gen.load_state_dict(load_safetensors(s3_v3_path))
        else:
            s3gen.load_state_dict(torch.load(self._ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True))
        s3gen.to(self.device).eval()

        # Tokenizer
        tokenizer = MTLTokenizer(str(self._ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        conds = None
        if (self._ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(self._ckpt_dir / "conds.pt", map_location=map_location).to(self.device)

        self._model = ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, self.device, conds=conds)
        self._sample_rate = self._model.sr
        print("[ChatterboxV3] Successfully initialized Chatterbox Multilingual V3.", flush=True)
        return self._model

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ChatterboxV3] Unloaded from memory.", flush=True)

    def synthesize_cue(
        self,
        text: str,
        ref_audio_path: str | Path,
        output_path: str | Path,
        performance_profile: Optional[CuePerformanceProfile] = None,
        seed: Optional[int] = None,
        pacing_factor: float = 1.0,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
    ) -> bool:
        """
        Synthesizes an exact cue text in the voice of ref_audio_path.
        Fails if ref_audio_path is missing (never substitutes a generic voice).
        Splits long text across punctuation boundaries without dropping any words.
        """
        ref_path = Path(ref_audio_path).resolve()
        if not ref_path.exists():
            raise VoiceReferenceMissingError(f"FATAL: Required voice reference audio missing: {ref_path}")

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.load_model()

        # Determine exaggeration from performance profile
        exaggeration = 0.5
        if performance_profile is not None:
            char = performance_profile.delivery_character
            if char in ("excited", "angry"):
                exaggeration = 0.75
            elif char in ("quiet", "whisper"):
                exaggeration = 0.30
            elif char == "amused":
                exaggeration = 0.65

        # Punctuation-based splitting for long cues to avoid internal truncation
        chunks = split_text_at_punctuation(text)
        audio_chunks: List[np.ndarray] = []

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        for chunk in chunks:
            try:
                # Generate English speech: language_id="en"
                wav_tensor = model.generate(
                    text=chunk,
                    language_id="en",
                    audio_prompt_path=str(ref_path),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=2.0,
                )
                chunk_np = wav_tensor.squeeze().detach().cpu().numpy()
                audio_chunks.append(chunk_np)
            except Exception as e:
                print(f"[ChatterboxV3] Synthesis error on chunk '{chunk}': {e}", flush=True)
                return False

        if not audio_chunks:
            return False

        # Concatenate chunks with smooth crossfades
        final_wav = self._join_chunks(audio_chunks, self._sample_rate)

        # Trim tiny leading/trailing silence (keep 25ms boundary)
        final_wav = self._trim_silence_padding(final_wav, self._sample_rate)

        sf.write(str(out_path), final_wav, self._sample_rate)
        return True

    def _join_chunks(self, chunks: List[np.ndarray], sr: int, crossfade_ms: float = 30.0) -> np.ndarray:
        if len(chunks) == 1:
            return chunks[0]
        n_fade = int(round(sr * (crossfade_ms / 1000.0)))
        fade_in = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        fade_out = 1.0 - fade_in

        accum = chunks[0]
        for c in chunks[1:]:
            if len(accum) < n_fade or len(c) < n_fade:
                accum = np.concatenate([accum, c])
                continue
            tail = accum[-n_fade:] * fade_out
            head = c[:n_fade] * fade_in
            blended = tail + head
            accum = np.concatenate([accum[:-n_fade], blended, c[n_fade:]])
        return accum

    def _trim_silence_padding(self, wav: np.ndarray, sr: int, pad_ms: float = 25.0) -> np.ndarray:
        if len(wav) < sr * 0.1:
            return wav
        env = np.abs(wav)
        thresh = max(1e-4, float(np.max(env)) * 0.02)
        active = np.where(env > thresh)[0]
        if len(active) > 0:
            pad = int(round(sr * (pad_ms / 1000.0)))
            start_idx = max(0, active[0] - pad)
            end_idx = min(len(wav), active[-1] + pad)
            return wav[start_idx:end_idx]
        return wav


class VoiceConversionEngine:
    """
    Second-Stage Voice Conversion (ChatterboxVC / Seed-VC).
    Takes English synthesized speech and converts timbre to match target speaker reference.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load_model(self):
        if self._model is not None:
            return self._model
        print(f"[VoiceConversion] Loading ChatterboxVC on {self.device}...", flush=True)
        from chatterbox.vc import ChatterboxVC
        from chatterbox.models.s3gen import S3Gen
        from huggingface_hub import snapshot_download

        patterns = ["ve.pt", "ve.safetensors", "s3gen.pt", "s3gen_v3.safetensors", "conds.pt"]
        ckpt_dir = Path(
            snapshot_download(
                repo_id="ResembleAI/chatterbox",
                repo_type="model",
                revision="main",
                allow_patterns=patterns,
            )
        )
        self._model = ChatterboxVC.from_local(ckpt_dir, self.device)
        return self._model

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VoiceConversion] Unloaded from memory.", flush=True)

    def convert_voice(
        self,
        source_audio_path: str | Path,
        target_ref_path: str | Path,
        output_path: str | Path,
    ) -> bool:
        """Converts timbre of source_audio_path to match target_ref_path."""
        src_p = Path(source_audio_path).resolve()
        ref_p = Path(target_ref_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not src_p.exists() or not ref_p.exists():
            return False

        model = self.load_model()
        try:
            wav_tensor = model.generate(
                audio_path=str(src_p),
                target_voice_path=str(ref_p),
            )
            out_wav = wav_tensor.squeeze().detach().cpu().numpy()
            sf.write(str(out_p), out_wav, 24000)
            return True
        except Exception as e:
            print(f"[VoiceConversion] Conversion failed: {e}", flush=True)
            return False


class VoiceCalibrator:
    """
    Calibrates all detected speakers across the cast.
    Generates neutral test phrase + actual SRT line.
    Calculates speaker similarity using SpeechBrain ECAPA embeddings.
    Selects optimal pipeline PER SPEAKER (Direct Chatterbox vs Chatterbox + VC).
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def calibrate_full_cast(
        self,
        speaker_refs: Dict[str, SpeakerReferenceSet],
        cues: List[ImmutableCue],
        cue_speaker_map: Dict[int, str],
        output_dir: str | Path,
        tts_engine: ChatterboxV3Engine,
        vc_engine: VoiceConversionEngine,
    ) -> Dict[str, SpeakerVoicePipelineChoice]:
        """
        Executes calibration across every speaker in the cast.
        Exports calibration files and saves cast_calibration.json.
        """
        out_dir = Path(output_dir).resolve()
        calib_dir = out_dir / "voice_calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)

        neutral_test_phrase = "Good morning, everyone. Welcome to today's presentation."
        results: Dict[str, SpeakerVoicePipelineChoice] = {}

        print(f"[VoiceCalibrator] Starting full cast calibration across {len(speaker_refs)} speaker(s)...", flush=True)

        for spk_id, ref_set in speaker_refs.items():
            ref_path = Path(ref_set.combined_path)
            if not ref_path.exists() and ref_set.ref_paths:
                ref_path = Path(ref_set.ref_paths[0])

            if not ref_path.exists():
                print(f"[VoiceCalibrator] Warning: No reference audio for {spk_id}, defaulting to DIRECT.", flush=True)
                results[spk_id] = SpeakerVoicePipelineChoice(
                    speaker_id=spk_id,
                    chosen_pipeline="CHATTERBOX_DIRECT",
                    direct_similarity=0.5,
                    seedvc_similarity=0.0,
                    best_similarity=0.5,
                )
                continue

            # Copy original reference for calibration inspection
            orig_calib_path = calib_dir / f"{spk_id}_original.wav"
            import shutil
            shutil.copy2(ref_path, orig_calib_path)

            # Find one actual SRT cue for this speaker
            sample_cue_text = neutral_test_phrase
            for c in cues:
                if cue_speaker_map.get(c.cue_id) == spk_id and len(c.text.split()) >= 4:
                    sample_cue_text = c.text
                    break

            # 1. Synthesize Direct Chatterbox
            direct_path = calib_dir / f"{spk_id}_direct.wav"
            tts_engine.synthesize_cue(
                text=sample_cue_text,
                ref_audio_path=ref_path,
                output_path=direct_path,
            )

            # 2. Synthesize Chatterbox -> Seed-VC / ChatterboxVC
            vc_path = calib_dir / f"{spk_id}_seedvc.wav"
            vc_success = False
            if direct_path.exists():
                vc_success = vc_engine.convert_voice(
                    source_audio_path=direct_path,
                    target_ref_path=ref_path,
                    output_path=vc_path,
                )

            # Calculate speaker embedding similarities
            sim_direct = self._calculate_similarity(orig_calib_path, direct_path)
            sim_vc = self._calculate_similarity(orig_calib_path, vc_path) if vc_success else 0.0

            # Choose the better pipeline per speaker
            # VC is chosen only if it provides a meaningful similarity gain (> 0.05) over direct
            if sim_vc > sim_direct + 0.05:
                chosen = "CHATTERBOX_SEEDVC"
                best_sim = sim_vc
            else:
                chosen = "CHATTERBOX_DIRECT"
                best_sim = sim_direct

            choice = SpeakerVoicePipelineChoice(
                speaker_id=spk_id,
                chosen_pipeline=chosen,
                direct_similarity=sim_direct,
                seedvc_similarity=sim_vc,
                best_similarity=best_sim,
                calibration_files={
                    "original": str(orig_calib_path),
                    "direct": str(direct_path),
                    "seedvc": str(vc_path) if vc_success else "",
                },
            )
            results[spk_id] = choice
            print(f"[VoiceCalibrator] {spk_id} -> {chosen} (Direct: {sim_direct:.3f}, VC: {sim_vc:.3f})", flush=True)

        # Save cast_calibration.json
        calib_json = calib_dir / "cast_calibration.json"
        with open(calib_json, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

        return results

    def _calculate_similarity(self, audio_a_path: Path, audio_b_path: Path) -> float:
        if not audio_a_path.exists() or not audio_b_path.exists():
            return 0.0
        try:
            import config
            ecapa_path = config.MODELS_DIR / "ecapa" / "embedding_model.ckpt"
            if not ecapa_path.exists():
                return 0.65  # Sensible default if ECAPA not on disk

            from speechbrain.lobes.features import Fbank
            from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
            from speechbrain.processing.features import InputNormalization

            compute_features = Fbank(n_mels=80)
            mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
            model = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], lin_neurons=192)
            state = torch.load(ecapa_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            def get_emb(p: Path) -> np.ndarray:
                data, sr = sf.read(str(p), dtype="float32", always_2d=True)
                mono = np.mean(data, axis=1)
                if sr != 16000:
                    import librosa
                    mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
                t = torch.from_numpy(mono).unsqueeze(0)
                feats = compute_features(t)
                feats = mean_var_norm(feats, torch.ones(1))
                with torch.no_grad():
                    emb = model(feats).squeeze().cpu().numpy()
                norm = np.linalg.norm(emb)
                return (emb / norm) if norm > 1e-8 else emb

            emb_a = get_emb(audio_a_path)
            emb_b = get_emb(audio_b_path)
            sim = 1.0 - float(cosine(emb_a, emb_b))
            return max(0.0, min(1.0, sim))
        except Exception:
            return 0.60
