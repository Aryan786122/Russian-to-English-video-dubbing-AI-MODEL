"""
High-Fidelity SFX & Foley Preserving Master Mixer.

Specification:
- Separates vocals and background using audio-separator (BS-RoFormer / Demucs).
- NEVER destroys the original soundtrack:
  * Outside dialogue windows: 100% untouched original audio.
  * Inside dialogue windows: smooth crossfade into background stem + overlay English dub.
- Transient / SFX Preservation during speech:
  * Detects claps, knocks, impacts, taps via high-frequency transient analysis.
  * Restores transient events from original audio into dialogue replacement windows.
- Preserves non-verbal human reactions (laughs, gasps, breaths, sighs, coughs) outside replaced lines.
- Produces:
  audio/original_master.wav
  audio/separated_background.wav
  audio/final_mix.wav
  output/final_dubbed.mp4
"""

from __future__ import annotations

import gc
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


class StemSeparator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def separate_stems(
        self,
        input_audio_path: str | Path,
        output_vocals_path: str | Path,
        output_background_path: str | Path,
    ) -> bool:
        """
        Separates vocal and background stems using audio-separator (BS-RoFormer) or Demucs.
        Outputs 48kHz stereo WAVs.
        """
        in_path = Path(input_audio_path).resolve()
        voc_path = Path(output_vocals_path).resolve()
        bg_path = Path(output_background_path).resolve()

        voc_path.parent.mkdir(parents=True, exist_ok=True)
        bg_path.parent.mkdir(parents=True, exist_ok=True)

        if voc_path.exists() and bg_path.exists():
            print(f"[StemSeparator] Using cached stems at {voc_path.name} and {bg_path.name}", flush=True)
            return True

        print(f"[StemSeparator] Running stem separation on {in_path.name}...", flush=True)

        # 1. Try audio-separator with BS-RoFormer
        try:
            from audio_separator.separator import Separator
            sep = Separator(output_dir=str(voc_path.parent))
            # Model: BS-RoFormer Viperx
            sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            out_files = sep.separate(str(in_path))
            if len(out_files) >= 2:
                # Find vocals and instrumental
                for f in out_files:
                    fp = Path(voc_path.parent) / f
                    if "vocals" in f.lower():
                        import shutil
                        shutil.move(fp, voc_path)
                    elif "instrumental" in f.lower() or "background" in f.lower():
                        import shutil
                        shutil.move(fp, bg_path)
                self._unload()
                return True
        except Exception as e:
            print(f"[StemSeparator] BS-RoFormer notice ({e}), falling back to Demucs...", flush=True)

        # 2. Fallback to Demucs (local HTDemucs model cached in HF cache)
        try:
            from demucs.api import Separator
            demucs_sep = Separator(model="htdemucs", device=self.device)
            audio_data, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
            if audio_data.shape[1] == 1:
                audio_data = np.repeat(audio_data, 2, axis=1)
            elif audio_data.shape[1] > 2:
                audio_data = audio_data[:, :2]

            # Resample to 44100 for Demucs if needed
            if sr != demucs_sep.samplerate:
                divisor = math.gcd(sr, demucs_sep.samplerate)
                audio_data = resample_poly(audio_data, demucs_sep.samplerate // divisor, sr // divisor, axis=0)
                sr = demucs_sep.samplerate

            audio_tensor = torch.from_numpy(audio_data.T.astype(np.float32))
            _, separated = demucs_sep.separate_tensor(audio_tensor)

            vocals_tensor = separated["vocals"]
            bg_tensor = sum(t for name, t in separated.items() if name != "vocals")

            vocals_np = vocals_tensor.detach().cpu().numpy().T
            bg_np = bg_tensor.detach().cpu().numpy().T

            # Resample back to 48kHz stereo
            if sr != 48000:
                g = math.gcd(sr, 48000)
                vocals_np = resample_poly(vocals_np, 48000 // g, sr // g, axis=0).astype(np.float32)
                bg_np = resample_poly(bg_np, 48000 // g, sr // g, axis=0).astype(np.float32)

            sf.write(str(voc_path), vocals_np, 48000, subtype="PCM_24")
            sf.write(str(bg_path), bg_np, 48000, subtype="PCM_24")
            print("[StemSeparator] Demucs separation completed successfully at 48kHz.", flush=True)
            self._unload()
            return True
        except Exception as e2:
            print(f"[StemSeparator] Separation error ({e2}). Copying original as fallback.", flush=True)
            import shutil
            shutil.copy2(in_path, voc_path)
            shutil.copy2(in_path, bg_path)
            self._unload()
            return False

    def _unload(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SFXPreservingMixer:
    def __init__(self, master_sr: int = 48000):
        self.master_sr = master_sr

    def mix_dub_master(
        self,
        original_audio_path: str | Path,
        background_audio_path: str | Path,
        dialogue_audio_path: str | Path,
        dialogue_intervals: List[Tuple[float, float]],
        output_mix_path: str | Path,
        crossfade_ms: float = 45.0,
        pad_sec: float = 0.05,
    ) -> Path:
        """
        Creates the final master mix preserving 100% untouched original audio outside dialogue windows,
        detecting and restoring transient claps/impacts inside dialogue windows, and overlaying English dialogue.
        """
        orig_p = Path(original_audio_path).resolve()
        bg_p = Path(background_audio_path).resolve()
        dlg_p = Path(dialogue_audio_path).resolve()
        out_p = Path(output_mix_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        orig_data, sr = sf.read(str(orig_p), dtype="float32", always_2d=True)
        bg_data, sr_bg = sf.read(str(bg_p), dtype="float32", always_2d=True)
        dlg_data, sr_dlg = sf.read(str(dlg_p), dtype="float32", always_2d=True)

        # Standardize all to 48kHz stereo
        orig_data = self._standardize_48k_stereo(orig_data, sr)
        bg_data = self._standardize_48k_stereo(bg_data, sr_bg)
        dlg_data = self._standardize_48k_stereo(dlg_data, sr_dlg)

        # Match exact length to original
        target_len = len(orig_data)
        bg_data = self._match_length(bg_data, target_len)
        dlg_data = self._match_length(dlg_data, target_len)

        print("[SFXMixer] Detecting transient SFX (claps, knocks, impacts) in original audio...", flush=True)
        transient_signal = self._extract_transients(orig_data, self.master_sr)

        # Build dialogue mask with smooth crossfades
        mask = np.zeros(target_len, dtype=np.float32)
        fade_len = int(round(self.master_sr * (crossfade_ms / 1000.0)))
        pad_samples = int(round(self.master_sr * pad_sec))

        for start_s, end_s in dialogue_intervals:
            s_idx = max(0, int(round(start_s * self.master_sr)) - pad_samples)
            e_idx = min(target_len, int(round(end_s * self.master_sr)) + pad_samples)
            mask[s_idx:e_idx] = 1.0

        # Smooth mask with Hann convolution
        if fade_len > 1:
            window = np.hanning(fade_len * 2)
            window = window / np.sum(window)
            smoothed_mask = np.convolve(mask, window, mode="same")
            smoothed_mask = np.clip(smoothed_mask, 0.0, 1.0)
        else:
            smoothed_mask = mask

        smoothed_mask_2d = smoothed_mask[:, None]

        # -------------------------------------------------------------
        # Audio Bed Synthesis:
        # Outside speech (mask=0): 100% original untouched audio
        # Inside speech (mask=1): background stem + restored transients
        # -------------------------------------------------------------
        bed = (orig_data * (1.0 - smoothed_mask_2d)) + ((bg_data + (transient_signal * 0.85)) * smoothed_mask_2d)

        # Overlay English dub with dialogue leveling
        master = bed + dlg_data

        # Final transparent peak limiting to -0.5 dBFS
        master = self._soft_limiter(master, ceiling=0.95)

        sf.write(str(out_p), master, self.master_sr, subtype="PCM_24")
        print(f"[SFXMixer] Master mix created successfully at {out_p}", flush=True)
        return out_p

    def _extract_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detects sharp high-frequency transient spikes (claps, taps, knocks)
        using onset detection and high-pass residual filtering.
        """
        mono = np.mean(audio, axis=1)
        # Compute spectral flux onset envelope
        hop_length = int(sr * 0.005) # 5ms resolution
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
        threshold = np.percentile(onset_env, 92)

        # Detect frames with intense abrupt onsets
        onset_frames = np.where(onset_env > threshold)[0]
        transient_mask = np.zeros(len(mono), dtype=np.float32)

        burst_len = int(sr * 0.04) # 40ms transient burst
        for f in onset_frames:
            center_sample = f * hop_length
            s0 = max(0, center_sample - int(burst_len * 0.2))
            s1 = min(len(mono), center_sample + int(burst_len * 0.8))
            transient_mask[s0:s1] = 1.0

        # High-pass filter above 1.5 kHz to isolate claps / impacts from vocal fundamentals
        from scipy.signal import butter, sosfilt
        sos = butter(4, 1500, btype="highpass", fs=sr, output="sos")
        filtered_transient = sosfilt(sos, audio, axis=0)

        # Window with transient mask
        transient_signal = filtered_transient * transient_mask[:, None]
        return transient_signal.astype(np.float32)

    def _standardize_48k_stereo(self, x: np.ndarray, sr: int) -> np.ndarray:
        if x.ndim == 1:
            x = np.column_stack([x, x])
        elif x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        elif x.shape[1] > 2:
            x = x[:, :2]

        if sr != self.master_sr:
            g = math.gcd(sr, self.master_sr)
            x = resample_poly(x, self.master_sr // g, sr // g, axis=0).astype(np.float32)
        return x.astype(np.float32)

    def _match_length(self, x: np.ndarray, target_len: int) -> np.ndarray:
        if len(x) == target_len:
            return x
        if len(x) < target_len:
            pad = np.zeros((target_len - len(x), 2), dtype=np.float32)
            return np.vstack([x, pad])
        return x[:target_len]

    def _soft_limiter(self, x: np.ndarray, knee: float = 0.88, ceiling: float = 0.95) -> np.ndarray:
        peak = np.max(np.abs(x))
        if peak <= knee:
            return x
        a = np.abs(x)
        over = a > knee
        if np.any(over):
            span = ceiling - knee
            x[over] = np.sign(x[over]) * (knee + span * np.tanh((a[over] - knee) / span))
        final_peak = np.max(np.abs(x))
        if final_peak > ceiling:
            x = x * (ceiling / final_peak)
        return x

    def mux_video_and_audio(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        duration_sec: Optional[float] = None,
    ) -> Path:
        """Muxes the final master audio with the original video container using ffmpeg."""
        vid_p = Path(video_path).resolve()
        aud_p = Path(audio_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(vid_p),
            "-i", str(aud_p),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
        ]
        if duration_sec is not None:
            cmd.extend(["-t", str(round(duration_sec, 3))])

        cmd.append(str(out_p))

        print(f"[SFXMixer] Muxing final video to {out_p.name}...", flush=True)
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[SFXMixer] Mux complete: {out_p}", flush=True)
        return out_p
