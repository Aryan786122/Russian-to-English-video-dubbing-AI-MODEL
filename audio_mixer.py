from pathlib import Path
import math
import numpy as np
import soundfile as sf
import torch


def separate_vocals_and_background(
    input_audio_path: str | Path,
    output_vocals_path: str | Path,
    output_background_path: str | Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> bool:
    """
    Separate an audio file into vocals.wav and background.wav (BGM + SFX)
    using Meta's Demucs deep learning stem separation.
    Handles mono and stereo transparently by ensuring 2-channel tensor format.
    """
    input_audio_path = Path(input_audio_path).resolve()
    output_vocals_path = Path(output_vocals_path).resolve()
    output_background_path = Path(output_background_path).resolve()

    output_vocals_path.parent.mkdir(parents=True, exist_ok=True)
    output_background_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running stem separation on '{input_audio_path.name}' via Demucs ({device})...")

    try:
        from demucs.api import Separator
        separator = Separator(model="htdemucs", device=device)

        # Load audio data via soundfile
        audio_data, sr = sf.read(str(input_audio_path), dtype="float32", always_2d=True)
        # Demucs expects stereo (2, samples)
        if audio_data.shape[1] == 1:
            audio_data = np.repeat(audio_data, 2, axis=1)
        elif audio_data.shape[1] > 2:
            audio_data = audio_data[:, :2]

        # Resample to separator sample rate (44100) if needed
        if sr != separator.samplerate:
            from scipy.signal import resample_poly
            divisor = math.gcd(sr, separator.samplerate)
            audio_data = resample_poly(audio_data, separator.samplerate // divisor, sr // divisor, axis=0)
            sr = separator.samplerate

        # Convert to tensor: (channels, samples)
        audio_tensor = torch.from_numpy(audio_data.T.astype(np.float32))

        # Run Demucs separation
        _, separated = separator.separate_tensor(audio_tensor)

        # Extract vocals and combine remaining stems (drums, bass, other) into background
        vocals_tensor = separated["vocals"]
        bg_tensor = sum(t for name, t in separated.items() if name != "vocals")

        # Convert back to (samples, channels) numpy array
        vocals_np = vocals_tensor.detach().cpu().numpy().T
        bg_np = bg_tensor.detach().cpu().numpy().T

        sf.write(str(output_vocals_path), vocals_np, sr)
        sf.write(str(output_background_path), bg_np, sr)
        print("Demucs stem separation successfully completed.")
        return True

    except Exception as e:
        print(f"Demucs separation error: {e}")
        # Fallback to avoid breaking pipeline
        data, sr = sf.read(str(input_audio_path), dtype="float32")
        sf.write(str(output_background_path), data, sr)
        sf.write(str(output_vocals_path), data, sr)
        return False


def preserve_and_mix_dub(
    original_wav_path: str | Path,
    background_wav_path: str | Path,
    voice_segments: list[dict],
    total_duration: float,
    output_master_wav_path: str | Path,
    output_preserved_sfx_path: str | Path | None = None,
    output_voice_solo_path: str | Path | None = None,
    crossfade_ms: float = 30.0,
    speech_intervals: list[tuple[float, float]] | None = None
) -> bool:
    """
    High-fidelity transient & Foley preserving dub mixer:
    1. Outside speech intervals: 100% original audio is preserved (claps, impacts, laughs, Foley, room tone).
    2. During Russian dialogue intervals: smoothly crossfades to Demucs background stem to suppress Russian voice.
    3. Overlays cloned English voice segments with headroom protection.
    4. Exports preview components: preserved SFX/BGM solo, voice solo, and master mix.
    """
    orig_path = Path(original_wav_path).resolve()
    bg_path = Path(background_wav_path).resolve()
    out_master = Path(output_master_wav_path).resolve()
    out_master.parent.mkdir(parents=True, exist_ok=True)

    if not orig_path.exists():
        raise FileNotFoundError(f"Original audio file not found: {orig_path}")
    if not bg_path.exists():
        bg_path = orig_path

    # Read original audio
    orig_data, sr = sf.read(str(orig_path), dtype="float32", always_2d=True)
    bg_data, bg_sr = sf.read(str(bg_path), dtype="float32", always_2d=True)

    # Align sample rates if needed
    if bg_sr != sr:
        from scipy.signal import resample_poly
        divisor = math.gcd(bg_sr, sr)
        bg_data = resample_poly(bg_data, sr // divisor, bg_sr // divisor, axis=0)

    # Align channels
    channels = orig_data.shape[1]
    if bg_data.shape[1] != channels:
        if bg_data.shape[1] == 1 and channels == 2:
            bg_data = np.repeat(bg_data, 2, axis=1)
        elif bg_data.shape[1] == 2 and channels == 1:
            bg_data = bg_data.mean(axis=1, keepdims=True)

    # Align length to authoritative total_duration exactly
    if total_duration and total_duration > 0:
        target_len = int(round(total_duration * sr))
    else:
        target_len = len(orig_data)

    def fit_length(arr, target):
        if len(arr) < target:
            pad = np.zeros((target - len(arr), arr.shape[1]), dtype=np.float32)
            return np.vstack([arr, pad])
        return arr[:target]

    orig_data = fit_length(orig_data, target_len)
    bg_data = fit_length(bg_data, target_len)

    # 1. Build replacement mask (1 = replace with background stem, 0 = keep pristine original)
    replacement_mask = np.zeros(target_len, dtype=np.float32)
    crossfade_samples = max(1, int(sr * (crossfade_ms / 1000.0)))

    # Use explicitly passed speech intervals, or extract from segments
    intervals = speech_intervals or []
    if not intervals:
        for seg in voice_segments:
            intervals.append((float(seg["start"]), float(seg["end"])))

    for start_sec, end_sec in intervals:
        s_idx = max(0, int(round(start_sec * sr)))
        e_idx = min(target_len, int(round(end_sec * sr)))
        if s_idx >= target_len:
            continue

        fade_in_start = max(0, s_idx - crossfade_samples)
        fade_out_end = min(target_len, e_idx + crossfade_samples)

        # Ramp up
        if s_idx > fade_in_start:
            ramp_up = np.linspace(0.0, 1.0, s_idx - fade_in_start, dtype=np.float32)
            replacement_mask[fade_in_start:s_idx] = np.maximum(replacement_mask[fade_in_start:s_idx], ramp_up)

        # Full replacement during speech
        replacement_mask[s_idx:e_idx] = 1.0

        # Ramp down
        if fade_out_end > e_idx:
            ramp_down = np.linspace(1.0, 0.0, fade_out_end - e_idx, dtype=np.float32)
            replacement_mask[e_idx:fade_out_end] = np.maximum(replacement_mask[e_idx:fade_out_end], ramp_down)

    # 2. Base track: original outside speech, background inside speech
    # This guarantees claps, impacts, laughs, and Foley outside speech intervals are 100% original!
    mask_2d = replacement_mask[:, np.newaxis]
    preserved_base = orig_data * (1.0 - mask_2d) + bg_data * mask_2d

    if output_preserved_sfx_path:
        out_sfx = Path(output_preserved_sfx_path).resolve()
        out_sfx.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_sfx), preserved_base, sr)

    # 3. Composite English voice track
    voice_master = np.zeros((target_len, channels), dtype=np.float32)
    for seg in voice_segments:
        start_sec = float(seg["start"])
        wav_file = seg.get("wav_path")
        if not wav_file or not Path(wav_file).exists():
            continue

        v_data, v_sr = sf.read(str(wav_file), dtype="float32", always_2d=True)
        if v_sr != sr:
            from scipy.signal import resample_poly
            divisor = math.gcd(v_sr, sr)
            v_data = resample_poly(v_data, sr // divisor, v_sr // divisor, axis=0)

        if v_data.shape[1] != channels:
            if v_data.shape[1] == 1 and channels == 2:
                v_data = np.repeat(v_data, 2, axis=1)
            elif v_data.shape[1] == 2 and channels == 1:
                v_data = v_data.mean(axis=1, keepdims=True)

        # Micro-fades (5ms) at voice edges to prevent any click
        edge_samples = min(len(v_data) // 4, max(1, int(sr * 0.005)))
        v_data[:edge_samples] *= np.linspace(0.0, 1.0, edge_samples, dtype=np.float32)[:, None]
        v_data[-edge_samples:] *= np.linspace(1.0, 0.0, edge_samples, dtype=np.float32)[:, None]

        start_idx = int(round(start_sec * sr))
        end_idx = min(start_idx + len(v_data), target_len)
        if start_idx >= target_len:
            continue

        chunk_len = end_idx - start_idx
        voice_master[start_idx:end_idx] += v_data[:chunk_len]

    if output_voice_solo_path:
        out_voice = Path(output_voice_solo_path).resolve()
        out_voice.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_voice), voice_master, sr)

    # 4. Final Master Mix with Headroom Guard
    master_mix = preserved_base + voice_master
    peak = float(np.max(np.abs(master_mix)))
    if peak > 0.98:
        master_mix = master_mix * (0.95 / peak)

    sf.write(str(out_master), master_mix, sr)
    return True


def duck_and_mix_dub(
    background_wav_path: str | Path,
    voice_segments: list[dict],
    total_duration: float,
    output_wav_path: str | Path,
    original_wav_path: str | Path | None = None,
    ducking_db: float = 4.5,
    attack_ms: float = 60.0,
    release_ms: float = 120.0,
    output_preserved_sfx_path: str | Path | None = None,
    output_voice_solo_path: str | Path | None = None,
    speech_intervals: list[tuple[float, float]] | None = None
) -> bool:
    """Compatibility wrapper that calls preserve_and_mix_dub."""
    orig_path = original_wav_path or background_wav_path
    return preserve_and_mix_dub(
        original_wav_path=orig_path,
        background_wav_path=background_wav_path,
        voice_segments=voice_segments,
        total_duration=total_duration,
        output_master_wav_path=output_wav_path,
        output_preserved_sfx_path=output_preserved_sfx_path,
        output_voice_solo_path=output_voice_solo_path,
        crossfade_ms=30.0,
        speech_intervals=speech_intervals
    )
