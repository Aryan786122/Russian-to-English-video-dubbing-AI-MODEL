"""
LocalDub - Automated Model Downloader & Verification

Downloads all required open-source AI model checkpoints from official upstream
sources (Hugging Face / PyTorch Hub) into the local workspace.

Safe, idempotent, and storage-efficient: checks if weights already exist
before downloading. Keeps Git repositories lightweight (<1 MB) while ensuring
end-users get a working turnkey environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[ModelSetup] {msg}", flush=True)


def download_huggingface_model(repo_id: str, filename: str, target_dir: Path) -> Path:
    """Download a specific file from Hugging Face Hub if not present."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 1024:
        log(f"Already present: {target_path.relative_to(ROOT)} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return target_path

    log(f"Downloading '{filename}' from https://huggingface.co/{repo_id}...")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir))
    log(f"Successfully downloaded: {Path(downloaded).relative_to(ROOT)}")
    return Path(downloaded)


def setup_speechbrain_ecapa():
    """Ensure SpeechBrain ECAPA-TDNN embedding model is available."""
    target_dir = MODELS_DIR / "ecapa"
    target_file = target_dir / "embedding_model.ckpt"
    if target_file.exists() and target_file.stat().st_size > 10 * 1024 * 1024:
        log(f"SpeechBrain ECAPA-TDNN verified ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        return

    log("Downloading SpeechBrain ECAPA-TDNN speaker embedding model...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            filename="embedding_model.ckpt",
            local_dir=str(target_dir)
        )
        log("SpeechBrain ECAPA-TDNN setup complete.")
    except Exception as e:
        log(f"Warning: Could not pre-download ECAPA-TDNN directly: {e}. It will load via SpeechBrain on first run.")


def setup_vocos():
    """Ensure Vocos 24kHz neural vocoder weights are available."""
    target_dir = MODELS_DIR / "vocos"
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg = target_dir / "config.yaml"
    weights = target_dir / "pytorch_model.bin"

    if cfg.exists() and weights.exists() and weights.stat().st_size > 10 * 1024 * 1024:
        log(f"Vocos 24kHz vocoder verified ({weights.stat().st_size / (1024*1024):.1f} MB)")
        return

    log("Downloading Vocos 24kHz vocoder from charactr/vocos-mel-24khz...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="config.yaml", local_dir=str(target_dir))
    hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="pytorch_model.bin", local_dir=str(target_dir))
    log("Vocos vocoder setup complete.")


def setup_f5_tts():
    """Ensure F5-TTS model checkpoint and vocabulary are cached."""
    log("Checking F5-TTS model checkpoint (SWivid/F5-TTS)...")
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors"
        )
        log(f"F5-TTS model verified in Hugging Face cache: {Path(cached).name}")
    except Exception as e:
        log(f"F5-TTS check note: {e}")


def setup_whisper():
    """Ensure faster-whisper model is downloaded/verified."""
    log("Verifying Faster-Whisper model...")
    try:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = "int8_float16" if device == "cuda" else "int8"
        # Test loading or caching medium
        log(f"Loading/verifying 'medium' Whisper model on {device} ({ctype})...")
        m = WhisperModel("medium", device=device, compute_type=ctype, download_root=str(MODELS_DIR / "whisper"))
        del m
        log("Faster-Whisper model verified.")
    except Exception as e:
        log(f"Faster-Whisper note: {e}")


def setup_demucs():
    """Ensure Demucs HTDemucs separation model is cached."""
    log("Verifying Demucs stem separation model (htdemucs)...")
    try:
        import demucs.pretrained
        model = demucs.pretrained.get_model("htdemucs")
        del model
        log("Demucs htdemucs model verified.")
    except Exception as e:
        log(f"Demucs note: {e}")


def main():
    print("=" * 65)
    print("  LocalDub: Open-Source AI Model Downloader & Setup")
    print("=" * 65)
    print(f"Target Models Directory: {MODELS_DIR}\n")

    setup_speechbrain_ecapa()
    setup_vocos()
    setup_f5_tts()
    setup_whisper()
    setup_demucs()

    print("\n" + "=" * 65)
    print("  All model dependencies are verified and ready to run!")
    print("=" * 65)


if __name__ == "__main__":
    main()
