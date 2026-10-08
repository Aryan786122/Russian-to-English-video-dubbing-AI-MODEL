import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Check for local high-quality models already on disk
LOCAL_WHISPER_LARGE = Path(r"C:\Users\aryan\Documents\Codex\2026-10-02\english-dub-that-sounds-like-the-2\work\models\whisper-large-v3")
LOCAL_WHISPER_TINY = Path(r"C:\Users\aryan\Documents\Codex\2026-10-02\english-dub-that-sounds-like-the-2\work\models\whisper-tiny")

if LOCAL_WHISPER_LARGE.exists() and (LOCAL_WHISPER_LARGE / "model.bin").exists():
    DEFAULT_WHISPER_MODEL = str(LOCAL_WHISPER_LARGE)
else:
    DEFAULT_WHISPER_MODEL = "large-v3"

SOURCE_LANGUAGE = "ru"
TARGET_TASK = "transcribe"  # Native Russian transcription first!

# Audio settings
SAMPLE_RATE = 44100
MAX_TEMPO_STRETCH = 1.22
MIN_TEMPO_STRETCH = 0.85
