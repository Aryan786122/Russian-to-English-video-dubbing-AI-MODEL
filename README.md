# LocalDub: Professional Local Video Dubbing Workstation

A 100% offline, fully local AI video dubbing workstation for **ANY** input video and **ANY** verified target-language subtitle file (`.srt` / `.vtt`).

---

## Core Architectural Contract

1. **The Subtitle File is 100% Authoritative:**
   - No re-transcription to obtain text
   - No machine re-translation
   - No summarization, censorship, or fuzzy matching
   - Subtitle text, timestamps (`start_ms`, `end_ms`), and cue order are immutable.

2. **Full-Video Speaker Analysis & Consolidation:**
   - Analyzes the **COMPLETE** source audio with `pyannote/speaker-diarization-community-1`
   - Global persistent identities (`SPEAKER_00`, `SPEAKER_01`, ...) across the entire film
   - Global consolidation pass merging split identities using speaker embeddings and non-overlapping patterns
   - Full-video reference mining: 15–30 seconds of clean, isolated composite reference per actor.

3. **Multi-Speaker TTS & Voice Refinement:**
   - **Primary Engine:** Chatterbox Multilingual V3 (`language_id="en"`)
   - **Timbre Refinement:** Optional Seed-VC / ChatterboxVC timbre transfer
   - **Full Cast Calibration:** Evaluates Direct vs VC pipelines per speaker using acoustic similarity.

4. **Exact Word Preservation & QC:**
   - Every word must be spoken completely
   - Local English Whisper ASR QC pass compares recognized words against normalized authoritative text
   - Multi-seed regeneration on mismatch; flags `HARD_QC_FAIL` if retries fail.
   - Punctuation-based splitting prevents internal model truncation.

5. **Pause Preservation & Timing Fitting:**
   - Safe pitch-preserving time-scale modification (strictly bounded within ±10–15%)
   - Safe silence borrowing from adjacent non-colliding intervals (never drops or chops words)
   - Preserves source performance rhythm and punctuation pauses.

6. **Sample-Accurate 48 kHz Canvas & SFX Preservation:**
   - Absolute sample positioning at 48,000 Hz: `round(start_sec * 48000)`
   - Zero cumulative drift across the entire timeline
   - **Never destroys the original soundtrack:** 100% untouched original audio outside dialogue windows
   - Transient SFX preservation detects claps, knocks, and impacts and restores them into dialogue windows
   - Preserves original non-verbal human reactions (laughs, gasps, breaths).

7. **Representative 5–10 Minute Preview:**
   - Automatically selects a multi-speaker 5–10 minute representative window
   - Automated Preview Acceptance Test verifies SRT integrity, speakers, words, timing, and audio bed
   - Full video render is unlocked only when the preview passes.

---

## Workstation UI (Gradio)

Run `run.bat` or execute in PowerShell:
```powershell
.\venv\Scripts\python.exe app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 7-Stage Workstation Workflow:
1. **Input & Validation:** Upload Video and Verified Subtitle. Probes metadata and runs automated SRT integrity verification.
2. **Full Analysis:** Complete video diarization, global consolidation, and 15–30s clean reference mining.
3. **Cast Calibration:** Audition original references, test Direct Chatterbox vs VC, and pick per-speaker pipelines.
4. **Subtitle / Speaker Map:** Interactive table with millisecond overlap and manual override controls.
5. **5–10m Master Preview:** Generates representative multi-scene preview with video player and audio stems.
6. **QC Dashboard:** Comprehensive word completeness, timing accuracy, and HTML acceptance report.
7. **Full Render:** Renders the complete movie with sample-accurate dialogue and preserved SFX.

---

## Command-Line Interface (CLI)

Run via CLI:
```powershell
# Run representative 5-10 minute preview and acceptance test:
.\venv\Scripts\python.exe cli.py "video.mp4" "subs.srt" --preview

# Render full video:
.\venv\Scripts\python.exe cli.py "video.mp4" "subs.srt" --full
```

---

## Automated Test Suite

Execute the unit test suite:
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
Tests cover:
- Deterministic SRT parsing, cue immutability, and corruption aborts
- Absolute sample-accurate 48 kHz timeline compositing and drift prevention
- Exact-word Whisper QC and difference detection
- SFX preservation, transient clap/impact extraction, and audio crossfading.
