"""
LocalDub Professional Video Dubbing Workstation Pipeline.

Core Orchestrator connecting:
- Deterministic SRT engine & automated integrity verification
- Full-video speaker diarization & global consolidation
- Full-video reference mining (15-30s clean composite audio)
- Exact temporal cue-to-speaker mapping with multi-speaker detection
- Source performance analysis (F0, RMS, pauses, emotion, laughter)
- Chatterbox Multilingual V3 TTS + optional Seed-VC voice conversion
- Voice calibration across the full cast
- Exact-word English ASR QC with multi-seed regeneration & HARD_QC_FAIL
- Pause preservation & safe duration fitting (librosa +-15%, silence borrowing)
- 48 kHz absolute sample-accurate timeline compositor (zero drift)
- High-fidelity audio separation & SFX/transient-preserving master mixer
- Representative 5-10 minute preview & acceptance testing
- Strict sequential GPU memory management for RTX 4050 6GB
- Strict offline local execution
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

import config
from cue_mapper import CueSpeakerAssignment, CueSpeakerMapper
from diarization_engine import DiarizationSegment, FullVideoDiarizer, SpeakerProfile
from performance_analyzer import CuePerformanceProfile, PerformanceAnalyzer
from preview_engine import PreviewAcceptanceReport, PreviewEngine, PreviewWindow
from reference_miner import ReferenceMiner, SpeakerReferenceSet
from sfx_mixer import SFXPreservingMixer, StemSeparator
from subtitle_engine import (
    ImmutableCue,
    SRTIntegrityError,
    export_cues_to_json,
    load_cues_from_json,
    parse_subtitles_deterministic,
    verify_srt_integrity,
)
from timeline_compositor import TimelineCompositor
from timeline_validator import probe_media_metadata
from timing_duration_engine import CueDurationFittingResult, TimingDurationEngine
from tts_engine import (
    ChatterboxV3Engine,
    SpeakerVoicePipelineChoice,
    VoiceCalibrator,
    VoiceConversionEngine,
    VoiceReferenceMissingError,
)
from tts_qc_engine import CueQCResult, WordCompletenessQC

ProgressCallback = Optional[Callable[[float, str], None]]


def file_sha256(path: Union[str, Path]) -> str:
    """Computes SHA256 hash of a file for cache verification."""
    p = Path(path).resolve()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class WorkstationProject:
    video_path: str
    subtitle_path: str
    video_hash: str
    subtitle_hash: str
    source_duration_sec: float
    project_dir: Path
    analysis_dir: Path
    refs_dir: Path
    tts_dir: Path
    qc_dir: Path
    audio_dir: Path
    output_dir: Path
    offline_strict: bool = True

    @classmethod
    def create(cls, video_path: str | Path, subtitle_path: str | Path, offline_strict: bool = True) -> WorkstationProject:
        v_path = Path(video_path).resolve()
        s_path = Path(subtitle_path).resolve()

        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")
        if not s_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {s_path}")

        v_hash = file_sha256(v_path)
        s_hash = file_sha256(s_path)

        meta = probe_media_metadata(v_path)
        source_dur = float(meta["source_duration"])

        stem = v_path.stem
        # Workspace layout keyed by video and subtitle SHA256
        proj_dir = config.OUTPUT_DIR / f"{stem}_{v_hash[:8]}_{s_hash[:8]}"
        proj_dir.mkdir(parents=True, exist_ok=True)

        analysis_dir = proj_dir / "analysis"
        refs_dir = proj_dir / "speaker_refs"
        tts_dir = proj_dir / "tts"
        qc_dir = proj_dir / "qc"
        audio_dir = proj_dir / "audio"
        output_dir = proj_dir / "output"

        for d in (analysis_dir, refs_dir, tts_dir, qc_dir, audio_dir, output_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Save source_info.json
        info_path = analysis_dir / "source_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_file": v_path.name,
                "subtitle_file": s_path.name,
                "video_sha256": v_hash,
                "subtitle_sha256": s_hash,
                "source_duration_sec": source_dur,
                "media_metadata": meta,
            }, f, indent=2)

        return cls(
            video_path=str(v_path),
            subtitle_path=str(s_path),
            video_hash=v_hash,
            subtitle_hash=s_hash,
            source_duration_sec=source_dur,
            project_dir=proj_dir,
            analysis_dir=analysis_dir,
            refs_dir=refs_dir,
            tts_dir=tts_dir,
            qc_dir=qc_dir,
            audio_dir=audio_dir,
            output_dir=output_dir,
            offline_strict=offline_strict,
        )


class DubbingWorkstation:
    def __init__(self, offline_strict: bool = True):
        self.offline_strict = offline_strict
        if offline_strict:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Modular engines
        self.diarizer = FullVideoDiarizer()
        self.miner = ReferenceMiner()
        self.cue_mapper = CueSpeakerMapper()
        self.perf_analyzer = PerformanceAnalyzer()
        self.tts_engine = ChatterboxV3Engine()
        self.vc_engine = VoiceConversionEngine()
        self.calibrator = VoiceCalibrator()
        self.word_qc = WordCompletenessQC()
        self.duration_engine = TimingDurationEngine()
        self.compositor = TimelineCompositor()
        self.separator = StemSeparator()
        self.mixer = SFXPreservingMixer()
        self.preview_engine = PreviewEngine()

    def _gpu_cleanup(self):
        """Strict VRAM cleanup for RTX 4050 6GB."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 1: Source Validation & Absolute Deterministic SRT Engine
    # =========================================================================
    def prepare_source_and_subtitles(
        self,
        proj: WorkstationProject,
        progress_cb: ProgressCallback = None,
    ) -> Tuple[List[ImmutableCue], Path]:
        """
        Extracts 48kHz original audio master, parses subtitles deterministically with pysubs2,
        and verifies automated SRT integrity before any TTS is permitted.
        """
        if progress_cb:
            progress_cb(0.05, "Extracting authoritative 48kHz audio master...")

        orig_wav = proj.audio_dir / "original_master.wav"
        if not orig_wav.exists():
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", proj.video_path,
                "-vn",
                "-ar", "48000",
                "-ac", "2",
                "-acodec", "pcm_s24le",
                str(orig_wav)
            ]
            subprocess.run(cmd, check=True)

        if progress_cb:
            progress_cb(0.10, "Parsing verified subtitles & running automated integrity check...")

        cues = parse_subtitles_deterministic(proj.subtitle_path)
        # Automated SRT integrity test before any TTS is allowed:
        # Aborts immediately on violation
        verify_srt_integrity(cues, source_duration_sec=proj.source_duration_sec)

        # Export immutable cues JSON
        cues_json = proj.analysis_dir / "cues.json"
        export_cues_to_json(cues, cues_json)
        print(f"[Workstation] Verified {len(cues)} authoritative cues. SRT integrity passed.", flush=True)

        return cues, orig_wav

    # =========================================================================
    # STAGE 2: Full-Video Speaker Diarization & Consolidation
    # =========================================================================
    def run_full_diarization_and_refs(
        self,
        proj: WorkstationProject,
        orig_wav: Path,
        cues: List[ImmutableCue],
        progress_cb: ProgressCallback = None,
    ) -> Tuple[List[DiarizationSegment], Dict[str, str], Dict[str, SpeakerReferenceSet], List[CueSpeakerAssignment], List[CuePerformanceProfile]]:
        """
        Runs full-video speaker diarization, global consolidation, reference mining,
        cue-to-speaker mapping, and source performance analysis across the COMPLETE audio.
        """
        # 1. Stem separation (vocals for clean diarization & reference mining)
        vocals_wav = proj.audio_dir / "vocals_48k.wav"
        bg_wav = proj.audio_dir / "separated_background.wav"
        if progress_cb:
            progress_cb(0.15, "Separating dialogue and background stems...")

        self.separator.separate_stems(orig_wav, vocals_wav, bg_wav)
        self._gpu_cleanup()

        # 2. Full-video diarization
        if progress_cb:
            progress_cb(0.30, "Running full-video diarization & global speaker consolidation...")

        diar_path = proj.analysis_dir / "full_diarization.json"
        map_path = proj.analysis_dir / "speaker_map.json"

        if diar_path.exists() and map_path.exists():
            print("[Workstation] Using cached full diarization and speaker map.", flush=True)
            with open(diar_path, "r", encoding="utf-8") as f:
                d_data = json.load(f)
            segments = [
                DiarizationSegment(
                    start=s["start"],
                    end=s["end"],
                    speaker=s["speaker"],
                    overlap=s.get("overlap", False),
                    confidence=s.get("confidence", 1.0),
                )
                for s in d_data
            ]
            with open(map_path, "r", encoding="utf-8") as f:
                sm_data = json.load(f)
            speaker_map = sm_data.get("mapping", {})
        else:
            segments, speaker_map, _ = self.diarizer.run_full_diarization(
                audio_path=vocals_wav if vocals_wav.exists() else orig_wav,
                output_dir=proj.project_dir,
                progress_cb=progress_cb,
            )
            self._gpu_cleanup()

        # 3. Full-video reference mining
        if progress_cb:
            progress_cb(0.55, "Mining 15-30s clean references across full video for all speakers...")

        refs_manifest_path = proj.refs_dir / "refs_manifest.json"
        if refs_manifest_path.exists():
            print("[Workstation] Using cached speaker references.", flush=True)
            with open(refs_manifest_path, "r", encoding="utf-8") as f:
                rm_data = json.load(f)
            speaker_refs = {
                spk: SpeakerReferenceSet(
                    speaker_id=spk,
                    total_duration_sec=v["total_duration_sec"],
                    samples_count=v["samples_count"],
                    quality_status=v["quality_status"],
                    ref_paths=v["ref_paths"],
                    combined_path=v["combined_path"],
                )
                for spk, v in rm_data.items()
            }
        else:
            speaker_refs = self.miner.mine_all_speakers(
                audio_path=orig_wav,
                segments=segments,
                output_dir=proj.project_dir,
                vocals_path=vocals_wav,
            )

        # 4. SRT Cue -> Speaker Mapping
        if progress_cb:
            progress_cb(0.70, "Mapping SRT cues to dominant speakers with exact overlap...")

        cue_map_path = proj.analysis_dir / "cue_speaker_map.json"
        if cue_map_path.exists():
            with open(cue_map_path, "r", encoding="utf-8") as f:
                cm_data = json.load(f)
            assignments = [CueSpeakerAssignment.from_dict(d) for d in cm_data]
        else:
            assignments = self.cue_mapper.map_cues_to_speakers(
                cues=cues,
                diarization_segments=segments,
                output_dir=proj.project_dir,
            )

        # 5. Source Performance Analysis
        if progress_cb:
            progress_cb(0.80, "Analyzing source performance dynamics, pauses & pitch contour...")

        perf_path = proj.analysis_dir / "cue_performance.json"
        if perf_path.exists():
            with open(perf_path, "r", encoding="utf-8") as f:
                pf_data = json.load(f)
            profiles = [
                CuePerformanceProfile(
                    cue_id=p["cue_id"],
                    start_sec=p["start_sec"],
                    end_sec=p["end_sec"],
                    speech_onset_sec=p["speech_onset_sec"],
                    speech_offset_sec=p["speech_offset_sec"],
                    internal_pauses_ms=p["internal_pauses_ms"],
                    speech_density=p["speech_density"],
                    mean_f0_hz=p["mean_f0_hz"],
                    f0_std=p["f0_std"],
                    f0_contour=p["f0_contour"],
                    mean_rms=p["mean_rms"],
                    peak_rms=p["peak_rms"],
                    energy_level=p["energy_level"],
                    speaking_rate_cps=p["speaking_rate_cps"],
                    delivery_character=p["delivery_character"],
                    has_laugh=p.get("has_laugh", False),
                    has_gasp=p.get("has_gasp", False),
                )
                for p in pf_data
            ]
        else:
            profiles = self.perf_analyzer.analyze_all_cues(
                audio_path=orig_wav,
                cues=cues,
                output_dir=proj.project_dir,
                vocals_path=vocals_wav,
            )

        return segments, speaker_map, speaker_refs, assignments, profiles

    # =========================================================================
    # STAGE 3: Full Cast Voice Calibration
    # =========================================================================
    def calibrate_cast(
        self,
        proj: WorkstationProject,
        speaker_refs: Dict[str, SpeakerReferenceSet],
        cues: List[ImmutableCue],
        assignments: List[CueSpeakerAssignment],
        progress_cb: ProgressCallback = None,
    ) -> Dict[str, SpeakerVoicePipelineChoice]:
        """
        Calibrates voice engine per speaker:
        Tests Chatterbox direct vs Chatterbox + VC, picking the winning pipeline per speaker.
        """
        if progress_cb:
            progress_cb(0.85, "Calibrating voice cloning pipeline across all speakers...")

        calib_json = proj.project_dir / "voice_calibration" / "cast_calibration.json"
        if calib_json.exists():
            print("[Workstation] Using cached cast calibration.", flush=True)
            with open(calib_json, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            choices = {
                spk: SpeakerVoicePipelineChoice(
                    speaker_id=spk,
                    chosen_pipeline=v["chosen_pipeline"],
                    direct_similarity=v["direct_similarity"],
                    seedvc_similarity=v["seedvc_similarity"],
                    best_similarity=v["best_similarity"],
                    calibration_files=v.get("calibration_files", {}),
                )
                for spk, v in c_data.items()
            }
            return choices

        cue_spk_dict = {a.cue_id: a.speaker_id for a in assignments}
        choices = self.calibrator.calibrate_full_cast(
            speaker_refs=speaker_refs,
            cues=cues,
            cue_speaker_map=cue_spk_dict,
            output_dir=proj.project_dir,
            tts_engine=self.tts_engine,
            vc_engine=self.vc_engine,
        )
        self.tts_engine.unload()
        self.vc_engine.unload()
        self._gpu_cleanup()
        return choices

    # =========================================================================
    # STAGE 4: Synthesis, QC & Fitting for a Cue Range (Preview or Full)
    # =========================================================================
    def render_cue_range(
        self,
        proj: WorkstationProject,
        cues: List[ImmutableCue],
        assignments: List[CueSpeakerAssignment],
        profiles: List[CuePerformanceProfile],
        speaker_refs: Dict[str, SpeakerReferenceSet],
        cast_choices: Dict[str, SpeakerVoicePipelineChoice],
        target_cues: List[ImmutableCue],
        progress_cb: ProgressCallback = None,
    ) -> Tuple[List[Tuple[np.ndarray, int, CueDurationFittingResult]], List[CueQCResult]]:
        """
        Synthesizes target cues with:
        1. Chatterbox Multilingual V3 (language_id="en", strict speaker reference)
        2. Optional Voice Conversion if selected during calibration
        3. Word-completeness QC via local English faster-whisper with multi-seed retries
        4. Safe duration fitting with librosa pitch-preserving stretch (bounded +-15%) and silence borrowing
        """
        asgn_map = {a.cue_id: a for a in assignments}
        prof_map = {p.cue_id: p for p in profiles}

        fitted_audio_results: List[Tuple[np.ndarray, int, CueDurationFittingResult]] = []
        qc_results: List[CueQCResult] = []

        total_to_render = len(target_cues)
        print(f"[Workstation] Synthesizing & QCing {total_to_render} cues with Chatterbox V3...", flush=True)

        # Preload TTS model
        self.tts_engine.load_model()

        for idx, cue in enumerate(target_cues, start=1):
            if progress_cb:
                pct = 0.10 + (0.75 * (idx / total_to_render))
                progress_cb(pct, f"Synthesizing & QCing cue {cue.cue_id}/{total_to_render}...")

            asgn = asgn_map.get(cue.cue_id)
            speaker_id = asgn.speaker_id if asgn else "SPEAKER_00"
            prof = prof_map.get(cue.cue_id)

            ref_set = speaker_refs.get(speaker_id)
            if not ref_set or not Path(ref_set.combined_path).exists():
                raise VoiceReferenceMissingError(f"Cue {cue.cue_id}: No voice reference available for {speaker_id}")

            ref_audio_path = Path(ref_set.combined_path)
            raw_cue_wav = proj.tts_dir / cue.raw_wav_filename
            final_cue_wav = proj.tts_dir / cue.final_wav_filename

            # Iterative generation & QC loop (up to 3 seeds)
            best_qc: Optional[CueQCResult] = None
            success = False

            for attempt in range(1, 4):
                seed = 42 + (attempt * 1337)
                temp = 0.8 if attempt == 1 else (0.7 if attempt == 2 else 0.85)

                tts_ok = self.tts_engine.synthesize_cue(
                    text=cue.text,
                    ref_audio_path=ref_audio_path,
                    output_path=raw_cue_wav,
                    performance_profile=prof,
                    seed=seed,
                    temperature=temp,
                )
                if not tts_ok:
                    continue

                # Voice Conversion stage if chosen for this speaker
                choice = cast_choices.get(speaker_id)
                if choice and choice.chosen_pipeline == "CHATTERBOX_SEEDVC":
                    vc_temp_wav = proj.tts_dir / f"cue_{cue.cue_id:06d}_vc.wav"
                    if self.vc_engine.convert_voice(raw_cue_wav, ref_audio_path, vc_temp_wav):
                        shutil.copy2(vc_temp_wav, raw_cue_wav)

                # English Whisper ASR QC verification
                qc = self.word_qc.verify_cue(
                    cue_id=cue.cue_id,
                    wav_path=raw_cue_wav,
                    expected_text=cue.text,
                    attempt_count=attempt,
                )
                best_qc = qc

                if qc.status == "PASS":
                    success = True
                    break
                else:
                    print(f"[Workstation] Cue {cue.cue_id} retry {attempt} needed: missing {qc.missing_words}", flush=True)

            if not success and best_qc is not None:
                best_qc.status = "HARD_QC_FAIL"
                print(f"[Workstation] FATAL QC: Cue {cue.cue_id} failed all retries: missing {best_qc.missing_words}", flush=True)

            qc_results.append(best_qc)

            # Duration fitting & safe silence borrowing
            raw_audio, sr = sf.read(str(raw_cue_wav), dtype="float32")
            fitted_audio, dur_result = self.duration_engine.fit_cue_audio(
                cue=cue,
                raw_audio=raw_audio,
                sr=sr,
                all_cues=cues,
                video_duration_sec=proj.source_duration_sec,
                performance_profile=prof,
            )

            # Save final fitted cue WAV
            sf.write(str(final_cue_wav), fitted_audio, sr)
            fitted_audio_results.append((fitted_audio, sr, dur_result))

        # Unload models after batch
        self.tts_engine.unload()
        self.vc_engine.unload()
        self.word_qc.unload()
        self._gpu_cleanup()

        # Export QC records
        self.word_qc.export_qc_report(qc_results, proj.project_dir)
        self.duration_engine.export_duration_qc([r[2] for r in fitted_audio_results], proj.project_dir)

        return fitted_audio_results, qc_results

    # =========================================================================
    # STAGE 5: 5-10 Minute Representative Preview
    # =========================================================================
    def generate_representative_preview(
        self,
        proj: WorkstationProject,
        cues: List[ImmutableCue],
        assignments: List[CueSpeakerAssignment],
        profiles: List[CuePerformanceProfile],
        speaker_refs: Dict[str, SpeakerReferenceSet],
        cast_choices: Dict[str, SpeakerVoicePipelineChoice],
        progress_cb: ProgressCallback = None,
    ) -> Tuple[Path, PreviewAcceptanceReport]:
        """
        Selects 5-10 minute representative window, renders cues, mixes audio bed,
        runs acceptance test, and produces output/preview_dubbed.mp4.
        """
        if progress_cb:
            progress_cb(0.10, "Selecting representative 5-10 minute preview window...")

        window = self.preview_engine.select_representative_window(
            cues=cues,
            assignments=assignments,
            total_video_duration_sec=proj.source_duration_sec,
        )

        preview_cues = [c for c in cues if c.start_sec >= window.start_sec and c.end_sec <= window.end_sec]

        # Render preview cues
        fitted_cues, qc_results = self.render_cue_range(
            proj=proj,
            cues=cues,
            assignments=assignments,
            profiles=profiles,
            speaker_refs=speaker_refs,
            cast_choices=cast_choices,
            target_cues=preview_cues,
            progress_cb=progress_cb,
        )

        # Composite dialogue canvas for preview window
        preview_dialogue_wav = proj.audio_dir / "preview_dialogue_48k.wav"
        self.compositor.create_dialogue_track(
            cues_audio=fitted_cues,
            source_duration_sec=proj.source_duration_sec,
            output_path=preview_dialogue_wav,
        )

        # Mix preview master audio
        preview_mix_wav = proj.audio_dir / "preview_mix_48k.wav"
        bg_wav = proj.audio_dir / "separated_background.wav"
        orig_wav = proj.audio_dir / "original_master.wav"

        intervals = [(c.start_sec, c.end_sec) for c in preview_cues]
        self.mixer.mix_dub_master(
            original_audio_path=orig_wav,
            background_audio_path=bg_wav,
            dialogue_audio_path=preview_dialogue_wav,
            dialogue_intervals=intervals,
            output_mix_path=preview_mix_wav,
        )

        # Mux preview video
        preview_mp4 = proj.output_dir / "preview_dubbed.mp4"
        self.mixer.mux_video_and_audio(
            video_path=proj.video_path,
            audio_path=preview_mix_wav,
            output_path=preview_mp4,
            duration_sec=window.duration_sec,
        )

        # Run Preview Acceptance Test
        dur_results = [f[2] for f in fitted_cues]
        calib_dict = {k: v.to_dict() for k, v in cast_choices.items()}

        report = self.preview_engine.run_acceptance_test(
            preview_cues=preview_cues,
            qc_results=qc_results,
            duration_results=dur_results,
            speaker_calibration=calib_dict,
            preview_video_path=str(preview_mp4),
            output_dir=proj.project_dir,
        )

        return preview_mp4, report

    # =========================================================================
    # STAGE 6: Full Video Render (Only after Preview / Explicit Confirmation)
    # =========================================================================
    def render_full_video(
        self,
        proj: WorkstationProject,
        cues: List[ImmutableCue],
        assignments: List[CueSpeakerAssignment],
        profiles: List[CuePerformanceProfile],
        speaker_refs: Dict[str, SpeakerReferenceSet],
        cast_choices: Dict[str, SpeakerVoicePipelineChoice],
        progress_cb: ProgressCallback = None,
    ) -> Path:
        """
        Renders all remaining cues, composites full 48kHz master dialogue track,
        preserves audio bed & transients, and exports output/final_dubbed.mp4.
        """
        print(f"[Workstation] Starting full render across all {len(cues)} cues...", flush=True)

        fitted_cues, qc_results = self.render_cue_range(
            proj=proj,
            cues=cues,
            assignments=assignments,
            profiles=profiles,
            speaker_refs=speaker_refs,
            cast_choices=cast_choices,
            target_cues=cues,
            progress_cb=progress_cb,
        )

        # Composite full master dialogue track
        dialogue_wav = proj.audio_dir / "english_dialogue.wav"
        if progress_cb:
            progress_cb(0.90, "Compositing 48kHz sample-accurate master dialogue canvas...")

        self.compositor.create_dialogue_track(
            cues_audio=fitted_cues,
            source_duration_sec=proj.source_duration_sec,
            output_path=dialogue_wav,
        )

        # Mix full master audio
        final_mix_wav = proj.audio_dir / "final_mix.wav"
        bg_wav = proj.audio_dir / "separated_background.wav"
        orig_wav = proj.audio_dir / "original_master.wav"

        if progress_cb:
            progress_cb(0.95, "Mixing master soundtrack with transient SFX & Foley preservation...")

        all_intervals = [(c.start_sec, c.end_sec) for c in cues]
        self.mixer.mix_dub_master(
            original_audio_path=orig_wav,
            background_audio_path=bg_wav,
            dialogue_audio_path=dialogue_wav,
            dialogue_intervals=all_intervals,
            output_mix_path=final_mix_wav,
        )

        # Mux final video
        final_mp4 = proj.output_dir / "final_dubbed.mp4"
        self.mixer.mux_video_and_audio(
            video_path=proj.video_path,
            audio_path=final_mix_wav,
            output_path=final_mp4,
        )

        if progress_cb:
            progress_cb(1.0, f"Full render complete: {final_mp4.name}")

        print(f"[Workstation] SUCCESS! Final dubbed video rendered at {final_mp4}", flush=True)
        return final_mp4
