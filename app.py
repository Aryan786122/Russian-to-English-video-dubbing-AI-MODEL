"""
LocalDub Workstation — Professional Local Video Dubbing Workstation.

UI Layout (7 Structured Stages):
1. INPUT: Video + Verified SRT/VTT upload, media probe, automated SRT integrity verification.
2. FULL ANALYSIS: Full-video diarization, global speaker consolidation, 15-30s reference mining.
3. CAST: Speaker cards, original sample, reference compilation, direct vs VC tests, chosen pipeline.
4. SUBTITLE / SPEAKER MAP: Interactive cue table with millisecond overlap, dominant speaker, manual re-assignment.
5. 5–10 MINUTE MASTER PREVIEW: Representative window selector, preview generator, video player.
6. QC DASHBOARD: Word completeness QC, duration fitting QC, speaker similarity, interactive HTML report.
7. FULL RENDER: Complete video render (unlocked after verification).
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

import config
from cue_mapper import CueSpeakerAssignment
from diarization_engine import DiarizationSegment, SpeakerProfile
from performance_analyzer import CuePerformanceProfile
from preview_engine import PreviewAcceptanceReport, PreviewWindow
from reference_miner import SpeakerReferenceSet
from subtitle_engine import ImmutableCue, SRTIntegrityError
from timing_duration_engine import CueDurationFittingResult
from tts_engine import SpeakerVoicePipelineChoice
from tts_qc_engine import CueQCResult
from workstation_pipeline import DubbingWorkstation, WorkstationProject

_gradio_tmp = Path(__file__).parent / "temp" / "gradio"
_gradio_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_gradio_tmp)

workstation = DubbingWorkstation(offline_strict=True)

APP_STATE: Dict[str, Any] = {
    "project": None,
    "cues": [],
    "orig_wav": None,
    "segments": [],
    "speaker_map": {},
    "speaker_refs": {},
    "assignments": [],
    "profiles": [],
    "cast_choices": {},
    "preview_report": None,
    "preview_mp4": None,
    "final_mp4": None,
}


def _get_project() -> WorkstationProject:
    p = APP_STATE.get("project")
    if p is None:
        raise gr.Error("Please create a project first by uploading Video and SRT in Tab 1.")
    return p


def _copy_upload(file_obj) -> Path:
    if file_obj is None:
        raise gr.Error("File upload is missing.")
    if isinstance(file_obj, dict) and "path" in file_obj:
        src = Path(file_obj["path"])
    elif hasattr(file_obj, "name") and not isinstance(file_obj, str):
        src = Path(file_obj.name)
    else:
        src = Path(str(file_obj))
    if not src.exists():
        raise gr.Error(f"Uploaded file not found: {src}")

    dst_dir = config.TEMP_DIR / "uploads"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        return dst.resolve()
    except Exception:
        return src.resolve()


# ---------------------------------------------------------------------------
# Tab 1: Input & Validation Handlers
# ---------------------------------------------------------------------------
def handle_open_project(video_file, srt_file, progress=gr.Progress()):
    if not video_file or not srt_file:
        raise gr.Error("Both Video and Verified Subtitle (.srt / .vtt) files are required.")

    vid_path = _copy_upload(video_file)
    srt_path = _copy_upload(srt_file)

    progress(0.1, desc="Probing media container & initializing workstation...")
    try:
        proj = WorkstationProject.create(vid_path, srt_path, offline_strict=True)
        cues, orig_wav = workstation.prepare_source_and_subtitles(proj, progress_cb=lambda p, m: progress(p, desc=m))

        APP_STATE["project"] = proj
        APP_STATE["cues"] = cues
        APP_STATE["orig_wav"] = orig_wav

        info = f"""
### Project Initialized Successfully
- **Video:** `{vid_path.name}` ({proj.source_duration_sec:.2f}s)
- **Subtitle:** `{srt_path.name}` ({len(cues)} authoritative cues)
- **SRT Integrity:** PASSED (Verified 100% ordered, non-overlapping boundaries, non-zero text)
- **Audio Master:** 48,000 Hz 24-bit Stereo
- **Offline Strict Mode:** ENABLED
"""
        return info, gr.update(interactive=True)
    except SRTIntegrityError as e:
        raise gr.Error(f"SRT Integrity Abort: {e}")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Initialization Failed: {e}")


# ---------------------------------------------------------------------------
# Tab 2: Full Analysis Handlers
# ---------------------------------------------------------------------------
def handle_run_full_analysis(progress=gr.Progress()):
    proj = _get_project()
    cues = APP_STATE.get("cues")
    orig_wav = APP_STATE.get("orig_wav")

    try:
        segments, speaker_map, speaker_refs, assignments, profiles = workstation.run_full_diarization_and_refs(
            proj=proj,
            orig_wav=orig_wav,
            cues=cues,
            progress_cb=lambda p, m: progress(p, desc=m),
        )

        APP_STATE["segments"] = segments
        APP_STATE["speaker_map"] = speaker_map
        APP_STATE["speaker_refs"] = speaker_refs
        APP_STATE["assignments"] = assignments
        APP_STATE["profiles"] = profiles

        # Build summary table
        rows = []
        for spk, ref in speaker_refs.items():
            rows.append({
                "Speaker ID": spk,
                "Reference Status": ref.quality_status,
                "Mined Duration": f"{ref.total_duration_sec:.1f}s",
                "Clean Samples": ref.samples_count,
            })
        df = pd.DataFrame(rows)

        summary_msg = f"### Full Analysis Complete\n- Detected **{len(speaker_refs)}** global speaker identities.\n- Mined clean 15-30s composite references across full video."
        return summary_msg, df, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Full Analysis Failed: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Cast Calibration Handlers
# ---------------------------------------------------------------------------
def handle_calibrate_cast(progress=gr.Progress()):
    proj = _get_project()
    cues = APP_STATE.get("cues")
    speaker_refs = APP_STATE.get("speaker_refs")
    assignments = APP_STATE.get("assignments")

    if not speaker_refs:
        raise gr.Error("Run Full Analysis (Tab 2) before calibrating cast.")

    try:
        choices = workstation.calibrate_cast(
            proj=proj,
            speaker_refs=speaker_refs,
            cues=cues,
            assignments=assignments,
            progress_cb=lambda p, m: progress(p, desc=m),
        )
        APP_STATE["cast_choices"] = choices

        rows = []
        for spk, c in choices.items():
            rows.append({
                "Speaker": spk,
                "Direct Similarity": c.direct_similarity,
                "Seed-VC Similarity": c.seedvc_similarity,
                "Chosen Pipeline": c.chosen_pipeline,
                "Best Similarity": c.best_similarity,
            })
        df = pd.DataFrame(rows)
        return df, "### Cast Calibration Complete. Per-speaker voice pipelines chosen."
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Calibration Failed: {e}")


# ---------------------------------------------------------------------------
# Tab 4: Cue Speaker Map Handlers
# ---------------------------------------------------------------------------
def get_cue_table_data():
    assignments: List[CueSpeakerAssignment] = APP_STATE.get("assignments", [])
    rows = []
    for a in assignments:
        rows.append({
            "Cue ID": a.cue_id,
            "Start": f"{a.start_ms / 1000.0:.2f}s",
            "End": f"{a.end_ms / 1000.0:.2f}s",
            "Speaker": a.speaker_id,
            "Confidence": f"{a.assignment_confidence * 100:.0f}%",
            "Multi-Speaker": "⚠️ YES" if a.multi_speaker else "NO",
            "Manual": "✏️" if a.manual_override else "",
            "Text": a.text,
        })
    return pd.DataFrame(rows)


def handle_manual_reassign(cue_id_input, new_speaker_input):
    proj = _get_project()
    assignments: List[CueSpeakerAssignment] = APP_STATE.get("assignments", [])
    if not cue_id_input or not new_speaker_input:
        raise gr.Error("Specify both Cue ID and New Speaker.")

    cue_id = int(cue_id_input)
    updated = workstation.cue_mapper.update_manual_assignment(
        assignments=assignments,
        cue_id=cue_id,
        new_speaker_id=str(new_speaker_input).strip(),
        output_dir=proj.project_dir,
    )
    APP_STATE["assignments"] = updated
    return get_cue_table_data(), f"Updated Cue {cue_id} to {new_speaker_input}"


# ---------------------------------------------------------------------------
# Tab 5: 5-10m Master Preview Handlers
# ---------------------------------------------------------------------------
def handle_render_preview(progress=gr.Progress()):
    proj = _get_project()
    cues = APP_STATE.get("cues")
    assignments = APP_STATE.get("assignments")
    profiles = APP_STATE.get("profiles")
    speaker_refs = APP_STATE.get("speaker_refs")
    cast_choices = APP_STATE.get("cast_choices")

    if not cast_choices:
        # Auto-calibrate if skipped
        cast_choices = workstation.calibrate_cast(proj, speaker_refs, cues, assignments)
        APP_STATE["cast_choices"] = cast_choices

    try:
        preview_mp4, report = workstation.generate_representative_preview(
            proj=proj,
            cues=cues,
            assignments=assignments,
            profiles=profiles,
            speaker_refs=speaker_refs,
            cast_choices=cast_choices,
            progress_cb=lambda p, m: progress(p, desc=m),
        )

        APP_STATE["preview_mp4"] = preview_mp4
        APP_STATE["preview_report"] = report

        status_text = "PASSED (Ready for Full Render)" if report.passed else "NEEDS REVIEW (Violations Detected)"
        report_summary = f"""
### Preview Acceptance Status: **{status_text}**
- **Reviewed Cues:** {report.cues_reviewed_count}
- **Word Completeness:** {report.word_qc_pass_rate * 100:.1f}%
- **Timing Pacing:** {report.timing_qc_pass_rate * 100:.1f}%
- **Voice Similarity:** {report.average_speaker_similarity:.3f}
"""
        unlock_full = gr.update(interactive=report.passed)
        return str(preview_mp4), report_summary, unlock_full
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Preview Generation Failed: {e}")


# ---------------------------------------------------------------------------
# Tab 6: QC Dashboard Handlers
# ---------------------------------------------------------------------------
def get_qc_html():
    proj = APP_STATE.get("project")
    if proj is None:
        return "<p>No active project.</p>"
    html_path = proj.qc_dir / "preview_report.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<p>Run Preview to generate QC Report.</p>"


# ---------------------------------------------------------------------------
# Tab 7: Full Render Handlers
# ---------------------------------------------------------------------------
def handle_full_render(progress=gr.Progress()):
    proj = _get_project()
    cues = APP_STATE.get("cues")
    assignments = APP_STATE.get("assignments")
    profiles = APP_STATE.get("profiles")
    speaker_refs = APP_STATE.get("speaker_refs")
    cast_choices = APP_STATE.get("cast_choices")

    try:
        final_mp4 = workstation.render_full_video(
            proj=proj,
            cues=cues,
            assignments=assignments,
            profiles=profiles,
            speaker_refs=speaker_refs,
            cast_choices=cast_choices,
            progress_cb=lambda p, m: progress(p, desc=m),
        )
        APP_STATE["final_mp4"] = final_mp4
        return str(final_mp4), f"### Full Dubbing Completed Successfully!\nOutput saved to `{final_mp4}`"
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Full Render Failed: {e}")


# ---------------------------------------------------------------------------
# Gradio Workstation UI Construction
# ---------------------------------------------------------------------------
def build_ui():
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: auto !important; }
    .header-box { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 24px; border-radius: 12px; margin-bottom: 20px; }
    """
    with gr.Blocks(title="LocalDub — Professional Local Dubbing Workstation", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["header-box"]):
            with gr.Column():
                gr.Markdown("# 🎙️ LocalDub Workstation")
                gr.Markdown("### Professional, Local, Subtitle-Authoritative Video Dubbing System")
                gr.Markdown("Zero cloud dependencies • Chatterbox Multilingual V3 • Seed-VC Refinement • Sample-Accurate 48kHz Timeline")

        with gr.Tabs():
            # -----------------------------------------------------------------
            # TAB 1: INPUT
            # -----------------------------------------------------------------
            with gr.Tab("1. Input & Validation"):
                gr.Markdown("### Authoritative Input Contract: Video + Verified Subtitle (.srt / .vtt)")
                with gr.Row():
                    video_in = gr.File(label="Original Video File", file_types=["video"])
                    srt_in = gr.File(label="Verified Target-Language SRT / VTT", file_types=[".srt", ".vtt"])

                btn_validate = gr.Button("Validate Source & Run SRT Integrity Test", variant="primary")
                input_status = gr.Markdown("Upload files and click validate.")

            # -----------------------------------------------------------------
            # TAB 2: FULL ANALYSIS
            # -----------------------------------------------------------------
            with gr.Tab("2. Full-Video Analysis"):
                gr.Markdown("### Full-Video Speaker Diarization & Reference Mining")
                gr.Markdown("Analyzes COMPLETE source audio. Extracts persistent speaker identities and mines 15-30s clean references.")
                btn_analyze = gr.Button("Run Full-Video Diarization & Mining", variant="primary", interactive=False)
                analysis_status = gr.Markdown()
                speaker_summary_df = gr.DataFrame(headers=["Speaker ID", "Reference Status", "Mined Duration", "Clean Samples"])

            # -----------------------------------------------------------------
            # TAB 3: CAST
            # -----------------------------------------------------------------
            with gr.Tab("3. Cast Calibration"):
                gr.Markdown("### Multi-Speaker Cast Voice Pipeline Calibration")
                gr.Markdown("Evaluates Chatterbox V3 Direct vs Seed-VC Timbre Transfer per speaker using acoustic similarity.")
                btn_calibrate = gr.Button("Calibrate Voice Pipelines Across Cast", variant="primary", interactive=False)
                cast_status = gr.Markdown()
                cast_df = gr.DataFrame(headers=["Speaker", "Direct Similarity", "Seed-VC Similarity", "Chosen Pipeline", "Best Similarity"])

            # -----------------------------------------------------------------
            # TAB 4: SUBTITLE / SPEAKER MAP
            # -----------------------------------------------------------------
            with gr.Tab("4. Subtitle / Speaker Map"):
                gr.Markdown("### Authoritative SRT Cues & Speaker Mapping")
                btn_refresh_cues = gr.Button("Refresh Mapping Table")
                cue_table = gr.DataFrame(headers=["Cue ID", "Start", "End", "Speaker", "Confidence", "Multi-Speaker", "Manual", "Text"])

                with gr.Row():
                    cue_id_box = gr.Number(label="Cue ID to Reassign", precision=0)
                    new_spk_box = gr.Textbox(label="New Speaker ID (e.g. SPEAKER_01)")
                    btn_reassign = gr.Button("Apply Manual Override")
                reassign_status = gr.Markdown()

            # -----------------------------------------------------------------
            # TAB 5: 5-10 MINUTE MASTER PREVIEW
            # -----------------------------------------------------------------
            with gr.Tab("5. 5-10m Master Preview"):
                gr.Markdown("### Representative 5–10 Minute Multi-Speaker Preview")
                gr.Markdown("Tests casting, exact words, timing, pauses, SFX preservation, and audio bed.")
                btn_preview = gr.Button("Render Representative Master Preview", variant="primary", interactive=False)
                preview_status = gr.Markdown()
                preview_video = gr.Video(label="Rendered Preview Video")

            # -----------------------------------------------------------------
            # TAB 6: QC DASHBOARD
            # -----------------------------------------------------------------
            with gr.Tab("6. QC Dashboard"):
                gr.Markdown("### Automated Acceptance & Word-Completeness QC")
                btn_refresh_qc = gr.Button("Refresh QC Report")
                qc_html_view = gr.HTML()

            # -----------------------------------------------------------------
            # TAB 7: FULL RENDER
            # -----------------------------------------------------------------
            with gr.Tab("7. Full Render"):
                gr.Markdown("### Full Video Render")
                gr.Markdown("Locked until analysis is complete and preview has verified cast, timing, and word completeness.")
                btn_full_render = gr.Button("Render Full Dubbed Video", variant="primary", interactive=False)
                full_status = gr.Markdown()
                final_video = gr.Video(label="Final Dubbed Video")

        # -----------------------------------------------------------------
        # Wire Events
        # -----------------------------------------------------------------
        btn_validate.click(
            handle_open_project,
            inputs=[video_in, srt_in],
            outputs=[input_status, btn_analyze],
        )

        btn_analyze.click(
            handle_run_full_analysis,
            outputs=[analysis_status, speaker_summary_df, btn_calibrate, btn_preview],
        )

        btn_calibrate.click(
            handle_calibrate_cast,
            outputs=[cast_df, cast_status],
        )

        btn_refresh_cues.click(
            get_cue_table_data,
            outputs=[cue_table],
        )

        btn_reassign.click(
            handle_manual_reassign,
            inputs=[cue_id_box, new_spk_box],
            outputs=[cue_table, reassign_status],
        )

        btn_preview.click(
            handle_render_preview,
            outputs=[preview_video, preview_status, btn_full_render],
        )

        btn_refresh_qc.click(
            get_qc_html,
            outputs=[qc_html_view],
        )

        btn_full_render.click(
            handle_full_render,
            outputs=[final_video, full_status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
