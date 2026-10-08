"""
LocalDub CLI — Professional Local Video Dubbing Command-Line Interface.
"""

import argparse
import sys
from pathlib import Path

from workstation_pipeline import DubbingWorkstation, WorkstationProject


def main():
    parser = argparse.ArgumentParser(
        description="LocalDub — Professional, Local, Subtitle-Authoritative Video Dubbing Workstation"
    )
    parser.add_argument("video_path", type=str, help="Path to source video file")
    parser.add_argument("subtitle_path", type=str, help="Path to verified target-language SRT/VTT")
    parser.add_argument(
        "--preview", action="store_true",
        help="Run representative 5-10 minute preview and acceptance test"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Render complete dubbed video"
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Allow internet access to download missing models if needed"
    )

    args = parser.parse_args()

    v_path = Path(args.video_path).resolve()
    s_path = Path(args.subtitle_path).resolve()

    if not v_path.exists():
        print(f"Error: Video file not found: {v_path}")
        sys.exit(1)
    if not s_path.exists():
        print(f"Error: Subtitle file not found: {s_path}")
        sys.exit(1)

    print("=" * 60)
    print("LocalDub Workstation (Local, Subtitle-Authoritative Dubbing)")
    print(f"Video: {v_path.name}")
    print(f"Subtitles: {s_path.name}")
    print("=" * 60)

    workstation = DubbingWorkstation(offline_strict=not args.online)

    def print_progress(pct: float, desc: str):
        print(f"[{pct * 100:5.1f}%] {desc}", flush=True)

    try:
        # Step 1: Open project & verify SRT
        proj = WorkstationProject.create(v_path, s_path, offline_strict=not args.online)
        cues, orig_wav = workstation.prepare_source_and_subtitles(proj, progress_cb=print_progress)

        # Step 2: Full analysis (diarization, consolidation, refs, mapping, performance)
        segments, speaker_map, speaker_refs, assignments, profiles = workstation.run_full_diarization_and_refs(
            proj=proj,
            orig_wav=orig_wav,
            cues=cues,
            progress_cb=print_progress,
        )

        # Step 3: Cast calibration
        cast_choices = workstation.calibrate_cast(
            proj=proj,
            speaker_refs=speaker_refs,
            cues=cues,
            assignments=assignments,
            progress_cb=print_progress,
        )

        # Step 4: Preview or Full Render
        if args.preview or not args.full:
            print("\n--- Running Representative 5-10 Minute Preview ---")
            preview_mp4, report = workstation.generate_representative_preview(
                proj=proj,
                cues=cues,
                assignments=assignments,
                profiles=profiles,
                speaker_refs=speaker_refs,
                cast_choices=cast_choices,
                progress_cb=print_progress,
            )
            print(f"\nPreview rendered: {preview_mp4}")
            print(f"Acceptance Status: {'PASSED' if report.passed else 'NEEDS REVIEW'}")
            print(f"Word Completeness: {report.word_qc_pass_rate * 100:.1f}%")
            print(f"Timing Accuracy: {report.timing_qc_pass_rate * 100:.1f}%")

        if args.full:
            print("\n--- Rendering Full Dubbed Video ---")
            final_mp4 = workstation.render_full_video(
                proj=proj,
                cues=cues,
                assignments=assignments,
                profiles=profiles,
                speaker_refs=speaker_refs,
                cast_choices=cast_choices,
                progress_cb=print_progress,
            )
            print(f"\nFull render complete: {final_mp4}")

        print("\nWorkstation run completed successfully.")
    except Exception as e:
        print(f"\nExecution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
