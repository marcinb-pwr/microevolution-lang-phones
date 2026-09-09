"""Command line validation for portable local projects."""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .measure import measure_interval, trajectory_json
from .project import Project, ProjectError, _read_csv


def main(argv=None):
    parser = argparse.ArgumentParser(prog="microevolution")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate manifest, tables, audio, and hashes")
    validate.add_argument("manifest", type=Path)
    extract = sub.add_parser("extract", help="measure all manual token intervals with Praat/Burg")
    extract.add_argument("manifest", type=Path)
    extract.add_argument("--max-formant-hz", type=float, default=5500)
    extract.add_argument("--window-s", type=float, default=0.025)
    extract.add_argument("--force", action="store_true", help="replace existing measurements")
    collect = sub.add_parser("collect-youtube", help="download YouTube audio with pinned provenance")
    collect.add_argument("manifest", type=Path)
    collect.add_argument("urls", nargs="+")
    collect.add_argument("--speaker-id", required=True)
    collect.add_argument("--yt-dlp", default="yt-dlp", help="yt-dlp executable")
    collect.add_argument("--max-videos", type=int,
                         help="maximum videos per channel/playlist (recommended for pilots)")
    collect.add_argument("--date-after", help="only videos uploaded on/after YYYY-MM-DD")
    transcribe = sub.add_parser("transcribe", help="create word-timestamp transcripts")
    transcribe.add_argument("manifest", type=Path)
    transcribe.add_argument("--model", default="small")
    transcribe.add_argument("--language")
    align = sub.add_parser("align", help="project word timestamps through a CMU lexicon")
    align.add_argument("manifest", type=Path)
    align.add_argument("--lexicon", type=Path, required=True)
    align.add_argument("--force", action="store_true")
    compare = sub.add_parser("compare", help="stochastically compare constant and temporal models")
    compare.add_argument("manifest", type=Path)
    compare.add_argument("--phone", required=True)
    compare.add_argument("--response", choices=["f1_hz", "f2_hz", "f0_hz", "duration_s"], default="f1_hz")
    compare.add_argument("--iterations", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        project = Project.load(args.manifest)
        print(f"valid: {len(project.recordings)} recordings, {len(project.tokens)} tokens")
    elif args.command == "extract":
        project = Project.load(args.manifest)
        run_id = datetime.now(timezone.utc).strftime("praat-%Y%m%dT%H%M%SZ")
        measured = 0
        for token in project.tokens:
            if token["f1_hz"] and not args.force:
                continue
            rec = project.recording(token["recording_id"])
            result = measure_interval(project.root / rec["local_audio_path"],
                                      float(token["original_start_s"]), float(token["original_end_s"]),
                                      max_formant_hz=args.max_formant_hz, window_s=args.window_s)
            token.update(f1_hz=_text(result["f1_hz"]), f2_hz=_text(result["f2_hz"]),
                         f0_hz=_text(result["f0_hz"]), trajectory=trajectory_json(result),
                         exclusion_reason=result["exclusion_reason"], run_id=run_id)
            if result["measurement_status"] == "rejected":
                token["review_status"] = "rejected"
            measured += 1
        token_path = project.root / project.manifest["tokens"]
        temporary = token_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(project.tokens[0]))
            writer.writeheader(); writer.writerows(project.tokens)
        temporary.replace(token_path)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        project.manifest["run"] = {"run_id": run_id, "code_revision": revision,
            "measurement_method": "Praat Burg via praat-parselmouth",
            "configuration": {"max_formant_hz": args.max_formant_hz, "window_s": args.window_s}}
        Path(args.manifest).write_text(json.dumps(project.manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{run_id}: processed {measured} token intervals")
    elif args.command == "collect-youtube":
        from .automatic import collect_youtube, write_rows
        root = args.manifest.resolve().parent
        if args.manifest.exists():
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        else:
            root.mkdir(parents=True, exist_ok=True)
            manifest = {"recordings": "recordings.csv", "tokens": "tokens.csv"}
        recording_path = root / manifest["recordings"]
        existing = _read_csv(recording_path) if recording_path.exists() else []
        additions = collect_youtube(args.urls, root, speaker_id=args.speaker_id,
                                    yt_dlp=args.yt_dlp, max_videos=args.max_videos,
                                    date_after=args.date_after)
        by_id = {row["recording_id"]: row for row in existing}
        for row in additions:
            if row["recording_id"] in by_id and row != by_id[row["recording_id"]]:
                raise ProjectError(f"recording already exists with different metadata: {row['recording_id']}")
            by_id[row["recording_id"]] = row
        from .project import RECORDING_FIELDS, TOKEN_FIELDS
        write_rows(recording_path, list(by_id.values()), RECORDING_FIELDS)
        token_path = root / manifest["tokens"]
        if not token_path.exists():
            write_rows(token_path, [], TOKEN_FIELDS)
        manifest["collection"] = {"method": "yt-dlp", "collected_at": datetime.now(timezone.utc).isoformat(),
                                  "source_urls": args.urls, "max_videos": args.max_videos,
                                  "date_after": args.date_after, "recording_ids": sorted(by_id)}
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"collected {len(additions)} recording(s)")
    elif args.command == "transcribe":
        from .automatic import transcribe_recording
        manifest, root, recordings = _unvalidated_recordings(args.manifest)
        for recording in recordings:
            output = root / "transcripts" / f"{recording['recording_id']}.json"
            transcribe_recording(root / recording["local_audio_path"], output,
                                 model_name=args.model, language=args.language)
        manifest["transcription"] = {"engine": "faster-whisper", "model": args.model,
                                     "language": args.language, "word_timestamps": True}
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"transcribed {len(recordings)} recording(s)")
    elif args.command == "align":
        from .automatic import align_words, read_lexicon, utc_run_id, write_rows
        from .project import TOKEN_FIELDS
        manifest, root, recordings = _unvalidated_recordings(args.manifest)
        token_path = root / manifest["tokens"]
        existing = _read_csv(token_path) if token_path.exists() else []
        if existing and not args.force:
            raise ProjectError("tokens table is not empty; pass --force to replace it")
        lexicon = read_lexicon(args.lexicon)
        run_id = utc_run_id("align")
        tokens = []
        for recording in recordings:
            transcript_path = root / "transcripts" / f"{recording['recording_id']}.json"
            if not transcript_path.exists():
                raise ProjectError(f"missing transcript: {transcript_path}")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            tokens.extend(align_words(recording["recording_id"], transcript, lexicon, run_id=run_id,
                                      data_origin=recording["data_origin"]))
        if not tokens:
            raise ProjectError("alignment produced no tokens; check transcript words and lexicon")
        write_rows(token_path, tokens, TOKEN_FIELDS)
        manifest["alignment"] = {"run_id": run_id, "method": "timestamped-word lexicon projection",
                                 "lexicon_sha256": hashlib.sha256(args.lexicon.read_bytes()).hexdigest()}
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        Project.load(args.manifest)
        print(f"{run_id}: aligned {len(tokens)} phone intervals")
    elif args.command == "compare":
        from .compare import compare_models
        result = compare_models(Project.load(args.manifest), phone=args.phone,
                                response=args.response, iterations=args.iterations, seed=args.seed)
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")


def _text(value):
    return "" if value is None else f"{value:.6g}"


def _unvalidated_recordings(manifest_path: Path):
    """Load collection-stage tables before phone tokens exist."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    recordings = _read_csv(root / manifest["recordings"])
    if not recordings:
        raise ProjectError("recordings table is empty")
    return manifest, root, recordings


if __name__ == "__main__":
    main()
