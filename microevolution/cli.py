"""Command line validation for portable local projects."""

import argparse
import csv
import hashlib
import json
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path

from .measure import measure_sound_interval, trajectory_json
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
    collect.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER[+KEYRING][:PROFILE][::CONTAINER]",
        help="let yt-dlp use an existing browser login (use only when authorized)",
    )
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
    compare.add_argument("--speaker-id", help="required for projects with multiple speakers")
    compare.add_argument("--reviews", type=Path, help="review database (default: reviews.sqlite3 beside manifest)")
    compare.add_argument("--include-unverified", action="store_true",
                         help="include automatic/unverified phone boundaries")
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        project = Project.load(args.manifest)
        print(f"valid: {len(project.recordings)} recordings, {len(project.tokens)} tokens")
    elif args.command == "extract":
        project = Project.load(args.manifest)
        run_id = datetime.now(timezone.utc).strftime("praat-%Y%m%dT%H%M%S%fZ")
        configuration = {"max_formant_hz": args.max_formant_hz, "window_s": args.window_s}
        previous = project.manifest.get("run", {})
        same_configuration = previous.get("configuration") == configuration
        previous_inputs = previous.get("measurement_inputs", {})
        measurement_inputs = {}
        measured = 0
        pending_by_recording = {}
        for token in project.tokens:
            rec = project.recording(token["recording_id"])
            measurement_input = hashlib.sha256(json.dumps({
                "audio_sha256": rec["audio_sha256"], "start": token["original_start_s"],
                "end": token["original_end_s"], "configuration": configuration,
            }, sort_keys=True).encode()).hexdigest()
            measurement_inputs[token["token_id"]] = measurement_input
            if (token["f1_hz"] and same_configuration
                    and previous_inputs.get(token["token_id"]) == measurement_input and not args.force):
                continue
            pending_by_recording.setdefault(token["recording_id"], []).append(token)
        import parselmouth
        for recording_id, tokens in pending_by_recording.items():
            rec = project.recording(recording_id)
            try:
                sound = parselmouth.Sound(str(project.root / rec["local_audio_path"]))
            except Exception as exc:
                sound = None
                load_error = exc
            for token in tokens:
                try:
                    if sound is None:
                        raise load_error
                    result = measure_sound_interval(
                        sound, float(token["original_start_s"]), float(token["original_end_s"]),
                        max_formant_hz=args.max_formant_hz, window_s=args.window_s)
                except Exception as exc:
                    result = {"f1_hz": None, "f2_hz": None, "f0_hz": None, "trajectory": [],
                              "measurement_status": "rejected",
                              "exclusion_reason": f"measurement_error:{type(exc).__name__}"}
                token.update(f1_hz=_text(result["f1_hz"]), f2_hz=_text(result["f2_hz"]),
                             f0_hz=_text(result["f0_hz"]), trajectory=trajectory_json(result),
                             exclusion_reason=result["exclusion_reason"], run_id=run_id)
                if result["measurement_status"] == "rejected":
                    token["review_status"] = "rejected"
                else:
                    token["review_status"] = "pending"
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
        try:
            import parselmouth
            parselmouth_version = parselmouth.__version__
        except (ImportError, AttributeError):
            parselmouth_version = "unknown"
        project.manifest["schema_version"] = 2
        project.manifest["run"] = {"run_id": run_id, "code_revision": revision,
            "measurement_method": "Praat Burg via praat-parselmouth",
            "configuration": configuration,
            "tool_versions": {"python": platform.python_version(),
                              "praat_parselmouth": parselmouth_version},
            "input_hashes": {r["recording_id"]: r["audio_sha256"] for r in project.recordings},
            "measurement_inputs": measurement_inputs}
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
                                    date_after=args.date_after,
                                    cookies_from_browser=args.cookies_from_browser)
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
                                  "date_after": args.date_after,
                                  "browser_cookies_used": bool(args.cookies_from_browser),
                                  "recording_ids": sorted(by_id)}
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
        from .project import ReviewStore
        project = Project.load(args.manifest)
        review_path = args.reviews or project.root / "reviews.sqlite3"
        result = compare_models(project, phone=args.phone, response=args.response,
                                iterations=args.iterations, seed=args.seed,
                                reviews=ReviewStore(review_path), speaker_id=args.speaker_id,
                                verified_only=not args.include_unverified)
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
