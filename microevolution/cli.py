"""Command line validation for portable local projects."""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .measure import measure_interval, trajectory_json
from .project import Project


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


def _text(value):
    return "" if value is None else f"{value:.6g}"


if __name__ == "__main__":
    main()
