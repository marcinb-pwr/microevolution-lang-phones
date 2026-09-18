"""Project loading, validation, reviews, and portable exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import wave
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


RECORDING_FIELDS = {
    "recording_id", "speaker_id", "source_url", "local_audio_path",
    "audio_sha256", "upload_date", "recording_date", "date_basis",
    "date_uncertainty", "session_id", "recording_setup", "style",
    "sample_rate", "quality_notes", "data_origin",
}
TOKEN_FIELDS = {
    "token_id", "recording_id", "word", "phone_label", "phone_set",
    "stress", "left_phone", "right_phone", "original_start_s",
    "original_end_s", "f1_hz", "f2_hz", "duration_s", "f0_hz",
    "trajectory", "alignment_quality", "review_status",
    "exclusion_reason", "run_id", "data_origin",
    "raw_phone_label", "alignment_method", "alignment_run_id",
    "alignment_qc_status", "alignment_qc_reasons", "measurement_status",
    "measurement_input_hash", "track_coverage", "boundary_sensitivity",
}
# Added fields are optional when reading schema-v1/v2 projects. Writers always
# emit them, which provides a migration without rewriting archived study data.
LEGACY_TOKEN_FIELDS = TOKEN_FIELDS - {
    "raw_phone_label", "alignment_method", "alignment_run_id",
    "alignment_qc_status", "alignment_qc_reasons", "measurement_status",
    "measurement_input_hash", "track_coverage", "boundary_sensitivity",
}


class ProjectError(ValueError):
    """A project is unsafe or internally inconsistent."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_optional_date(value: str, field: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ProjectError(f"invalid {field}: {value!r}; expected YYYY-MM-DD") from exc


def _float(value: str, field: str, *, optional: bool = False) -> float | None:
    if optional and value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"invalid {field}: {value!r}") from exc


@dataclass
class Project:
    root: Path
    manifest: dict
    recordings: list[dict[str, str]]
    tokens: list[dict[str, str]]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "Project":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        recordings = _read_csv(root / manifest["recordings"])
        tokens = _read_csv(root / manifest["tokens"])
        project = cls(root, manifest, recordings, tokens)
        project.validate()
        return project

    def validate(self, verify_hashes: bool = True) -> None:
        if not self.recordings:
            raise ProjectError("recordings table is empty")
        if not self.tokens:
            raise ProjectError("tokens table is empty")
        for label, rows, required in (
            ("recordings", self.recordings, RECORDING_FIELDS),
            ("tokens", self.tokens, LEGACY_TOKEN_FIELDS),
        ):
            missing = required - set(rows[0])
            if missing:
                raise ProjectError(f"{label} missing fields: {', '.join(sorted(missing))}")
        recording_ids = [r["recording_id"] for r in self.recordings]
        token_ids = [t["token_id"] for t in self.tokens]
        for label, identifiers in (("recording", recording_ids), ("token", token_ids)):
            duplicates = sorted({x for x in identifiers if identifiers.count(x) > 1})
            if duplicates:
                raise ProjectError(f"duplicate {label} identifiers: {duplicates}")
        known = set(recording_ids)
        roots = {r["data_origin"] for r in self.recordings}
        if roots - {"observed", "synthetic"}:
            raise ProjectError("data_origin must be observed or synthetic")
        for recording in self.recordings:
            _parse_optional_date(recording["upload_date"], "upload_date")
            _parse_optional_date(recording["recording_date"], "recording_date")
            audio = (self.root / recording["local_audio_path"]).resolve()
            if self.root not in audio.parents:
                raise ProjectError(f"audio path escapes project: {audio}")
            if not audio.is_file():
                raise ProjectError(f"audio file does not exist: {audio}")
            if verify_hashes and recording["audio_sha256"]:
                actual = hashlib.sha256(audio.read_bytes()).hexdigest()
                if actual != recording["audio_sha256"]:
                    raise ProjectError(f"audio hash mismatch: {recording['recording_id']}")
        origin_by_recording = {r["recording_id"]: r["data_origin"] for r in self.recordings}
        duration_by_recording = {}
        for recording in self.recordings:
            with wave.open(str(self.root / recording["local_audio_path"]), "rb") as wav:
                if int(recording["sample_rate"]) != wav.getframerate():
                    raise ProjectError(f"sample rate mismatch: {recording['recording_id']}")
                duration_by_recording[recording["recording_id"]] = wav.getnframes() / wav.getframerate()
        for token in self.tokens:
            for field in TOKEN_FIELDS:
                token.setdefault(field, "")
            if token["recording_id"] not in known:
                raise ProjectError(f"token {token['token_id']} references absent recording")
            start = _float(token["original_start_s"], "original_start_s")
            end = _float(token["original_end_s"], "original_end_s")
            if start < 0 or end <= start:
                raise ProjectError(f"token {token['token_id']} has reversed/invalid interval")
            if end > duration_by_recording[token["recording_id"]]:
                raise ProjectError(f"token {token['token_id']} exceeds original audio duration")
            _float(token["f1_hz"], "f1_hz", optional=True)
            _float(token["f2_hz"], "f2_hz", optional=True)
            if token["data_origin"] != origin_by_recording[token["recording_id"]]:
                raise ProjectError(f"token {token['token_id']} mixes data origins")

    def recording(self, recording_id: str) -> dict[str, str]:
        return next(r for r in self.recordings if r["recording_id"] == recording_id)

    def effective_date(self, recording: dict[str, str]) -> date | None:
        """Return a real recording date, or an explicitly labelled upload proxy."""
        return _parse_optional_date(
            recording["recording_date"] or recording["upload_date"], "effective date"
        )

    def context_wav(self, token: dict[str, str], padding_s: float = 0.5) -> bytes:
        """Extract context from the original WAV without changing its time mapping."""
        recording = self.recording(token["recording_id"])
        path = self.root / recording["local_audio_path"]
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            first = max(0, round((float(token["original_start_s"]) - padding_s) * rate))
            last = min(source.getnframes(), round((float(token["original_end_s"]) + padding_s) * rate))
            source.setpos(first)
            frames = source.readframes(last - first)
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setparams(source.getparams())
                target.setnframes(last - first)
                target.writeframes(frames)
        return output.getvalue()


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reviews (
                token_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS versioned_reviews (
                token_id TEXT NOT NULL, token_revision TEXT NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (token_id, token_revision)
            )""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def set(self, token_id: str, status: str, reason: str = "", *, revision: str = "") -> None:
        if status not in {"pending", "accepted", "rejected"}:
            raise ProjectError(f"invalid review status: {status}")
        with self._connect() as db:
            if revision:
                db.execute("""INSERT INTO versioned_reviews(token_id, token_revision, status, reason)
                    VALUES (?, ?, ?, ?) ON CONFLICT(token_id, token_revision) DO UPDATE SET
                    status=excluded.status, reason=excluded.reason,
                    updated_at=CURRENT_TIMESTAMP""", (token_id, revision, status, reason))
                return
            db.execute("""INSERT INTO reviews(token_id, status, reason) VALUES (?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET status=excluded.status,
                reason=excluded.reason, updated_at=CURRENT_TIMESTAMP""", (token_id, status, reason))

    def all(self, revisions: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
        with self._connect() as db:
            if revisions is None:
                rows = db.execute("SELECT token_id, status, reason, updated_at FROM reviews").fetchall()
            else:
                rows = db.execute("SELECT token_id, status, reason, updated_at, token_revision "
                                  "FROM versioned_reviews").fetchall()
                rows = [row[:4] for row in rows if revisions.get(row[0]) == row[4]]
        return {r[0]: {"review_status": r[1], "exclusion_reason": r[2], "reviewed_at": r[3]} for r in rows}


def token_revision(project: Project, token: dict[str, str]) -> str:
    """Identify the exact audio, interval, alignment and measurement reviewed."""
    recording = project.recording(token["recording_id"])
    identity = {"audio_sha256": recording["audio_sha256"],
                "start": token["original_start_s"], "end": token["original_end_s"],
                "alignment_quality": token["alignment_quality"],
                "alignment_run_id": token.get("alignment_run_id", ""),
                "measurement_input_hash": token.get("measurement_input_hash", ""),
                "run_id": token["run_id"]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def merged_tokens(project: Project, reviews: ReviewStore) -> list[dict[str, str]]:
    revisions = {t["token_id"]: token_revision(project, t) for t in project.tokens}
    overrides = reviews.all(revisions)
    return [{**token, "token_revision": revisions[token["token_id"]],
             **overrides.get(token["token_id"], {})} for token in project.tokens]


def select_tokens(project: Project, reviews: ReviewStore, *, speaker_id: str,
                  phone: str | None = None, data_origin: str = "observed",
                  statuses: tuple[str, ...] = ("accepted",),
                  verified_only: bool = True, automatic_qc: bool = False) -> list[dict[str, str]]:
    """Apply the same research-safe selection used by analysis and exports."""
    recordings = {r["recording_id"]: r for r in project.recordings}
    selected = []
    for token in merged_tokens(project, reviews):
        recording = recordings[token["recording_id"]]
        verified = token["alignment_quality"].lower() in {"manual", "verified", "manually_verified"}
        auto_eligible = (automatic_qc and token.get("alignment_qc_status") == "accepted"
                         and token.get("measurement_status", "measured") == "measured")
        if (recording["speaker_id"] == speaker_id and token["data_origin"] == data_origin
                and token["review_status"] in statuses and (phone is None or token["phone_label"] == phone)
                and (verified or auto_eligible or not verified_only)):
            selected.append(token)
    return selected


def export_zip(project: Project, reviews: ReviewStore, selected_ids: Iterable[str]) -> bytes:
    """Export exactly the displayed selection with provenance and run identity."""
    wanted = set(selected_ids)
    tokens = [t for t in merged_tokens(project, reviews) if t["token_id"] in wanted]
    recording_ids = {t["recording_id"] for t in tokens}
    recordings = [r for r in project.recordings if r["recording_id"] in recording_ids]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rows, base_fields in (("tokens.csv", tokens, sorted(TOKEN_FIELDS)),
                                        ("recordings.csv", recordings, sorted(RECORDING_FIELDS))):
            buffer = io.StringIO()
            extra_fields = [field for row in rows for field in row if field not in base_fields]
            fields = base_fields + list(dict.fromkeys(extra_fields))
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            archive.writestr(name, buffer.getvalue())
        archive.writestr("run.json", json.dumps(project.manifest.get("run", {}), indent=2))
        archive.writestr("selection.json", json.dumps({"token_ids": sorted(wanted)}, indent=2))
    return output.getvalue()
