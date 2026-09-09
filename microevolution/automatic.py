"""Reproducible, opt-in adapters for collecting and annotating public audio.

Network access and speech models are deliberately kept behind commands.  The
functions which turn metadata and word timestamps into project rows are pure so
that an automated run can be audited and tested without downloading anything.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .project import RECORDING_FIELDS, TOKEN_FIELDS, ProjectError


def utc_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%SZ")


def collect_youtube(urls: Iterable[str], root: Path, *, speaker_id: str,
                    yt_dlp: str = "yt-dlp") -> list[dict[str, str]]:
    """Download URLs as mono WAV files and return provenance-complete rows.

    yt-dlp's printed JSON is used rather than scraping YouTube pages. Existing
    files are reused by yt-dlp, making an interrupted collection resumable.
    """
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for url in urls:
        command = [yt_dlp, "--no-playlist", "--print-json", "-x", "--audio-format", "wav",
                   "--audio-quality", "0", "-o", str(audio_dir / "%(id)s.%(ext)s"), url]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise ProjectError("yt-dlp is required for collection; install the 'automatic' extra") from exc
        except subprocess.CalledProcessError as exc:
            raise ProjectError(f"yt-dlp failed for {url}: {exc.stderr.strip()}") from exc
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
        if not lines:
            raise ProjectError(f"yt-dlp returned no metadata for {url}")
        info = json.loads(lines[-1])
        video_id = str(info["id"])
        audio = audio_dir / f"{video_id}.wav"
        if not audio.is_file():
            raise ProjectError(f"yt-dlp did not create expected audio: {audio}")
        with wave.open(str(audio), "rb") as wav:
            rate = wav.getframerate()
        upload = str(info.get("upload_date") or "")
        upload = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else ""
        row = {field: "" for field in RECORDING_FIELDS}
        row.update(recording_id=f"yt-{video_id}", speaker_id=speaker_id,
                   source_url=str(info.get("webpage_url") or url),
                   local_audio_path=audio.relative_to(root).as_posix(),
                   audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                   upload_date=upload, date_basis="upload_date_proxy",
                   session_id=f"youtube-{video_id}", recording_setup="unknown",
                   style="online_video", sample_rate=str(rate),
                   quality_notes="automatically downloaded; recording date unknown",
                   data_origin="observed")
        rows.append(row)
    return rows


def transcribe_recording(audio: Path, output: Path, *, model_name: str = "small",
                         language: str | None = None) -> dict:
    """Transcribe one recording with faster-whisper word timestamps."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ProjectError("faster-whisper is required for transcription; install the 'automatic' extra") from exc
    model = WhisperModel(model_name)
    segments, info = model.transcribe(str(audio), language=language, word_timestamps=True)
    words = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is not None and word.end is not None:
                words.append({"word": word.word.strip(), "start": word.start, "end": word.end,
                              "probability": word.probability})
    document = {"engine": "faster-whisper", "model": model_name,
                "language": info.language, "words": words}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def read_lexicon(path: Path) -> dict[str, list[str]]:
    """Read a CMU-style lexicon (WORD PHONE1 PHONE2 ...)."""
    entries = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ProjectError(f"invalid lexicon line {number}")
        entries.setdefault(parts[0].upper().split("(", 1)[0], parts[1:])
    return entries


def align_words(recording_id: str, transcript: dict, lexicon: dict[str, list[str]],
                *, run_id: str, data_origin: str = "observed") -> list[dict[str, str]]:
    """Project timestamped words to phones using a pronunciation lexicon.

    Boundaries are proportional estimates, not acoustic forced alignment, and
    are labelled accordingly so they cannot be mistaken for manual boundaries.
    """
    tokens = []
    for word_index, item in enumerate(transcript.get("words", [])):
        spelling = "".join(c for c in str(item["word"]).upper() if c.isalpha() or c == "'")
        phones = lexicon.get(spelling)
        start, end = float(item["start"]), float(item["end"])
        if not phones or start < 0 or end <= start:
            continue
        width = (end - start) / len(phones)
        for phone_index, raw_phone in enumerate(phones):
            phone = raw_phone.rstrip("012")
            stress = raw_phone[len(phone):]
            left = phones[phone_index - 1].rstrip("012") if phone_index else ""
            right = phones[phone_index + 1].rstrip("012") if phone_index + 1 < len(phones) else ""
            row = {field: "" for field in TOKEN_FIELDS}
            a, b = start + phone_index * width, start + (phone_index + 1) * width
            row.update(token_id=f"{recording_id}-w{word_index:06d}-p{phone_index:02d}",
                       recording_id=recording_id, word=spelling, phone_label=phone,
                       phone_set="ARPAbet", stress=stress, left_phone=left, right_phone=right,
                       original_start_s=f"{a:.6f}", original_end_s=f"{b:.6f}",
                       duration_s=f"{b-a:.6f}", alignment_quality="lexicon_projected",
                       review_status="pending", run_id=run_id, data_origin=data_origin)
            tokens.append(row)
    return tokens


def write_rows(path: Path, rows: list[dict[str, str]], fields: set[str]) -> None:
    ordered = sorted(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
