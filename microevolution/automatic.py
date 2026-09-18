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
import re
import shutil
import wave
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .project import RECORDING_FIELDS, TOKEN_FIELDS, ProjectError


def utc_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%SZ")


def collect_youtube(urls: Iterable[str], root: Path, *, speaker_id: str,
                    yt_dlp: str = "yt-dlp", max_videos: int | None = None,
                    date_after: str | None = None,
                    cookies_from_browser: str | None = None) -> list[dict[str, str]]:
    """Download video or channel URLs and return provenance-complete rows.

    yt-dlp's printed JSON is used rather than scraping YouTube pages. Existing
    files are reused by yt-dlp, making an interrupted collection resumable. A
    channel URL is intentionally treated as a playlist; use ``max_videos`` while
    piloting to avoid accidentally downloading an entire channel.
    """
    if max_videos is not None and max_videos < 1:
        raise ProjectError("max_videos must be positive")
    if date_after:
        try:
            date.fromisoformat(date_after)
        except ValueError as exc:
            raise ProjectError("date_after must use YYYY-MM-DD") from exc
    if cookies_from_browser is not None and not cookies_from_browser.strip():
        raise ProjectError("cookies_from_browser must name a browser")
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for url in urls:
        command = [yt_dlp, "--yes-playlist", "--print-json", "-x", "--audio-format", "wav",
                   "--audio-quality", "0", "-o", str(audio_dir / "%(id)s.%(ext)s")]
        if max_videos is not None:
            command.extend(["--playlist-end", str(max_videos)])
        if date_after:
            command.extend(["--dateafter", date_after.replace("-", "")])
        if cookies_from_browser:
            command.extend(["--cookies-from-browser", cookies_from_browser])
        command.append(url)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise ProjectError("yt-dlp is required for collection; install the 'automatic' extra") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            hints = []
            lowered = stderr.lower()
            if "older than" in lowered or "update" in lowered:
                hints.append("update yt-dlp (python -m pip install --upgrade yt-dlp)")
            if "403" in lowered or "sign in" in lowered or "not a bot" in lowered:
                hints.append("retry with --cookies-from-browser BROWSER (for example, chrome or safari)")
            advice = f"\nSuggested action: {'; then '.join(hints)}." if hints else ""
            detail = stderr or f"process exited with status {exc.returncode}"
            raise ProjectError(f"yt-dlp failed for {url}: {detail}{advice}") from exc
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
        if not lines:
            raise ProjectError(f"yt-dlp returned no metadata for {url}")
        for line in lines:
            info = json.loads(line)
            video_id = str(info["id"])
            audio = audio_dir / f"{video_id}.wav"
            if not audio.is_file():
                raise ProjectError(f"yt-dlp did not create expected audio: {audio}")
            with wave.open(str(audio), "rb") as wav:
                rate = wav.getframerate()
            upload = str(info.get("upload_date") or "")
            upload = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else ""
            channel = str(info.get("channel") or info.get("uploader") or "unknown")
            channel_id = str(info.get("channel_id") or info.get("uploader_id") or "unknown")
            row = {field: "" for field in RECORDING_FIELDS}
            row.update(recording_id=f"yt-{video_id}", speaker_id=speaker_id,
                       source_url=str(info.get("webpage_url") or url),
                       local_audio_path=audio.relative_to(root).as_posix(),
                       audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                       upload_date=upload, date_basis="upload_date_proxy",
                       session_id=f"youtube-{video_id}", recording_setup="unknown",
                       style="online_video", sample_rate=str(rate),
                       quality_notes=(f"automatically downloaded from channel {channel} "
                                      f"({channel_id}); recording date unknown"),
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


def segment_words(words: list[dict], *, min_s: float = 2.0, max_s: float = 15.0,
                  pause_s: float = 0.6) -> list[dict]:
    """Make bounded utterances without changing the words' recording coordinates."""
    valid = [w for w in words if float(w["end"]) > float(w["start"]) >= 0]
    segments, current = [], []
    for word in valid:
        if current:
            span = float(word["end"]) - float(current[0]["start"])
            gap = float(word["start"]) - float(current[-1]["end"])
            if span > max_s or (gap >= pause_s and
                                float(current[-1]["end"]) - float(current[0]["start"]) >= min_s):
                segments.append(_segment(current)); current = []
        current.append(word)
    if current:
        segments.append(_segment(current))
    return segments


def _segment(words):
    return {"start": float(words[0]["start"]), "end": float(words[-1]["end"]),
            "text": " ".join(str(w["word"]).strip() for w in words), "words": words}


def export_mfa_corpus(audio: Path, transcript: dict, corpus: Path, *, recording_id: str,
                      speaker_id: str, context_s: float = 0.05) -> list[dict]:
    """Write segment WAV/LAB pairs and return their exact source offsets."""
    import wave
    segments = segment_words(transcript.get("words", []))
    speaker = re.sub(r"[^A-Za-z0-9_-]", "_", speaker_id) or "speaker"
    target = corpus / speaker; target.mkdir(parents=True, exist_ok=True)
    mapping = []
    with wave.open(str(audio), "rb") as source:
        rate, total = source.getframerate(), source.getnframes()
        for index, segment in enumerate(segments):
            first = max(0, int(round((segment["start"] - context_s) * rate)))
            last = min(total, int(round((segment["end"] + context_s) * rate)))
            source.setpos(first); frames = source.readframes(last - first)
            stem = f"{recording_id}__{index:05d}"
            wav_path = target / f"{stem}.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setparams(source.getparams()); output.writeframes(frames)
            (target / f"{stem}.lab").write_text(segment["text"] + "\n", encoding="utf-8")
            mapping.append({"stem": stem, "recording_id": recording_id,
                            "offset_s": first / rate, "duration_s": (last-first) / rate})
    return mapping


def _textgrid_tiers(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Parse MFA's long TextGrid format without adding a runtime dependency."""
    tiers, name = {}, None
    xmin = xmax = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"'); tiers.setdefault(name, [])
        elif name and line.startswith("xmin ="):
            xmin = float(line.split("=", 1)[1])
        elif name and line.startswith("xmax ="):
            xmax = float(line.split("=", 1)[1])
        elif name and line.startswith("text =") and xmin is not None and xmax is not None:
            text = line.split("=", 1)[1].strip().strip('"').replace('""', '"')
            tiers[name].append((xmin, xmax, text)); xmin = xmax = None
    return tiers


def import_mfa_textgrids(aligned: Path, mappings: list[dict], recordings: dict[str, dict],
                         *, run_id: str) -> list[dict[str, str]]:
    """Import ARPA phone tiers and restore original-recording coordinates."""
    by_stem = {m["stem"]: m for m in mappings}; tokens = []
    for grid in aligned.rglob("*.TextGrid"):
        mapping = by_stem.get(grid.stem)
        if not mapping:
            continue
        tiers = _textgrid_tiers(grid)
        phones = next((v for k, v in tiers.items() if k.lower() in {"phones", "phone"}), [])
        words = next((v for k, v in tiers.items() if k.lower() in {"words", "word"}), [])
        for index, (local_start, local_end, raw_phone) in enumerate(phones):
            if not raw_phone or raw_phone.lower() in {"sil", "sp", "spn", "<eps>"}:
                continue
            start, end = mapping["offset_s"] + local_start, mapping["offset_s"] + local_end
            word = next((label for a, b, label in words if label and a < local_end and b > local_start), "")
            canonical = raw_phone.rstrip("012").upper(); stress = raw_phone[len(canonical):]
            reasons = qc_phone_interval(start, end, mapping["duration_s"] + mapping["offset_s"])
            row = {field: "" for field in TOKEN_FIELDS}
            row.update(token_id=f'{mapping["stem"]}-p{index:04d}', recording_id=mapping["recording_id"],
                       word=word.upper(), raw_phone_label=raw_phone, phone_label=canonical,
                       phone_set="ARPAbet", stress=stress, original_start_s=f"{start:.6f}",
                       original_end_s=f"{end:.6f}", duration_s=f"{end-start:.6f}",
                       alignment_quality="mfa_acoustic", alignment_method="mfa",
                       alignment_run_id=run_id, alignment_qc_status="excluded" if reasons else "accepted",
                       alignment_qc_reasons=";".join(reasons), review_status="pending", run_id=run_id,
                       data_origin=recordings[mapping["recording_id"]]["data_origin"])
            tokens.append(row)
    return tokens


def qc_phone_interval(start: float, end: float, recording_end: float) -> list[str]:
    reasons = []
    if start < 0 or end <= start or end > recording_end + 1e-6: reasons.append("invalid_bounds")
    duration = end - start
    if duration < .015: reasons.append("collapsed_phone")
    if duration > .5: reasons.append("stretched_phone")
    return reasons


def run_mfa(corpus: Path, dictionary: str, acoustic_model: str, aligned: Path, *,
            executable: str = "mfa", fine_tune: bool = False, retries: int = 1) -> None:
    """Run pinned external MFA, retrying only a bounded number of failed runs."""
    command = [executable, "align", str(corpus), dictionary, acoustic_model, str(aligned),
               "--clean", "--output_format", "long_textgrid"]
    if fine_tune: command.append("--fine_tune")
    errors = []
    for _ in range(retries + 1):
        try:
            subprocess.run(command, check=True, text=True, capture_output=True); return
        except FileNotFoundError as exc:
            raise ProjectError("MFA executable not found; install and download the pinned models") from exc
        except subprocess.CalledProcessError as exc:
            errors.append((exc.stderr or exc.stdout or str(exc)).strip())
            shutil.rmtree(aligned, ignore_errors=True); aligned.mkdir(parents=True, exist_ok=True)
    raise ProjectError(f"MFA failed after {retries + 1} attempt(s): {errors[-1]}")
