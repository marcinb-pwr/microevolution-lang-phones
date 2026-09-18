import csv
import json
import shutil
import subprocess

import pytest

from microevolution.automatic import (align_words, collect_youtube, import_mfa_textgrids,
                                      read_lexicon, segment_words)
from microevolution.compare import compare_models
from microevolution.project import Project, ProjectError, ReviewStore, token_revision
from test_project import make_project


def test_alignment_preserves_coordinates_and_marks_estimates(tmp_path):
    lexicon_path = tmp_path / "lexicon.txt"
    lexicon_path.write_text("BLACK B L AE1 K\n")
    tokens = align_words("r1", {"words": [{"word": " black!", "start": 1, "end": 1.4}]},
                         read_lexicon(lexicon_path), run_id="align-1")
    assert [t["phone_label"] for t in tokens] == ["B", "L", "AE", "K"]
    assert tokens[2]["stress"] == "1"
    assert tokens[0]["original_start_s"] == "1.000000"
    assert tokens[-1]["original_end_s"] == "1.400000"
    assert {t["alignment_quality"] for t in tokens} == {"lexicon_projected"}


def test_mfa_import_restores_segment_offset_and_automatic_qc(tmp_path):
    grid = tmp_path / "rec__00000.TextGrid"
    grid.write_text('''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
 class = "IntervalTier"
 name = "words"
 intervals [1]:
 xmin = 0.1
 xmax = 0.4
 text = "pawn"
item [2]:
 class = "IntervalTier"
 name = "phones"
 intervals [1]:
 xmin = 0.15
 xmax = 0.30
 text = "AO1"
''')
    rows = import_mfa_textgrids(tmp_path, [{"stem": "rec__00000", "recording_id": "rec",
        "offset_s": 10.0, "duration_s": 2.0}], {"rec": {"data_origin": "observed"}}, run_id="mfa-1")
    assert rows[0]["original_start_s"] == "10.150000"
    assert rows[0]["original_end_s"] == "10.300000"
    assert rows[0]["phone_label"] == "AO"
    assert rows[0]["alignment_qc_status"] == "accepted"
    assert rows[0]["review_status"] == "pending"


def test_segmentation_splits_long_utterances_and_retains_absolute_times():
    words = [{"word": str(i), "start": i * 2.0, "end": i * 2.0 + .5} for i in range(10)]
    segments = segment_words(words, max_s=5)
    assert len(segments) > 1
    assert segments[1]["start"] == float(segments[1]["words"][0]["start"])


def test_model_comparison_is_reproducible_and_recording_level(tmp_path):
    manifest = make_project(tmp_path, recording_date="2020-01-01")
    rec_path, tok_path = tmp_path / "recordings.csv", tmp_path / "tokens.csv"
    with rec_path.open(newline="") as f:
        recs = list(csv.DictReader(f))
    with tok_path.open(newline="") as f:
        toks = list(csv.DictReader(f))
    for i, (when, value) in enumerate((("2021-01-01", "710"), ("2022-01-01", "730")), 2):
        shutil.copy(tmp_path / "source.wav", tmp_path / f"source{i}.wav")
        recs.append(dict(recs[0], recording_id=f"r{i}", local_audio_path=f"source{i}.wav",
                         recording_date=when, session_id=f"session{i}"))
        toks.append(dict(toks[0], token_id=f"t{i}", recording_id=f"r{i}", f1_hz=value))
    for path, rows in ((rec_path, recs), (tok_path, toks)):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    project = Project.load(manifest)
    reviews = ReviewStore(tmp_path / "reviews.sqlite3")
    for token in project.tokens:
        reviews.set(token["token_id"], "accepted", revision=token_revision(project, token))
    first = compare_models(project, phone="AE", iterations=50, seed=7, reviews=reviews)
    assert first == compare_models(project, phone="AE", iterations=50, seed=7, reviews=reviews)
    assert first["n_recordings"] == 3
    assert first["slope_per_year"] > 0


def test_exploratory_comparison_can_include_pending_projected_tokens(tmp_path):
    manifest = make_project(tmp_path, recording_date="2020-01-01")
    rec_path, tok_path = tmp_path / "recordings.csv", tmp_path / "tokens.csv"
    with rec_path.open(newline="") as handle:
        recordings = list(csv.DictReader(handle))
    with tok_path.open(newline="") as handle:
        tokens = list(csv.DictReader(handle))
    tokens[0]["alignment_quality"] = "lexicon_projected"
    for index, (when, value) in enumerate((("2021-01-01", "710"),
                                           ("2022-01-01", "730")), 2):
        shutil.copy(tmp_path / "source.wav", tmp_path / f"source{index}.wav")
        recordings.append(dict(recordings[0], recording_id=f"r{index}",
                               local_audio_path=f"source{index}.wav",
                               recording_date=when, session_id=f"session{index}"))
        tokens.append(dict(tokens[0], token_id=f"t{index}", recording_id=f"r{index}",
                           f1_hz=value))
    for path, rows in ((rec_path, recordings), (tok_path, tokens)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    result = compare_models(Project.load(manifest), phone="AE", iterations=10,
                            include_pending=True, verified_only=False)

    assert result["n_recordings"] == 3
    assert result["included_review_statuses"] == ["accepted", "pending"]
    assert result["included_unverified_boundaries"] is True


def test_comparison_rejects_too_few_recordings(tmp_path):
    project = Project.load(make_project(tmp_path, recording_date="2020-01-01"))
    with pytest.raises(ProjectError, match="three dated recordings"):
        compare_models(project, phone="AE", iterations=2)


def test_comparison_reports_unknown_phone_and_available_labels(tmp_path):
    project = Project.load(make_project(tmp_path, recording_date="2020-01-01"))
    with pytest.raises(ProjectError, match=r"phone 'A01'.*available phones: AE"):
        compare_models(project, phone="A01", iterations=2)


@pytest.mark.parametrize("maximum,date_after,message", [
    (0, None, "max_videos"),
    (1, "last Tuesday", "date_after"),
])
def test_collection_rejects_unsafe_filters_before_downloading(
        tmp_path, maximum, date_after, message):
    with pytest.raises(ProjectError, match=message):
        collect_youtube(["https://youtube.example/channel"], tmp_path,
                        speaker_id="host", max_videos=maximum, date_after=date_after)


def test_collection_adds_browser_cookies_to_downloader_command(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        assert command[-3:] == ["--cookies-from-browser", "safari", "https://youtube.example/watch"]
        raise subprocess.CalledProcessError(1, command, stderr="download failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ProjectError, match="download failed"):
        collect_youtube(["https://youtube.example/watch"], tmp_path,
                        speaker_id="host", cookies_from_browser="safari")


def test_collection_explains_outdated_yt_dlp_403(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr="WARNING: version is older than 90 days\nHTTP Error 403: Forbidden")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ProjectError) as caught:
        collect_youtube(["https://youtube.example/watch"], tmp_path, speaker_id="host")
    message = str(caught.value)
    assert "pip install --upgrade yt-dlp" in message
    assert "--cookies-from-browser BROWSER" in message
