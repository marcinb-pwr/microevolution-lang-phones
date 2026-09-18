import csv
import hashlib
import json
import wave

import pytest

from microevolution.project import (Project, ProjectError, ReviewStore, export_zip,
                                    merged_tokens, select_tokens, token_revision)


def make_project(tmp_path, *, recording_date="", start="0.1", end="0.2"):
    audio = tmp_path / "source.wav"
    with wave.open(str(audio), "wb") as w:
        w.setparams((1, 2, 8000, 0, "NONE", "not compressed")); w.writeframes(b"\0\0" * 8000)
    rec = {"recording_id":"r1", "speaker_id":"s", "source_url":"", "local_audio_path":"source.wav",
           "audio_sha256":hashlib.sha256(audio.read_bytes()).hexdigest(), "upload_date":"",
           "recording_date":recording_date, "date_basis":"unknown", "date_uncertainty":"",
           "session_id":"session1", "recording_setup":"", "style":"", "sample_rate":"8000",
           "quality_notes":"", "data_origin":"observed"}
    tok = {"token_id":"t1", "recording_id":"r1", "word":"black", "phone_label":"AE",
           "phone_set":"ARPAbet", "stress":"1", "left_phone":"L", "right_phone":"K",
           "original_start_s":start, "original_end_s":end, "f1_hz":"700", "f2_hz":"1700",
           "duration_s":"0.1", "f0_hz":"120", "trajectory":"[]", "alignment_quality":"manual",
           "review_status":"pending", "exclusion_reason":"", "run_id":"run1", "data_origin":"observed"}
    for name, row in (("recordings.csv", rec), ("tokens.csv", tok)):
        with (tmp_path / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row); writer.writeheader(); writer.writerow(row)
    (tmp_path / "project.json").write_text(json.dumps({"recordings":"recordings.csv", "tokens":"tokens.csv", "run":{"run_id":"run1"}}))
    return tmp_path / "project.json"


def test_missing_date_remains_none_and_context_is_original_audio(tmp_path):
    project = Project.load(make_project(tmp_path))
    assert project.effective_date(project.recordings[0]) is None
    with wave.open(__import__("io").BytesIO(project.context_wav(project.tokens[0], 0.05))) as excerpt:
        assert excerpt.getframerate() == 8000
        assert excerpt.getnframes() == pytest.approx(1600, abs=1)


def test_review_survives_reopening_and_export_keeps_ids(tmp_path):
    project = Project.load(make_project(tmp_path))
    ReviewStore(tmp_path / "reviews.db").set(
        "t1", "accepted", "checked", revision=token_revision(project, project.tokens[0]))
    reopened = ReviewStore(tmp_path / "reviews.db")
    assert merged_tokens(project, reopened)[0]["review_status"] == "accepted"
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(export_zip(project, reopened, ["t1"]))) as exported:
        assert "t1" in exported.read("tokens.csv").decode()
        assert json.loads(exported.read("run.json"))["run_id"] == "run1"


def test_export_has_stable_schema_for_partially_reviewed_selection(tmp_path):
    project = Project.load(make_project(tmp_path))
    second = dict(project.tokens[0], token_id="t2", original_start_s="0.3", original_end_s="0.4")
    project.tokens.append(second)
    reviews = ReviewStore(tmp_path / "reviews.db")
    reviews.set("t1", "accepted", revision=token_revision(project, project.tokens[0]))
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(export_zip(project, reviews, ["t1", "t2"]))) as exported:
        rows = list(csv.DictReader(io.StringIO(exported.read("tokens.csv").decode())))
    assert len(rows) == 2
    assert {"reviewed_at", "token_revision"} <= rows[0].keys()
    assert rows[1]["reviewed_at"] == ""


def test_review_is_invalidated_when_interval_changes(tmp_path):
    project = Project.load(make_project(tmp_path))
    reviews = ReviewStore(tmp_path / "reviews.db")
    reviews.set("t1", "accepted", revision=token_revision(project, project.tokens[0]))
    project.tokens[0]["original_end_s"] = "0.25"
    assert merged_tokens(project, reviews)[0]["review_status"] == "pending"


def test_automatic_qc_is_separate_from_human_review_but_explicitly_eligible(tmp_path):
    project = Project.load(make_project(tmp_path))
    token = project.tokens[0]
    token.update(alignment_quality="mfa_acoustic", alignment_qc_status="accepted",
                 measurement_status="measured", review_status="pending")
    selected = select_tokens(project, ReviewStore(tmp_path / "reviews.db"), speaker_id="s",
                             statuses=("accepted", "pending"), automatic_qc=True)
    assert [row["token_id"] for row in selected] == ["t1"]
    assert selected[0]["review_status"] == "pending"
    assert not select_tokens(project, ReviewStore(tmp_path / "reviews2.db"), speaker_id="s")


@pytest.mark.parametrize("start,end", [("0.3", "0.2"), ("-1", "0.2")])
def test_invalid_intervals_rejected(tmp_path, start, end):
    with pytest.raises(ProjectError, match="interval"):
        Project.load(make_project(tmp_path, start=start, end=end))
