import csv
import json
import sys
from types import SimpleNamespace

from microevolution import cli
from test_project import make_project


def test_extract_loads_recording_once_for_multiple_tokens(tmp_path, monkeypatch):
    manifest = make_project(tmp_path)
    token_path = tmp_path / "tokens.csv"
    with token_path.open(newline="") as handle:
        tokens = list(csv.DictReader(handle))
    tokens.append(dict(tokens[0], token_id="t2", original_start_s="0.3",
                       original_end_s="0.4"))
    with token_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tokens[0])
        writer.writeheader()
        writer.writerows(tokens)

    loaded = []
    fake_sound = object()
    monkeypatch.setitem(sys.modules, "parselmouth", SimpleNamespace(
        Sound=lambda path: loaded.append(path) or fake_sound,
        __version__="test",
    ))
    monkeypatch.setattr(cli, "measure_sound_interval", lambda sound, start, end, **kwargs: {
        "f1_hz": 700, "f2_hz": 1700, "f0_hz": 120, "trajectory": [],
        "measurement_status": "measured", "exclusion_reason": "",
    })

    cli.main(["extract", str(manifest), "--force"])

    assert len(loaded) == 1
    assert loaded[0].endswith("source.wav")
    assert len(list(csv.DictReader(token_path.open(newline="")))) == 2
    assert json.loads(manifest.read_text())["run"]["tool_versions"]["praat_parselmouth"] == "test"
