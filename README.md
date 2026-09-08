# Phoneme microevolution: provenance-first pilot

This restart provides a small vertical slice for **local WAV recordings and manually
checked phone intervals**. It validates portable project tables, measures formants
with Praat's Burg analysis, keeps every token in original-recording coordinates,
and presents timelines, conventional vowel space, original-audio playback,
persistent review, and reproducible exports.

The older PocketSphinx and exploratory embedding pipeline remains under `scripts/`
for historical reference. Its peak-picking `Formants` output must be described as
spectral peaks, not as validated F1/F2 measurements.

## Quick start

Python 3.10 or newer is required. The historic Python 3.6 `Pipfile` is retained only
as a record of the earlier environment.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python demo/generate_audio.py
microevolution validate demo/project.json
streamlit run app.py
```

The default project is a three-period, reproducibly generated **synthetic interface demonstration**.
It contains no human speech and its illustrative F1/F2 values are not empirical
results. Its rows are marked `data_origin=synthetic`; observed and synthetic data
must be separate projects and the viewer requires an explicit origin selection.
Generated WAVs are ignored by Git and verified against hashes in the checked-in
manifest, avoiding opaque binary diffs while keeping the demonstration repeatable.

## Make an observed project

Copy `demo/project.json`, `recordings.csv`, and `tokens.csv` as templates. Put WAV
files below the project directory and add their SHA-256 hashes. Token intervals use
seconds in the original WAV—not in a concatenated excerpt. Leave unknown dates and
unknown acoustic values empty. `recording_date` takes priority; `upload_date` is
shown as an explicit proxy. If both are empty, the token remains visible in review
but never enters the timeline.

Validate before extraction:

```bash
microevolution validate path/to/project.json
microevolution extract path/to/project.json --max-formant-hz 5500 --window-s 0.025
```

Extraction records a run identifier, code revision, method, and settings. It stores
five relative-position track samples and the median of the central three. Intervals
shorter than 40 ms, silence/near-silence, and intervals without trackable F1/F2 are
explicitly rejected with missing acoustic values. Use `--force` only when deliberately
starting a new measurement run.

## Calibration before interpretation

The default ceiling is not universally appropriate. Select a stratified calibration
subset spanning periods and recording conditions, measure it manually in Praat with
the same Burg window/ceiling, and compare central F1/F2 plus full tracks. Record the
comparison and selected configuration, then freeze it before the main extraction.
Automated tests verify data integrity and audio coordinate behavior; they are not a
claim of acoustic agreement with an external Praat calibration set.

Review choices are stored in `reviews.sqlite3` beside the manifest and survive app
restarts. Exports contain exactly the current filtered token IDs, the referenced
recordings, review overrides, selection metadata, and the run record. Audio itself
is intentionally not duplicated into exports; its path and pinned hash remain in
the recording table.

## Scientific interpretation

The views are descriptive. A plotted point is one token; the interface separately
reports independent recordings. Token spread is not uncertainty in a mean, and
neither an embedding nor a changing centroid alone establishes phonetic change.
Recording/session clustering, lexical context, recording setup, and exclusion rates
must be addressed by later inferential work.
