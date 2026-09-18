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

Extraction records a schema version, run identifier, code revision, tool versions,
settings, audio hashes, and per-token input fingerprints. A changed interval,
audio file, or configuration is remeasured rather than silently retaining stale
values. One failed interval does not abort the batch, and unavailable F0 does not
discard usable F1/F2. It stores
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

Review choices are stored in `reviews.sqlite3` beside the manifest, are tied to the
specific audio/interval/measurement revision, and survive app restarts. Changing a
boundary or remeasuring therefore returns that token to pending review. Exports
contain exactly the current filtered token IDs, the referenced
recordings, review overrides, selection metadata, and the run record. Audio itself
is intentionally not duplicated into exports; its path and pinned hash remain in
the recording table.

## Scientific interpretation

The views are descriptive. A plotted point is one token; the interface separately
reports independent recordings. Token spread is not uncertainty in a mean, and
neither an embedding nor a changing centroid alone establishes phonetic change.
Recording/session clustering, lexical context, recording setup, and exclusion rates
must be addressed by later inferential work.

## Automated YouTube-to-comparison workflow

Automation is optional because downloading and speech recognition have substantial
system/model dependencies. Install it with `pip install -e '.[automatic]'`; `ffmpeg`
must also be available to `yt-dlp`. Only collect material that you are permitted to
download and retain. YouTube upload dates are stored as proxies, never represented
as recording dates.

```bash
microevolution collect-youtube study/project.json --speaker-id speaker-a \
  'https://www.youtube.com/watch?v=VIDEO_ID'
microevolution transcribe study/project.json --model small --language en
microevolution align study/project.json --lexicon cmudict.txt
microevolution extract study/project.json
microevolution compare study/project.json --phone AE --response f1_hz \
  --iterations 5000 --seed 2024 --include-pending --include-unverified \
  --output study/AE-f1-models.json
```

YouTube changes its download requirements frequently. If collection reports that
`yt-dlp` is out of date, update the environment that provides the executable:

```bash
python -m pip install --upgrade yt-dlp
```

If YouTube still responds with HTTP 403 or requires sign-in, an authorized local
browser session can be supplied without exporting a cookie file:

```bash
microevolution collect-youtube study/project.json --speaker-id speaker-a \
  --cookies-from-browser safari \
  'https://www.youtube.com/watch?v=VIDEO_ID'
```

Chrome and Firefox are also accepted by `yt-dlp`. Close the browser first if its
cookie database is locked. The project records only that browser cookies were
used, not the browser/profile name or any cookie values. Never commit cookies.

Collection pins the source URL, audio hash, sample rate, and downloader metadata.
Transcription preserves word timestamps in per-recording JSON sidecars. Alignment
expands those words through a CMU-style pronunciation lexicon. Its within-word
phone boundaries are proportional estimates marked `lexicon_projected`, **not**
acoustic forced alignment; they therefore remain pending until review. Extraction
then applies the existing Praat/Burg measurement and rejection rules. To avoid
repeatedly decoding long audio files, extraction loads each recording only once.

The example comparison is explicitly exploratory: `--include-pending` admits
measured tokens that have not been reviewed, while `--include-unverified` admits
the projected phone boundaries. Omit those flags after reviewing and accepting
tokens in the Streamlit app.

`compare` evaluates an intercept-only model against linear time. By default it only
uses accepted, observed tokens with manually verified boundaries for one speaker;
use `--speaker-id` in a multi-speaker project and `--include-unverified` only for an
explicit exploratory run. It first averages eligible tokens within each recording, permutes dates at the recording level for a
finite-sample p-value, and resamples recordings for the slope interval. The seed,
iteration count, model errors, effect estimate, and warning are written to JSON.
This guards against token-level pseudoreplication but remains exploratory: it does
not correct recording conditions, uncertain dates, lexical imbalance, or speaker
sampling, and it cannot establish causal language change.

### Example: starting from a chess channel

First identify what a “speaker” means for the study. If one presenter narrates all
videos, use one stable speaker ID. If guests or multiple hosts occur, do **not**
label the channel as one speaker: collect them into separate projects or correct
the recording table before analysis. Start with a small sample from the channel's
`/videos` page rather than downloading its entire archive:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[automatic]'

microevolution collect-youtube chess-study/project.json \
  --speaker-id chess-host \
  --max-videos 10 \
  --date-after 2020-01-01 \
  'https://www.youtube.com/@EXAMPLE_CHESS_CHANNEL/videos'
```

This creates `project.json`, `recordings.csv`, an initially empty `tokens.csv`, and
one hashed WAV per video under `chess-study/audio/`. `--max-videos` applies per URL.
Remove it only when you intentionally want every video. Re-running collection is
safe when the downloaded metadata is unchanged, and individual watch URLs can be
mixed with channel or playlist URLs.

Next obtain a CMU-style pronunciation dictionary. Each line is a word followed by
ARPAbet phones; for example:

```text
CHESS CH EH1 S
GAMBIT G AE1 M B IH0 T
KNIGHT N AY1 T
```

Then run the remaining stages:

```bash
microevolution transcribe chess-study/project.json --model small --language en
microevolution align chess-study/project.json --backend projection --lexicon chess-lexicon.txt
microevolution validate chess-study/project.json
microevolution extract chess-study/project.json
microevolution compare chess-study/project.json --phone AE --response f1_hz \
  --iterations 5000 --seed 2024 --output chess-study/AE-f1-models.json
```

This legacy reproduction path proportionally divides Whisper word timestamps and
skips words absent from its small dictionary. It is retained for compatibility,
not recommended for new acoustic work; use the MFA workflow below for production.

The comparison needs at least three dated videos spanning two upload dates. A chess
channel can also change microphones, rooms, editing, content format, and guests over
time; those changes may look like phonetic change. Treat output as a screening
result, use actual recording dates when known, review speaker identity, and model
recording conditions before making substantive claims.

## Automatic MFA alignment and continuous formants

Production alignment uses Montreal Forced Aligner (MFA) 3.4.2 and English (US)
ARPA acoustic model 3.0.0 with its matching dictionary. Install MFA in a separate conda environment (the
Python package is intentionally not imported by this project), record the exact
installed versions, then download the pinned model generation:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa version
mfa model inspect acoustic english_us_arpa
pip install -e '.[automatic,test]'
```

Use the full downloaded ARPA dictionary augmented only with ARPAbet-compatible
chess pronunciations. Transcribe and align with:

```bash
microevolution transcribe project.json --model small --language en
microevolution align project.json --backend mfa \
  --lexicon /path/to/english_us_arpa.dict --acoustic-model english_us_arpa \
  --retries 1 --fine-tune --force
microevolution extract project.json
microevolution compare project.json --phone AO --automatic-qc --output results/ao.json
```

`align` makes pause/length-bounded utterances, retains their exact source offsets,
and restores MFA phone times to recording coordinates. It never marks a token as
human verified. Automatic QC failures remain in `tokens.csv` with reasons. The
legacy proportional method remains available only as `--backend projection` for
reproduction of old results.

Extraction constructs one continuous-recording Praat Formant object per recording;
only RMS, duration, and pitch checks use the phone crop. This preserves real
neighbouring waveform support for short phones. Every measurement input digest
includes the audio hash, boundaries, alignment run, algorithm, and settings.
Changing any of them invalidates the cached value. `project.json` retains append-only
`alignment_runs` and `runs` provenance histories while keeping the former singular
keys for older readers. Existing tables are migrated in memory; the new columns
are written by the next alignment/extraction, so archived data need not be edited.

To rerun the bundled demonstration exactly:

```bash
python demo/generate_audio.py
microevolution validate demo/project.json
microevolution extract demo/project.json --force
pytest -q
```

The repository does not contain the seven original study WAVs. Run the four-way
projection/MFA and crop/continuous validation on those secured originals before
interpreting longitudinal results; the implementation and synthetic real-Praat
regression can be validated without them.
