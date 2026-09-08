# Frozen synthetic interface demonstration

Every file and row in this directory is **synthetic** and is labelled `data_origin=synthetic`.
The formant values are illustrative inputs, not measurements of a human voice and not calibration evidence.
The generated WAV files are ignored by Git because binary WAV diffs are not handled
reliably in the development environment. Create the deterministic fixtures with
`python demo/generate_audio.py`; their SHA-256 hashes are pinned in `recordings.csv`
and `project.json`.
Use a separate project manifest for observed recordings; the application requires an explicit origin filter.
