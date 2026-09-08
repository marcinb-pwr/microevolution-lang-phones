"""Validated Praat/Burg measurements from manually checked intervals."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median


def measure_interval(audio_path: str | Path, start_s: float, end_s: float,
                     *, max_formant_hz: float = 5500, window_s: float = 0.025) -> dict:
    """Measure central F1/F2 and a five-point trajectory using Praat's Burg method."""
    import parselmouth

    result = {"f1_hz": None, "f2_hz": None, "f0_hz": None, "trajectory": [],
              "measurement_status": "rejected", "exclusion_reason": ""}
    if start_s < 0 or end_s <= start_s:
        result["exclusion_reason"] = "invalid_interval"
        return result
    sound = parselmouth.Sound(str(audio_path)).extract_part(start_s, end_s, preserve_times=False)
    if sound.duration < 0.04:
        result["exclusion_reason"] = "interval_too_short"
        return result
    values = sound.values
    rms = float((values ** 2).mean() ** 0.5)
    if rms < 1e-4:
        result["exclusion_reason"] = "silence_or_near_silence"
        return result
    formants = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                                     maximum_formant=max_formant_hz,
                                     window_length=window_s, pre_emphasis_from=50)
    pitch = sound.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=400)
    trajectory = []
    for position in (0.2, 0.35, 0.5, 0.65, 0.8):
        time = position * sound.duration
        f1 = formants.get_value_at_time(1, time)
        f2 = formants.get_value_at_time(2, time)
        trajectory.append({"position": position, "f1_hz": _finite(f1), "f2_hz": _finite(f2)})
    central = trajectory[1:4]
    f1s = [p["f1_hz"] for p in central if p["f1_hz"] is not None]
    f2s = [p["f2_hz"] for p in central if p["f2_hz"] is not None]
    if len(f1s) < 2 or len(f2s) < 2:
        result["trajectory"] = trajectory
        result["exclusion_reason"] = "untrackable_formants"
        return result
    f0 = pitch.get_mean(0, sound.duration, "Hertz")
    result.update(f1_hz=median(f1s), f2_hz=median(f2s), f0_hz=_finite(f0),
                  trajectory=trajectory, measurement_status="measured")
    return result


def _finite(value):
    import math
    return float(value) if value is not None and math.isfinite(value) else None


def trajectory_json(measurement: dict) -> str:
    return json.dumps(measurement["trajectory"], separators=(",", ":"))
