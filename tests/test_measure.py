import sys
import types
import pytest

from microevolution.measure import measure_interval


def test_missing_pitch_does_not_discard_formants(monkeypatch):
    class Formants:
        def get_value_at_time(self, number, time):
            return 700 if number == 1 else 1700

    class Sound:
        duration = 0.045

        class Values:
            def __pow__(self, exponent): return self
            def mean(self): return 0.01

        values = Values()

        def __init__(self, path=None): pass
        def extract_part(self, *args, **kwargs): return self
        def to_formant_burg(self, **kwargs): return Formants()
        def to_pitch(self, **kwargs): raise RuntimeError("not enough periods")

    praat = types.ModuleType("parselmouth.praat")
    praat.call = lambda *args: None
    module = types.ModuleType("parselmouth")
    module.Sound = Sound
    module.praat = praat
    monkeypatch.setitem(sys.modules, "parselmouth", module)
    monkeypatch.setitem(sys.modules, "parselmouth.praat", praat)

    result = measure_interval("short.wav", 0, 0.045)
    assert result["measurement_status"] == "measured"
    assert result["f1_hz"] == 700
    assert result["f0_hz"] is None


@pytest.mark.parametrize("duration_ms", [40, 45, 50, 55, 60, 65, 70, 75, 80, 100, 150])
def test_real_praat_continuous_context_tracks_short_intervals(tmp_path, duration_ms):
    parselmouth = pytest.importorskip("parselmouth")
    import numpy as np
    rate, duration = 48000, 0.4
    times = np.arange(int(rate * duration)) / rate
    # Deterministic voiced, vowel-like harmonic signal with ample real context.
    signal = sum((1 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * times)
                 for harmonic in range(1, 35))
    sound = parselmouth.Sound(signal, sampling_frequency=rate)
    from microevolution.measure import RecordingMeasurements, measure_sound_interval
    analysis = RecordingMeasurements(sound)
    start = .2 - duration_ms / 2000
    result = measure_sound_interval(sound, start, start + duration_ms / 1000,
                                    formants=analysis.formants)
    assert len(result["trajectory"]) == 5
    assert sum(p["f1_hz"] is not None and p["f2_hz"] is not None
               for p in result["trajectory"]) == 5
    assert result["measurement_status"] == "measured"
