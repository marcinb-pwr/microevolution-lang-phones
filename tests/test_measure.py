import sys
import types

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
