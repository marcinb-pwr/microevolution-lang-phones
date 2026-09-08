"""Generate the ignored, frozen synthetic WAV fixtures used by the demo.

The files are deterministic derivatives rather than source artifacts, so they are
deliberately excluded from Git. Their expected hashes remain in the manifest.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


RATE = 16_000
DURATION_S = 1.5
PERIODS = ((2014, 690, 1680), (2018, 710, 1640), (2022, 730, 1600))


def generate(output_directory: Path | None = None) -> None:
    output = output_directory or Path(__file__).parent / "audio"
    output.mkdir(parents=True, exist_ok=True)
    for year, f1_hz, f2_hz in PERIODS:
        frames = bytearray()
        for sample_number in range(round(RATE * DURATION_S)):
            time_s = sample_number / RATE
            envelope = 0.2 if 0.45 <= time_s <= 1.05 else 0.02
            value = envelope * (
                math.sin(2 * math.pi * 120 * time_s)
                + 0.35 * math.sin(2 * math.pi * f1_hz * time_s)
                + 0.2 * math.sin(2 * math.pi * f2_hz * time_s)
            )
            frames.extend(struct.pack("<h", max(-32767, min(32767, int(value * 18000)))))
        with wave.open(str(output / f"synthetic_{year}.wav"), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            wav.writeframes(frames)


if __name__ == "__main__":
    generate()
