"""A WORD AS HER MOUTH'S POSES --- an instrument's yardstick, never hers.

`wordPoses(name, pitch, step)` turns one of the certified word tables
(`body/speech.py`) into the sequence of seven articulator levels her mouth
would hold, one per tick, so an instrument can ask whether her own mouth
passed through a word (`measured/scripts/doctor.py`, `saidalone.py`,
`hermouth.py`).  It lived in the words guide (`measure/bootstrap.py`) until
that guide was removed on his word, 2026-09-06: she is born from nothing, and
nothing moves her mouth through a word.  The yardstick stays.
"""
from __future__ import annotations

import numpy as np

from body import speech
from body.muscles import FRONT_HZ, OPEN_HZ, PITCH_HZ


def wordPoses(name: str, pitch: float, step: int) -> list:
    T = np.asarray(speech.tracks(name, pitch=pitch), np.float64)
    out = []
    for k in range(T.shape[1] // step):
        f0, f1, f2, f3, amp, nasal, close, hiss = T[:, k * step:(k + 1) * step].mean(axis=1)
        out.append(np.array([min(1.0, amp),
                             np.clip((f0 - PITCH_HZ[0]) / (PITCH_HZ[1] - PITCH_HZ[0]), 0, 1),
                             np.clip((f1 - OPEN_HZ[0]) / (OPEN_HZ[1] - OPEN_HZ[0]), 0, 1),
                             np.clip((f2 - FRONT_HZ[0]) / (FRONT_HZ[1] - FRONT_HZ[0]), 0, 1),
                             np.clip(nasal, 0, 1), float(close > 0.5), float(hiss)], np.float32))
    return out
