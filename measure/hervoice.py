"""HIS WORDS IN HER VOICE --- the path he chose on 2026-09-12, written down so it
can be changed later without touching anything else.

    python -m measure.hervoice --from HIS.wav                # 20 s of him, loudest stretch
    python -m measure.hervoice --from HIS.wav --seconds 30
    python -m measure.hervoice --from HIS.wav --tract 1.3 --pitch 0.2 0.8 --share 0.7 out.wav

TESTING ONLY.  Nothing in `body/` or `mind/` imports this, and nothing ever may.
It does not teach her anything: it is what her memory WOULD let her do if a
sound reached it as her ear's levels (24 a piece) plus its voicing and pitch,
instead of one number.  Today her memory keeps one level per piece, so her own
mouth gives back his rhythm and pauses and none of his words; this instrument
gives them back (recogniser: 14-20 of 41 words in his 20 s) and is the measure
of what that decision is worth.

THE PATH, one step at a time --- each step is one function below:

  1. HIS AIR, as his body would hand it to her: 16 kHz, into her register once
     (`toHer`; at REGISTER 1.0 that is the identity), then the loudest
     `--seconds` of it.                                            `hisAir()`
  2. HER EAR on it: one piece every 11 ms, each named from the last `LISTENS`
     of air, 24 band levels a piece --- exactly `door`, exactly what her body
     hears of him.                                                  `earOf()`
  3. HIS THREE LINES, read by the outside app (praat) per piece: is it voiced,
     is it air, and where his pitch sits between his own 10th and 90th
     percentile.  Her sound lines do not carry these; this is what would have
     to be on her row for her to do it herself.                  `hisLines()`
  4. HER CARRIER: her own mouth --- the flow folds, the open tract --- saying an
     open vowel, with THREE of her seven muscles following his lines each
     piece: `loud` off where his piece is silent, `hiss` = 1 where it is air,
     `pitch` = his contour placed on her larynx at `--pitch lo hi` (his word:
     0.4-1.0, about 400 Hz median).  Her voice switching off with his is what
     puts the edges between his words (150 switches against his 144); her
     pitch moving with his is his melody.                         `carrier()`
  5. THE SHAPE: each piece of her carrier is filtered so its 24 bands become
     his 24 --- divided by her STEADY voice (one mean spectrum for her voiced
     pieces, one for her air), never by the wobbling piece itself (that tore
     the voice apart: voiced 12%).  Before that, his envelope is moved UP her
     log-spaced bands by `--tract` (his word: 1.5, a small child's tract; 1.3
     a child; 1.0 is him on helium), and `--share` < 1 lets her own tract's
     colour through (1.0 is all his shape).                         `shape()`
  6. Praat on the result: her F0, F1, F2, HNR --- and, with `--judge`, the
     recogniser's word count against his raw.

WHAT TO CHANGE FOR WHAT:
  higher / lower child        --pitch  (0.2 0.8 = 332 Hz, 0.4 1.0 = 399, 0.55 1.0 = 445)
  further from his timbre     --tract  (1.0 -> 1.3 -> 1.5 moves F1 469 -> 566 -> 645 Hz)
  more of her own colour      --share  (0.7 measured HNR 23.5 against 20.3 at 1.0)
  the window her ear listens  hearing.LISTENS (hers, not this file's)
  her folds, her tract        body/speech.py, body/muscles.py (hers, not this file's)
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                            # noqa: E402
from body.hearing import (EDGES_HZ, LISTENS, SOUND_BANDS, SOUND_HOPS,  # noqa: E402
                          SOUND_SLIDES, TICK_SECONDS, door, grabFor, toHer)
from body.muscles import Voice                                     # noqa: E402

RATE = int(speech.RATE)
HOP = int(round(RATE * TICK_SECONDS))            # one piece
GRAB = grabFor(RATE, LISTENS)                    # what her ear listens to for one piece
PARTS = 7
OUT = os.path.join("measured", "hervoice.wav")

#: HIS CHOICES, 2026-09-12 --- the knobs.  Change here, nothing else moves.
PITCH = (0.4, 1.0)      # his contour on her pitch line: lo..hi of 0..1 (250..600 Hz)
TRACT = 1.5             # his envelope moved up her log bands by a small child's tract
SHARE = 1.0             # 1.0 = all his shape; less lets her own tract's colour through
VOICE_ROW = (0.7, 0.5, 0.5, 0.5, 0.05, 0.0, 0.0)   # her open vowel: loud, pitch, open, front, nasal, close, hiss
AIR_LOUD = 0.45                                    # her breath line where his piece is air


# ------------------------------------------------------------- 1. his air
def hisAir(path: str, seconds: float = 20.0) -> np.ndarray:
    with wave.open(path, "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        r0 = float(w.getframerate())
    n = int(round(len(y) * RATE / r0))
    y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    y = toHer(y)                                          # into her register, once (identity at 1.0)
    half = int(RATE * 0.5)
    power = np.array([float(np.abs(y[k:k + half]).mean()) for k in range(0, max(1, len(y) - half), half)])
    if power.size:
        at = int(np.argmax(np.convolve(power, np.ones(int(seconds * 2)) / (seconds * 2), "same")))
        y = y[max(0, at * half - int(RATE * seconds / 2)):][:int(RATE * seconds)]
    return y


# ------------------------------------------------------------- 2. her ear
def earOf(pcm: np.ndarray) -> np.ndarray:
    """`(pieces, 24)`: what her ear says of every 11 ms piece of this air."""
    z = np.concatenate([np.zeros(GRAB - HOP, np.float32), np.asarray(pcm, np.float32)])
    return np.asarray([np.asarray(door(z[e - GRAB:e], RATE, LISTENS, SOUND_HOPS), np.float32)[:, -1]
                       for e in range(GRAB, len(z) + 1, HOP)])


# ------------------------------------------------------------- 3. his lines
def hisLines(pcm: np.ndarray, pieces: int) -> tuple:
    """Per piece: voiced?, and his pitch contour in 0..1 (10th..90th percentile)."""
    import parselmouth as pm
    s = pm.Sound(np.asarray(pcm, np.float64), sampling_frequency=RATE)
    pit = s.to_pitch(time_step=TICK_SECONDS, pitch_floor=60, pitch_ceiling=400)
    f0 = np.array([pit.get_value_at_time((k + 0.5) * TICK_SECONDS) for k in range(pieces)])
    voiced = np.isfinite(f0) & (f0 > 0)
    if not voiced.any():
        return voiced, np.zeros(pieces)
    lo, hi = np.percentile(f0[voiced], [10, 90])
    contour = (np.nan_to_num(f0, nan=lo) - lo) / max(hi - lo, 1e-6)
    return voiced, np.clip(contour, 0.0, 1.0)


# ------------------------------------------------------------- 4. her carrier
def carrier(his: np.ndarray, voiced: np.ndarray, contour: np.ndarray, pitch=PITCH) -> tuple:
    """Her own mouth, three lines following his: `(pcm, kind)`; kind 0 silent,
    1 voice, 2 air."""
    sound = his.sum(axis=1) > 0
    pl = np.clip(pitch[0] + (pitch[1] - pitch[0]) * contour, 0.0, 1.0)
    v = Voice(SOUND_BANDS, SOUND_SLIDES)
    out, kind = [], []
    for k in range(len(his)):
        row = np.array(VOICE_ROW, np.float32)
        if not sound[k]:
            row[0] = 0.0; kind.append(0)
        elif voiced[k]:
            row[1] = pl[k]; kind.append(1)
        else:
            row[0] = AIR_LOUD; row[6] = 1.0; kind.append(2)
        v.say(row.reshape(PARTS, 1))
        out.append(np.asarray(v.pcm, np.float32).ravel().copy())
    return np.concatenate(out), np.asarray(kind)


# ------------------------------------------------------------- 5. the shape
def shape(car: np.ndarray, kind: np.ndarray, his: np.ndarray, tract=TRACT, share=SHARE) -> np.ndarray:
    own = earOf(car)
    refV = own[kind == 1].mean(axis=0) if (kind == 1).any() else own.mean(axis=0)
    refA = own[kind == 2].mean(axis=0) if (kind == 2).any() else refV
    band = float(EDGES_HZ[1] / EDGES_HZ[0])
    x = np.arange(SOUND_BANDS, dtype=np.float64)
    up = np.log(tract) / np.log(band)

    def child(b):                                    # his envelope on a child's tract, energy kept
        o = np.interp(x - up, x, b, left=0.0, right=0.0).astype(np.float32)
        return o * (b.sum() / max(o.sum(), 1e-9))

    frame = 4 * HOP
    win = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / RATE)
    bandOf = np.searchsorted(EDGES_HZ, freqs) - 1
    ok = (bandOf >= 0) & (bandOf < SOUND_BANDS)
    out = np.zeros(len(car) + frame, np.float32)
    src = np.concatenate([car, np.zeros(frame, np.float32)])
    for t in range(min(len(his), len(own))):
        if kind[t] == 0:
            continue
        ref = refV if kind[t] == 1 else refA
        ratio = (child(his[t]) / np.maximum(ref, 1e-4)) ** share
        g = np.zeros_like(freqs)
        g[ok] = ratio[bandOf[ok]]
        s = t * HOP
        out[s:s + frame] += np.fft.irfft(np.fft.rfft(src[s:s + frame] * win) * g, n=frame).astype(np.float32) * win
    return out[:len(car)]


def write(path: str, pcm: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pk = float(np.abs(pcm).max()) or 1.0
    with wave.open(path, "wb") as g:
        g.setnchannels(1); g.setsampwidth(2); g.setframerate(RATE)
        g.writeframes((np.clip(pcm / pk * 0.85, -1, 1) * 32767).astype("<i2").tobytes())


# ------------------------------------------------------------- 6. praat
def report(path: str, his: np.ndarray | None = None, judge: bool = False) -> None:
    import parselmouth as pm
    from parselmouth.praat import call
    s = pm.Sound(path)
    f = s.to_pitch(pitch_floor=150, pitch_ceiling=900).selected_array["frequency"]; vv = f[f > 0]
    fm = s.to_formant_burg(max_number_of_formants=5, maximum_formant=6500)
    tt = np.linspace(0.1, s.duration - 0.1, 400)
    F1 = np.nanmedian([fm.get_value_at_time(1, t) for t in tt]); F2 = np.nanmedian([fm.get_value_at_time(2, t) for t in tt])
    print("  her voice (praat):  F0 %.0f Hz   F1 %.0f   F2 %.0f   voiced %.0f%%   HNR %.1f dB" % (
        np.median(vv) if vv.size else 0, F1, F2, 100 * vv.size / max(1, f.size), call(s.to_harmonicity(), "Get mean", 0, 0)))
    if judge and his is not None:
        from measure import judge as J
        m = J._model()
        ref = [w for w, c in J.hear(his, RATE, model=m)[1]]
        with wave.open(path, "rb") as w:
            hers = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        free, words, _, _ = J.hear(hers, RATE, model=m)
        got = [w for w, c in words]
        print("  the judge: %d of his %d words   | %s" % (sum(1 for w in got if w in ref), len(ref), free[:90]))


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    args = sys.argv[1:]

    def opt(name, n, default):
        if name in args:
            i = args.index(name); vals = args[i + 1:i + 1 + n]
            del args[i:i + 1 + n]
            return [float(v) for v in vals] if n > 1 else float(vals[0])
        return default
    seconds = opt("--seconds", 1, 20.0); pitch = tuple(opt("--pitch", 2, list(PITCH)))
    tract = opt("--tract", 1, TRACT); share = opt("--share", 1, SHARE)
    judge = "--judge" in args; args = [a for a in args if a != "--judge"]
    if "--from" not in args:
        print(__doc__.split("\n\n")[1]); return 1
    src = args[args.index("--from") + 1]; args = [a for a in args if a not in ("--from", src)]
    out = args[0] if args else OUT
    his = hisAir(src, seconds)
    bands = earOf(his)
    voiced, contour = hisLines(his, len(bands))
    car, kind = carrier(bands, voiced, contour, pitch)
    write(out, shape(car, kind, bands, tract, share))
    print("HIS WORDS IN HER VOICE --- %.0f s of %s;  pitch %.2f-%.2f, tract x%.2f, share %.2f" % (seconds, src, pitch[0], pitch[1], tract, share))
    print("  written %s" % out)
    report(out, his, judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
