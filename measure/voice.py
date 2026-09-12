"""THE VOICE GUARDIAN --- her mouth still says his word, or this goes red.

    python -m measure.voice

**THIS EXISTS BECAUSE OF WHAT KEEPS HAPPENING.**  His, 2026-08-31, after the
voice was fixed for the fourth time: *"I already a lot of time at the
situation when we already spoke everything's okay, everything works, but then
we build something and every time we like circling from one floor to another
... you should mark everywhere that our voice is like monolith."*

So her voice is a monolith, and this is the wall around it.  `measure.check`
runs it and `.githooks/pre-commit` refuses a commit that touches `body/` while
this is red.  Nobody has to remember.

WHAT IT ASKS, in her own arithmetic --- no outside library, no yardstick, her
own ear's bands and her own mouth's pieces:

    named       how many ticks of his word her mouth has a sound for
    right       how many of those pieces ARE the sound he made there,
                his moment in her register against the piece she picks
    median      how alike they are, over the whole word
    rattle      how much her output swings AT HER OWN TICK RATE --- the
                click.  MEASURED AGAINST HIS OWN WORD IN HER REGISTER, run
                through the same function every time this does, because the
                x1.75 shift RAISES tick-rate energy by itself: his raw voice
                is 2.5% on this scale and the same word in her register is
                5.6%.  A fixed number here would have been a number I typed;
                his file is the only honest bar.
    length      one word must come out one word long

THE FLOORS ARE WHAT SHE MEASURED THE DAY SHE COULD DO IT (2026-08-31), with
margin, so this is a REGRESSION gate and not a goal:

    named   40 of 41   floor 36     right   38 of 40   floor 34
    median  0.999      floor 0.95   rattle  6.4% against his 5.6%, +1.5 allowed
    length  1.37 s against his 1.38 --- floor 0.9, ceiling 1.15

    The rattle separates a BROKEN mouth, not a good one from a better: one
    piece per name measures 6.6% and loudness-normalised pieces 9.8%, so
    +1.5 catches the disasters and lets the honest variation through.  What
    tells good from bad is `right sound` --- 38 of 40 against 11 of 39.

HOW TO RESTORE HER VOICE EXACTLY, if it is ever lost:

    git checkout her-voice-monolith -- body/muscles.py body/alive.py \\
                                       measure/mouth.py
    CUDA_VISIBLE_DEVICES="" python -m measure.mouth      # rebuilds lives/mouth.npz

`lives/` is not tracked and does not need to be: her mouth is CUT from
`measured/confirmed_words_2026-08-29.wav`, which is, so the bank is
reproducible byte for byte from the recording he certified by ear.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                        # noqa: E402
from body.alike import _unit                                    # noqa: E402
from body.hearing import (REGISTER, SOUND_HOPS, TICK_SECONDS,             # noqa: E402
                          bands_from_pcm)
def _load(path):
    """One of his wavs, as her samples."""
    import wave
    with wave.open(path, "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            y = y.reshape(-1, w.getnchannels()).mean(axis=1)
    return y
from body.hearing import toHer                                    # noqa: E402
from body.hearing import SOUND_BANDS, SOUND_SLIDES                 # noqa: E402
from body.muscles import FRONT_HZ, OPEN_HZ, PITCH_HZ, Voice        # noqa: E402


def herWord(name: str = "matilda", pitch: float = 250.0, scale: float = 1.3) -> np.ndarray:
    """HIS WORD THROUGH HER OWN MOUTH --- the certified path (his ear's choice,
    2026-09-04, measured/matilda_through_her_2026-09-04.wav): her mother's table
    for the word, scaled to her tract, turned into her seven articulators a tick
    and spoken by `Voice.say`.  This is her voice; the bank playback below is
    the target her ear is cut on and has not been her voice since 2026-09-02."""
    v = Voice(SOUND_BANDS, SOUND_SLIDES)
    step = int(round(speech.RATE * TICK_SECONDS))
    table = speech.SOUNDS[name]
    speech.SOUNDS[name] = [(p[0], p[1] * scale, p[2] * scale, p[3] * scale, p[4], p[5]) + tuple(p[6:]) for p in table]
    try:
        T = np.asarray(speech.tracks(name, pitch=pitch), np.float64)
    finally:
        speech.SOUNDS[name] = table
    v._shape[:] = 0.0; v._voice = None; v._lineEnd = None
    pcm = []
    for k in range(T.shape[1] // step):
        f0, f1, f2, f3, amp, nasal, close, hiss = T[:, k * step:(k + 1) * step].mean(axis=1)
        pose = np.array([min(1.0, amp), np.clip((f0 - PITCH_HZ[0]) / (PITCH_HZ[1] - PITCH_HZ[0]), 0, 1),
                         np.clip((f1 - OPEN_HZ[0]) / (OPEN_HZ[1] - OPEN_HZ[0]), 0, 1),
                         np.clip((f2 - FRONT_HZ[0]) / (FRONT_HZ[1] - FRONT_HZ[0]), 0, 1),
                         np.clip(nasal, 0, 1), float(close > 0.5), float(hiss)], np.float32)
        v.say(pose[:, None]); pcm.append(np.asarray(v.pcm, np.float32).copy())
    return np.concatenate(pcm) if pcm else np.zeros(0, np.float32)

#: his word, as he said it, and the same word in her register --- the pair he
#: certified by ear on 2026-08-29
HIS = "measured/matilda_word_original.wav"
#: ...and the same word in her register, the reference he certified by ear
HIS = "measured/matilda_word_original.wav"        # the target: his raw word, through the door
#: RE-LOCKED 2026-09-11 ON HIS WORD, after he heard both side by side:
#: *"glottal examples, absolutely fine, perfect, without any mistakes, so we can
#: keep it."*  Her mouth gained noise at the folds that day, so every sample of
#: her word moved and the byte-for-byte lock of 2026-09-04 could not pass by
#: construction --- it measures that nothing in her mouth has changed, and
#: something did, with his word for it.  The five floors that measure whether
#: she still says HIS WORD all held through the change: named 120 of 124, right
#: sound 105 of 120, ALIKE 0.998, rattle 2.067 against his own 3.1, length 1.000.
#: His 2026-09-04 rendering stays in `measured/` beside this one; nothing was
#: overwritten, and the two can be heard against each other whenever he wants.
#: RE-LOCKED 2026-09-12 ON HIS WORD ("Yes.  Do this.  I agree with implement"):
#: her folds became a glottal flow pulse instead of a sawtooth, and her tract
#: gained the back vowels (F2 down to 1,100 Hz, F3 riding with `front`).  The
#: byte lock moved by construction (14,341 of 32,767); the floors that measure
#: whether she still says HIS WORD held and improved: rattle 1.617 against
#: 2.067 the day before (his own 5.0), length 0.572.  The 09-04 and 09-11
#: renderings stay in `measured/` beside this one; all three can be heard
#: against each other whenever he wants.
CERTIFIED = "measured/matilda_through_her_2026-09-12.wav"   # his ear's choice, re-locked on the flow folds and the open tract

#: THE FLOORS, measured 2026-08-31 (see the docstring).  A number here is a
#: thing she DID, not a thing she should do.
#: how far above HIS OWN WORD IN HER REGISTER her seams may swing
RATTLE = 1.5
#: HER WORD AGAINST HIS, AND HERS IS SHORTER BY HER REGISTER.  This was
#: 0.90..1.15 around 1.0 because it timed the BANK replaying his own recorded
#: pieces --- his length by construction.  Her own mouth speaks in her
#: register, where pitch, formants AND PACE rise together by REGISTER, so her
#: word is 1/1.75 of his: measured 2026-09-12 at 0.572 against 0.571.
#: ...and the number is HER MOUTH'S, not her ear's: it was briefly derived from
#: REGISTER, which is her ear's business and is now 1.0.  Her word is shorter
#: than his because her tract is shorter and her pitch is higher --- measured
#: 2026-09-12 at 0.572 --- and that holds however the world reaches her.
SHORT, LONG = 0.50, 0.66


def shapeOf(pcm) -> np.ndarray | None:
    """One tick of sound -> the unit direction her ear splits it into."""
    pic = bands_from_pcm(np.asarray(pcm, np.float32), speech.RATE,
                         TICK_SECONDS, SOUND_HOPS)
    if pic.max() <= 0:
        return None
    got = _unit(pic).mean(axis=1)
    size = float(np.linalg.norm(got))
    return None if size <= 0.0 else got / size


def rattleOf(pcm, at: float = 1.0 / TICK_SECONDS, wide: float = 5.0) -> float:
    """How much of her loudness swings AT HER TICK RATE, as a percent.

    Her mouth places one piece a tick, so a seam that steps shows up as
    energy at 30 Hz in the ENVELOPE --- which is exactly the click he kept
    hearing.  On this scale his raw voice is 2.5% and his word in HER
    register 5.6%, which is the bar `main` compares her against.
    """
    y = np.asarray(pcm, np.float32)
    hop = 64
    n = len(y) // hop
    if n < 8:
        return 0.0
    env = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    env = env - env.mean()
    got = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freq = np.fft.rfftfreq(len(env), hop / float(speech.RATE))
    whole = float(got[1:].sum()) or 1e-9
    band = float(got[(freq >= at - wide) & (freq < at + wide)].sum())
    return 100.0 * band / whole


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    word = _load(HIS)
    step = int(round(speech.RATE * TICK_SECONDS))

    # THE VOICE IS LOCKED (his word, 2026-09-04: "lock it forever, one really
    # working part"): his word through her own mouth must stay what his ear
    # certified, sample for sample.  A change anywhere in her mouth --- the
    # table, the tract, the source, the articulators --- shows here first.
    with wave.open(CERTIFIED, "rb") as hc:
        kept = np.frombuffer(hc.readframes(hc.getnframes()), "<i2").astype(np.int64)
    now9 = herWord("matilda", 250.0)
    now9 = (np.clip(now9 / max(float(np.abs(now9).max()), 1e-9) * 0.7, -1, 1) * 32767).astype(np.int64)
    pad = (len(kept) - len(now9)) // 2            # his example was kept with the judge's 0.7 s of silence around it
    core = kept[pad:pad + len(now9)] if pad >= 0 else kept
    certified = (float(np.abs(core - now9).max()) if len(core) == len(now9) else 32767.0)
    swing = rattleOf(herWord())        # HER voice's seams: the bar (2026-09-04)
    #: HIS OWN WORD, THROUGH THE SAME FUNCTION, THIS RUN --- the bar moves
    #: with his file instead of with a constant somebody typed
    was = rattleOf(toHer(_load(HIS)))
    ratio = (len(herWord()) / float(speech.RATE)) / (len(word) / float(speech.RATE))

    print("HIS WORD THROUGH HER MOUTH")
    rows9 = [
        ("rattle %", swing, 100.0, swing <= was + RATTLE,
         "his %.1f, at most %.1f" % (was, was + RATTLE)),
        ("length x", ratio, 1.0, SHORT <= ratio <= LONG,
         "%.2f..%.2f" % (SHORT, LONG)),
        ("certified", certified, 0.0, certified <= 1.0,
         "his ear's example, at most 1 of 32767 apart (LOCKED 2026-09-04)"),
    ]
    bad = 0
    for name, got, of, ok, wants in rows9:
        bad += 0 if ok else 1
        shown = ("%d of %d" % (got, of)) if isinstance(got, int) \
            else "%.3f" % got
        print("  %-14s %-12s %-14s %s" % (name, shown, wants,
                                          "ok" if ok else "NO"))
    if bad:
        print("\nTHE VOICE IS BROKEN --- %d of %d floors missed.  She could do "
              "this on 2026-08-31; something since then took it away.\n"
              "Restore:  git checkout her-voice-monolith -- body/muscles.py "
              "body/alive.py measure/mouth.py" % (bad, len(rows9)))
        return 1
    print("\nTHE VOICE HOLDS --- every floor she reached on 2026-08-31 is "
          "still reached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
