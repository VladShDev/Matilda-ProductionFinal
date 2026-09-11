"""HER EAR'S SPLIT --- raw samples into the bands she actually has.

Copied unchanged from the old tree's `body/sound.py`.  It is the ONE answer to
"what is a band", and her mouth and her ears both go through it --- so there is
no path in this body that treats her own voice as a special kind of sound.

What did NOT come across yet: the room (`NEAR/(NEAR+far)`, `FAINTEST`, "THEY
ADD"), the head shadow and the two ears.  Those are her hearing and they arrive
with her body.
"""


from __future__ import annotations

import os as _os

import numpy as np

from .speech import SHORTEST_CYCLE

#: her ear's range and its band edges --- geometric, like a cochlea
SOUND_BANDS = 24
#: HER CLOCK.  30 a second, his decision --- *"30 tick per second it anought
#: for everithing"*.
#:
#: IT WAS 1.5 s BECAUSE I COPIED IT ACROSS with `bands_from_pcm` and never
#: asked.  At 1.5 s her sensing was 47x slower than balancing needs (32 ms), her
#: ears could never drain what arrived so ~10% of everything said to her was
#: destroyed oldest-first, and she could not make a syllable at all: one held
#: mouth per tick is silent for 1.5 s or sounding for 1.5 s, and a syllable is
#: 300-600 ms with an onset and an offset inside it.
#:
#: EVERY TIME IN THIS TREE IS DERIVED FROM THIS, ONCE.  A constant written
#: "per tick" is a bug --- it means something different the moment her clock
#: changes.
#: HER CLOCK, AND IT IS SETTABLE FROM OUTSIDE --- his standing ask, repeated
#: 2026-09-01: *"as you remember i ask you make t/s like env var to avoid this
#: problems"*.  `MATILDA_FPS` sets it once for her body AND her mind: her body
#: starts her mind as a child process, so one export reaches all three.
#:
#: IT IS NOT A FRAME RATE.  Everything of hers is derived from this --- her
#: hormones, her look-back, the length of an experience, and her physics
#: timestep (`ragdoll.DT`) --- so a life lived at one value cannot be compared
#: with a life lived at another, and a tape must be fresh when it changes.
#: NINETY IS HERS --- his 11 ms sound tick, 2026-09-03 ("tick has to be 11ms
#: but it is for sound so its like blueprint for resc channels"); thirty was
#: hers until then, and every tape from before is of another clock.
TICK_SECONDS = 1.0 / float(_os.environ.get("MATILDA_FPS", "90") or "90")
#: ...AND HOW OFTEN SHE LOOKS, which is the second clock in her and belongs
#: beside the first.  His: *"5 times per second just to be able controle her
#: body by walking"* --- so this is a REQUIREMENT of her body, not a budget.
#:
#: A look costs 63.8 ms in three stages and no stage is over 25, so it is spread
#: one stage a tick and lands every `1/LOOKS_PER_SECOND` seconds with no tick
#: ever over budget.  Anything that remembers ACROSS looks has to be derived
#: from this the same way sound is derived from `TICK_SECONDS`: `body/bind.py`
#: held 0.45 and 0.04 per LOOK, chosen when a look was every 1.5 s, and at 0.2 s
#: they meant something else entirely --- a thing she stopped finding was
#: forgotten in 1.54 s instead of 11.55 s.  Nothing raised.
LOOKS_PER_SECOND = 5.0
LOOK_SECONDS = 1.0 / LOOKS_PER_SECOND
#: ...AND ONE TICK IS ONE FRAME OF SOUND.  His: *"she has frame our tick
#: |...---___...-~~| --- THAT PART OF SOUND IS THE ONLY ONE WE CAN DISCUSS AT
#: ALL"*.
#:
#: It was 64 slides in a 1.5 s tick --- 23.4 ms each.  At 33 ms a tick, 64
#: slides is 8 samples and no band split is possible on that.  One tick, one
#: slide, one sound: 533 samples, and the whole `slides` dimension leaves her
#: body.  A syllable is then 10-20 ticks --- which is a CHAIN, and chains are
#: built.
SOUND_SLIDES = 1
#: ...AND HOW MANY PLACES THAT ONE FRAME CAN SIT IN.
#:
#: **HER FRAME STAYS ONE TICK WIDE** --- his rule, and it does not move.  What
#: this adds is WHERE it sits.  `body/ears.py align` slides her frame along an
#: arriving sound and takes the wholest placement that is a sound she can make,
#: because a frame landing across two of his sounds averages them into an index
#: that is NEITHER and she can never answer a smear.  63 of 91 when it was
#: written.
#:
#: AND IT HAD NOTHING TO CHOOSE FROM.  Her ear handed it ONE column, so there was
#: one placement and the offset came back 0 every time --- measured, 40 of 40.
#: It did not break; it went inert when the clock went from 1.5 s (64 slides to
#: slide along) to 33 ms (one), and nothing said so because it still returns a
#: perfectly good id.
#:
#: So the same one-tick window is taken at `SOUND_HOPS` offsets a quarter-tick
#: apart.  The window stays 533 samples --- the band split is untouched --- and
#: only where it begins moves.
#: **AND IT IS 1, BECAUSE MORE MADE HER DEAF TO HERSELF.**
#:
#: He asked for it --- *"fix the ear so align has something to choose"* --- and
#: `align` did start choosing: offsets 0, 1, 2 and 3 instead of always 0.  Then
#: the only test that matters was run: say each of her OWN 93 sounds to her and
#: count how many come back as themselves.
#:
#:     SOUND_HOPS = 1     84 of 93    90.3%
#:     SOUND_HOPS = 2     32 of 93    34.4%
#:     SOUND_HOPS = 4     24 of 93    25.8%
#:
#: **She could name nine of her own sounds in ten, and this took her to one in
#: four.**  What `align` picks by is the WHOLEST placement, and among four
#: windows of one sound the wholest is often a window of a DIFFERENT one --- a
#: cleaner-looking smear beats the true frame.  Wholeness tells a whole sound
#: from a smear when there is one sound to find; it cannot tell WHICH.
#:
#: The machinery stays and the number is 1: nothing slides, `align` has one
#: placement, and her naming is back at 90%.  Turning it up again needs a
#: reason to believe the choosing rule, and the rule is what failed.
SOUND_HOPS = 1

#: THE ONE DOOR (his rule, 2026-09-04): the world enters her ears in HER
#: register, once, and what she stores is that.  His voice, her mother's word,
#: anything in the room passes `door` --- resampled by REGISTER (pitch, formants
#: and pace rise together, his certified 1.75 of 2026-08-31) and then banded ---
#: and nothing before it converts: her mother keeps his words raw and speaks
#: them raw, so he hears her as he spoke.  Her own voice is already hers: her
#: echo is her own line and does not pass the door.  Before this the shift
#: lived at her mouth (the bank played his piece x1.75, retired 2026-09-02) and
#: then in the bank's cutter alone, so his live word and her own were named in
#: two registers and she never answered him (her56: four sayings, no reply).
REGISTER = 1.75


def toHer(pcm, want: int | None = None) -> np.ndarray:
    """His sound in her register: every REGISTER-th sample, by interpolation.
    `want` samples out (the whole, shortened by REGISTER, when None)."""
    y = np.asarray(pcm, np.float32).ravel()
    if want is None:
        want = int(len(y) / REGISTER)
    if want <= 0 or y.size < 2:
        return np.zeros(max(want, 0), np.float32)
    i = np.arange(want, dtype=np.float64) * REGISTER
    return np.interp(np.minimum(i, len(y) - 1), np.arange(len(y)), y).astype(np.float32)


def grabFor(rate: float = 16000.0, seconds=None) -> int:
    """How many raw samples one tick of her ear takes through the door."""
    want = int(round(rate * (TICK_SECONDS if seconds is None else seconds)))
    return int(np.ceil(want * REGISTER)) + 1


def door(pcm, rate: float = 16000.0, seconds=None, hops: int = 1) -> np.ndarray:
    """One tick of the world, as her ear takes it: the LAST grabFor() raw
    samples (padded in front when fewer), into her register, into her bands."""
    seconds = TICK_SECONDS if seconds is None else seconds
    want = int(round(rate * seconds))
    grab = grabFor(rate, seconds)
    y = np.asarray(pcm, np.float32).ravel()
    if y.size < grab:
        y = np.pad(y, (grab - y.size, 0))
    return bands_from_pcm(toHer(y[-grab:], want), rate, seconds, hops)


def wordFrames(pcm, rate: float = 16000.0) -> list:
    """A whole word of his through the door, one frame a tick of hers."""
    y = np.asarray(pcm, np.float32).ravel()
    grab = grabFor(rate)
    stride = int(round(rate * TICK_SECONDS * REGISTER))
    return [door(y[i:i + grab], rate) for i in range(0, len(y) - grab + 1, max(1, stride))]

LOW_HZ = 50.0
HIGH_HZ = 8000.0

EDGES_HZ = np.geomspace(LOW_HZ, HIGH_HZ, SOUND_BANDS + 1).astype(np.float32)
BAND_HZ = np.sqrt(EDGES_HZ[:-1] * EDGES_HZ[1:]).astype(np.float32)

#: below this a slide is silence, whole
GATE = 0.02


def _bins(n: int, rate: float) -> np.ndarray:
    """Which band each FFT bin belongs to, or -1 for out of range.  Built once
    per (window size, rate) pair, which in practice means once."""
    hz = np.fft.rfftfreq(n, 1.0 / rate)
    at = np.searchsorted(EDGES_HZ, hz) - 1
    at[(hz < LOW_HZ) | (hz >= HIGH_HZ)] = -1
    return at.astype(np.int64)


_BIN_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _quiet(chunk, rate: float) -> bool:
    """Is there no sound in this piece of window at all?

    **IT ASKS THE LOUDEST PIECE, NOT THE AVERAGE, AND THAT IS THE WHOLE BUG.**

    `GATE` was applied to a whole window.  When a window was one slide of a
    1.5 s frame that was a short piece of sound; at 33 ms the window IS the
    frame, so the test quietly became "is the average of a whole frame quiet"
    --- and **her hiss is bursty.**  MEASURED 2026-08-26 on the sounds she lost:

        sound   whole-frame rms   loudest short piece
          147           0.01834               0.04855
          148           0.01223               0.03237
          290           0.01828               0.05083
          358           0.01556               0.02750
          416           0.01584               0.02601
          205           0.01911               0.04050

    Every one of them is a HIGH-HISS articulation --- her fricatives --- and
    every one is loud in bursts and quiet on average.  So six of her sounds were
    zeroed whole before anything else happened to them, and her alphabet went
    from 92 to 90 at the clock change without a word.

    THE PIECE IS ONE CYCLE OF HER HIGHEST PITCH, and that is a fact about her
    body rather than a number anybody picked: nothing her voice makes can vary
    faster than its own shortest cycle, so a piece that long is the smallest
    window in which "is there sound here" is a question about her at all.  At
    her LOWEST pitch (64 samples) only three of the six come back; at her
    highest (26) all six do, and true silence is still silent.
    """
    n = max(1, int(round(rate * SHORTEST_CYCLE)))
    if chunk.size <= n:
        return float(np.sqrt(np.mean(chunk * chunk))) < GATE
    # the loudest piece decides
    ends = (chunk.size // n) * n
    pieces = chunk[:ends].reshape(-1, n)
    return float(np.sqrt((pieces * pieces).mean(axis=1)).max()) < GATE


def bands_from_pcm(pcm, rate: float = 16000.0,
                   seconds: float = TICK_SECONDS,
                   hops: int = 1) -> np.ndarray:
    """Raw microphone samples -> `(bands, hops)`, 0..1.

    Silence in is silence out: a window quieter than `GATE` is zeroed whole,
    before anything else happens to it.

    `hops` is HOW MANY PLACES her one-tick frame may sit in.  One is what it
    always was --- the last `seconds` of sound, a single column.  More takes the
    same `seconds`-wide window at that many offsets, `seconds/hops` apart, and
    needs `seconds * (1 + (hops-1)/hops)` of sound to do it.
    """
    got = np.asarray(pcm, np.float32).ravel()
    out = np.zeros((SOUND_BANDS, max(hops, 1)), np.float32)
    if got.size == 0:
        return out
    if np.abs(got).max() > 1.5:          # int16 arriving as int16
        got = got / 32768.0

    want = int(round(rate * seconds))
    hops = max(int(hops), 1)
    step = max(1, want // hops)
    need = want + (hops - 1) * step
    # PAD THE FRONT, NEVER THE END.  Silence before he spoke is the truth;
    # padding the end would invent sound he has not made yet.
    if got.size < need:
        got = np.pad(got, (need - got.size, 0))
    got = got[-need:]

    key = (want, int(rate))
    if key not in _BIN_CACHE:
        _BIN_CACHE[key] = _bins(want, rate)
    at = _BIN_CACHE[key]
    window = np.hanning(want).astype(np.float32)
    held = at >= 0

    # OLDEST COLUMN FIRST, so `align`'s offset counts backwards from now.
    for k in range(hops):
        began = (hops - 1 - k) * step
        chunk = got[began:began + want]
        if chunk.size < want:
            chunk = np.pad(chunk, (0, want - chunk.size))
        if _quiet(chunk, rate):
            continue                      # nothing in it clears the gate
        power = np.abs(np.fft.rfft(chunk * window)) / (want * 0.5)
        out[:, k] = np.bincount(at[held], power[held], minlength=SOUND_BANDS)
    return np.clip(out, 0.0, 1.0)
