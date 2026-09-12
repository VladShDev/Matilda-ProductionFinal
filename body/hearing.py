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

#: HER REGISTER.  His order, 2026-09-12: *"it first has to be converted to her
#: register, and then cut to eleven millisecond pieces --- just once."*  The
#: outside is brought into her register where it arrives (`window.say`), once,
#: and the door after it only cuts.  1.75 is his certified number of 2026-08-31.
#:
#: FREQUENCIES BY REGISTER, TIME NOT AT ALL.  It used to be a resample, which
#: raised pitch and formants and also ran his voice 1.75x fast --- so it was
#: thrown out, and with it the register itself.  `toHer` now stretches time by
#: REGISTER (WSOLA, overlap-add on aligned grains) and resamples by the same,
#: so a register would land his voice where hers lives and last exactly as
#: long as he spoke.
#:
#: AND THE MEASURED REGISTER FOR THE MOUTH SHE HAS IS 1.0.  His 1.75 was
#: certified on 2026-08-31 against the recorded-piece bank.  Swept 2026-09-12
#: against her seven-muscle tract, judged by praat and by her own recogniser:
#: every step above 1.0 made her worse on every count (follows +0.888 -> +0.768,
#: F1 +0.571 -> +0.342, HNR 12.9 -> 4.9 dB; words through her ear 12 -> 1 of
#: 19).  Her similarity is a shape with loudness divided out, and her own rows
#: are named by the same ear, so his /a/ and hers already land together; a
#: shift only pushed him toward her hiss.  At 1.0 `toHer` is the identity and
#: the structure --- once, where his air arrives; the door only cuts --- stands
#: ready for any register he names.
REGISTER = 1.0

#: the grains `toHer` works in, at her 16 kHz: a 25 ms window, a 10 ms hop, and
#: how far it may slide a grain to line it up with the last (6 ms)
_GRAIN, _GRAIN_HOP, _GRAIN_SEEK = 400, 160, 96

#: HOW MUCH AIR HER EAR LISTENS TO FOR ONE PIECE.  The piece is one tick and the
#: step is one tick --- that never moves.  But a frequency cannot be told in
#: less time than about one of its cycles: in 11 ms of air nothing below ~90 Hz
#: apart is separable, and fifteen of her 24 bands are narrower than that.
#: Measured 2026-09-12 with her own judge on his voice: named from 11 ms of
#: air, 6 of his 19 words survive her ear; named from the last 44 ms, 14 (16
#: with her ear bypassed altogether).  So each piece is named from the last
#: four ticks of air --- his and hers alike, the one gate --- and stored as
#: one piece.  A cochlea does the same: low tones take longer to hear.
LISTENS = 4 * TICK_SECONDS


def _stretch(y: np.ndarray, factor: float) -> np.ndarray:
    """The same voice, `factor` times longer, pitch untouched (WSOLA)."""
    N, Hs, Sr = _GRAIN, _GRAIN_HOP, _GRAIN_SEEK
    Ha = Hs / factor
    w = np.hanning(N).astype(np.float32)
    n = int(len(y) * factor)
    out = np.zeros(n + 2 * N, np.float32)
    norm = np.zeros(n + 2 * N, np.float32)
    po, pi, prev = 0, 0.0, None
    while po + N <= out.size and int(pi) + N + Sr + Hs < len(y):
        i = int(pi)
        if prev is not None and i - Sr >= 0:
            offs = np.arange(-Sr, Sr + 1, 2)
            i += int(offs[int(np.argmax([float(np.dot(y[i + d:i + d + N], prev)) for d in offs]))])
        out[po:po + N] += y[i:i + N] * w
        norm[po:po + N] += w
        prev = y[i + Hs:i + Hs + N]
        po += Hs
        pi += Ha
    return out[:n] / np.maximum(norm[:n], 1e-6)


def toHer(pcm, want: int | None = None) -> np.ndarray:
    """HIS SOUND IN HER REGISTER --- frequencies x REGISTER, and it lasts
    exactly as long as he spoke.  Stretched by REGISTER, then resampled by
    REGISTER.  `want` trims or front-pads to that many samples; the whole, as
    long as it came, when None.  A piece too short to overlap grains (under
    ~50 ms) is plainly resampled, pace and all --- the door never sends one."""
    y = np.asarray(pcm, np.float32).ravel()
    n = len(y) if want is None else int(want)
    if n <= 0 or y.size < 2:
        return np.zeros(max(n, 0), np.float32)
    if abs(REGISTER - 1.0) < 1e-6:
        out = y
    elif y.size < _GRAIN + _GRAIN_HOP + 2 * _GRAIN_SEEK:
        i = np.arange(int(len(y) / REGISTER), dtype=np.float64) * REGISTER
        out = np.interp(np.minimum(i, len(y) - 1), np.arange(len(y)), y)
    else:
        s = _stretch(y, REGISTER)
        i = np.arange(0, len(s), REGISTER, dtype=np.float64)
        out = np.interp(np.minimum(i, len(s) - 1), np.arange(len(s)), s)[:len(y)]
    out = np.asarray(out, np.float32)
    if out.size < n:
        out = np.pad(out, (n - out.size, 0))
    return out[-n:]


def grabFor(rate: float = 16000.0, seconds=None) -> int:
    """How many raw samples one tick of her ear takes: one tick's worth.  (It
    was REGISTER times more, resampled down inside the door; the register
    moved to where his air arrives, once, and the door only cuts.)"""
    return int(round(rate * (TICK_SECONDS if seconds is None else seconds)))


def door(pcm, rate: float = 16000.0, seconds=None, hops: int = 1) -> np.ndarray:
    """One piece of the world, as her ear takes it: the LAST `seconds` of air
    (`LISTENS` when None; padded in front when fewer), into her bands.  NOTHING
    IS CONVERTED HERE: his air was brought into her register once, as it
    arrived (`window.say`).  His order, 2026-09-12: first her register, then
    the 11 ms pieces, once."""
    seconds = LISTENS if seconds is None else seconds
    grab = grabFor(rate, seconds)
    y = np.asarray(pcm, np.float32).ravel()
    if y.size < grab:
        y = np.pad(y, (grab - y.size, 0))
    return bands_from_pcm(y[-grab:], rate, seconds, hops)


def wordFrames(pcm, rate: float = 16000.0) -> list:
    """A whole word of his through the door, one frame a tick of hers: each
    frame named from the last `LISTENS` of air, stepped one tick.  The word
    arrives already in her register (it came off the queue, `window.say`), so
    this only cuts --- nothing converted a second time."""
    y = np.asarray(pcm, np.float32).ravel()
    grab = grabFor(rate, LISTENS)
    stride = int(round(rate * TICK_SECONDS))
    y = np.concatenate([np.zeros(grab - stride, np.float32), y])
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
