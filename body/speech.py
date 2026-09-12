"""A mouth, as sound: the source-filter synthesiser.  HERS AND HER MOTHER'S.

It lived in `say.py` first, as part of the tool that stands in for YOUR mouth.
It moved here the day the room got a mother, because a mother needs a mouth too
and a world actor importing a command-line tool would have the layering
backwards: the way a voice is MADE is a fact about bodies in the world, and the
tools (`say.py`, `teach.py`) borrow it from here.

**IT IS NOW HER MOUTH TOO, AND THAT IS THE POINT OF THIS FILE.**  It used to say
"nothing about HER is in this file", and while that was true she could not
imitate anybody.  Her mouth computed a spectrum directly --- a 1/f source times
three resonance magnitudes, read off at her own band centres --- and her
mother's went through real samples and a real FFT.  Measured: her mouth against
her mother's SAME WORD scored 0.134.  Four systematic differences, none of them
about control:

  * her mother's sawtooth falls 6 dB an octave and lip radiation lifts it 6, so
    her mother's source is FLAT; hers fell at 1/f, a 44 dB tilt across her range
  * her mother's three formants are SUMMED at 1 / 0.55 / 0.25; hers were
    MULTIPLIED, which is a different filter, not a different setting
  * her mother's spectrum has discrete harmonics; hers was a smooth envelope
  * hers was normalised to a peak every slide, so she could not be quiet

**IMITATION CANNOT CONVERGE ON A TARGET IT CANNOT REACH.**  So both mouths are
one function now, and the only difference left between them is what a difference
between two speakers should be: the numbers they send it.

HOW THE VOICE IS MADE.  A glottal pulse train --- what vocal folds do --- pushed
through three resonators at the formant frequencies of each sound, sliding
between them.  That is a source-filter model, which is what a vocal tract is.
What comes out is raw microphone samples, exactly what a phone posts to
`/window`; everything after that (the frequency split, the head shadow, which
ear hears it louder) happens in her body, from a point in her room.

WHY A REAL WORD AND NOT A BEEP.  A tone is one band and tells her nothing about
speech.  A word is a SHAPE MOVING ACROSS BANDS: the formants slide, the nasals
dip, the vowels hold.  Her ears run at 23 ms a slide precisely so that shape
survives, and there is no point having built that if the only thing she ever
hears is a sine wave.
"""

from __future__ import annotations

import math

import numpy as np

#: HER VOICE'S RANGE --- an infant's f0, babble through cry.
#:
#: It was written twice in `muscles.py`, twenty lines apart, and her HEARING
#: needs it too: the gate that decides whether a window is silence has to look
#: at a piece no longer than her shortest cycle, or a sound that is loud in
#: bursts averages away to nothing.  One place, and both read it.
PITCH_HZ = (250.0, 600.0)
#: ...so her SHORTEST cycle, in seconds.  A window with no piece this long that
#: clears the gate has no sound in it.
SHORTEST_CYCLE = 1.0 / PITCH_HZ[1]

RATE = 16000

#: A voice pitched where a mother's is.  Not a fact about her --- a fact about
#: whoever is doing the talking.
PITCH = 210.0

#: `(how long, F1, F2, F3, how loud, nasal)` --- the formants of each sound in
#: turn.  These are the measured formants of English vowels; the nasals are a
#: low murmur with the mouth shut, which is what /m/ is.
SOUNDS: dict[str, list[tuple]] = {
    #  m: lips closed, sound out through the nose.  a: open.  i: front, high.
    "mommy": [(0.09, 280, 1100, 2400, 0.30, True),
              (0.19, 730, 1090, 2440, 1.00, False),
              (0.09, 280, 1100, 2400, 0.30, True),
              (0.26, 280, 2250, 2900, 0.85, False)],
    "dada":  [(0.03, 300, 1700, 2600, 0.15, False),
              (0.20, 730, 1090, 2440, 1.00, False),
              (0.03, 300, 1700, 2600, 0.15, False),
              (0.26, 730, 1090, 2440, 0.90, False)],
    "baba":  [(0.03, 250, 900, 2300, 0.15, False),
              (0.20, 730, 1090, 2440, 1.00, False),
              (0.03, 250, 900, 2300, 0.15, False),
              (0.26, 730, 1090, 2440, 0.90, False)],
    "nini":  [(0.09, 280, 1700, 2600, 0.30, True),
              (0.20, 280, 2250, 2900, 0.95, False),
              (0.09, 280, 1700, 2600, 0.30, True),
              (0.26, 280, 2250, 2900, 0.85, False)],
    "aaa":   [(0.55, 730, 1090, 2440, 1.00, False)],
    #  g: closure at the back, so F2 starts LOW and climbs.  i as in "give":
    #  front and high, F2 up near 1900.  v: back down to a weak buzz.  The
    #  climb-then-fall is the whole shape, and it is nothing like "mommy"
    #  (nasal, F2 flat and low) or "dada" (open, F2 parked at 1090).
    "give":  [(0.05, 250, 1600, 2400, 0.25, False),
              (0.24, 400, 1900, 2500, 1.00, False),
              (0.21, 280, 1000, 2200, 0.50, False)],
    #  "up": one open back vowel snapping shut at the lips.  Its SHAPE is a
    #  fall into silence --- the only word in the table that ends by closing,
    #  which is exactly what a lifted baby hears as her feet leave the bed.
    "up":    [(0.20, 640, 1190, 2390, 1.00, False),
              (0.10, 250, 900, 2300, 0.12, False)],
    #  "milk": m, then i, then the tongue-side l, then the back stop.  F2
    #  starts LOW under the nose and LEAPS high at once --- the mirror of
    #  "mommy", whose F2 parks low through the whole first half.
    "milk":  [(0.08, 280, 1100, 2400, 0.30, True),
              (0.15, 400, 1900, 2500, 1.00, False),
              (0.12, 360, 1050, 2800, 0.80, False),
              (0.07, 300, 1400, 2300, 0.15, False)],
    # HER NAME, his word --- added 2026-09-04 ("add"), in the shape the separate
    # judge heard as "matilda" at 1.00 from this mouth: soft t and d as dips,
    # the vowels of the English table, a fall of pitch across the word.
    # THE t AND THE d ARE STOPS (2026-09-04): a seventh field, `True`, seals the
    # mouth for the first STOP_SEALED of the segment and releases it with a
    # hiss --- a closure and a burst, not a dip in loudness.  Measured through
    # her own articulators by the separate judge at 250/285/320 Hz: dips gave
    # 0.58 / 0.00 / 0.87 for "matilda"; the stops give 1.00 / 0.85 / 0.71.
    # MAMA (2026-09-04, his second word): m a m a --- the m is the nasal murmur of
    # the table, the a its open vowel; two syllables, the second a little longer,
    # as the word is said to a baby.  Pitch falls across it as every word does.
    # THE DOCTOR'S THINGS (2026-09-04): the names her mother says of what is
    # shown to her --- a ball, a rattle, a bear --- English formants, a stop
    # where the word has one.  Milk and mama are already here.
    "ball": [(0.04, 250, 900, 2300, 0.30, False, True), (0.22, 570, 840, 2410, 1.00, False),
             (0.14, 380, 1100, 2500, 0.80, False)],
    "rattle": [(0.08, 420, 1300, 1600, 0.60, False), (0.18, 660, 1720, 2410, 1.00, False),
               (0.05, 400, 1800, 2500, 0.30, False, True), (0.14, 380, 1100, 2500, 0.80, False)],
    "bear": [(0.04, 250, 900, 2300, 0.30, False, True), (0.20, 530, 1840, 2480, 1.00, False),
             (0.16, 420, 1300, 1600, 0.80, False)],
    "mama": [(0.10, 280, 1100, 2400, 0.3, True), (0.20, 730, 1090, 2440, 1.0, False),
             (0.10, 280, 1100, 2400, 0.3, True), (0.24, 730, 1090, 2440, 1.0, False)],
    "matilda": [(0.09, 280, 1100, 2400, 0.3, True), (0.18, 730, 1090, 2440, 1.0, False),
                (0.05, 400, 1800, 2500, 0.3, False, True), (0.14, 390, 1990, 2550, 1.0, False),
                (0.09, 380, 1100, 2500, 0.8, False), (0.05, 400, 1800, 2500, 0.3, False, True),
                (0.20, 620, 1220, 2500, 1.0, False)],
}


#: How often the filter may be retuned, in samples --- 4 ms.  A formant
#: transition runs 30-50 ms, so a slide is still a slide with a dozen steps in
#: it, and the cost of the whole synthesiser drops by this factor.  SHE uses a
#: coarser one: her block is one of her slides, 375 samples, because 23 ms is
#: genuinely how often her nervous system may say something new.
BLOCK = 64

#: Three resonators in a row are much peakier than three added together, so the
#: whole thing comes out quieter.  This puts it back where it was, and it is the
#: only number in this file that exists for arithmetic rather than for anatomy.
CASCADE_GAIN = 260.0
#: THE NOSE, AND IT NEVER MOVES.  A nasal murmur sits near here in everybody,
#: because a nose is a tube of fixed length that no muscle can reshape --- which
#: is exactly why `nasal` deserves one line and not three.
NOSE_HZ = 280.0
NOSE_WIDE = 60.0
#: How much quieter a nose is than a mouth: a small, soft, damped opening.
#: MEASURED against her own /a/ at the same effort, so a nasal lands at about
#: 0.6x an open vowel --- quieter, as a nasal is, and not absent.
NOSE_QUIET = 0.27
#: HER CONSONANTS --- turbulence at a constriction, shaped by the SHORT cavity
#: in front of it, not by the whole tract behind.  Measured 2026-09-03: his
#: certified word in her register has 18 sounds; a sweep of 4,116 poses of
#: her seven articulators reached none of six of them --- all six with the
#: similarity law's SPREAD digit at 5 while her whole alphabet spans 0-3.
#: Her noise went through the vowel cascade (bandwidths 90/110/160 Hz), so a
#: fricative was a hiss with a vowel painted on it, and a sealed mouth gated
#: it to nothing.  The front cavity is short, so its resonance is high and
#: BROAD, and it sits higher the fronter the constriction (a shorter cavity):
#: its centre rides on F2, its width is TURB_WIDE.  Turbulence needs flow AND
#: a squeeze, so it peaks half-shut and is silent sealed; a release lets a
#: BURST of the same turbulence out over BURST_S.  Vowels are untouched:
#: with `hiss` and `close` at zero this path is exactly zero.
#: MEASURED 2026-09-03, 36 constant sets x 350 random poses, named by her ear
#: reaches (exact / one step away): the vowel-only mouth 3 / 9; this set
#: 11 / 5, the best of the grid (0.03-0.10 gain and 600-1500 Hz all near it;
#: a loud or very wide turbulence drowns the vowel and the alphabet SHRINKS
#: to 11 ids).  Her alphabet is 58 ids at this setting against 37 before.
TURB_OVER_F2 = 2.6          # the front cavity's centre, as a multiple of F2
TURB_WIDE = 600.0           # ...and its bandwidth, Hz
TURB_GAIN = 0.03            # turbulence against the voice, at the same amp
BURST_S = 0.008             # a stop's release burst decays over this


def _glottis(f0: np.ndarray, start: float = 0.0):
    """What vocal folds do: a train of pulses, one per period of the pitch.

    Takes the pitch SAMPLE BY SAMPLE, because a mouth that cannot change pitch
    within a word is not a mouth she can say anything with.  Her mother's falls
    slightly across a word, because a real voice does --- a flat pitch is the
    single thing that makes synthetic speech sound synthetic --- and that now
    lives in `word()`, where it is a fact about speaking rather than about the
    machine.
    """
    phase = start + np.cumsum(np.asarray(f0, np.float64)) / RATE
    # VOCAL FOLDS AS FLOW, NOT A SAWTOOTH.  His word, 2026-09-12 ("maybe she
    # just needs a more advanced synthesizer"), and measured the same day on
    # her held vowel through her own lips: the sawtooth --- every harmonic at
    # -6 dB an octave, then the lips' +6 --- came out FLAT, +4.0 dB of tilt
    # where a voice is about -12: the harsh, bright "beast" he heard.  A
    # glottal flow pulse (Rosenberg 1971: opening 40% of the period as a
    # raised cosine, closing 16%, closed 44%) falls at -12 an octave, so after
    # the lips she is -6.9; shimmer 7.6% -> 0.5%, jitter 0.80% -> 0.03%, and
    # her memory stopped choosing an air row for his vowels four times in
    # five (22% -> 5%).  Centred, so the closed phase is rest.  The lips
    # (`np.diff` below) make the closure spike, which is the voice.
    t = phase - np.floor(phase)
    T1, T2 = 0.40, 0.16
    g = np.zeros_like(t)
    a = t < T1
    g[a] = 0.5 * (1.0 - np.cos(np.pi * t[a] / T1))
    b = (t >= T1) & (t < T1 + T2)
    g[b] = np.cos(np.pi * (t[b] - T1) / (2.0 * T2))
    return 2.0 * g - 1.0


def _resonate(x: np.ndarray, freq: np.ndarray, wide: float,
              block: int = BLOCK, start=None):
    """One formant: a two-pole resonator, retuned every `block` samples.

    **EXACT, AND WITHOUT A PYTHON LOOP OVER SAMPLES.**  It used to run the
    recursion one sample at a time, which cost about 35 ms a word --- fine for a
    mother who speaks occasionally, hopeless for a mouth that is used every
    tick.  The same filter, in closed form:

    A two-pole resonator is a conjugate pair, and a conjugate pair is twice the
    real part of ONE complex pole `p`.  A one-pole filter has an answer rather
    than a recursion --- `y[n] = p^n * sum(x[k] / p^k)` --- and a running sum is
    `cumsum`.  So a block is three array operations, the state carries into the
    next block as one complex number, and the arithmetic is the same arithmetic.

    Why it needs blocks at all: `1/p^k` grows like `1/r^k`, so over a whole
    1.5 s tick it would overflow.  Over one block it reaches about 3e5 at the
    widest formant, which float64 carries with ten digits to spare.
    """
    n = (x.size // block) * block
    xb = np.asarray(x[:n], np.float64).reshape(-1, block)
    blocks = xb.shape[0]
    r = math.exp(-math.pi * wide / RATE)
    turn = 2.0 * np.pi * np.asarray(freq, np.float64).reshape(blocks) / RATE
    p = r * np.exp(1j * turn)                       # one pole per block

    # `p**k` and `1/p**k` WITHOUT A COMPLEX POWER OR A COMPLEX DIVISION, both of
    # which are slow enough to matter at 24,000 samples a tick.  The magnitude
    # `r**k` depends only on the width, so it is one shared vector; the angle is
    # the only thing that differs block to block.
    k = np.arange(block)
    shrink = r ** k
    ang = turn[:, None] * k
    swing, rise = np.cos(ang), np.sin(ang)
    up = shrink * (swing + 1j * rise)
    down = (swing - 1j * rise) / shrink             # 1/up, exactly
    zero_state = up * np.cumsum(xb * down, axis=1)

    # THE STATE IS THE ONLY THING THAT CANNOT BE VECTORISED, and it is one
    # number per block: where the filter had got to when the block ended.
    ends = np.empty(blocks, np.complex128)
    over = p ** block
    # WHERE THE FILTER WAS when this call began: zero for a word rendered
    # whole (her mother), and for HER the state the last tick ended in ---
    # her mouth used to restart every resonator ninety times a second.
    carry = 0.0 + 0.0j if start is None else complex(start)
    first = carry
    for b in range(blocks):
        carry = zero_state[b, -1] + over[b] * carry
        ends[b] = carry
    came_in = np.concatenate(([first], ends[:-1]))

    y = zero_state + (up * p[:, None]) * came_in[:, None]
    # 1/((1-pz)(1-p'z)) = A/(1-pz) + conj, with A = p / (p - p') = p / 2j*Im(p)
    weight = p / (2j * np.imag(p))

    # **UNIT PEAK GAIN, WHICHEVER FREQUENCY IT IS TUNED TO.**  This was `1 - r`,
    # which is only half the normalisation: the true peak of a two-pole
    # resonator is `1 / ((1-r) * |1 - r*exp(-2j*w0)|)`, and that second factor
    # runs from about 0.018 at the bottom of her range to 2.0 at the top.  A
    # HUNDREDFOLD tilt in favour of low formants, which is why a low first
    # formant drowned everything else and why `front` and `nasal` measured dead
    # --- and, once the formants were cascaded, why her loudest sound came out
    # at 2.6 where a microphone saturates at 1.
    #
    # It is not a tuning constant.  It is the height of the peak, divided out.
    height = (1.0 - r) * np.abs(1.0 - r * np.exp(-2j * turn))
    out = (2.0 * np.real(weight[:, None] * y) * height[:, None]).ravel()
    return out if start is None else (out, ends[-1] if blocks else carry)


def voiced(f0, f1, f2, f3, amp, nasal, close=None, hiss=None,
           block: int = BLOCK, state=None):
    """A voice, as raw microphone samples.  **ONE MOUTH, SHARED.**

    Every argument is one value per sample: how fast the folds buzz, the three
    formants, how hard she is blowing, and how much of it goes out through the
    nose.  Her mother reads them off a table of English sounds; she orders them
    herself, five lines over 64 slides.  Nothing below can tell which.

    Not normalised and not faded --- those are things a WORD has, and this is
    not necessarily a word.
    """
    want = min(len(f0), len(f1), len(f2), len(f3), len(amp), len(nasal))
    n = (want // block) * block
    if n <= 0:
        return np.zeros(0, np.float32)
    nose = np.asarray(nasal, np.float64)[:n]
    # A MOUTH WITH NO CONSONANTS IN IT is what these two default to, so every
    # caller that predates them --- her mother's word table, every test ---
    # gets exactly the sound it always got.
    shut = (np.zeros(n) if close is None
            else np.clip(np.asarray(close, np.float64)[:n], 0.0, 1.0))
    air = (np.zeros(n) if hiss is None
           else np.clip(np.asarray(hiss, np.float64)[:n], 0.0, 1.0))

    def per_block(v) -> np.ndarray:
        return np.asarray(v, np.float64)[:n].reshape(-1, block).mean(axis=1)

    # **A CASCADE, NOT A SUM.**  A vocal tract is one tube and the source passes
    # through all of it, so the formants MULTIPLY.  These used to be filtered
    # separately and added at 1 / 0.55 / 0.25 --- the other standard way to build
    # a formant synthesiser, and one whose levels have to be tuned by hand.  The
    # hand-tuning was wrong in a way nobody could hear until she had to USE it.
    # Measured on the summed version (`py -m measure.mouth`):
    #
    #     sweeping her tongue across its WHOLE range moved the sound by 0.035
    #     sweeping her nose across its whole range moved it by 0.000, every step
    #
    # A low first formant rings loudest in this filter, so F1 drowned the other
    # two and `front` and `nasal` were dials she could turn for a lifetime
    # without hearing anything happen.  In a cascade each formant shapes what
    # the one before it made, which is what a tube does, and it is what makes an
    # /i/ an /i/.
    # HER SOURCE IS NOT ALWAYS HER FOLDS.  A fricative is turbulence at a
    # constriction shaped by the tract in front of it, so the noise goes through
    # the SAME cascade the buzz does --- that is what makes an /s/ an /s/ and
    # not a hiss with a vowel painted on it.  `hiss` crossfades the two, so she
    # can voice, whisper, or anything between, on one line.
    #
    # THE NOISE IS THE SAME NOISE EVERY TIME, on purpose.  Her body reproduces
    # exactly on a re-run --- her eye is a few thousand dot products for that
    # reason --- and a mouth seeded off the clock would make a replayed tape
    # disagree with the life it came from.
    # **NO TURBULENCE WITHOUT A CONSTRICTION.**  The comment above says what a
    # fricative IS --- *"turbulence at a constriction shaped by the tract in
    # front of it"* --- and then the noise was mixed in from `hiss` alone, so
    # she could hiss with a wide-open mouth.  A tract cannot: air only roughens
    # where it is squeezed.
    #
    # It matters because she BABBLES `hiss`, uniformly over 0..1, so on average
    # half her source was white noise.  Measured 2026-08-23 against his real
    # voice, energy by band:
    #
    #                0-300 Hz  300-800  800-2000  2000-4000  4000-8000
    #        him       41.8%     36.7%     11.4%       6.8%       3.2%
    #        her        8.5%      5.0%     59.0%       7.4%      20.2%
    #
    # A fifth of everything she made was hiss.  Her missing bottom end is her
    # being an infant and is not a fault; that 20% at the top is a mouth doing
    # something a mouth cannot do.
    #
    # `shut` is her constriction and it is already computed two lines up --- the
    # gate is her own articulator, not a number anyone picked, and a caller that
    # passes no `close` gets `shut = 0` and therefore no noise, which is exactly
    # what a mouth with no consonants in it should sound like.
    # HER MOUTH IS CONTINUOUS.  With `state` (a dict, hers) every part of
    # this picks up where the last tick left it and hands its end back;
    # without it (her mother, a whole word in one call) nothing changes.
    S = state if state is not None else {}
    f0a = np.asarray(f0, np.float64)[:n]
    source = _glottis(f0a, float(S.get("phase", 0.0)))
    S["phase"] = float((S.get("phase", 0.0) + f0a.sum() / RATE) % 1.0)
    rng = S.get("rng") or np.random.default_rng(0)
    S["rng"] = rng
    # NOISE AT HER THROAT AS WELL AS AT HER TEETH.  His word, 2026-09-11, after
    # hearing the two side by side.  A real mouth makes air-sound two ways: at a
    # pinch in the tract (sss, shh, fff --- which needs the pinch, and is what
    # this line built) and AT THE FOLDS THEMSELVES, air rushing through an open
    # glottis with the mouth wide open --- which is /h/, and which is BREATHING.
    # Gated by `shut`, she had only the first: opening her mouth took the noise
    # away and gave her back the buzz, so blowing hard with an open mouth came
    # out as a vowel.  That is the "a" he heard in her first exhale, and it is
    # almost certainly why the lungs of 2026-09-04 "sounded bad" --- the breath
    # was never air, it was a low drone.  Measured 2026-09-11 on the same pose,
    # mouth open at 0.85 and blowing full: 0.239 peak gated, 0.585 ungated.
    # It also gives her the unvoiced consonants she has never had --- every /s/,
    # /f/ and /h/ of hers has had her folds humming underneath it.
    # `hiss` now says plainly what share of her source is air rather than voice,
    # and `close` goes back to being one thing: how pinched her tract is.
    air = np.maximum(air * shut, air)
    if air.any():
        rough = rng.standard_normal(n)
        source = (1.0 - air) * source + air * rough
    def res(key, x, freq, wide):
        if state is None:
            return _resonate(x, freq, wide, block)
        y, S[key] = _resonate(x, freq, wide, block, start=S.get(key, 0j))
        return y
    out = res("f1", source, per_block(f1), 90.0)
    out = res("f2", out, per_block(f2), 110.0)
    out = res("f3", out, per_block(f3), 160.0)
    out = out * np.asarray(amp, np.float64)[:n] * CASCADE_GAIN

    # LIPS.  Sound leaving a mouth is radiated, not piped, and radiation lifts
    # it by 6 dB an octave -- a first difference, which is the standard model.
    # Without it a glottal source falls at 6 dB an octave unopposed and the
    # upper formants are buried: measured, "mommy" came out entirely below
    # 700 Hz and the /i/ at the end, whose F2 is 2250, was simply not there.
    # A word with no F2 is not a word.
    # ...AND A SHUT MOUTH DOES NOT LET IT OUT.  `close` gates the oral branch
    # BEFORE lip radiation and not after, which is the whole of how she gets a
    # stop: shutting stops the flow, and releasing it steps the pressure back up
    # --- and lip radiation is a first difference, so a step becomes a
    # TRANSIENT.  That burst is not modelled anywhere here; it is what a
    # derivative does to an edge, which is also what it is in a real mouth.
    #
    # The nose is deliberately NOT gated.  A sealed mouth with an open nose is
    # exactly /m/, and she has had `nasal` all along with nothing to seal.
    out = out * (1.0 - shut)
    lips = np.diff(out, prepend=float(S.get("lip", 0.0)))
    S["lip"] = float(out[-1]) if n else 0.0
    # THE FRONT CAVITY: her fricatives and her bursts (see TURB_OVER_F2).
    squeeze = 4.0 * shut * (1.0 - shut)             # flow and a squeeze, silent sealed
    turb = np.clip(np.asarray(hiss, np.float64)[:n] if hiss is not None else np.zeros(n), 0.0, 1.0) * squeeze
    release = np.clip(-np.diff(shut, prepend=shut[0] if n else 0.0), 0.0, 1.0)
    if turb.any() or release.any():
        rough = rng.standard_normal(n)
        if release.any():
            k = int(BURST_S * RATE)
            env = np.exp(-np.arange(k) / max(1.0, BURST_S * RATE / 3.0))
            turb = turb + np.convolve(release, env)[:n]
        cavity = np.clip(TURB_OVER_F2 * per_block(f2), 0.0, RATE * 0.45)
        hissed = res("cavity", rough * turb, cavity, TURB_WIDE)
        hissed = hissed * np.asarray(amp, np.float64)[:n] * TURB_GAIN
        lips = lips + np.diff(hissed, prepend=float(S.get("hisslip", 0.0)))
        S["hisslip"] = float(hissed[-1]) if n else 0.0
    if not nose.any():
        out_ = lips.astype(np.float32)
        return out_ if state is None else (out_, S)

    # ...AND A SHUT MOUTH DOES NOT RADIATE FROM THE LIPS AT ALL.  The sound
    # leaves through the nose, and **A NOSE IS A FIXED TUBE** --- that is the
    # whole reason `nasal` is one line rather than a set of formants: she cannot
    # reshape it, so it always sounds the same, a low murmur around 280 Hz with
    # no treble lift behind it.  Which is what /m/ is.
    #
    # Two earlier versions of this did nothing measurable.  The first put a
    # second resonator at the SAME frequency as the first, only narrower, so a
    # nasal was an oral sound with a slightly different F1.  The second dropped
    # lip radiation and kept everything else, which is only 6 dB an octave.
    # Measured (`py -m measure.mouth`): 0.000 and 0.056 across her whole range.
    # ONE resonator, so it does NOT get `CASCADE_GAIN` --- that number exists to
    # undo the attenuation of three in a row, and handing it to a single one
    # made a nasal 174 to 668 times LOUDER than the vowel beside it.  Which the
    # sweep reported as a working line: `nasal` moved the sound 0.688, all of it
    # between 0.000 and 0.125, because the murmur simply drowned the mouth.
    # A line that is really a switch measures almost the same as a line that
    # works, and only the LEVEL column told them apart.
    murmur = res("nose", source, np.full(n // block, NOSE_HZ), NOSE_WIDE)
    murmur = murmur * np.asarray(amp, np.float64)[:n] * NOSE_QUIET
    out_ = ((1.0 - nose) * lips + nose * murmur).astype(np.float32)
    return out_ if state is None else (out_, S)


#: A STOP: sealed for this fraction of its segment, then released with this
#: much hiss.  Searched 2026-09-04 over sealed 0.4/0.7 x hiss 0.3/0.8 through
#: her own articulators, judged at 250 and 320 Hz: 0.4 / 0.8 scored 0.85 mean
#: against 0.29 for the dips it replaced.
STOP_SEALED = 0.4
STOP_HISS = 0.8


def tracks(name: str, pitch: float = PITCH,
           seconds: float | None = None) -> np.ndarray:
    """A word, as the eight numbers a mouth is given: `(8, samples)`.

    `f0, f1, f2, f3, amp, nasal, close, hiss` --- exactly what `voiced` takes
    (2026-09-04: a table entry may carry a seventh field, a stop, which becomes
    a closure and a burst), and exactly
    what HER five articulator lines mean.  Pulled out of `word()` so that
    driving her mouth with her mother's word is a change of units and nothing
    else: if this returned one thing and her mouth were fed another, every
    comparison between them would be measuring the difference between two
    drivers rather than between two mouths.  That mistake was made once and cost
    an entire measurement --- her word stretched over a 1.5 s tick against her
    mother's 0.63 s one, scored as if they were the same sound.

    `seconds` pads (or clips) to a fixed length, so a word can be placed in a
    tick the same way for both of them.
    """
    parts = SOUNDS[name]
    said = int(RATE * sum(p[0] for p in parts))
    n = said if seconds is None else int(round(RATE * seconds))
    out = np.zeros((8, n), np.float64)
    # walk the formants smoothly from one sound into the next: the TRANSITION
    # is the cue, not the steady middle
    at = 0
    for k, part in enumerate(parts):
        secs, a, b, c, loud, nose = part[:6]
        stop = bool(part[6]) if len(part) > 6 else False
        m = min(int(RATE * secs), max(0, n - at))
        if m <= 0:
            break
        nxt = parts[min(k + 1, len(parts) - 1)]
        w = np.linspace(0.0, 1.0, m) ** 2
        for row, here, there in ((1, a, nxt[1]), (2, b, nxt[2]),
                                 (3, c, nxt[3]), (4, loud, nxt[4])):
            out[row, at:at + m] = here + (there - here) * w
        out[5, at:at + m] = 1.0 if nose else 0.0
        if stop:
            sealed = int(STOP_SEALED * m)
            out[6, at:at + sealed] = 1.0
            out[7, at + sealed:at + m] = STOP_HISS
        at += m

    # A REAL VOICE FALLS ACROSS A WORD.  A flat pitch is the single thing that
    # makes synthetic speech sound synthetic.  It belongs to the speaking, not
    # to the mouth, which is why it is here and not in `voiced`.
    out[0] = pitch * (1.0 - 0.12 * np.clip(np.arange(n) / max(said - 1, 1), 0, 1))
    return out


def word(name: str, pitch: float = PITCH) -> np.ndarray:
    """
    DEAD 2026-08-25 --- no callers.  It synthesises a NAMED word, which is her
    mother's mouth and not hers: her brain commands a similarity index and the
    table turns it into articulators.  Kept because `body/mother.py` will need
    it the day a mother is connected, and never before.
A spoken word, as raw microphone samples."""
    out = np.asarray(voiced(*tracks(name, pitch)), np.float32)
    n = out.size

    edge = int(RATE * 0.012)                       # no clicks at either end
    out[:edge] *= np.linspace(0, 1, edge)
    out[-edge:] *= np.linspace(1, 0, edge)
    peak = float(np.abs(out).max()) or 1.0
    return (out / peak * 0.7).astype(np.float32)
