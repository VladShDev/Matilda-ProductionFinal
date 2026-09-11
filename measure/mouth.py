"""HIS RECORDINGS INTO HER MOUTH --- the one place a sound piece is cut.

    python -m measure.mouth                          # rebuild lives/mouth.npz
    python -m measure.mouth --say WORD.wav OUT.wav   # ...and say a word with it

His three steps, 2026-08-31, verbatim: *"You grip my voice from the mic.  Then
you align all that track with all prerecorded pieces what we have now already
...  Then all these pieces that contain that track You store in her ...  And
then when she see in experience those ticks, you need just go by them and
reproduce them."*  This file is step one: the pieces, cut and filed so steps
two and three can be a lookup.

**EVERY PIECE HAS ITS OWN ID; HER EAR'S NAME IS ONLY THE INDEX.**  His, the
same night: *"Each of prelearned sound sounds also has to feel just one tick
and has his own ID."*  It was built the other way first --- one piece per name
--- and he heard it immediately: *"it's again half of my word"*.

MEASURED, on his certified word, the night it was found:

    442 moments of his voice collapse into 52 names
    inside ONE name, the two most unlike moments scored 0.000 --- nothing
    in common: one name, two different sounds

    one piece per name    11 of 39 ticks were the sound he actually made
    every piece its own   38 of 40                       (the ceiling is 38)
    his whole word        3.47 away before, 1.29 after

So filing one piece per name threw the other 441 away and she said the right
sequence of WRONG sounds.  That is the half-word he heard, and it is exactly
what he named while it was being measured: *"maybe when you make offset, you
lose that part which is outside of track"*.

**THE NAME STAYS HER MIND'S WORD FOR IT.**  `alike_of` is what her ear
quantises to, what `align` answers in, and what her mind stores, suggests and
commands --- and her similarity law reads that id ARITHMETICALLY
(`expect.near`), so a piece id in that channel would silently mean nothing.
The name indexes; the piece sounds.  His older rule holds untouched --- *cut
by lookup, not by scan*: the name is the indexed probe (free) and the choice
inside it is ~15 dot products, measured at 0.211 ms against her 33 ms tick.

**A PIECE IS FILED UNDER THE SOUND HER EAR NAMES, NOT UNDER THE SOUND IT
MAKES.**  Her ear names 33 ms of arriving air, raw, at his own pitch; her
mouth speaks in her register.  A bank keyed by its own shifted audio lives in
a DIFFERENT index space from the one her ear produces, so his voice could
never select a piece --- measured 2026-08-31: not one id of his word played
into her room matched an id computed for it offline, and nothing raised,
because an id nothing matches is simply a quiet mouth.  After: 86% of the ids
his live voice made were pieces her mouth has.

    name  = alike_of(bands(33 ms of him, RAW))   what her ear calls it
    ear   = that frame's unit shape              how she finds the piece in it
    clip  = that same moment x1.75               what her mouth says instead

So the table IS the translation from what she heard to how she says it, which
is what a baby's mouth is.

**THE CLIP COVERS 58 ms OF HIM AND LASTS 33 ms OF HER.**  x1.75 raises pitch,
formants and pace together --- the certified mouth, not re-opened here.  The
clip begins where her ear's frame began and runs past it: consecutive pieces
overlap in his time by 25 ms, and that overlap is what raises her register
without stealing time from the word.  The alternative --- cutting 33 ms and
stretching it back to length --- was built and measured the same day: a phase
vocoder smeared her (flatness 0.050 against her certified 0.019) and a
waveform stretcher made noise (0.377).

WHY IT MUST BE HER EAR'S RATE AND NOT HIS.  A piece covering 58 ms of him
placed every 33 ms says the word 1.75x too slowly --- his 1.38 s word came out
1.47 s and the syllables re-entered each other.  He heard it at once: *"it
sounds like she tell few times Matilda, but you cut piece where the end of
first word and then start another Matilda but not till the end"*.  Her ear is
a draining queue (`window.speech`), so it cannot advance faster than he
speaks: what her ear advanced in a tick is what her mouth must cover in a
tick.

**NOTHING IS LOUDNESS-NORMALISED.**  Measured 2026-08-31, and it was mine:
flattening every piece to one RMS turns his syllable envelope into a 30 Hz
square wave --- the click he kept hearing.  With normalisation the modulation
at her tick rate was 9.2%, without it 3.5%, his own voice 2.0%.  A piece keeps
the loudness it was said with; her `lvl` scales it at speaking time.
"""
from __future__ import annotations

import os
import sys
import wave

import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                        # noqa: E402
from body.alike import FLOOR, SILENCE, _unit, alike_of          # noqa: E402
from body.hearing import (REGISTER, SOUND_HOPS, TICK_SECONDS, toHer,  # noqa: E402
                          bands_from_pcm)

#: HER REGISTER.  The certified shift --- his voice into hers, pitch, formants
#: and pace together.  1.75 was frozen by his ear on 2026-08-29; on 2026-09-04
#: he chose a toddler's tract (1.3 x an adult's, `body/muscles.py`) so that her
#: word can be heard by a recogniser, and his voice enters her register by
#: the same factor.  The bank is re-cut and the guardian re-certified with it.
#: THE REGISTER IS THE EAR'S (body/hearing.REGISTER, his 1.75): the cutter has none of
#: its own since 2026-09-04 --- it files his pieces under what her ear makes of them.


#: too quiet to be a sound he made --- the same bar the mom's capture draws
QUIET = 0.04
#: two moments are ONE piece above this: she cannot tell them apart, so a
#: second copy is only weight in the bucket she has to search
SAME = 0.995
#: his certified recordings
SOURCES = ("measured/confirmed_words_2026-08-29.wav",)


def _load(path: str) -> np.ndarray:
    """A wav off disk at her rate, mono, -1..1."""
    with wave.open(path, "rb") as h:
        rate, n = h.getframerate(), h.getnframes()
        y = np.frombuffer(h.readframes(n), "<i2").astype(np.float32) / 32768.0
    if rate != speech.RATE:
        i = np.arange(0.0, len(y) - 1, rate / float(speech.RATE))
        y = np.interp(i, np.arange(len(y)), y).astype(np.float32)
    return y


def _hers(chunk: np.ndarray, want: int) -> np.ndarray:
    """His moment in her register: x1.75, cut to one tick of her."""
    # EXACTLY `step` samples of hers from the grab: `arange(0, len-1, REGISTER)`
    # gave 178 at REGISTER 1.75 by luck and 177 at 1.3, one short of a tick, so
    # every piece was dropped as too short (2026-09-04).
    return toHer(chunk, want)


def heardAs(frame: np.ndarray):
    """One arriving tick -> `(her ear's name for it, its shape)`.

    The ONE way a sound is turned into something she can look up, used when
    the pieces are cut and again when one is chosen.  `(None, None)` when
    there is nothing there.
    """
    pic = bands_from_pcm(np.asarray(frame, np.float32), speech.RATE,
                         TICK_SECONDS, SOUND_HOPS)
    if pic.max() <= 0:
        return None, None
    got = _unit(pic).mean(axis=1)
    size = float(np.linalg.norm(got))
    if size <= 0.0:
        return None, None
    return int(alike_of(pic)), (got / size).astype(np.float32)


def cut(sources=SOURCES) -> tuple:
    """His recordings -> `(pieces, rows, ears)`.  Her whole mouth.

        pieces   piece id -> one tick of her, ready to play
        rows     her ear's name -> the piece ids that wear it
        ears     piece id -> the shape her ear heard, unit length
    """
    step = int(round(speech.RATE * TICK_SECONDS))
    grab = int(math.ceil(step * REGISTER)) + 1        # enough of his for a whole tick of hers
    pieces: list = []
    ears: list = []
    rows: dict = {}
    for path in sources:
        raw = _load(path)
        # QUARTER-TICK STEPS, BECAUSE WHAT FALLS BETWEEN THEM IS LOST ---
        # his, the night it was found.  The duplicate test throws away only
        # what she can already say, so a finer step costs nothing but time.
        for i in range(0, len(raw) - grab, step // 4):
            window = raw[i:i + step]
            if float(np.abs(window).max()) < QUIET:
                continue
            name, heard = heardAs(window)
            if name is None or name in (SILENCE, FLOOR):
                continue
            clip = _hers(raw[i:i + grab], step)
            if len(clip) < step:
                continue
            kin = rows.setdefault(name, [])
            if any(float(np.dot(heard, ears[j])) > SAME for j in kin):
                continue
            at = len(pieces)
            kin.append(at)
            pieces.append(clip)
            ears.append(heard)
            # ...AND UNDER THE NAME IT HAS WHEN SHE SAYS IT.  A piece is
            # filed by the sound HER EAR GAVE HIS VOICE, but the piece itself
            # is her register --- so one of her own sounds coming back through
            # the air was a stranger.  Measured 2026-09-01 by the deaf-ear
            # test (`measure.hears`): her own sounds came back as themselves
            # 1 time in 16, where her old synthesized voice managed 84 of 93.
            #
            # It is ONE piece with TWO names: what she heard, and what she
            # sounds like saying it.  Nothing is duplicated and nothing is
            # averaged --- her mouth can be reached from either side of the
            # same moment, which is what makes parroting possible at all.
            mine, _ = heardAs(clip)
            if mine is not None and mine not in (SILENCE, FLOOR) and mine != name:
                rows.setdefault(mine, []).append(at)
    return pieces, rows, ears


def pick(rows, ears, frame: np.ndarray):
    """The piece she heard, or None --- her ear's name, then inside it.

    Two lookups and no scan: the name is one quantise, and the choice is a
    dot product against the pieces already wearing that name.
    """
    name, heard = heardAs(frame)
    if name is None:
        return None, None
    kin = rows.get(name)
    if not kin:
        return name, None
    got = np.stack([ears[j] for j in kin]) @ heard
    return name, kin[int(np.argmax(got))]


def save(pieces, rows, ears, path: str = "lives/mouth.npz") -> None:
    """Her mouth to disk, in the shape `muscles.Voice` loads.

    `ids` are the PIECE ids; `row` is the name each piece wears and `ear` the
    shape her ear heard it as, so her body can choose inside a name without
    re-cutting anything.  A bank written before this has neither and still
    loads --- one piece per name, which is what she had.
    """
    # A PIECE IS FILED UNDER WHAT IT ACTUALLY MAKES.  His, 2026-09-02: *"file
    # it under what it actually makes"*.
    #
    # `cut` gives every piece TWO names --- what her ear called HIS voice, and
    # what it sounds like when SHE says it --- and this wrote ONE name per
    # piece, so the second was overwritten and **which one survived depended
    # on dict ordering**.  Measured 2026-09-02: 12 of her pieces played into
    # her own room came back under their filed name **3 times**, and the three
    # were the ones where the survivor happened to be the right one.  Asking
    # her to say 154 handed her a piece that sounds like 220.
    #
    # So the PRIMARY name is now the clip's own sound, computed here from the
    # clip itself --- the same `heardAs` her ear uses, on the audio she will
    # actually make.  Nothing is re-cut and no signature moves.
    row = np.zeros(len(pieces), np.int64)
    for j, clip in enumerate(pieces):
        mine, _ = heardAs(clip)
        row[j] = (int(mine) if mine is not None
                  and mine not in (SILENCE, FLOOR) else 0)
    # ...AND EVERY NAME IT CAN BE REACHED BY SURVIVES, as pairs, because the
    # heard side is what makes parroting possible: her mouth is reached from
    # either side of the same moment, and one array per piece could never
    # hold that.
    pairName, pairPiece = [], []
    for name, kin in rows.items():
        for j in kin:
            pairName.append(int(name))
            pairPiece.append(int(j))
            if row[j] == 0:            # its own sound was silence: keep the
                row[j] = int(name)     # heard name rather than lose the piece
    np.savez(path, ids=np.arange(len(pieces), dtype=np.int64), row=row,
             ear=np.stack(ears).astype(np.float32),
             pairName=np.asarray(pairName, np.int64),
             pairPiece=np.asarray(pairPiece, np.int64),
             **{("clip_%d" % i): c for i, c in enumerate(pieces)})


def say(pieces, rows, ears, word: np.ndarray) -> tuple:
    """A heard word -> the pieces she would say, and the air they make.

    Steps two and three of his three, run offline: her ear names each tick,
    the nearest piece inside that name is the one she heard, and her mouth
    plays them in that order.  The seam is her mouth's own 2 ms ramp
    (`muscles.Voice.FADE`) --- measured 2026-08-31, longer crossfades make the
    modulation WORSE, not better: 3.5% at 2 ms, 5.7% at 8, 7.6% at 16.
    """
    step = int(round(speech.RATE * TICK_SECONDS))
    got = []
    for k in range(len(word) // step):
        _, one = pick(rows, ears, word[k * step:(k + 1) * step])
        got.append(one)
    ramp = int(round(speech.RATE * 0.002))
    out = np.zeros(len(got) * step, np.float32)
    last = np.float32(0.0)
    for k, one in enumerate(got):
        ch = np.zeros(step, np.float32) if one is None else pieces[one].copy()
        up = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        ch[:ramp] = ch[:ramp] * up + last * (1.0 - up)
        last = ch[-1]
        out[k * step:(k + 1) * step] = ch
    return got, out


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    pieces, rows, ears = cut()
    step = int(round(speech.RATE * TICK_SECONDS))
    print("her mouth: %d pieces in %d names (%.1f a name), each %d samples"
          % (len(pieces), len(rows), len(pieces) / max(1, len(rows)), step))
    if "--say" in sys.argv:
        at = sys.argv.index("--say")
        got, out = say(pieces, rows, ears, _load(sys.argv[at + 1]))
        out = out / (float(np.abs(out).max()) or 1.0) * 0.9
        with wave.open(sys.argv[at + 2], "wb") as h:
            h.setnchannels(1)
            h.setsampwidth(2)
            h.setframerate(int(speech.RATE))
            h.writeframes((out * 32767).astype("<i2").tobytes())
        print("said %d of %d ticks, %.2f s -> %s"
              % (sum(1 for i in got if i is not None), len(got),
                 len(out) / speech.RATE, sys.argv[at + 2]))
        return 0
    save(pieces, rows, ears)
    print("written lives/mouth.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
