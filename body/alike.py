"""THE SIMILARITY INDEX --- one integer that sounds alike share.

Copied from the old tree's `brain/utter.py`, and **it lives in her body now**.
Her brain never sees a waveform: her ears hand it `input.sound.id`, her mouth
hands back `input.echo.id`, and she commands `output.sound.id`.  All three are
this integer.  Nothing about a spectrum reaches her.

WHAT IT IS.  Three numbers off the shape of the sound, with loudness already
divided out --- because loudness is how HARD she said it, not what she said:

    centre   where the energy sits across her bands
    spread   how far it is smeared around that
    peak     how much of it is in its loudest band --- tonal against broad

quantised at `ALIKE_STEPS` each.  None of them survives a change of what was
said and all of them survive it being said louder, further away, **or by a
different mouth** --- which is the whole test, and is why his voice and hers
land in the same space and can be compared as integers.

**SILENCE IS A SOUND, AND IT IS ID 0.**  The owner, 2026-08-25: *"we also have
to have silence --- if she try to repeat silence with white noise it has to be
silence similarity"*.  It used to return `-1` with the note *"silence is not a
resemblance"*, and that made a silent moment fall out of everything: it could
not be stored, compared, commanded or repeated, so a gap inside a word had no id
and a word with a pause could not be held together.

Her structure already agrees --- `"sound": {"id": 0.0, "lvl": 0.0}` is the
default on both sides --- so **she is born commanding silence and hearing
silence**, and that is the right value rather than an absence.

    id 0          SILENCE            her mouth at rest
    id 1..512     a sound            1 + (centre, spread, peak)

White noise is broad with no peak and lands somewhere in 1..512 like any other
sound, so trying to repeat silence with noise gives a different id and does not
match.  That is the whole point of the change.
"""

from __future__ import annotations

import numpy as np

#: how finely each of the three is told apart.  The same number her sight has
#: used since `representations` was written --- one answer to "how alike".
ALIKE_STEPS = 8
#: A SOUND IS THE SHAPE OF THE MOUTH, NOT THE PITCH OF THE LARYNX (2026-09-04,
#: his "finish her voicing").  The three digits are read from band
#: SOUND_FLOOR_BAND up --- above 627 Hz --- so the fundamental (his at 133 Hz,
#: hers at 250-600) is not what a name is.  Measured before: his word held 22%
#: of its energy in one low bin, its names sat where her larynx cannot, and
#: her mouth reached 11 of its 26 names in 20,000 poses; the copy of his word
#: through names could not carry it.  With the floor at 6: 23 of 26 reached,
#: her mouth's alphabet 80 -> 138 names, and the vowel of a sound is told
#: across pitch (nearest pose by shape: open/front/nasal errors 0.03-0.10
#: against 0.3 by chance; scratchpad/earname.py, earnn.py).  One law for both
#: ears, the bank and her mother: everything named before this is another
#: alphabet, and the bank was re-cut.
SOUND_FLOOR_BAND = 6
#: ...so the whole space is this many sounds, plus silence at 0.  Measured
#: 2026-08-23: about 250 of them are reachable by her mouth, and it saturates
#: --- 107 in the first 500 babbles, 234 after 5,000, +3 in the last 500.
ALIKE_KINDS = ALIKE_STEPS ** 3
#: what she makes when she makes nothing
SILENCE = 0
#: ...AND WHAT THE WORLD GIVES WHEN IT GIVES NOTHING.  His, 2026-08-26: *"when
#: we make her prelearn file it has to contain id 1 silent flor"*.
#:
#: TWO DIFFERENT NOTHINGS, AND SHE HAS TO BE ABLE TO TELL THEM APART.  `SILENCE`
#: is HER mouth at rest --- she made nothing.  `FLOOR` is the ROOM at rest ---
#: nobody is speaking, and what arrives is the hiss that is always there.
#:
#: WITHOUT IT THE FLOOR WEARS A REAL SOUND'S NAME.  MEASURED on 55 s of his
#: actual room, cut into her frames:
#:
#:     the quietest third   id 283  51.9%   282  31.1%   284  5.5%   = 88.5%
#:     the whole recording  id 283  47.9%   282  23.0%   284  8.6%   = 79.5%
#:     silence (id 0)       132 of 12,034 frames = 1.1%
#:
#: ...and **her mouth can make 283**.  So four fifths of everything her ear named
#: from his room was room tone wearing the id of a sound she can produce --- she
#: could babble his room back at herself and be paid for it, a sound arrived on
#: 38 ticks in 40, and a word could never end because a word ends on a gap.
#:
#: IT IS A LOOKUP AND NOT A GATE.  `GATE` asks *is this quiet* and needs a
#: number; this asks *is this the thing that is always here*, which is the same
#: habituation her boredom already uses (`hormones._usual`) and has no number in
#: it at all.
FLOOR = 1


def one_ear(picture: np.ndarray) -> np.ndarray:
    """`(places, bands, slides)` -> `(bands, slides)`: the louder side wins.

    A sound is measured by its loudest reading, not by the mean of a head that
    happened to be turned.  Her mouth is one place and passes through.
    """
    got = np.asarray(picture, np.float32)
    return got.max(axis=0) if got.ndim == 3 else got


def _unit(picture: np.ndarray) -> np.ndarray:
    """Each slide as a direction: what frequencies were loud TOGETHER.

    Both sides are non-negative loudness per band, so the shape across bands is
    what a formant is.  A silent slide becomes zeros.
    """
    got = np.asarray(one_ear(picture), np.float64)
    size = np.linalg.norm(got, axis=0)
    return np.divide(got, size, out=np.zeros_like(got), where=size > 1e-12)


def _index_of(shapes: np.ndarray) -> np.ndarray:
    """`(n, bands)` of per-band shape -> `(n,)` indices in `1..ALIKE_KINDS`.

    `0` where there was no sound at all, which is silence.
    """
    got = np.atleast_2d(np.asarray(shapes, np.float64))
    if got.shape[1] > SOUND_FLOOR_BAND + 1:
        got = got[:, SOUND_FLOOR_BAND:]          # the envelope, not the pitch
    total = got.sum(axis=1)
    bands = got.shape[1]
    out = np.full(got.shape[0], SILENCE, np.int64)
    live = (total > 1e-9) & (bands >= 2)
    if not live.any():
        return out
    w = got[live] / total[live][:, None]
    at = np.arange(bands, dtype=np.float64)
    last = float(bands - 1)
    centre = (w * at).sum(axis=1) / last
    spread = np.sqrt((w * (at[None, :] - (centre * last)[:, None]) ** 2).sum(axis=1)) / last
    peak = w.max(axis=1)
    steps = int(ALIKE_STEPS)
    c = np.clip((centre * steps).astype(np.int64), 0, steps - 1)
    s = np.clip((np.minimum(spread * 2.0, 0.999) * steps).astype(np.int64), 0, steps - 1)
    k = np.clip((np.minimum(peak * 2.0, 0.999) * steps).astype(np.int64), 0, steps - 1)
    # ...AND ONE PAST SILENCE, so `0` is free to mean her mouth at rest.
    out[live] = 1 + (c * steps + s) * steps + k
    return out


def alike_of(sound: np.ndarray) -> int:
    """A `(bands, slides)` sound -> its index.  `0` is silence."""
    unit = _unit(sound)
    if unit is None or not np.size(unit):
        return SILENCE
    unit = np.asarray(unit, np.float64)
    shape = unit.mean(axis=1) if unit.ndim > 1 else unit
    return int(_index_of(shape[None, :])[0])
