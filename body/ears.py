"""HER EARS --- an arriving sound, cut into her own frames.

The owner, 2026-08-25: *"she has frame our tick `|...---___...-~~|` --- THAT
PART OF SOUND IS THE ONLY ONE WE CAN DISCUSS AT ALL, then this part becomes an
exp chain, so the smallest parts of sound become something meaningful"*, and
*"those pieces have to be FOUND IN the input sound ... but that has to happen
SIMULTANEOUSLY with the coming input sound, so she is able to reproduce then"*.

**THE TICK DOES THE CUTTING.**  One frame is one sound and it is the smallest
thing there is, so none of the old tree's cutting comes across: no
`cut_by_lookup`, no `nearest_part`, no `cut_into_parts`, no `match`.  There is
nothing to cut.

WHAT IS LEFT IS THE OFFSET, and it is real.  Her clock starts a frame when it
starts; he starts a word when he speaks.  A frame that lands across two of his
sounds averages them into an index that is NEITHER --- a smear that matches
nothing her mouth can make, so she could never answer it:

    his word          [ aaa ][ mmm ][ aaa ]
    her frames     |........|........|........|
                       ^ half of one sound and half of the next

    aligned        |......|......|......|
                     136     201     136          three she can actually use

SO THE OFFSET IS CHOSEN BY HER OWN SIMILARITY --- his words: *"an offset that
aligns the tick with her similarity"*.  Slide the frame and take the first
placement whose index is a sound **she can make**.  That is not a score anybody
invented: it is one integer being in a table of 92, asked at every offset at
once.

AND IT COSTS NOTHING, because `alike_of` is a mean over the frame and the mean
of every window is one prefix sum away.  The old tree measured the other shape
--- sliding her whole alphabet along his voice --- at **47 seconds a tick**
against a 1.5 second budget.
"""

from __future__ import annotations

import numpy as np

from .alike import ALIKE_KINDS, SILENCE, _index_of, _unit, level_of


def frames(heard: np.ndarray, slides: int):
    """EVERY PLACEMENT OF HER FRAME on an arriving sound, and how clean each is.

    `(bands, n)` in; out is `(indices, wholeness, level)`, one per placement.
    A running mean, so his length costs nothing.

    **WHOLENESS IS FREE.**  Each slide is already a unit direction, so the norm
    of a window's MEAN is 1.0 when every slide points the same way --- one sound
    --- and falls as the window straddles two.  It is the same arithmetic
    `alike_of` already does, read one line earlier, and it is what tells a whole
    sound from a smear of two.
    """
    unit = _unit(heard)
    if unit.size == 0 or unit.shape[1] < slides:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    bands = unit.shape[0]
    run = np.concatenate([np.zeros((bands, 1), np.float64),
                          np.cumsum(np.asarray(unit, np.float64), axis=1)],
                         axis=1)
    block = (run[:, slides:] - run[:, :-slides]) / float(slides)
    return _index_of(block.T), np.linalg.norm(block, axis=0), level_of(block.T)


def align(heard: np.ndarray, knows, slides: int) -> tuple:
    """WHERE HER FRAME SITS ON THIS SOUND, and what it is.  `(offset, index)`.

    Among the placements that ARE a sound she can make, the WHOLEST one wins ---
    the frame holding one sound rather than the end of one and the start of the
    next.  His words: *"an offset that aligns the tick with her similarity"*.

    MEASURED 2026-08-25, on frames deliberately landing across two of her own
    sounds, all 91 of them:

        first makeable placement      6 of 20   ->  scaled up, poor
        the WHOLEST makeable one     63 of 91   (69%)

    A blend often lands on a makeable index by accident, so being makeable alone
    cannot tell a whole sound from a smear.  The 28 that still miss are frames
    where the blend is itself wholer than either sound in it --- two of her
    sounds that are close together.

    When no placement is a sound of hers, the newest frame is used as it falls:
    she heard something and it is not one of hers, which is the truth and is
    itself worth an experience.
    """
    # THE NAME, AS A LEVEL.  The piece's own name divided by how many names
    # there are: a level in 0..1 like every other line of hers, the same sound
    # always the same level, and invertible --- a level times `ALIKE_KINDS` is a
    # name her mouth can reach for, which is what lets her say it back.
    #
    # The names are packed `1 + (centre*8 + spread)*8 + peak`
    # (`alike._index_of`), so the leading digit of this level is the spectral
    # centre and two sounds with the same centre land within an eighth of each
    # other.  The ordering is the shape's.
    #
    # NO RULER: the name comes from the piece itself.  `knows` is left in the
    # signature because her body still hands it in; nothing reads it, and her
    # alphabet is whatever her life puts on the line.
    got, whole, lvl = frames(heard, slides)
    if not got.size:
        return 0, SILENCE, 0.0
    best = int(np.argmax(whole))
    return got.size - 1 - best, int(got[best]), float(lvl[best])
