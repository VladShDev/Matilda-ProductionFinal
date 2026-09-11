"""THE MARK ON HER WALLS AND HER CEILING, as a bitmap her eye can read.

The owner, twice.  First: *"add our logo --- project Matilda - AGI --- on all
walls."*  Then, when the first pass came back as tasteful signage: *"it has to
be visible in her view, partially.  It was for her EXPLORATION.  So those words
have to almost fill each wall, stretched almost for full width and length, but
without geometry / words broken."*

WHY IT IS IN HER EYE AND NOT ONLY ON THE SPECTATOR'S PAGE.  `light.one_look`
renders six flat planes with one repeating checker, the bed slab, whatever is
in `room.things` --- and her room is empty --- and the window.  That is her
whole world.  Her cot's bars are not in it, and the checker is IDENTICAL
EVERYWHERE: it gives her edges and optic flow, and it cannot tell her WHICH
part of a wall she is looking at or how far she turned, because every square is
the same square.  A letterform can.  His words: *"the room for her is nothing,
she doesn't see it at all, but she tries."*

MEASURED, on her real eye in her cot (`measured/a_room_she_can_tell_apart.md`):
a 20 degree head turn changed her picture by 0.0779 per cell without it and
**0.1371 with it --- 1.76x**.  That is the number this exists for.  Her piece
count goes 22 -> 34 at rest and her DISTINCT `alike` kinds only 6 -> 7, which
is the honest half and is written up there.

WHY THE LETTERS ARE DRAWN HERE AND NOT FETCHED.  `body/` has numpy and nothing
else, and it is not going to grow a font dependency for this.  The page draws
the same words from its own font stack; these are not the same glyphs and
cannot be, and what is matched is the FIT --- contained at 96%, centred, its
own ratio kept, clear of both corners --- which is the shape he approved.

WHY IT IS NEUTRAL.  `light.COLOURS` keeps her room in neutrals on purpose, so
that a colour means exactly one thing: something is THERE.  Ink with a hue in
it would make every wall a coloured object, which is the one thing that note
forbids.  It is painted, not tinted: a covering ink measured 3.5x bigger pieces
than a multiplying one at 95% of the benefit.
"""

from __future__ import annotations

import numpy as np

#: 5 wide, 8 tall.  Row 6 is the baseline, row 7 is descender space; capitals
#: use rows 0..6 and lowercase 2..6, so a word sits on one line.
GLYPHS = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "M": ("10001", "11011", "10101", "10001", "10001", "10001", "10001", "00000"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001", "00000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110", "00000"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111", "00000"),
    "p": ("00000", "00000", "11110", "10001", "10001", "11110", "10000", "10000"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000", "00000"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110", "00000"),
    "j": ("00010", "00000", "00010", "00010", "00010", "00010", "00010", "01100"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110", "00000"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110", "00000"),
    "t": ("00000", "00100", "11111", "00100", "00100", "00100", "00011", "00000"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111", "00000"),
    "i": ("00100", "00000", "00100", "00100", "00100", "00100", "00100", "00000"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "00110", "00000"),
    "d": ("00001", "00001", "01111", "10001", "10001", "10001", "01111", "00000"),
}
CELL_W = 5

#: The page's layout: the small words are 0.385 of the big one and tracked at
#: 0.36 of their own size.  4 against 10 is 0.400 --- the nearest whole number
#: of pixels to 0.385 that leaves the small type readable.
BIG, SMALL = 10, 4
#: How much of a surface the ink may take.  A word running off an edge is a
#: broken word and a word wrapping a corner is two broken words, so it stops
#: short of both: on a 20 m wall this leaves 0.4 m clear at each end.
FILL = 0.96
#: What the ink is.  Darker than every surface it lands on --- ceiling 0.79,
#: wall 0.60 --- and grey, for the reason in the module docstring.
INK = 0.40
#: Which of `light._SIDES` wear it, in that tuple's own order: NOT the floor,
#: which is under her, and which the spectator's page does not mark either.
WEARS = (False, True, True, True, True, True)


def _put(sheet, ch, x, y, s):
    rows = GLYPHS.get(ch)
    if rows is None:
        return
    for r, bits in enumerate(rows):
        for c, b in enumerate(bits):
            if b == "1":
                sheet[y + (r - 6) * s:y + (r - 5) * s,
                      x + c * s:x + (c + 1) * s] = 1.0


def _word(sheet, text, x, y, s, gap):
    for ch in text:
        _put(sheet, ch, x, y, s)
        x += CELL_W * s + gap
    return x


def _ink() -> np.ndarray:
    """The words, cropped to their own ink box --- which is the only honest
    source for how wide this mark is against how tall."""
    sheet = np.zeros((200, 900), np.float32)
    x, y = 20, 120
    x = _word(sheet, "project", x, y, SMALL, max(1, int(SMALL * 0.36)))
    x += int(BIG * 0.30)
    x = _word(sheet, "Matilda", x, y, BIG, 1)
    x += int(BIG * 0.34)
    _word(sheet, "- AGI", x, y, SMALL, max(1, int(SMALL * 0.36)))
    rows = np.nonzero(sheet.any(axis=1))[0]
    cols = np.nonzero(sheet.any(axis=0))[0]
    return sheet[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _tile(wall: float = 8.0):
    """CONTAIN, at `FILL`: the ink grows until it meets whichever edge stops it
    first, and keeps its own shape.  Stretching a letter is the one thing he
    ruled out.  `wall` is the ratio of the tile it is centred in."""
    ink = _ink()
    ih, iw = ink.shape
    th = int(round(ih / FILL))
    tw = int(round(th * wall))
    s = min(tw * FILL / iw, th * FILL / ih)
    w, h = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    out = np.zeros((th, tw), np.float32)
    # nearest neighbour, because the ink is binary and this leaves it binary
    ys = np.clip((np.arange(h) * ih / h).astype(np.int64), 0, ih - 1)
    xs = np.clip((np.arange(w) * iw / w).astype(np.int64), 0, iw - 1)
    out[(th - h) // 2:(th - h) // 2 + h,
        (tw - w) // 2:(tw - w) // 2 + w] = ink[ys][:, xs]
    return out, iw / ih


#: The sheet her eye samples, and the ink's OWN width over its own height ---
#: measured off the pixels, never declared, exactly as the page measures its.
SHEET, RATIO = _tile()
