"""HER ALPHABET --- what her mouth can make, found the way she finds it.

    python -m measure.alphabet              # sweep, report, write measured/alphabet.npz
    python -m measure.alphabet --ticks 7200 # ...that many ticks a pass (default 7200)

TESTING ONLY.  His word, 2026-09-12: *"she will find them during her real life
by exp, but now we need that table to be able to test her without spending time
on real life ... we just have to make the right table with all her possibilities,
the mapping of her muscles, breathing and so, to recreate her real speech."*
Nothing in `body/` or `mind/` imports this, and nothing ever may.

HOW IT IS FOUND --- HER OWN LAW, `mind/discover.py`, and nothing else:
  ONE line at a time, one smallest step, up to its top, back down, then the next
  line starts at its own zero; every other line stands where she left it; her
  mouth CARRIES between ticks and is never reset (a table swept from a mouth at
  rest and played from a mouth in motion disagreed by 42 grains --- measured
  2026-09-12 --- and that was the root of every failed reproduction).  Seven
  passes, each entering the sweep on a different line, so `loud` --- her breath
  --- is not the first thing swept to zero in every pass.

HOW IT IS HEARD --- THE ONE GATE her body uses for every sound, his and hers:
  `align` on her ear bands, `SOUND_HOPS`, at `speech.RATE` (`alive.py`).  Each
  row is kept with her TWO sound lines: the similarity as a level, and how loud.
  Rows that make nothing are kept too --- the floor is a level like any other,
  and a gap needs no branch.

WHAT IS WRITTEN: `measured/alphabet.npz` with `rows` (N, 7) her seven
articulators (loud, pitch, open, front, nasal, close, hiss), `level` (N,) and
`loud` (N,) --- what her ear said of each row.  The step of this sweep is the
instrument's own choice, not her body's.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body.ears import align                                        # noqa: E402
from body.hearing import SOUND_BANDS, SOUND_SLIDES                 # noqa: E402
from body.muscles import Voice                                     # noqa: E402

#: the sweep's step --- fine enough that the table covers his voice (measured
#: 2026-09-12: 313 distinct levels, 94% of his ticks reachable)
STEP = 1.0 / 512.0
#: THE ROW THE SWEEP BEGINS FROM --- nothing more.  In her life she begins from
#: wherever she is; the instrument has to begin somewhere.  Not a pose she
#: returns to, not a gap between steps: her steps carry, one into the next.
START = np.array([0.6, 0.6, 0.5, 0.5, 0.1, 0.1, 0.1], np.float32)
PARTS = 7
OUT = os.path.join("measured", "alphabet.npz")


def heard(bands) -> tuple:
    """HER TWO SOUND LINES of one tick of air, by the one gate: `(level, loud)`.
    `alive.py:1627` and `:1676` read his voice and hers with exactly this."""
    b = np.asarray(bands, np.float32)
    if not b.size or float(b.max()) <= 0.0:
        return 0.0, 0.0
    off, _id, lev = align(b, None, SOUND_SLIDES)
    at = max(0, b.shape[1] - 1 - int(off))
    return float(lev), float(b[:, at].max())


def sweep(ticks: int = 7200, step: float = STEP) -> tuple:
    """`(rows, level, loud)` --- every row she lived in the sweep and what her
    ear said of it.  One mouth, carried, for the whole of it."""
    v = Voice(SOUND_BANDS, SOUND_SLIDES)
    rows, level, loud = [], [], []
    for start in range(1, PARTS + 1):
        row = START.copy()
        line, way = start, 1
        for _ in range(int(ticks)):
            at = float(row[line - 1])
            now = at + step * way
            if now >= 1.0:                       # the top of its range: turn back
                now, way = 1.0, -1
            elif now <= 0.0:                     # the bottom: this line is done
                now, way = 0.0, 1
                line = line % PARTS + 1          # ...and the next starts at its own 0
            row = row.copy()
            row[line - 1] = min(1.0, max(0.0, now))
            lv, ld = heard(v.say(row.reshape(PARTS, 1)))
            rows.append(row.copy()); level.append(lv); loud.append(ld)
    return (np.asarray(rows, np.float32), np.asarray(level, np.float64),
            np.asarray(loud, np.float64))


def load(path: str = OUT) -> tuple:
    """The table, or a fresh sweep when there is none."""
    if os.path.exists(path):
        got = np.load(path)
        return got["rows"], got["level"], got["loud"]
    rows, level, loud = sweep()
    save(rows, level, loud, path)
    return rows, level, loud


def save(rows, level, loud, path: str = OUT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, rows=rows, level=level, loud=loud,
                        step=np.float64(STEP), made=np.float64(time.time()))


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    ticks = 7200
    args = sys.argv[1:]
    if "--ticks" in args:
        ticks = int(args[args.index("--ticks") + 1])
    print("HER ALPHABET --- her seven articulators, swept by her own law, heard by her own ear")
    began = time.time()
    rows, level, loud = sweep(ticks)
    took = time.time() - began
    live = level > 0.0
    kinds = len(np.unique(np.round(level[live] * 512.0)))
    print("  %-26s %d   (%d passes of %d ticks, one carried mouth)" % ("rows she lived", len(rows), PARTS, ticks))
    print("  %-26s %d of 512" % ("distinct levels she made", kinds))
    print("  %-26s %.4f .. %.4f" % ("her level runs", level[live].min() if live.any() else 0.0,
                                     level[live].max() if live.any() else 0.0))
    print("  %-26s %.0f%%" % ("rows that make a sound", 100.0 * live.mean()))
    print("  %-26s %.0f s" % ("swept in", took))
    save(rows, level, loud)
    print("  written to %s  (an instrument; she never reads it)" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
