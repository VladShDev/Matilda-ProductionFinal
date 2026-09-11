"""WHAT HER MOUTH CAN MAKE --- computed once, kept as a table.

The owner, 2026-08-24: *"2 prelearn indexes of all souns manualy once keep value
list in db"*, and 2026-08-25: *"main idea is make prelearning to each sound has
similarity id to be able compare input sound with her learned echo"*.

    sound index  ->  the articulators that make it        at most 513 rows

**IT IS A PROPERTY OF HER BODY, NOT OF HER LIFE.**  It changes only if her mouth
changes.  It is not memory, it cannot be forgotten, and it puts no hole in her
life the way the old bootstrap did --- a tape born with a seeded alphabet was
a tape with an hour of living it had never done.

WHAT IT REPLACES.  Everything.  *"If our voice can do this, we can reproduce"*
was a search across her whole alphabet, and it cost 47 seconds a tick.  Here it
is `index in table` --- true or false --- and it is true the **first time she
ever hears a sound**, before she has babbled it once.  None of `goal_like`,
`chains_like`, `cut_by_lookup`, `nearest_part` or `match` comes across.

    she commands       out.sound.id 137  ->  table[137]  ->  her mouth
    it comes back      in.echo.id   137                      her mouth's spindle
    someone speaks     in.sound.id  137                      the same space

A BABBLE HOLDS ITS SHAPE.  Each articulator is drawn and HELD for a span, not
rolled every slide --- a per-slide roll is white noise and would never hold a
vowel.  Measured in the old tree: without holding, a band of her voice stayed
lit for 28 ms and a vowel needs 100-200.

AND IT SATURATES, which is why there is no size to pick.  Measured 2026-08-23:
107 distinct sounds in the first 500 babbles, 234 after 5,000, **+3 in the last
500**.  The sweep stops when nothing new has arrived for a while, and that is
her whole alphabet.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .alike import ALIKE_KINDS, FLOOR, SILENCE, alike_of
from .hearing import SOUND_SLIDES
from .muscles import Voice, VOICE_PARTS

#: how long an articulator holds a value, in slides.  Her body's grain for a
#: held shape --- the same one her limbs draw at.
HOLD_SLIDES = 15
#: how finely each articulator is swept.  Not a tuning --- it is how many
#: levels of each her body is walked at, and the table is what it finds.
GRID = 4
#: how many draws without a new sound before her alphabet is done.  Not a size:
#: a run this long with nothing new means nothing new is coming.
DRY = 3000

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "sounds.json")


def babble(voice: Voice, rng) -> np.ndarray:
    """DEAD 2026-08-25 --- no callers.

    The random-babble sweep.  Superseded by `sweep`, which walks her whole
    articulator space on a grid and then draws: measured, the grid found 88
    distinct sounds where 20,000 random babbles found 65.  Kept because a
    babble is how she will explore when nothing is known, and this is what one
    looks like.

    One utterance of hers: every articulator drawn and HELD for a span."""
    want = np.empty((voice.parts, SOUND_SLIDES), np.float32)
    for part in range(voice.parts):
        at = 0
        while at < SOUND_SLIDES:
            span = min(1 + int(rng.integers(HOLD_SLIDES)), SOUND_SLIDES - at)
            want[part, at:at + span] = rng.random()
            at += span
    return want


def held(voice: Voice, lvl) -> int:
    """One mouth, HELD --- and what it sounds like.  `-1` if it made no sound.

    A sound IS a mouth held: `say` slides toward what it is told within one
    slide, so holding one shape for the whole utterance is the cleanest thing
    her mouth can produce, and it is what a vowel is.
    """
    lvl = np.asarray(lvl, np.float32)
    want = np.repeat(lvl[:, None], SOUND_SLIDES, axis=1)
    voice._shape = lvl.copy()
    heard = np.asarray(voice.say(want), np.float64)
    lit = np.nonzero(heard.max(axis=0) > 1e-6)[0]
    if not lit.size:
        return -1
    return alike_of(heard[:, int(lit.min()):int(lit.max()) + 1])


def sweep(steps: int = GRID, seed: int = 0, most: int = 20000) -> dict:
    """Everything her mouth can make, keyed by what it sounds like.

    THE GRID FIRST, because a body's reach is not a matter of luck: every
    articulator at `steps` evenly spaced levels, every combination.  It is
    deterministic and it hits the corners --- fully open, fully closed --- that
    a random draw almost never lands on.  Measured: a grid of 4 found **88**
    distinct sounds where 20,000 random babbles found 65.

    THEN DRAWS, for the cells that fall between grid points, until nothing new
    has arrived for `DRY` tries.

    Her articulators are reset before each one, so the table says what she CAN
    make and not what she can make from where her mouth happens to be.
    """
    voice = Voice(24, SOUND_SLIDES)
    # TWO RESERVED ROWS, AND NEITHER IS A SOUND SHE MAKES.
    #
    #     0   SILENCE   her mouth at rest --- SHE gives nothing
    #     1   FLOOR     the room at rest  --- the WORLD gives nothing
    #
    # His, 2026-08-26.  The floor's row carries the same rest mouth because
    # there is no mouth of hers that makes it: it is what arrives, not what she
    # produces, and reserving the id is what stops a real sound taking the name.
    table = {SILENCE: [0.0] * voice.parts,
             FLOOR: [0.0] * voice.parts}
    grid = np.linspace(0.0, 1.0, steps)
    at = np.zeros(voice.parts, np.int64)
    while True:
        lvl = grid[at]
        if lvl[0] > 0.0:                        # not blowing is silence
            index = held(voice, lvl)
            # ...AND A RESERVED ROW IS NEVER TAKEN BY A MOUTH.  `_index_of`
            # returns 1 for the flattest possible sound, so without this her
            # own voice could claim the floor's name.
            if index > FLOOR and index not in table:
                table[index] = [float(v) for v in lvl]
        k = voice.parts - 1
        while k >= 0:
            at[k] += 1
            if at[k] < steps:
                break
            at[k] = 0
            k -= 1
        if k < 0:
            break

    rng = np.random.default_rng(seed)
    dry = 0
    for _ in range(most):
        lvl = rng.random(voice.parts).astype(np.float32)
        index = held(voice, lvl)
        if index <= FLOOR or index in table:
            dry += 1
            if dry >= DRY:
                break
            continue
        table[index] = [float(v) for v in lvl]
        dry = 0
    return table


def walk(voice: Voice, target_frames, tries: int = 2500, seed: int = 0):
    """The shapes of her mouth that best SAY a certified letter --- one pose
    per tick, fitted against the letter's frames with her mouth state carried
    through, so the transitions are her own flesh moving and not a cut.

    HIS DESIGN, 2026-08-29: *"when Kurzor moves and find her pieces, we
    produce her sound with all her parts and save in table.  So she would
    reproduce sound by ID, but sound will sounds like right now with
    breathing, with opening mouth, with all of these nature things."*

    The recorded voice is the TARGET of this search and never enters her:
    what is stored is articulator poses --- positions of HER body.  Her
    ceiling is her throat's gamut (measured 2026-08-29: mean fit 0.71 on the
    certified word, consonants 0.38-0.56); what comes out is the word in her
    own small voice, which is what a child is.
    """
    rng = np.random.default_rng(seed)
    poses = []
    state = np.zeros(voice.parts, np.float32)
    for f in target_frames:
        t = np.asarray(f, np.float64).ravel()
        tn = float(np.linalg.norm(t))
        best, bc = np.zeros(voice.parts, np.float32), -1.0
        if tn > 1e-6:
            for k in range(tries):
                s = (rng.random(voice.parts).astype(np.float32) if k < tries * 3 // 5
                     else np.clip(best + rng.normal(0, 0.07, voice.parts)
                                  .astype(np.float32), 0, 1))
                voice._shape = state.copy()
                h = np.asarray(voice.say(
                    np.repeat(s[:, None], SOUND_SLIDES, axis=1)), np.float64).ravel()
                hn = float(np.linalg.norm(h))
                c = float(t @ h / (tn * hn)) if hn > 0 else -1.0
                if c > bc:
                    bc, best = c, s
        poses.append([float(x) for x in best])
        voice._shape = state.copy()
        voice.say(np.repeat(np.asarray(best, np.float32)[:, None],
                            SOUND_SLIDES, axis=1))
        state = voice._shape.copy()
    return poses


def save(table: dict, where: str = TABLE) -> str:
    with open(where, "w", encoding="utf-8") as fh:
        json.dump({"parts": list(VOICE_PARTS),
                   "kinds": ALIKE_KINDS,
                   "sounds": {str(k): v for k, v in sorted(table.items())}},
                  fh, indent=1)
    return where


def load(where: str = TABLE) -> dict:
    with open(where, encoding="utf-8") as fh:
        got = json.load(fh)
    return {int(k): v for k, v in got["sounds"].items()}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--most", type=int, default=20000)
    args = ap.parse_args()
    table = sweep(args.grid, args.seed, args.most)
    where = save(table)
    print(f"{len(table)} of {ALIKE_KINDS + 1} sounds  ->  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
