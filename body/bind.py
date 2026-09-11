"""What moves together is one thing --- HERS, in her brain, on her own retina.

Her grouping finds SURFACES.  A person is several of them --- measured on her
own retina, her lumps covered 23% of him and leaked 23% onto a pillow and a
painting, because his lit face and his shadowed chest are separated by a real
brightness step and no rule about colour will ever join them.

Nothing STATIC can join them either.  The owner's screen is flat, so his face
and the wall behind him are painted on one quad at one depth: parallax cannot
separate them, and neither can any amount of looking.  What separates them is
that his face and his chest MOVE TOGETHER and the wall does not.

And it has to be RELATIVE motion, not motion.  Her retina rides her head, and
she swings it further than her whole field in a tick --- measured 2.70 per cell
per tick undriven against 83.55 driven.  Against a slowly-forgotten average
(the rule that works on a camera bolted to a wall) everything stands apart from
a smear: measured on her eye, 27% of her field marked as "a thing" with nothing
happening at all, 99.9% while she was driven.

So: her head moves EVERYTHING at once, and the middle of all that movement IS
her head.  Take it out, and what is left is what moved on its own.  Two pieces
whose leftovers agree are one thing.

AND ONE THING IN HER ROOM DEFEATED ALL OF THAT: a screen is the only object in
her world that changes without moving.  A video replaces every cell of itself
between looks while her body barely shifts, so the loudest mover in her room
was a thing that never moved at all --- the top box lay 63% of its area on the
TV and 7% on her.

What tells them apart is NOT how unstable the contents were.  That was tried
and it is written up below: weighting each piece by how well it matched what it
was took her from 96% covered to 24%.  It is the BOUNDARY.  A panel is a rigid
boundary with changing contents; a thing that moved is the opposite.  Two
readings of that, and each carries a different half:

  * `_outside` --- the ground JUST BEYOND a box.  A thing that moved covers new
    ground, so the ground beyond it is never quiet; beyond a screen's rim
    nothing moves at all.  Measured: 88x the view's own change around her
    against 0.00x around the screen.
  * `walls` --- keep walking outward.  A fragment INSIDE a screen has a lively
    neighbourhood, because the neighbourhood of a piece of screen is more
    screen.  So follow the change to where it stops: around a thing it dies out
    into still background, around a screen it stops at a rim that is in exactly
    the same place every look.  Measured: the screen's change-region boundary
    moved 0.0000 and held still on 95% of looks.

================================================================================
**THIS WAS `measure/binding.py` UNTIL 2026-08-17, AND IT WAS CALLED BY NOTHING.**

It was written in `measure/` on purpose --- *"the honest order is to find out
whether it works before putting it in her"* --- and then it stayed there, so she
went on finding a hundred surfaces a tick and never joining any two of them.
The owner's whole ask on 2026-08-17 was to *"finally collect all pieces
together"*: she does not see him as one thing, and she does not know she has met
him before.

It is MOVED and not copied.  `measure/binding.py` now imports this file, so
`measure.screen`'s check still scores exactly the rule she runs, and there is
one answer to "what is one thing" (locked rule 4).  Nothing above this line
changed on the way across; what is NEW below it is the two stages that were
missing between her surfaces and this rule:

    `follow`   a surface keeps its NAME from look to look, so "the same thing,
               nearer" is a sentence she can form.  Her rule, moved out of
               `orient._closest`, which held exactly one thing.
    `Seen`     the whole stage, carried tick to tick, so `brain/run.py` holds
               one object rather than three parallel dictionaries.

It reads her `eye` sheet --- her LEVELS, the sensor lines themselves.  Never
`/see`, `/eye` or `/fly`: those divide a frame by its own 99.5th percentile, and
measuring change on that packed picture says 99.9% of the view changed on 7
looks of 9.  That is an exposure, not a room.

**AND ON HER OWN RETINA THE MOVING-TOGETHER HALF DOES NOT FIRE YET.  SAY SO
FIRST.**  Measured 2026-08-17 on 24 of her own looks --- her in her cot, a real
photograph of a real person on a screen 0.30 m off her face, the television
FLYING so that there is a rigid thing genuinely moving across her view, all
three arms scored over the same captured frames (`measured/binding_in_her.md`):

    two surfaces BOTH on the flying screen   agree 0.216
    one on it and one on the room            agree 0.150
    a join needs                                   0.70
    surfaces joined into a thing        3 pairs in 24 looks of ~35 surfaces

The separation is real and it is a tenth of what a join asks for.  The reason is
measurable and it is not the rule: `parts.find` re-derives every surface from
scratch each look, so a surface KEEPS ITS NAME (79%, below) while its extent
does not --- its centre lands somewhere else on the thing, and that shift is the
same size as the movement being measured.  `measure.watch` never sees this
because it smooths a box over eighteen looks a second (`HOLD`); she looks once a
tick.  Nothing here is tuned to get round it, and nothing should be until the
jitter itself is measured and dealt with.

**WHAT DOES WORK, AND IT IS THE HALF THAT WAS MISSING:** `follow` --- a surface
keeps its name from look to look.  0% of names survived before `swung` existed;
79-80% survive now, and what her eyes hold survives 14 looks of 24 against 9.

AND IN AN EMPTY ROOM IT HALVES HER KEEP-RATE, WHICH IS THE POINT.  50 ticks of
her real brain on a real body, two seeds, against the commit before this one:
she kept her hold on 60% and 48% of ticks BEFORE and 26% and 24% after --- and
her eyes came off the rail, **|pan| median 0.96 -> 0.24, pegged at the edge of
their travel on 25 ticks of 50 -> 6.**  `orient` already records what the old
number was: *"every match a fresh blob of wall sitting where the last one had
been, and her eyes walked to the corner of their travel and stayed there."*
When there is a thing she holds it longer; when there is only wall she stops
pretending.
================================================================================
"""

from __future__ import annotations

import numpy as np

#: SHE LIVES IN HER BODY, not in her brain.  Deciding what one thing IS, and
#: whether it is the same thing again, is a sense --- her brain is handed
#: things with ids and never a picture.  It was `brain/bind.py` in the old
#: tree, where sight and reasoning shared a folder.
from .hearing import LOOK_SECONDS
from . import orient, parts

#: How far apart two pieces may sit and still be candidates for being one
#: thing, as a fraction of her field.  Only a candidacy test --- what actually
#: joins them is whether they move together.  MEASURED on 40 of her looks with
#: him in view, scored against the detector on her own retina:
#:
#:     NEAR  SCALE   covered   leaked
#:     0.22   0.55      77%      59%
#:     0.35   0.55      80%      58%
#:     0.10   0.55      82%      59%
#:     0.22   0.30      86%      54%
#:     0.10   0.30      85%      48%   <- here
NEAR = 0.10

#: The comparative rule's one number, exactly as `parts.find` uses it: two
#: groups join when what separates them is small against what is already inside
#: each, and `SCALE / size` is how much slack a group of that size still gets.
#: Larger groups have to agree more closely, so a big thing does not swallow
#: the room by accident.  See the table on `NEAR`: this is the one that matters,
#: and tighter is better on both numbers at once.
SCALE = 0.30

#: How much of each look enters a piece's remembered movement.  Her tick is
#: 1.5 s and a person's movement lasts several of them; too short and a pause
#: dissolves the binding it just earned.
#:
#: IT IS ALSO HOW MUCH OF A LOOK ENTERS A PIECE'S BEING THERE AT ALL (`follow`),
#: and that is deliberate: "how much of lately" is one question, and answering
#: it twice with two numbers is the second implementation this project keeps
#: being burned by.
#: ...AND IT IS PER SECOND, NOT PER LOOK.  This was 0.45 A LOOK, chosen when a
#: look was every 1.5 s.  Her clock is 30 a second now and she looks 5 times a
#: second, so the same 0.45 was being applied 7.5x as often and meant something
#: completely different:
#:
#:                                        a look every 1.5 s   every 0.2 s
#:     half a thing's name gone after           1.74 s           0.232 s
#:     the name gone entirely after            11.55 s           1.541 s
#:
#: She forgot a thing she stopped finding in a second and a half.  Nothing
#: raised, the suite stayed green, and the docstring above still said "her tick
#: is 1.5 s" --- exactly what `hearing.py` means by *a constant written "per
#: tick" is a bug*.
#:
#: **AND IT IS NOT WHY SHE FRAGMENTS.**  I wrote that here before measuring it
#: and it is false.  30 of her looks, the same rows through both settings in one
#: process: a name survives to the next look **94.3% either way**, 60.6 things
#: against 60.9, the biggest 72.66% of the field in both.  These two constants
#: only bite where a thing GOES MISSING for a while or an agreement has to build
#: over time, and in a room where every surface is found every look, neither
#: happens.  The fragmentation is upstream of them and still open.
#:
#: The change stays because a per-look constant is a bug by his own clock rule
#: whatever it is worth today, and this restores the wall-clock behaviour that
#: was measured in 2026-08 EXACTLY while staying right if the clock changes
#: again: *"if we whnt faster simply swithc and go"*.
KEEPS_PER_SECOND = 1.0 - (1.0 - 0.45) ** (1.0 / 1.5)
KEEPS = 1.0 - (1.0 - KEEPS_PER_SECOND) ** LOOK_SECONDS

#: How fast a binding is forgotten once the two stop agreeing.  The owner's
#: rule, 2026-08-15: *"when she sees some object long enough, its shape has to
#: become some meaning"* --- so a binding EARNED by moving together survives
#: the thing going still, and only disagreement takes it away.
#: Per second, for the same reason and by the same arithmetic as `KEEPS`.  At
#: 0.04 a look it was half gone in 3.4 s instead of 25.5 --- and this is the one
#: that carries *"when she sees some object long enough, its shape has to become
#: some meaning"*, so shortening it 7.5x shortens the meaning itself.  Same
#: caveat as `KEEPS`: measured, it changes nothing in a room where nothing goes
#: missing, and what it protects is the case that room cannot show.
FORGETS_PER_SECOND = 1.0 - (1.0 - 0.04) ** (1.0 / 1.5)
FORGETS = 1.0 - (1.0 - FORGETS_PER_SECOND) ** LOOK_SECONDS

#: How thick a boundary is, as a share of the SMALLER SIDE OF THE THING it goes
#: round --- not of the view.  MEASURED: fixed to the view, a rim was too thick
#: for her at 2.6 m (her box never once qualified in 39 looks) and too thin for
#: the screen.  A share of the thing itself works at every distance, which is
#: exactly what a rule that must survive her crawling towards something needs.
RIM = 0.18

#: How far a change-region's boundary may sit from where it sat last look and
#: still count as not having moved, in view units.  MEASURED on a fixed fly-by
#: of her room: the screen's change-region boundary moved **0.0000** and was
#: inside 0.01 on 95% of looks.  This is one cell of the grid the regions are
#: knitted on (2/127), so it says "not one cell" and not a tuned tolerance.
STILL = 0.02
#: How coarse that grid is.  `measure.watch.moving` knits its regions at 4 and
#: this is the same knitting on the same kind of picture; two different answers
#: to "what is connected" is exactly the second implementation this project
#: keeps being burned by.
STEP = 4

#: Below this, a thing she has not seen lately is gone.  ONE floor, already
#: here: `bind` has always dropped a forgotten pairing at 0.01, and a piece
#: nobody has seen is forgotten the same way for the same reason.
GONE = 0.01


def _res(known: list[dict]) -> np.ndarray:
    """Each piece's movement with HER OWN taken out.

    The median and not the mean: a big thing moving across her view would drag
    a mean until her head appeared to be following it, and then the thing would
    look still and the room would look like it was moving.  A median is what
    MOST of her view did, which is what her head did.
    """
    v = np.asarray([[k["vx"], k["vy"]] for k in known], np.float64)
    if not len(v):
        return v
    v = v - np.median(v, axis=0)
    # DISCOUNTING A PIECE BY HOW MUCH IT CHANGED WAS TRIED AND IS WRONG.  The
    # idea was sound --- a screen's cells change colour where they stand and a
    # real thing translates --- but measured, weighting each piece by
    # `1/(1+(fit/median fit)^2)` took her from 96% covered to 24%: her own body
    # is small and its pieces are as unstable as the screen's, so the weight
    # suppressed the thing it was meant to find along with the thing it was
    # meant to reject.  `fit` is still carried on every track for whoever wants
    # to try again; what it needs is the PANEL's rigid boundary, not the
    # instability of its contents.
    return v


def bind(known: list[dict], held: dict | None = None) -> tuple[list[list[int]], dict]:
    """Group the pieces that move together.  `(groups, what to keep)`.

    `held` is what this returned last look, so a binding earned by moving
    together outlives the movement that earned it.
    """
    held = dict(held or {})
    n = len(known)
    if n < 2:
        return [[i] for i in range(n)], held

    res = _res(known)
    mag = np.linalg.norm(res, axis=1)
    # THE NOISE FLOOR IS MEASURED, NOT CHOSEN.  A piece's edge lands a cell
    # differently every look whether or not anything moved, so some leftover is
    # always there.  The median leftover across her whole view IS that noise:
    # half of everything she sees is doing no more than jitter, by definition.
    # Pairs quieter than that are not evidence of anything and are left out ---
    # otherwise two motionless pieces agree perfectly (0 against 0) and the
    # whole room binds into one thing.
    floor = float(np.median(mag))

    box = np.asarray([[k["x"] - k["w"], k["y"] - k["h"],
                       k["x"] + k["w"], k["y"] + k["h"]] for k in known])
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            near = (min(box[a, 2], box[b, 2]) + NEAR >= max(box[a, 0], box[b, 0])
                    and min(box[a, 3], box[b, 3]) + NEAR >= max(box[a, 1], box[b, 1]))
            if not near:
                continue
            key = (known[a]["id"], known[b]["id"])
            both = mag[a] + mag[b]
            if both > floor * 2.0:
                # HOW ALIKE THEIR LEFTOVERS ARE, as a share of how much there
                # is to be alike about.  0 is moving as one body; 1 is moving
                # in unrelated directions.  A ratio and not a distance, so a
                # slow shift of his shoulder counts the same as a fast one.
                agree = 1.0 - float(np.linalg.norm(res[a] - res[b]) / both)
                was = held.get(key, 0.0)
                held[key] = (1.0 - KEEPS) * was + KEEPS * max(0.0, agree)
            elif key in held:
                # nothing moved: the binding is not renewed, only forgotten,
                # which is what makes it a memory rather than a measurement
                held[key] = held[key] * (1.0 - FORGETS)
            pairs.append((key, a, b))

    # ...AND JOIN THEM CHEAPEST-FIRST, `find`'s own rule one level up: two
    # groups become one when what separates them is small against what is
    # already inside each of them.
    parent = list(range(n))
    inside = [0.0] * n
    size = [1] * n

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(pairs, key=lambda p: -held.get(p[0], 0.0))
    for key, a, b in order:
        strength = held.get(key, 0.0)
        if strength <= 0.0:
            continue
        x, y = root(a), root(b)
        if x == y:
            continue
        apart = 1.0 - strength
        if apart <= min(inside[x] + SCALE / size[x], inside[y] + SCALE / size[y]):
            parent[y] = x
            inside[x] = max(inside[x], inside[y], apart)
            size[x] += size[y]

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    alive = {k["id"] for k in known}
    held = {key: v for key, v in held.items()
            if v > GONE and key[0] in alive and key[1] in alive}
    return sorted(groups.values(), key=len, reverse=True), held


def _outside(pic: np.ndarray, was: np.ndarray, box: tuple) -> float:
    """How much the view changed in a rim JUST OUTSIDE this box.

    A THING THAT MOVED COVERS NEW GROUND.  Its outline is the one place where
    something that was not there a moment ago now is --- so the ground just
    beyond it is never quiet.  A PANEL IS THE OPPOSITE: its contents may
    replace themselves entirely and the ground just beyond its rim does not
    move a hair, because the rim is a piece of furniture and not an edge of
    anything that travelled.

    Measured on a fixed fly-by of her room with his real camera on the screen,
    39 looks, against `measure.where`'s exact geometry:

        what          just outside the rim    against the view's own change
        her, driven         0.03693                     88x
        the bed             0.00560                     13x
        THE SCREEN          0.00000                    0.00x

    Zero.  Not small --- zero, in every condition tried, resting and driven,
    while the same screen's inside changed 0.024 a look.  That is the whole
    difference between a thing that moved and a thing that only changed.
    """
    h, w = pic.shape[0], pic.shape[1]
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int((x0 + 1) * 0.5 * (w - 1))))
    x1 = max(0, min(w - 1, int((x1 + 1) * 0.5 * (w - 1))))
    y0 = max(0, min(h - 1, int((y0 + 1) * 0.5 * (h - 1))))
    y1 = max(0, min(h - 1, int((y1 + 1) * 0.5 * (h - 1))))
    r = max(2, int(RIM * min(x1 - x0, y1 - y0)))
    gx0, gy0 = max(0, x0 - r), max(0, y0 - r)
    gx1, gy1 = min(w - 1, x1 + r), min(h - 1, y1 + r)
    if gx1 - gx0 < 2 or gy1 - gy0 < 2:
        return 0.0
    # ONE SUM FOR THE WHOLE PICTURE, ASKED PER BOX IN CONSTANT TIME.
    #
    # The rim's mean is the outer rectangle's sum minus the inner one's, over
    # how many cells that leaves --- exactly what `d[keep].mean()` computed,
    # and the same number, because `keep` IS the outer rectangle with the inner
    # one taken out.
    #
    # It used to take `|pic - was|`, average it over her three cones and build
    # a boolean mask FOR EVERY BOX.  Profiled on six looks: **8,484 calls to
    # `np.mean` --- 1,414 a look --- on arrays of a few hundred numbers.**  That
    # is Python round-trips, not arithmetic, and it was 42% of what a look cost.
    # The same shape as `look_back` scoring her whole life, and `driver` reading
    # 361 ticks to use two: do it once for everything.
    add = _summed(pic, was)
    outer = _box_sum(add, gy0, gx0, gy1, gx1)
    inner = _box_sum(add, y0, x0, y1, x1)
    cells = ((gy1 - gy0 + 1) * (gx1 - gx0 + 1)
             - (y1 - y0 + 1) * (x1 - x0 + 1))
    return float((outer - inner) / cells) if cells > 0 else 0.0


#: THE LAST SUMMED PICTURE, kept because every box in one look asks the same
#: pair.  Two looks are two entries and no more: it is keyed by the arrays
#: themselves, so a new look replaces it and nothing accumulates.
_SUMMED: dict = {}


def _summed(pic: np.ndarray, was: np.ndarray) -> np.ndarray:
    """`|pic - was|` averaged over her cones, summed up and to the left.

    A summed-area table: the sum of any rectangle is four lookups.  Built once
    a look and asked by every box.
    """
    key = (id(pic), id(was), pic.shape)
    got = _SUMMED.get(key)
    if got is not None:
        return got
    d = np.abs(np.asarray(pic, np.float64) - np.asarray(was, np.float64))
    if d.ndim == 3:
        d = d.mean(axis=2)
    add = np.zeros((d.shape[0] + 1, d.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(d, axis=0), axis=1, out=add[1:, 1:])
    _SUMMED.clear()
    _SUMMED[key] = add
    return add


def _box_sum(add: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    """The sum inside a rectangle, inclusive, in four lookups."""
    return float(add[y1 + 1, x1 + 1] - add[y0, x1 + 1]
                 - add[y1 + 1, x0] + add[y0, x0])


def _changed(a: np.ndarray, b: np.ndarray) -> list[dict]:
    """The connected regions where the view changed, as boxes in `-1..1`.

    Knitted exactly the way `measure.watch.moving` knits its regions, on the
    same coarseness, because two answers to "what is connected" is the second
    implementation this project keeps being burned by.
    """
    d = np.abs(a - b)
    if d.ndim == 3:
        d = d.mean(axis=2)
    d = d[::STEP, ::STEP]
    # CHANGED MEANS CHANGED AT ALL, and that is a measurement and not a mercy.
    # Her render is exact, so a cell either changed or it did not: measured on
    # a fly-by of her room, the share of cells above 0, above 1e-6, above 1e-4
    # and above 1e-3 was THE SAME NUMBER on all nine looks (2.86%--3.15%).
    # There is no threshold to pick because there is nothing in between.
    #
    # This only holds on her LEVELS.  Taking the same measurement off `/fly`'s
    # packed picture said 99.9% of the view changed on 7 looks of 9, because
    # that picture is divided by its own 99.5th percentile and the screen is
    # bright enough to move it -- an exposure, not a room.
    on = (d > 0.0).ravel()
    if not on.any():
        return []
    h, w = d.shape
    # A PAIR IS JOINED WHEN BOTH ITS CELLS CHANGED, and the pairs are a grid
    # rather than a list --- `parts._knit` takes them that way now, which is
    # what took it off numpy's scatter and halved it.
    grid = on.reshape(h, w)
    lab = parts._knit(h, w, grid[:-1] & grid[1:], grid[:, :-1] & grid[:, 1:])
    lab = np.where(on, lab, -1)
    # EVERY REGION'S CORNERS IN ONE PASS.  This walked the whole label array
    # once per tag --- `lab == tag` over every cell, for every region it found.
    # `np.minimum.at` does all of them together, and the answer is identical
    # because a bounding box is a min and a max and nothing else.
    live = lab >= 0
    if not live.any():
        return []
    tags = lab[live]
    ys, xs = np.divmod(np.flatnonzero(live), w)
    n = int(tags.max()) + 1
    count = np.bincount(tags, minlength=n)
    lox = np.full(n, w, np.int64); hix = np.full(n, -1, np.int64)
    loy = np.full(n, h, np.int64); hiy = np.full(n, -1, np.int64)
    np.minimum.at(lox, tags, xs); np.maximum.at(hix, tags, xs)
    np.minimum.at(loy, tags, ys); np.maximum.at(hiy, tags, ys)
    out = []
    for tag in np.flatnonzero(count >= 6):
        x0 = lox[tag] / (w - 1) * 2 - 1
        x1 = hix[tag] / (w - 1) * 2 - 1
        y0 = loy[tag] / (h - 1) * 2 - 1
        y1 = hiy[tag] / (h - 1) * 2 - 1
        out.append({"x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                    "w": (x1 - x0) / 2, "h": (y1 - y0) / 2})
    return out


def _holds(wall: dict, x: float, y: float) -> bool:
    """Does this wall stand round that spot?

    A SPOT AND NOT A BOX.  Asking whether a group's box was mostly inside a
    wall let the screen's own groups walk out of it: measured, a group made
    entirely of screen came back 0.210 wide at x+0.329 while the screen it
    lived in was 0.173 wide at x+0.134, because a tracked box is smoothed and
    drifts.  Where a piece IS does not drift.
    """
    return (wall["x"] - wall["w"] <= x <= wall["x"] + wall["w"]
            and wall["y"] - wall["h"] <= y <= wall["y"] + wall["h"])


def _apart(a: dict, b: dict) -> float:
    """How far one box's boundary sits from another's."""
    return max(abs(a["x"] - a["w"] - (b["x"] - b["w"])),
               abs(a["x"] + a["w"] - (b["x"] + b["w"])),
               abs(a["y"] - a["h"] - (b["y"] - b["h"])),
               abs(a["y"] + a["h"] - (b["y"] + b["h"])))


def walls(pic, was, older, is_the_room: float = 0.85) -> list[dict]:
    """The change-regions whose OUTER BOUNDARY is where it was last look.

    A fragment inside a screen cannot be told from a thing by its own rim ---
    the neighbourhood of a piece of screen is more screen, and that changes.
    `_outside` only ever reads the hard zero when a box lands on the WHOLE
    rectangle, and a group almost never does.

    So keep walking outward.  Around a thing that moved the change dies out,
    because the ground beyond it is still.  Around a screen the change runs on
    until it hits the rim and then stops dead --- **at the same place every
    look**.  Measured on a fixed fly-by of her room with his camera playing:
    the screen's change-region boundary moved 0.0000 between looks and stayed
    inside one cell on 95% of them.

    A region that fills the view is NOT a wall, it is the world.  When the
    screen is held against her face it covers everything she has, and then
    finding the man on it is the whole job --- the same reason `boxes` already
    refuses to call the room a thing.

    THIS READS HER RENDER, WHERE A CELL EITHER CHANGED OR IT DID NOT.  On a
    camera stream every cell changes every frame from sensor noise alone
    (`watch` measures it at 0.0092 a frame), so the regions become one region
    that fills the view --- which the rule above then refuses to call a wall,
    and `boxes` falls back to the rim on its own.  It degrades to the old
    answer rather than to a wrong one, but it does not yet WORK there.
    """
    now = _changed(pic, was)
    then = _changed(was, older)
    if not now or not then:
        return []
    out = []
    for b in now:
        if b["w"] > is_the_room and b["h"] > is_the_room:
            continue
        if min(_apart(b, t) for t in then) <= STILL:
            out.append(b)
    return out


def boxes(known: list[dict], groups: list[list[int]], most: int = 3,
          is_the_room: float = 0.85,
          pic: np.ndarray | None = None,
          was: np.ndarray | None = None,
          older: np.ndarray | None = None) -> list[dict]:
    """One box per group, MOST-MOVING first, minus the one that is the room.

    Not biggest first.  What this rule is for is the thing that moved, and the
    thing that moved is routinely small: measured on a fixed view of her own
    room she is 3% of it, so ranking by size can never once pick her --- it
    covered 96% of her and leaked 95%, because the only groups big enough to
    be chosen were the room.  Ranking by how much a group moved covers the
    same 96% without the room in it.

    AND HOW MUCH OF THAT MOVEMENT WAS THE GROUP'S OWN.  Ranking by movement
    alone picked the TV every time on her own room, because a video replaces
    every cell of itself between looks while her body barely shifts.  What
    tells them apart is not how unstable the contents were --- that was tried,
    and weighting by `fit` took her from 96% covered to 24% --- but whether
    the OUTLINE went anywhere.  A group whose boundary is quieter than the view
    it sits in did not move, however much its inside changed (`_outside`), and
    neither did anything standing inside a boundary that is exactly where it
    was last look (`walls`).

    `pic`, `was` and `older` are this look and the two before it.  With two
    there is a rim to read; with one there is no boundary at all and this ranks
    by movement alone, which is what it did before and what it must keep doing
    when nobody kept the last picture.
    """
    res = _res(known)
    mag = np.linalg.norm(res, axis=1) if len(res) else np.zeros(len(known))
    # A STALE SPEED WAS LEFT ALONE, AND THAT IS A MEASUREMENT.  `measure.watch`
    # keeps a thing's last speed for as long as it keeps the thing, so a piece
    # that stopped being found goes on reporting the speed it had when it
    # vanished and never changes again: on a fly-by the top-ranked group came
    # back with the same box and the same `moved` of 0.1127 on four looks
    # running.  That corpse is real.  Discounting it is not: cutting stale
    # pieces out altogether took covering him from 32% to 17%, and ageing them
    # by `1/(1+looks missed)` still cost 12 points on his room while buying
    # nothing on hers (her own top-box share 61% aged against 64% left alone).
    #
    # IN HER, THE CORPSE NEVER ARRIVES: `Seen.look` hands `bind` only the
    # pieces THIS look actually found, and a piece she has lost keeps its name
    # and its history but stops being one of them.  Nothing is discounted from
    # the far end, which is what the measurement above says not to do.
    # `measure.watch` is the panel's tracker on the camera's frames and still
    # has it; that is written up in `measured/binding.md` for whoever owns it.

    # THE FLOOR IS WHAT THE WHOLE VIEW DID, and it is the MEAN and not the
    # median.  For velocity the median is right --- half of what she sees is
    # doing no more than jitter, by definition.  For a picture it is not: most
    # of a still room does not change AT ALL, so the median change is exactly
    # 0.000000 and every rim with any life in it scores full marks against it.
    # Measured, that made the rule toothless: the screen kept the top box on 51
    # looks in 100 instead of losing it.  The mean is what the view did.
    #
    # Measured against the other candidate --- a box's rim against its OWN
    # inside --- which is the wrong comparison and says the screen moved more
    # than she did: her rim came to 0.085 of her own change against the
    # screen's 0.148, while against THE VIEW it is 88x against 0.00x.
    seeing = (pic is not None and was is not None
              and getattr(pic, "shape", None) == getattr(was, "shape", None))
    floor = 0.0
    if seeing:
        whole = np.abs(pic - was)
        if whole.ndim == 3:
            whole = whole.mean(axis=2)
        floor = float(whole.mean())
    walled = (walls(pic, was, older, is_the_room)
              if seeing and older is not None and older.shape == pic.shape else [])
    out = []
    for grp in groups:
        xs = [known[i]["x"] for i in grp]
        ys = [known[i]["y"] for i in grp]
        ws = [known[i]["w"] for i in grp]
        hs = [known[i]["h"] for i in grp]
        x0 = min(x - w for x, w in zip(xs, ws))
        x1 = max(x + w for x, w in zip(xs, ws))
        y0 = min(y - h for y, h in zip(ys, hs))
        y1 = max(y + h for y, h in zip(ys, hs))
        if (x1 - x0) / 2 > is_the_room and (y1 - y0) / 2 > is_the_room:
            continue
        head = max(grp, key=lambda i: known[i].get("seen", 0.0))
        moved = float(sum(mag[i] for i in grp))
        # DID ITS OUTLINE GO ANYWHERE?  A share and not a threshold: a rim as
        # lively as the view scores a half, a rim far livelier approaches one,
        # and a rim that never moves is nought however hard the inside churns.
        edge = _outside(pic, was, (x0, y0, x1, y1)) if seeing else 0.0
        # A DEAD RIM IS NOUGHT, not a half.  Guarding the division with
        # `edge + floor > 0` handed a rim that never moved the same 0.5 as a
        # missing picture, which is the one case this rule exists to punish.
        own = 1.0 if not seeing else (edge / (edge + floor) if edge > 0.0 else 0.0)
        # ...AND NOTHING INSIDE A WALL WENT ANYWHERE.  A group whose pieces
        # stand inside a change-region that is exactly where it was last look
        # is a piece of a picture, however hard it appears to travel.  MOST of
        # them, because a group straddling a screen's edge is half a thing.
        if walled:
            behind = sum(any(_holds(wall, known[i]["x"], known[i]["y"])
                             for wall in walled) for i in grp)
            if behind * 2 >= len(grp):
                own = 0.0
        out.append({"x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                    "w": (x1 - x0) / 2, "h": (y1 - y0) / 2,
                    "area": (x1 - x0) * (y1 - y0) / 4,
                    "vx": known[head]["vx"], "vy": known[head]["vy"],
                    "id": known[head]["id"], "pieces": len(grp),
                    "moved": moved, "edge": edge, "own": own,
                    "carried": moved * own if seeing else moved,
                    "seen": sum(known[i].get("seen", 0.0) for i in grp)})
    out.sort(key=lambda t: -t["carried"])
    return out[:most]


# --- A SURFACE KEEPS ITS NAME ---------------------------------------------------
#
# Everything above answers "which of these surfaces are one thing, THIS LOOK".
# It cannot start without the thing this file was missing for two days: a piece
# that is the same piece as the one before it.  `bind` reads `k["id"]`, and
# every id it ever saw came from `measure.watch`, a tracker on the CAMERA's
# frames belonging to the panel.  Her brain had none.  What it had was
# `orient._closest` --- the identical question, asked of exactly ONE thing, the
# thing her eyes were on.
#
# So the rule moves here and is asked of all of them at once.  It is her rule
# and it is unchanged: her `alike` bucket first (`parts._alike` is built out of
# exactly what does NOT change when a thing moves or comes closer), nearest in
# her viewport second (a thing is somewhere near where it was), `parts.like` to
# separate two candidates sitting equally close --- and NO BAR anywhere, for
# `find`'s own measured reason: no fixed distance is right for both a face
# crossing her field and a bottle held still.


def _closest(one, rows: np.ndarray):
    """Which of `rows` is most likely to be `one`, seen again --- or `None`.

    Inside the `alike` bucket first, which is what `parts._alike` is for: it is
    built out of exactly what does NOT change when a thing moves or comes
    closer, so leaving the bucket means it stopped looking like itself.  Then
    nearest in her viewport, because a thing is somewhere near where it was;
    then `parts.like` to separate two candidates sitting equally close.
    """
    if one is None or rows is None or not len(rows):
        return None
    same = np.flatnonzero(rows[:, parts.ALIKE] == float(one[parts.ALIKE]))
    if not len(same):
        return None
    mine = rows[same]
    gap = np.linalg.norm(mine[:, :2] - np.asarray(one[:2], np.float32), axis=1)
    near = np.flatnonzero(gap <= gap.min() + 1e-6)
    if len(near) > 1:                       # equally close: the more alike one
        near = near[[int(np.argmin(parts.like(one, mine[near])))]]
    return int(same[int(near[0])])


def swung(rows: np.ndarray, swing) -> np.ndarray:
    """Last look's things, put where SHE MOVED THEM: a copy, x and y only.

    **WITHOUT THIS, NOTHING SHE SEES IS EVER THE SAME THING TWICE.**  Measured
    on her own retina in her cot with a real photograph of a person on the
    screen, 24 looks, her driven: matching by her `alike` bucket and nearest in
    the viewport carried a surface's name across **0 look-to-look steps in 23**.
    Her head throws her view further than its own width in a tick, so "nearest
    to where it was" points at whatever swung into that place, and the mutual
    check --- correctly --- refuses all of it.  Nothing can be bound to anything
    if no two looks share a single name, and 29 surfaces came back as 29
    things on every look of that run.

    So she is told what she just did first, which is not a new idea in this
    project: `orient.changed` already does exactly this to her PICTURE, for
    exactly this reason, and measured leaves 0.49 of the movement it is
    removing (`measured/she_moved_too.md`).  This is the same two lines run the
    other way --- her rays are laid out linearly in tan across her field, so a
    place is an angle by `arctan` and an angle is a place by `tan` --- and it
    is `orient`'s own optics, not a second copy of them.

    `swing` is `orient.turned`: how far her view moved between the last look and
    this one, from her own outgoing gaze order and her own canals.  Neither is
    the room and neither is a conclusion; one is a copy of what she ordered, the
    other is an organ.
    """
    out = np.array(rows, np.float32, copy=True)
    if not len(out):
        return out
    across, up, about = (float(v) for v in swing)
    if not (across or up or about):
        return out
    # where each thing WAS, as an angle off her axis.  `find`'s y counts DOWN
    # from the top row and her field's angles count up, which is the one
    # negation in here.
    ox = np.arctan(np.clip(out[:, X], -1.0, 1.0) * orient._EDGE)
    oy = np.arctan(-np.clip(out[:, Y], -1.0, 1.0) * orient._EDGE)
    ox, oy = ox - across, oy - up
    turn, rise = np.cos(about), np.sin(about)
    bx, by = turn * ox - rise * oy, rise * ox + turn * oy
    # ...AND WHAT HAS SWUNG PAST A QUARTER TURN IS BEHIND HER, where `tan` comes
    # back through the other side and would hand a thing at 143 degrees a place
    # near the middle of her view.  She can swing that far in one tick.
    gone = (np.abs(bx) >= orient._BEHIND) | (np.abs(by) >= orient._BEHIND)
    out[:, X] = np.where(gone, 9.0, np.tan(np.where(gone, 0.0, bx)) / orient._EDGE)
    out[:, Y] = np.where(gone, 9.0, -np.tan(np.where(gone, 0.0, by)) / orient._EDGE)
    return out


def again(rows: np.ndarray, was: np.ndarray) -> dict[int, int]:
    """Which row of `rows` is which row of `was`: `{new: old}`.

    **MUTUAL NEAREST, OR NOT MATCHED AT ALL**, and that is not a new rule ---
    it was `parts.nearing`'s (never called, deleted 2026-09-03), written for
    the failure it stops: *"Taking
    each new thing's closest old one lets several new things claim the same old
    one... Requiring the match to hold from both sides drops those instead of
    believing them, and costs nothing and invents nothing: it is the same
    distance read twice."*  It is the same failure here and it is worse, because
    a tracker that cannot refuse does not merely mis-report --- it FOLLOWS the
    wrong thing and drives her eyes with it.  Measured on her real body before
    this check existed: she matched something on 15 of 16 ticks, every match a
    fresh blob of wall sitting where the last one had been, and her eyes walked
    to the corner of their travel and stayed there.

    LOSING IT IS A REAL ANSWER.  Her head throws her own view 105 degrees in a
    tick, so most looks genuinely do not contain what the last one held.
    """
    out: dict[int, int] = {}
    if rows is None or was is None or not len(rows) or not len(was):
        return out
    for i in range(len(rows)):
        j = _closest(rows[i], was)
        if j is None:
            continue
        if _closest(was[j], rows) == i:
            out[i] = j
    return out


def follow(known: list[dict], rows: np.ndarray, tick: int,
           swing=(0.0, 0.0, 0.0)) -> list[dict]:
    """Carry every surface's NAME into this look.  The pieces found NOW.

    `known` is everything she has a name for, newest measurement first ---
    including what she has lost lately, because a thing she looks away from and
    back at is the same thing and she has to be able to say so.  What comes back
    is only the pieces THIS look actually found, which is what `bind` may reason
    about: a piece nobody saw has no speed, and inventing one for it is the
    stale-speed trap `boxes` is written up against.

    **THE SPEED IS RAW AND THAT IS DELIBERATE.**  `measure.watch` smooths a box
    and a speed because the CAMERA moves 0.0092 a frame from its own sensor
    noise; her render is EXACT (`_changed`, measured: a cell either changed or
    it did not).  So there is nothing here to smooth away, and `bind`'s own
    `KEEPS` is already where "how much of a look is remembered" is decided ---
    smoothing here as well would answer that question twice.

    `seen` is how much of lately a piece has been there --- the count that makes
    the owner's rule work: *"when she sees some object long enough, its shape
    has to become some meaning."*  It grows while she keeps finding it and
    fades while she does not, and below `GONE` she has forgotten it.
    """
    rows = np.zeros((0, parts.COLUMNS), np.float32) if rows is None else \
        np.asarray(rows, np.float32)
    was = (np.stack([k["row"] for k in known]) if known
           else np.zeros((0, parts.COLUMNS), np.float32))
    # WHERE SHE LEFT THEM, NOT WHERE THEY WERE.  See `swung`: without this,
    # measured, a name survived 0 of 23 look-to-look steps on her own retina.
    left = swung(was, swing)
    pairs = again(rows, left)

    # a name nobody is using.  Counted off what she still holds rather than off
    # a module-wide tally, so two of her in one process are two of her.
    naming = max((k["id"] for k in known), default=0)

    fresh, met = [], set()
    for i in range(len(rows)):
        x, y, area, w, h = (float(v) for v in rows[i, :5])
        j = pairs.get(i)
        if j is None:
            naming += 1
            fresh.append({"x": x, "y": y, "w": w, "h": h, "area": area,
                          "vx": 0.0, "vy": 0.0, "n": int(tick), "seen": KEEPS,
                          "alike": int(round(float(rows[i, parts.ALIKE]))),
                          "fit": 0.0, "row": np.asarray(rows[i]),
                          "id": naming})
            continue
        met.add(j)
        k = known[j]
        gap = max(1, int(tick) - int(k["n"]))
        # HOW WELL IT MATCHED, KEPT.  A piece that changed colour where it
        # stood is not the same surface having moved --- it is a new surface in
        # the same place, which is what every cell of a SCREEN does while a
        # video plays on it.  Anything reasoning about movement has to be able
        # to tell those two apart.
        k["fit"] = float(parts.like(rows[i], k["row"].reshape(1, -1))[0])
        # **HOW FAR IT WENT, RAW --- AND THE REGISTRATION IS NOT USED HERE.**
        # It reads as an obvious improvement to measure the displacement from
        # where she LEFT it rather than from where it was, and it is not: `_res`
        # takes the MEDIAN of these out, and that median IS her head, which is
        # this file's oldest argument.  Registering first and then subtracting
        # the median subtracts her twice.  Measured over 24 of her looks with
        # the television flying, the same frames scored both ways:
        #
        #                                  two pieces both      one on it,
        #                                  on the screen        one not
        #     speed raw, matched registered      0.216            0.150
        #     speed registered too               0.132            0.098
        #
        # and it cost 73 ms a look against 41.  So `swung` earns its place at
        # the MATCHING, where the question is "which of these is which", and
        # nowhere else.
        k["vx"], k["vy"] = (x - k["x"]) / gap, (y - k["y"]) / gap
        k["x"], k["y"], k["w"], k["h"], k["area"] = x, y, w, h, area
        k["row"] = np.asarray(rows[i])
        k["n"] = int(tick)

    # **THE ORDER OF THIS LIST IS LOAD-BEARING AND IT COST A DAY.**  `bind`
    # remembers a pairing under `(known[a]["id"], known[b]["id"])` with `a < b`
    # BY POSITION, so the key is only the same key next look if the list holds
    # its order.  Sorting it --- newest first, which reads perfectly sensible
    # --- flipped roughly half the keys every look, every remembered agreement
    # restarted from zero, and NOTHING EVER BOUND: measured on her own retina
    # with the television flying, 36 surfaces came back as 36 things on all 20
    # looks and the strongest agreement reached 0.431 against the 0.70 a join
    # needs.  So it is append-only, exactly as `measure.watch.track`'s is, and
    # a name only ever moves off the end of it.
    keep = []
    for at, k in enumerate(known):
        if at in met:
            k["seen"] = (1.0 - KEEPS) * k["seen"] + KEEPS
            keep.append(k)
            continue
        # NOT FOUND IS NOT GONE, YET.  She keeps the name while she still
        # half-remembers it, and the same 0.01 floor `bind` prunes a forgotten
        # pairing at says when she does not.
        k["seen"] *= (1.0 - KEEPS)
        if k["seen"] > GONE:
            keep.append(k)
    keep.extend(fresh)
    known[:] = keep
    return [k for k in known if k["n"] == int(tick)]


# --- WHAT SHE IS LOOKING AT ------------------------------------------------------
#
# One object, carried tick to tick, holding the three things this stage
# remembers: what each surface is called, which pairs of them have been moving
# together, and what each GROUP is called.  `brain/run.py` held three parallel
# dictionaries for the eyes alone and it is exactly the shape rule 3 is about.

#: A thing's row, in `parts.find`'s own convention for the first five columns
#: --- centre in -1..1, size a FRACTION of the view --- so `orient` reads a
#: bound thing with the same arithmetic it read a surface with, and the panel
#: draws it with none.
X, Y, AREA, W, H = 0, 1, 2, 3, 4
#: ...and then the four things only a bound thing has.
#: the name that survives to the next look.  It rides in a float32 column, so
#: it is exact up to 16,777,216 --- about ten days of her life at twenty new
#: names a tick.  Past that two things could share a name and `held_again`
#: would silently follow the wrong one; if she is ever meant to live that long
#: this column has to stop being a float.
ID = 5
PIECES = 6       #: how many surfaces are in it
CARRIED = 7      #: how much movement of its own it carried (`boxes`)
SEEN = 8         #: how much of lately it has been there
COLUMNS = 10

#: WHICH COLUMN OF A BOUND THING SAYS WHAT IT LOOKS LIKE.
#:
#: A thing had nine columns and not one of them was an appearance: `x, y, area,
#: w, h, name, pieces, carried, seen`, where `name` is a running integer from a
#: counter.  That is a TRACKING id --- it says "the same thing as last look" and
#: it cannot say "a thing like this one", which is the question her whole memory
#: is built on.  So there was nothing to file a bound thing under, and
#: `brain/run.py` filed her raw SURFACES instead: measured on her own retina, 62
#: surfaces stored and the 3 things she had made of them thrown away.  The
#: owner, finding out: *"so those objects don't reach her brain at all?"*
#:
#: It is the appearance of the surface inside it she has met most --- the same
#: piece `boxes` already names the group after, and the steadiest name a group
#: can have.  Nothing new is invented: it is `parts._alike` of a row she already
#: carried, moved one level up.
LOOKS = 9


class Seen:
    """Everything she is looking at, and it is the same everything next tick."""

    def __init__(self) -> None:
        #: every surface she has a name for, lately
        self.pieces: list[dict] = []
        #: which pairs of them have been moving together
        self.agreed: dict = {}
        #: what each group is called, as the surfaces that were in it
        self.names: dict[int, frozenset] = {}
        self._named = 0
        #: how many surfaces the last look found, and how many things they made
        self.surfaces = 0
        self.same = 0

    def look(self, rows, tick: int, swing=(0.0, 0.0, 0.0),
             pic=None, was=None, older=None) -> np.ndarray:
        """This look's THINGS: `(n, COLUMNS)`, the one that carried most first.

        `rows` is `parts.find`'s answer for this tick --- her tape already has
        it, so nothing here looks at her eye a second time.  `swing` is
        `orient.turned`, how far her view moved since the last look, and without
        it nothing she sees is ever the same thing twice (see `swung`).  `pic`,
        `was` and `older` are this look and the two before it, IN HER LEVELS,
        for the boundary rule; without them this ranks by movement alone, which
        is what it must keep doing when nobody kept the last picture.
        """
        knew = {k["id"] for k in self.pieces}
        here = follow(self.pieces, rows, tick, swing)
        self.surfaces = len(here)
        #: how many of them she had a name for already --- the one number that
        #: says whether any of this can work at all
        self.same = sum(k["id"] in knew for k in here)
        if not here:
            self.names = {}
            return np.zeros((0, COLUMNS), np.float32)
        groups, self.agreed = bind(here, self.agreed)
        drawn = boxes(here, groups, most=len(groups),
                      pic=pic, was=was, older=older)
        return self._name(drawn, groups, here)

    def _name(self, drawn: list[dict], groups: list[list[int]],
              here: list[dict]) -> np.ndarray:
        """Give each group the name of the group it came out of.

        **A GROUP IS ITS SURFACES**, so the group that shares most of them with
        one she had is that one, still here.  Nothing is compared that was not
        already carried: `follow` named the surfaces and this only reads which
        of them ended up together.  Biggest overlap first and one claim each,
        which is the same refusal `again` makes one level down.

        `boxes` already returns an `id` --- the longest-seen surface in the
        group --- and it is not enough on its own: the moment that one surface
        is lost the whole thing is renamed, and a face she looked away from
        comes back a stranger.
        """
        # which surfaces each drawn box is made of.  `boxes` names a group by
        # its longest-seen surface, so that is the key back to the group.
        whose = {}
        for grp in groups:
            head = max(grp, key=lambda i: here[i].get("seen", 0.0))
            whose[here[head]["id"]] = frozenset(here[i]["id"] for i in grp)
        ours = [whose.get(t["id"], frozenset()) for t in drawn]

        was = self.names
        share = sorted(((len(ours[k] & then), k, name)
                        for k in range(len(drawn)) for name, then in was.items()
                        if ours[k] & then), reverse=True)
        mine: dict[int, frozenset] = {}
        got_name: dict[int, int] = {}
        used = set()
        for _shared, k, name in share:
            if k in got_name or name in used:
                continue
            got_name[k] = name
            used.add(name)
        for k in range(len(drawn)):
            if k not in got_name:
                self._named += 1
                got_name[k] = self._named
            mine[got_name[k]] = ours[k]
        self.names = mine
        # ...AND WHAT IT LOOKS LIKE, so a thing can be remembered by what it
        # resembles and not only recognised as the one from last look.  `boxes`
        # named this group after its longest-seen surface; this is that
        # surface's own `alike`, which is the bucket her tape is already
        # indexed on (`reps_by_alike`).  See `LOOKS`.
        seems = {k["id"]: float(k["row"][parts.ALIKE]) for k in here
                 if k.get("row") is not None}
        got = np.zeros((len(drawn), COLUMNS), np.float32)
        for k, t in enumerate(drawn):
            got[k] = (t["x"], t["y"], t["area"], t["w"], t["h"],
                      got_name[k], t["pieces"], t["carried"], t["seen"],
                      seems.get(t["id"], 0.0))
        return got


def held_again(held, things: np.ndarray):
    """The thing she was holding, in this look's things --- or `None`, which
    means she has lost it and is holding nothing.

    A LOOKUP AND NOT A MATCH, and that is the whole point of this file: a thing
    keeps its name from look to look, so "is that still it" stops being a guess
    re-argued every tick.  What used to be here --- `orient._closest`: her
    `alike` bucket, nearest in the viewport, `parts.like` to break a tie,
    mutual-nearest or nothing --- is `again` now, asked of every surface rather
    than of the one she happened to be holding.  The failure it refuses is
    unchanged and worth keeping written down: a tracker that cannot say no does
    not merely mis-report, it FOLLOWS the wrong thing and drives her eyes with
    it, and measured on her real body before the refusal existed she matched
    something on 15 of 16 ticks --- every match a fresh blob of wall sitting
    where the last one had been --- and her eyes walked to the corner of their
    travel and stayed there.

    LOSING IT IS A REAL ANSWER.  Her head throws her own view 105 degrees in a
    tick, so some looks genuinely do not contain what the last one held.  She is
    allowed to have lost it; `orient`'s reflex then picks what to look at next,
    which is exactly what it is for.
    """
    if held is None or things is None or not len(things):
        return None
    same = np.flatnonzero(things[:, ID] == float(held[ID]))
    return None if not len(same) else things[int(same[0])].copy()
