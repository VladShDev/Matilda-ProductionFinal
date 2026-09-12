"""THE VISUAL ORIENTING REFLEX: a born-in lean of her neck toward whatever is
most salient in her OWN eye, tuned by experience afterward, never replacing
it.

`contract/manifest.py`'s own WIRING docstring already names "orienting to a
sound" as one of "the few connections she is born with" --- but nothing ever
turned that intention into a line of code, for either sense.  A sibling
project (ni_v2) hit the identical gap and measured why nothing general ever
finds it on its own: with a real 600-tick life, mean yaw with a sound present
was statistically identical to mean yaw alone (0.195 vs 0.202).  Nothing in a
generic look-back rewards facing the right way, so there is no gradient for
it to discover --- exactly the way a real newborn does not LEARN to root or
to turn toward a sound either.  It is a reflex, present at birth, refined by
living with it.  ni_v2's fix (`choose.py`'s `ORIENT_BIAS`/`lateral_cue`) is
the direct ancestor of this file.

SIGHT, NOT SOUND, because sound is the wrong sense for what she needs it for
right now: the far bottle never sounds (`sounds=False` in every offer), so an
ear-only reflex would help her find a voice but not a meal.  Her retina is
already gated to salience by the contrast fix (Phase B): a flat, boring wall
already reads near-zero everywhere, so "the loudest cell in her own eye" IS
the newborn pop-out cue already, with nothing extra computed or peeked at ---
this reads her own `eye` sensor lines, the same ones her memory does, never
the room's ground truth.

WHY `neck.twist` AND NOT `neck.bend`/`neck.side`: `body/joints.py` names
`twist` the azimuth that turns the face about the head-neck bone --- the
"look toward something on your left or right" motion, regardless of how her
head currently happens to be tilted (she rests ATNR-turned).  `bend` nods,
`side` tilts an ear toward a shoulder; neither points her face at anything.

STILL A LEAN, NOT A COMMAND: same size as ni_v2's two innate leans (0.25), a
PULL toward the cue on top of whatever `choose.act` already decided, so a
confident memory can still overrule it, the same way `INNATE_BIAS` never
forced the withdrawal it leaned toward.

WHAT THE CUE IS TAKEN FROM, AND WHY IT IS NOT THE PICTURE ANY MORE
------------------------------------------------------------------
The paragraph above was true of an eye that returned `|raw - baseline|`.  Hers
returns the picture itself, so her whole view is bright and "stands out from
the rest of the view" stopped being a test anything could pass ---
`measured/looking.md` §1, 0 of 12 ticks, and `measured/tracking.md` §2, 0 of 10.

The obvious repair is to hand it the CHANGE between her last two looks instead,
and on a body whose eyes cannot move that works (15 of 16 ticks).  On a body
whose eyes CAN it is worse than having no eyes at all --- 26.8 degrees of error
against 25.9 welded --- because **a change detector on a moving eye sees its own
movement** (`measured/tracking.md` §7).  Turn her eyes and every cell in her
field changes at once, the peak lands at the edge of her travel, she turns
further toward it and changes everything again.

So she is told what she just did before she believes what she sees, which is
what a real one gets and what §7 said was missing.  Between her last look and
this one her view swung by exactly two things she already has:

    her own outgoing gaze order   she issued it, and `_glimpse` applies it
                                  straight to her look with no solver between
    her canals                    `turning`, which is `ragdoll.turned` --- her
                                  head's rotation from the end of one tick to
                                  the end of the next, and with `EYE_SLIDES`
                                  at 1 that is exactly the interval between her
                                  last glimpse and this one

Neither is the room and neither is a conclusion: one is a copy of what she
ordered, the other is an organ.  `light.Retina.changed` already says the
comparison belongs to the brain --- *"comparing them is a thing a brain does,
not a thing an eye does for it"* --- and this is the rest of that sentence: a
brain that compares two looks accounts for having moved between them.

AND THE OVERLAP IS THE ONLY GATE.  Un-rotating this look's rays into the last
look's frame leaves some of them pointing outside it --- she has swung her view
onto ground the last look never covered, so there is nothing there to have
changed.  Those cells are not compared.  That is geometry and not a threshold:
when she has swung her whole field away, nothing overlaps and there is no cue,
which is the honest answer and not a tuning.

Measured on her live body, her own brain driving (`measured/she_moved_too.md`):
between a third and a half of the difference between two of her looks is her own
movement.  Registering by her own lines leaves **0.49, 0.58 and 0.73** of the
unregistered residual in three runs of 19 ticks at different moments of one
life --- it depends on how hard she is thrashing, so it is a range and not a
number.  Within any one run every one of the eight sign combinations is worse
(0.898 .. 2.414 against 0.490).
"""
from __future__ import annotations

import numpy as np

from .light import _lift, _mod

#: NOTHING FROM `bind` IS IMPORTED HERE, AND THAT IS THE LAYERING.  `bind` reads
#: this file --- for her optics, which is where they were measured --- so this
#: file may not read `bind`.  What she is holding and whether it is still there
#: is `bind.held_again`; a reflex points her eyes and does not decide identity.

#: same size as the reflex it is modelled on --- a lean, not a command.
ORIENT_BIAS = 0.25
#: which way the pull actually turns her.  NOT derivable from the geometry
#: alone: `order` is a position in `neck.twist`'s own local (parent-frame)
#: axis, and which physical direction that axis's positive end points is a
#: fact about her joint chain at read-time, not about the room.  Set here
#: after a live measurement on the running body (facing-vs-bearing
#: agreement); flip the sign if a re-check ever finds it backwards.
SIGN = 1.0
#: the brightest cell must clear the rest of the view by this share of its
#: own peak before it counts as a real cue.  A view that is uniformly near
#: zero (Phase B's gate on a boring wall) must never be mistaken for "the
#: thing is at column 0" just because argmax has to return something.
STRENGTH_FLOOR = 0.15

#: HER EYE'S OWN THREE FACTS, and they are here for the reason `SIGN` is here:
#: they are facts about the body she was born with, her brain may not import
#: `body/` to ask, and a reflex that is present at birth has its calibration
#: present at birth too.  All three were MEASURED on the running body rather
#: than copied, which is also how they get re-checked: if her field of view,
#: her eye travel or her canals ever change, the probe says so and these lines
#: change with them.
#:
#:   how wide she sees   100.0 deg, corner to corner of her viewport
#:   how far eyes turn    35.0 deg at gaze.pan = 1.0        (`measured/tracking.md`)
#:   a full canal        114.6 deg of head turn across one tick
#:
#: The last one was fitted on her live body by asking which value cancels most
#: of her own movement out of the difference between two of her looks --- the
#: residual left over, as a share of the unregistered one:
#:
#:     0.5 rad  0.808     1.5 rad  0.646     2.5 rad  0.759     4.0 rad  1.509
#:     1.0 rad  0.704     2.0 rad  0.656     3.0 rad  0.934
#:
#: Flat-bottomed between 1.5 and 2.0 and steep either side of that, so it is a
#: real minimum and not a slope she is sitting on.
_FIELD = np.radians(100.0)
_REACH = np.radians(35.0)
_CANAL_FULL = 2.0
#: her viewport's edge in ray-space: her rays are laid out to `tan(FIELD/2)`,
#: so a thing at x = 1.0 is `arctan(_EDGE)` off her axis.  A conversion, not a
#: setting --- it moves only if her field of view does.
_EDGE = float(np.tan(_FIELD * 0.5))
#: a quarter turn: past it a ray points behind her and has no cell at all.  A
#: fact about a camera, not a setting.
_BEHIND = float(np.pi * 0.5)

#: the off-axis angle of every row and column, built once per shape.  A fact
#: about how her rays are laid out (`light._spread` with `FOVEA == 1` is linear
#: in tan across the field), not a state.
_LAID_OUT: dict = {}


def _across(n: int, xp=np) -> np.ndarray:
    # KEYED BY DEVICE AS WELL AS SIZE.  The same 512 laid out on the host and
    # on the card are two different arrays, and handing one to the other's
    # `meshgrid` is a crash at best and a silent host round-trip at worst.
    key = (n, xp.__name__)
    got = _LAID_OUT.get(key)
    if got is None:
        # SINGLE PRECISION, AND IT IS MEASURED, NOT ASSUMED.  Her field was
        # laid out in float64, which every array downstream inherits --- and a
        # GeForce runs double precision at a fraction of its float32 rate, so
        # the registration cost MORE on the card than on the host (25.6 ms
        # against 20.7).  In float32 the same registration is 3.04 ms on the
        # card against 20.12 on the host, and it returns THE SAME CUE on 6 of
        # 6 scenes tested --- her cue is one cell out of 262,144, not a
        # quantity, so the last bits of a tangent cannot move it.
        got = xp.arctan(_EDGE * xp.linspace(-1.0, 1.0, n, dtype=xp.float32))
        _LAID_OUT[key] = got
    return got


def turned(gaze_now, gaze_before, canals) -> tuple[float, float, float]:
    """How far her view swung between her last look and this one, radians:
    `(across, up, about her own axis)`, all in her view's own frame.

    Nothing here is looked up anywhere.  `gaze_*` are her own outgoing eye
    orders --- `_glimpse` rotates her look by `pan * EYE_REACH` toward her right
    and `tilt * EYE_REACH` upward, straight, with no solver in between, so a
    copy of what she ordered IS where her eyes went.  `canals` is her `turning`
    sensor, six lines by their own names, a rate split into opposed pairs
    because a hair cell fires one way only.

    THE SIGNS ARE HER ANATOMY AND WERE MEASURED, not derived.  `frame()` builds
    `right = cross(fwd, up)`, so a positive turn about `up` swings her face to
    her LEFT while a positive `gaze.pan` aims it to her right --- they subtract.
    All eight combinations were tried on her live body over one run's ticks and
    this is the only one that removes more than it adds (0.490 of the
    unregistered residual; the next best is 0.898 and the worst is 2.414).
    """
    return (float((gaze_now[0] - gaze_before[0]) * _REACH
                  - (canals["up"] - canals["down"]) * _CANAL_FULL),
            float((gaze_now[1] - gaze_before[1]) * _REACH
                  - (canals["left"] - canals["right"]) * _CANAL_FULL),
            float((canals["front"] - canals["back"]) * _CANAL_FULL))


def changed(eye_sheet: np.ndarray, before: np.ndarray | None, swing):
    """What changed in her view that her own movement does not explain, and
    where that question could be asked at all: `(energy, seen)`.

    `eye_sheet` and `before` are `(RETINA_H, RETINA_W, cones, slides)`, manifest
    order: the same array `light.py` builds and her tape stores, reshaped.
    Summed over cone and slide because colour and the last four glimpses do not
    matter for WHERE something is, only that it is there.

    With no previous look there is nothing to compare, and the answer is the
    picture itself --- which is what this reflex read for its whole life.

    THE LAST LOOK IS PUT BACK WHERE SHE WAS LOOKING WHEN SHE TOOK IT.  Every
    cell of this look is turned by `swing` back into the last look's frame and
    read there, so a thing that merely sat still while she swung past it lands
    on itself and cancels.  Her rays are laid out linearly in tan across her
    field, so the whole conversion is `arctan` out and `tan` back --- her own
    optics, the same two lines `follow` uses, and nothing tuned.

    AND WHAT LANDS OUTSIDE THE LAST LOOK IS NOT COMPARED.  She has swung her
    view onto ground her last one never covered; there is nothing there that
    could have changed, and a difference taken against the edge of an old
    picture is the artefact that pinned her gaze at the rail.  `seen` is where
    the two looks actually overlap.
    """
    # WHEREVER THE PICTURE LIVES, THIS RUNS THERE.  His standing order ---
    # *"run her only on GPU"* --- and `light.py`'s own idiom: same math, same
    # code, the array module taken from the data.  Measured 2026-09-01: this
    # registration is 20.7 ms on the host --- nearly two of her 11.1 ms ticks,
    # which is why it is not on the host any more --- and her clock
    # fell 22 -> 17 ticks a second when the reflex was wired.  Not one number
    # below changes; only where it is computed.
    xp = _mod(eye_sheet)
    energy = xp.abs(eye_sheet).sum(axis=(2, 3))            # (H, W)
    if before is None:
        return energy, xp.ones(energy.shape, bool)
    was = xp.abs(before).sum(axis=(2, 3))
    h, w = energy.shape
    if h <= 1 or w <= 1 or was.shape != energy.shape:
        return energy, xp.ones(energy.shape, bool)
    across, up, about = swing
    bx, by = xp.meshgrid(_across(w, xp), _across(h, xp)[::-1])
    # her own axis first: the whole picture turned about its middle
    turn, rise = np.cos(-about), np.sin(-about)
    ox = (turn * bx - rise * by) + across
    oy = (rise * bx + turn * by) + up
    # ...AND WHAT HAS SWUNG PAST A QUARTER TURN IS BEHIND HER.  `tan` comes back
    # through the other side there and would hand a ray at 143 degrees a cell
    # near the middle of the last picture --- a real cell, a real difference and
    # a complete fiction.  She can swing that far in one tick, so it is asked.
    ahead = (xp.abs(ox) < _BEHIND) & (xp.abs(oy) < _BEHIND)
    # back to a cell, the inverse of `_across`.  ROUNDED, not truncated: with no
    # swing at all every cell must land back on itself, and `tan(arctan(x))`
    # comes home a hair under as often as a hair over.
    cx = (xp.tan(xp.where(ahead, ox, 0.0)) / _EDGE + 1.0) * ((w - 1) / 2.0)
    cy = (1.0 - xp.tan(xp.where(ahead, oy, 0.0)) / _EDGE) * ((h - 1) / 2.0)
    seen = ahead & (cx >= -0.5) & (cx <= w - 0.5) & (cy >= -0.5) & (cy <= h - 0.5)
    col = xp.clip(xp.rint(cx), 0, w - 1).astype(xp.int32)
    row = xp.clip(xp.rint(cy), 0, h - 1).astype(xp.int32)
    return xp.abs(energy - was[row, col]), seen


def _stands_out(energy: np.ndarray, seen: np.ndarray):
    """The peak, and whether it is one: `(pan, tilt)` in -1..1, or `None`.

    ONE ANSWER TO "WHERE IS THE MOST SALIENT CELL", asked of whatever map it is
    handed (`salience_bearing`, its never-called twin, deleted 2026-09-03), so the
    registration is not paid for twice.  Row 0 is the top of her field, so tilt
    is negated: looking at something high means aiming UP.
    """
    xp = _mod(energy)
    h, w = energy.shape if energy.ndim == 2 else (0, 0)
    if h <= 1 or w <= 1 or int(seen.sum()) < 2:
        return None
    live = energy[seen]
    peak = float(live.max())
    if peak <= 1e-9:
        return None                                        # nothing seen at all
    if peak - float(live.mean()) < STRENGTH_FLOOR * peak:
        return None                                        # everywhere looks the same
    # the ONE number that comes back to the host: which cell won
    flat = int(xp.argmax(xp.where(seen, energy, -xp.inf)))
    row, col = flat // w, flat % w
    return (float(np.clip((col - (w - 1) / 2.0) / ((w - 1) / 2.0), -1.0, 1.0)),
            float(np.clip(((h - 1) / 2.0 - row) / ((h - 1) / 2.0), -1.0, 1.0)))


def salience_aim(eye_sheet: np.ndarray, before=None, swing=(0.0, 0.0, 0.0)):
    """WHERE to point her eyes: `(pan, tilt)` in -1..1, or `None`.

    A thing above her is as worth looking at as a thing beside her, and until
    her eyes existed there was nothing that could act on the vertical half, so
    it was never returned.

    `before` is her last look and `swing` is how far her view moved since, from
    `turned`.  Without them this is the reflex exactly as it shipped: the peak
    of the picture itself.
    """
    return _stands_out(*changed(eye_sheet, before, swing))




def look(want: np.ndarray, gaze_at: tuple, aim) -> np.ndarray:
    """Point her eyes at what stands out, on every slide of her gaze order.

    **THIS USED TO PULL HER NECK** (`lean`, below), and her neck is what made
    her blind: it never turned her head at all, it spun her chest, and her gaze
    swung 80 degrees a tick across a 100 degree field. An eye turns without
    shoving anything. It is also what her fovea is FOR --- the dense middle of
    her retina is only worth having if it lands on what she is examining.

    A PULL toward the target rather than a jump to it, the same shape `lean`
    always had: she has somewhere to get to, not merely more of something.
    """
    if aim is None:
        return want
    lo, hi = gaze_at
    out = want.copy()
    slides = max(1, (hi - lo) // 2)
    for k, v in enumerate(aim):
        target = 0.5 + 0.5 * SIGN * float(v)
        s = lo + k * slides
        out[s:s + slides] += ORIENT_BIAS * (target - out[s:s + slides])
    return np.clip(out, 0.0, 1.0).astype(np.float32)



def what_to_hold(found: np.ndarray, aim=None):
    """The thing she picks up, when she is not already holding one.

    **THE REFLEX ABOVE CHOOSES IT, AND THIS IS THE JOINT BETWEEN THE TWO
    HALVES.**  `salience_aim` answers "what stands out" and forgets it a tick
    later; `held_again` answers "is that still it" and cannot start.  Handing
    one to the other is the whole mechanism: the born-in reflex says WHICH
    thing, and holding on to it is what her eyes then do.  Nothing new decides
    anything --- and it keeps `orient` one idea rather than two.

    Measured, this is also the difference between tracking and drifting.  When
    she picked up whatever lay nearest her fovea she picked up WALL, because in
    a nursery most of what is under the middle of her eye is nursery: a large
    self-similar region whose pieces cannot be told from their neighbours look
    to look, so her hold walked patch by patch and her eyes walked to the rail
    with it.  The salience peak lands on the thing that is not the room.

    WITHOUT A CUE, THE FOVEA, and that is not a fallback invented here either:
    `body/light.py` builds her retina dense in the middle and says what the
    dense middle is for --- it "is only worth having if it lands on the thing
    being examined".  With nothing standing out, what she is examining is
    whatever her fovea is on.

    Area breaks a tie, because two things equally close to the cue are not
    equally looked at.  `bind.Seen.look` hands these over most-moving first
    rather than biggest first --- the reason is written up in `bind.boxes`, and
    it is why size is asked for here explicitly rather than taken from the
    order they arrive in.
    """
    if found is None or not len(found):
        return None
    # `salience_aim`'s tilt counts UP from the middle and `find`'s y counts
    # DOWN from the top row.  One negation, in the one place the two meet.
    at = (np.zeros(2, np.float32) if aim is None
          else np.array([float(aim[0]), -float(aim[1])], np.float32))
    off = np.linalg.norm(found[:, :2] - at, axis=1)
    close = np.flatnonzero(off <= off.min() + 1e-6)
    return found[int(close[int(np.argmax(found[close, 2]))])].copy()


def follow(want: np.ndarray, gaze_at: tuple, at, thing) -> np.ndarray:
    """Put her eyes where the thing she is holding actually is.

    `at` is where her eyes are ALREADY pointed, `(pan, tilt)` in -1..1 of
    `EYE_REACH` --- and it has to be, because this is a correction and not a
    destination.  `run.py` rebuilds her decision from `choose` every tick and
    re-centres any gaze line nobody drove, so a pull laid on 0.5 each tick can
    only ever reach one small offset and then let go: measured, it saturated at
    8.75 degrees, and she leaned toward what changed without ever holding her
    eyes on it.  An eye that is tracking does not lean, it STAYS, and staying
    means this tick's order starts from where the last one left her.

    THE ERROR IS THE THING'S OWN COORDINATES, converted and not tuned.  `find`
    gives its place as -1..1 of her viewport; her rays are laid out to
    `tan(FIELD/2)` at the edge, so the angle to it is `arctan(x tan(FIELD/2))`,
    and `EYE_REACH` is what one unit of `gaze.pan` is worth.  Every number in
    that sentence is a fact about her eye that was already written down.

    ALL OF IT, NOT A FRACTION.  A partial correction is what a lean is, and the
    reason to move an eye rather than a neck is that an eye can simply be
    pointed --- if she overshoots, the next look says so, which is the only
    signal a real one gets either.

    AND WHAT HER EYES CANNOT REACH IS HANDED BACK, not thrown away --- it is
    the second return, and nothing reads it yet (`turn_head`, built to give it
    to her neck, was never called and was deleted 2026-09-03).  An eye has 35 degrees
    and a thing can be anywhere, so a tracker that only ever wrote to the eyes
    would peg them at the edge of their travel and sit there, which is measured
    and is exactly what it did.
    """
    if thing is None:
        return want, 0.0
    lo, hi = gaze_at
    out = want.copy()
    err = np.arctan(np.asarray(thing[:2], np.float64) * _EDGE) / _REACH
    # her viewport's y counts DOWN from the top and her tilt counts up.
    reach = (float(at[0]) + float(err[0]), float(at[1]) - float(err[1]))
    aim = np.clip(reach, -1.0, 1.0)
    slides = max(1, (hi - lo) // 2)
    for k, v in enumerate(aim):
        s = lo + k * slides
        out[s:s + slides] = 0.5 + 0.5 * SIGN * float(v)
    return (np.clip(out, 0.0, 1.0).astype(np.float32), reach[0] - float(aim[0]))


#: WHICH WAY HER NECK TURNS HER FACE, and it is not her eyes' way --- measured
#: on the running body 2026-08-15, one axis ordered and held, everything else
#: limp, read off the same spindles she reads:
#:
#:     neck.twist ordered 0.95   achieved 0.754   her face yaw  -35.7 deg
#:     neck.twist ordered 0.05   achieved 0.246   her face yaw  +35.6 deg
#:
#: Antisymmetric to a tenth of a degree, and INVERTED against `gaze.pan`, where
#: a positive order aims at her right.  Hence its own constant: one sign for
#: two different pieces of anatomy would be a coincidence, not a fact.
NECK_SIGN = -1.0


