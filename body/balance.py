"""Her inner ear: which way she is being pulled, and which way she is turning.

Both are read IN HER OWN HEAD'S FRAME, and neither is a coordinate.  Nothing
here says "you are lying down" or "you are facing the window" --- a real
vestibular organ cannot say either, and being told would remove the one thing she
would otherwise have to work out.

**THE OTOLITHS** are a mass on hair cells.  They report which way that mass is
dragged, and they physically CANNOT separate gravity from acceleration --- both
are the same displacement of the same mass, and no amount of anatomy fixes that.
So this does not separate them either.  One reading, and telling "I am tipped"
from "I am being moved" is hers, exactly as it is ours.

**THE CANALS** report rotation RATE, not angle.  Angle is something she would
have to integrate, and whether she ever does is her business.

**HAIR CELLS FIRE ONE WAY ONLY.**  Bend them one direction and they fire; bend
them the other and they go quiet.  So every axis is an opposed pair, and every
line stays 0..1 like everything else she has.  The six names are for us reading a
log --- to her they are six lines that differ, and nothing more.
"""

from __future__ import annotations

import numpy as np

from .hearing import TICK_SECONDS
from .ragdoll import DT, G, STEPS_PER_TICK

#: What a full 1 g reads as, leaving room above it for a real jolt to be felt as
#: bigger than merely lying still.  A hair cell saturates; so does this.
ONE_G = 0.7
#: How fast a turn has to be to saturate a canal, in radians across a whole
#: tick.  About a third of a turn in 1.5 s --- brisk for a newborn's head, and
#: well short of what being picked up and swung would do.
FULL_TURN = 2.0
#: HOW SLUGGISH THE OTOLITH MASS IS --- the fraction of a jolt that reaches it
#: in one tick.  An otolith is a stone sitting in jelly on hair cells; it has
#: mass, the jelly has drag, and it physically CANNOT follow a spike lasting a
#: thirtieth of a second.  Nothing is being smoothed for tidiness: leaving it
#: out is what claims she has a massless organ.
#:
#: (her clock, so a rate here is a time of his and not a count of ticks)
#:
#: Measured 2026-09-01, why it is here: with the acceleration correct but
#: undamped, her own ragdoll's per-tick jitter still drove the reading to its
#: ceiling on 95% of ticks --- a solver wobble of four millimetres becomes
#: 1.4 g when a second difference is divided by a thirtieth of a second
#: squared.  It follows a real jolt within four ticks and ignores one tick of
#: tremor.
#:
#: A half-life in his seconds, so it settles the same at any clock.
SETTLES = 1.0 - 0.5 ** (TICK_SECONDS / 0.0268)


def _split(v: float) -> tuple[float, float]:
    """One signed reading as an opposed pair: `(positive, negative)`.  A hair
    cell bent the wrong way says nothing, it does not say minus something."""
    return (float(np.clip(v, 0.0, 1.0)), float(np.clip(-v, 0.0, 1.0)))


class Balance:
    """Her inner ear across ticks --- it has to remember, because acceleration
    is a CHANGE in how she is moving and one snapshot cannot hold a change.

    It owns the two positions it needs and nothing else.  It was a bare
    `read(ragdoll, was_at)` with the caller holding the memory, which is how
    the second position went missing for as long as it did.
    """

    def __init__(self) -> None:
        #: where her head was at the end of the last tick, and how far it had
        #: moved to get there.  Both `None` until she has lived twice ---
        #: before that there is no change to speak of and she feels gravity
        #: alone, which is what lying still feels like anyway.
        self.was_at = None
        self.was_moved = None
        #: what her otolith mass is actually doing right now, which lags what
        #: her skull is doing --- see `SETTLES`
        self.drag = None

    def read(self, ragdoll):
        """`(pulled, turning)` --- six lines each, in `DIRECTIONS` order."""
        return read(ragdoll, self)


def read(ragdoll, memory=None) -> tuple[dict[str, float], dict[str, float]]:
    """`(pulled, turning)` --- six lines each, in `DIRECTIONS` order.

    `memory` is a `Balance`, holding where her head was and how it was moving.
    Without one she feels gravity alone and being carried is silent.
    """
    up, right, fwd = ragdoll.frame()
    here = np.asarray(ragdoll.ear, np.float32)

    # what an otolith mass is actually dragged by: gravity, plus however hard
    # she is being accelerated.  A head thrown upward feels heavier; one in
    # free fall feels nothing at all, which is exactly right.
    #
    # ACCELERATION IS A SECOND DIFFERENCE, AND IT WAS A FIRST ONE.  This read
    # `moved / dt**2` --- but `moved` is how far she went in one tick, which is
    # a SPEED, and dividing a speed by dt is not an acceleration unless she was
    # motionless the tick before.  With `dt = 1/30` that multiplies her head's
    # every drift by NINE HUNDRED, so the clamp below was pinned and the organ
    # reported the same number for ever: measured 2026-09-01, `1.000` at the
    # median, at the 99th, at its biggest, lying still AND being dropped from
    # 1.1 m.  Six of her lines carried one constant.
    #
    # The change in how far she moves is the acceleration, and now free fall
    # cancels gravity exactly --- she goes WEIGHTLESS, which is the real signal
    # a falling body has and the one a newborn's Moro reflex fires on.
    drag = np.array([0.0, G, 0.0], np.float32)
    if memory is not None and memory.was_at is not None:
        moved = here - np.asarray(memory.was_at, np.float32)
        if memory.was_moved is not None:
            got = (moved - np.asarray(memory.was_moved, np.float32))
            drag = drag - got / (STEPS_PER_TICK * DT) ** 2
        memory.was_moved = moved
    if memory is not None:
        memory.was_at = here.copy()
        # ...AND THE STONE TAKES ITS TIME.  Her skull's jolt is not her
        # otolith's reading; the mass follows, it does not teleport.
        memory.drag = drag if memory.drag is None else             memory.drag + SETTLES * (drag - memory.drag)
        drag = memory.drag

    strength = float(np.linalg.norm(drag))
    felt = (drag / strength) * min(1.0, strength / abs(G) * ONE_G) if strength > 1e-6 \
        else np.zeros(3, np.float32)

    turn = np.asarray(ragdoll.turned, np.float32) / FULL_TURN

    pulled, turning = {}, {}
    # her own three axes, each as a pair.  `turned` arrives as
    # (about right, about up, about forward), so it is reordered to match.
    for names, along, spin in (
        (("up", "down"), float(felt @ up), float(turn[1])),
        (("left", "right"), -float(felt @ right), -float(turn[0])),
        (("front", "back"), float(felt @ fwd), float(turn[2])),
    ):
        pos, neg = names
        pulled[pos], pulled[neg] = _split(along)
        turning[pos], turning[neg] = _split(spin)
    return pulled, turning
