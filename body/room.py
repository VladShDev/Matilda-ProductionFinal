"""Her room: a floor, four walls, a cot, and whatever you put in it.

Small and nearly empty on purpose.  Everything in here is PASSIVE --- it stops
her, it does not move her and it never decides anything.  Gravity, the floor,
the rails and a ball in the way are all the same kind of fact: a place her body
cannot be.

THE COT HAS SIDES NOW (2026-08-16, the owner's call).  The bed was a mattress
in the open, and the viewer drew it as one: the page's own comment refused to
draw slats because *"draw a bar and the page promises her arm a surface her
physics has never heard of."*  So the surface came first.  `COT` below is real
geometry, `cot()` is the same `stop()` the floor and the walls go through, and
the page draws boxes at the numbers `as_json()` sends.  Nothing about her
changed to carry it: a slat arrives on her skin as `pressed`, exactly like the
mattress, because being refused a place is the only thing any of this ever was.

WHY THE RAILS.  Unlike gravity, muscle pull has no fixed direction, and real
coordinated pulling against floor friction accumulates genuine net translation
over enough substeps --- a real body scooting, not a bug.  Nothing else bounds
where her own frame ends up.  Measured in the previous project: a real driven
life, seeded four ways, walked her 4 to 17 metres from the crib inside about a
hundred ticks.  The rails sit wider than the mattress so a limp body can still
sprawl a limb off the edge onto the floor beside it --- just not across the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: metres. She is about 0.6 m long.  HALF-extents.
#: 2.0 x 1.4 m of floor until 2026-08-11 morning ("too small"); 3.2 x 2.4
#: that afternoon; then the owner's evening order: "make room bigger x5 and
#: add walls --- she has to know why she stops," which made it 16 x 12.
#: Squared up to a true 20 x 20 at the Matilda fork, to match the room as
#: originally specified rather than the aspect ratio that grew by successive
#: doublings.  Same reasoning as before applies at the new size: a hall a
#: scooting baby cannot cross by accident, so travel is a REAL undertaking
#: with a visible destination.  The second half of the original order lives
#: in `light._planes`: walls and floor carry a two-tone pattern, because a
#: flat-coloured wall filling her view is ZERO CONTRAST --- she used to stop
#: against a surface her retina honestly could not see.
#: Ceiling stays 2.5 m: rooms grow sideways, not upward.
#: World-side and fingerprint-neutral; her retina re-baselines on restart.
#: NOTE: several `ni_one/measured/*.md` mobility numbers were captured at the
#: old 16x12 size and are about HER, not the room, so they should hold
#: qualitatively --- but a resize is a legitimate trigger to re-run them, not
#: something to assume still holds exactly.
ROOM = {"x": 10.0, "y": 2.5, "z": 10.0}
#: HIS TRAINING PEN, 2026-08-30 --- the square in front of her mirror,
#: closed on two sides by what exists (the mirror's +x wall, the +z wall)
#: and on the two open sides by LOW FENCES, *"little bit lower than she...
#: same way like bed do"*: real to her body, real to her eye, a doorway on
#: the mom's side so it is a fence and not a box.
#: ...and NO DOOR --- his correction: *"when I don't watch her, she would
#: be able to leave training space, but I don't want it."*  The mom's
#: carrying hands lift her over the fence; nothing walks out.
PEN = {"x0": 8.0, "x1": 10.0, "z0": 8.0, "z1": 10.0, "h": 0.40}
#: The mattress she is laid on.
BED = {"x": 0.42, "z": 0.28, "top": 0.34}
#: How much it dishes toward the middle.  An infant sleeping surface is a
#: shallow NEST, not a table, and the difference is not decoration: a real
#: newborn cannot roll over for months, but a limp stick figure with a
#: spherical head rolls at once.  Measured on a flat mattress --- laid
#: carefully on her back, she was face down by tick 3 and off the edge by tick
#: 4, and everything she could see after that was bedding.
BASIN = 0.05
#: ...and its sides SLOPE to the floor rather than dropping.  As a cliff it
#: teleported any joint crossing the edge a third of a metre in one solver
#: pass: with every muscle at zero her pelvis went 0.000 -> -0.152 -> 0.085 ->
#: -0.177 -> 0.288 and she had walked into the wall by tick 30.
SKIRT = 0.45
#: Things this small YIELD to her body instead of stopping it.  A bottle, a
#: toy, a rattle --- all of them together weigh less than her head, and a
#: room where a 100-gram bottle is an immovable wall is not a room, it is a
#: minefield.  Anything larger (a future stool, a crib wall) stays solid,
#: because a thing that can bear her weight refusing to move IS the fact
#: about it.  See `stop()` for what this replaced and the measured damage.
YIELDS = 0.12
#: HER COT'S SIDES: uprights, and a rail across the top of them.
#:
#: THE GAP IS THE WHOLE POINT.  A real cot is spaced so a limb goes through and
#: a head does not, and hers is the same: 0.058 m clear, against a hand 0.048
#: across, an elbow 0.044, a knee 0.052, a foot 0.056 --- and a neck 0.060, a
#: pelvis 0.100, a chest 0.104, a head 0.110.  Everything she can put through it
#: she can pull back; her body cannot follow, which is the first fact about a
#: container there is.  A solid wall would have cost the same and told her
#: nothing.
#:
#: AND IT SAYS 0.058 AND MEANT IT ONLY FROM 2026-08-17.  Measured that morning
#: (`measured/cot.md`), the sides passed everything up to 0.0684 and 0.0704
#: across --- 10.4 and 10.6 mm more than they claim, her NECK straight through a
#: gap her head cannot follow through, which is what this spacing is FOR.
#: It was never the number: it was `cot()` answering a squeeze in the middle of
#: a gap with a push straight ALONG the wall.  The fix is written up there and
#: sits in `cot()`; the sizes above are now the sizes she meets.
#:
#: MEASURED (`measured/cot.md`).  120 driven ticks, three seeds, identical
#: orders: without sides she scooted 5.1, 5.6 and 9.2 metres from the mattress
#: and every joint ended up outside it.  With them her pelvis never passed
#: |x| 0.36, |z| 0.23 and nothing was outside the frame at the end.  Limp, it
#: changes nothing at all: she settles in the same place either way, with her
#: feet through the end bars, which is what a foot 0.056 across in a 0.058 gap
#: does.
#:
#: HEIGHT.  0.42 m of side above a mattress her whole body is 0.6 m long on:
#: over her standing centre, so she is contained, and under her sitting reach,
#: so the top rail is something she can have her hands on.  `low` is where the
#: front comes to when a person drops it --- a lip to lean over, not a way out.
COT = {
    "rail": 0.42,    # how far the top of a side stands above the mattress
    "low": 0.14,     # ...and where the front one comes to, dropped
    "slat": 0.011,   # radius of an upright, and of the rail lying across them
    "gap": 0.058,    # the clear space between two uprights
}


#: HER MIRROR, and it is a real thing in her room now.
#:
#: It was two square metres of flat blue drawn ONLY in the spectator's page, on
#: a wall ten metres away, in no `.py` file anywhere: she could not see it, it
#: reflected nothing, and her body did not know it existed.  The page's own
#: comment admitted the placement was a guess --- *"neither corner nor wall was
#: specified"*.
#:
#: IT STAYS ON THE WALL, AND THAT IS THE OWNER'S CALL.  It was moved onto her
#: cot side once, on the reasoning that a 2 m panel ten metres away subtends
#: eleven degrees and is not much to learn from --- and his answer settled it:
#: *"before mirror was huge on a wall, why you relocate it? i cant point her
#: there."*  A mirror she can be CARRIED TO is a thing he can use; one bolted to
#: her cot is a thing that happens to her.  What *"properly"* meant was that it
#: reflects, not that it moves.
#:
#: Two centimetres proud of the wall, which is what stops the two z-fighting in
#: the drawing and what `light._reflect` steps over on the way in.
MIRROR = {
    "axis": 0,                       # mounted on the +x wall, facing -x
    "at": ROOM["x"] - 0.02,
    "y": 1.0,
    "z": ROOM["z"] - 1.0,            # into the corner, where it has always been
    "half_w": 1.0,                   # 2 x 2 m, the size it has always been
    "half_h": 1.0,
}


def _side(axis: int, at: float, along: float) -> dict:
    """One side of it: which way it faces, where its face is, how long it is,
    and where every upright in it stands.

    Evenly spaced with one at each end, at whichever whole count comes nearest
    the wanted gap --- so a side is symmetric, its corners are posts, and the
    gap is a consequence of the cot's real size rather than a number that
    happens not to divide it.
    """
    n = max(1, round(2.0 * along / (2.0 * COT["slat"] + COT["gap"])))
    step = 2.0 * along / n
    return {"axis": axis, "at": at, "along": along, "step": step,
            "slats": [round(-along + i * step, 5) for i in range(n + 1)]}


#: The four of them, in two facing pairs: the ends across her, then the sides
#: along her.  The sides stop a slat's width short so that no two uprights
#: stand in the same place --- the corner belongs to the end, which is how a
#: cot is actually built.
SIDES = [_side(0, -BED["x"], BED["z"]), _side(0, BED["x"], BED["z"]),
         _side(2, -BED["z"], BED["x"] - COT["slat"]),
         _side(2, BED["z"], BED["x"] - COT["slat"])]
#: ...and the same four as the arrays `cot()` reads, PAIRED, because it reads
#: them 720 times a tick (8 solver passes x 90 substeps) and at nineteen joints
#: what a numpy line costs is the CALL, not the arithmetic.  Shaped `(pair,
#: face)` against her `(joints, 1, 1)`, the two facing sides of a pair share
#: every constant that does not depend on which way they face, and both her
#: (x, z) and her (z, x) come out of `pos` as plain strided views instead of
#: two fancy-index copies.
_PLANE = np.array([[s["at"] for s in SIDES[:2]],
                   [s["at"] for s in SIDES[2:]]], np.float32)
_AHALF = np.array([[SIDES[0]["along"]], [SIDES[2]["along"]]], np.float32)
_STEP = np.array([[SIDES[0]["step"]], [SIDES[2]["step"]]], np.float32)
_HALFSTEP = _STEP * 0.5
#: where the fold starts, so `mod(a + _AOFF, step) - halfstep` is her SIGNED
#: offset from the NEAREST upright --- which is what turns eleven of them into
#: one question, and why a side may have as many as it likes
_AOFF = _AHALF + _HALFSTEP
#: ...and where a side runs out, so the fold does not go on inventing uprights
#: out past the corner where there is nothing
_ENDS = _AHALF + COT["slat"]
#: which way is OUT of the cot for each face, for the one joint in a million
#: that lands exactly on a bar's axis and has no direction of its own
_OUT = np.sign(_PLANE)


@dataclass
class Thing:
    """Something you put in her room.  A sphere, because a sphere is the only
    shape that needs no orientation to be in the way."""

    what: str
    at: np.ndarray
    size: float = 0.08
    #: WHETHER IT ANSWERS BEING TOUCHED.  A rattle is not decoration --- it is
    #: the first thing in her world whose behaviour depends on HER.  Milk and
    #: a mother happen to her on someone else's clock; a rattle only ever
    #: sounds because she moved, and that is the whole reason a baby reaches.
    #: The room holds no opinion about whether that is good.  It rattles.
    sounds: bool = False
    #: WHETHER IT FEEDS HER WHEN HER MOUTH FINDS IT, and how much is still in
    #: it.  The rattle is only half the reason a baby reaches: measured over
    #: 10,376 ticks (`measured/reaching.md`), a thing that pays purely in
    #: novelty bought her a 2.5x rise in reach and then lost all of it, which
    #: is what a habituating reward looks like.  The other half is WANTING:
    #: rooting for milk is the oldest wanted thing there is, and it is the
    #: same touch at the same mouth the mother's milk already arrives as.
    #:
    #: It EMPTIES as she drinks.  A bottle that never runs out would abolish
    #: hunger, and a baby with no hunger has nothing to move for --- the exact
    #: trap the mother's habituation was built to get out of.  Refilling it is
    #: a person's job, like putting it there was.
    feeds: float = 0.0
    left: float = 0.0
    #: ITS SHAPE --- "ball", "cube" or "pyramid".  His ask, 2026-09-13: the
    #: pen's three things as three shapes, so her eye has different shapes to
    #: look at.  Her flesh still meets each as its bounding sphere (`size`);
    #: her eye and the page draw the shape itself.
    shape: str = "ball"

    def __post_init__(self) -> None:
        self.at = np.asarray(self.at, np.float32).reshape(3)

    def as_json(self) -> dict:
        return {"what": self.what, "at": self.at.tolist(), "size": float(self.size),
                "sounds": bool(self.sounds), "feeds": float(self.feeds),
                "left": round(float(self.left), 4), "shape": str(self.shape)}


@dataclass
class Room:
    """Where she is.  It holds no opinion about her."""

    things: dict[str, Thing] = field(default_factory=dict)
    #: HOW HIGH THE FRONT SIDE STANDS, above the mattress.  The one part of a
    #: cot that moves --- and it moves because a person moved it, so the room
    #: still decides nothing.  Up to begin with: a side left down is a 0.34 m
    #: fall onto a floor, and she cannot yet choose not to take it.
    rail: float = COT["rail"]

    def __post_init__(self) -> None:
        self.drop(self.rail)

    def drop(self, to: float | bool = COT["low"]) -> float:
        """Slide the front side.  `True` puts it back up, `False` drops it, a
        number is metres above the mattress --- never below it, never above the
        other three.  It is a place, not a behaviour: what changes is where her
        body can be, and she finds that out the way she finds out about a wall.
        """
        if to is True:
            to = COT["rail"]
        elif to is False:
            to = COT["low"]
        self.rail = float(min(max(float(to), 0.0), COT["rail"]))
        up = BED["top"] + COT["rail"]
        #: where each side's top edge is, off the floor --- `(pair, face)`, and
        #: the front is the far face of the second pair
        self._top = np.array([[up, up], [up, BED["top"] + self.rail]], np.float32)
        #: ...the axis of the rail lying across it, a slat's radius below that...
        self._axle = self._top - COT["slat"]
        #: ...and the height where its uprights stop and that rail takes over
        self._band = self._axle - COT["slat"]
        return self.rail

    @property
    def rest_top(self) -> float:
        """Where a newborn is laid down.  She is put on the bed, not built in
        mid-air --- a body assembled in the air is crushed into itself by the
        floor on the very first substep."""
        return BED["top"]

    # --- what is under her ----------------------------------------------------

    def under(self, pos: np.ndarray, radius: np.ndarray) -> np.ndarray:
        """How high the ground is beneath each joint.

        A nest in the middle, sloping to the floor at the sides, and continuous
        everywhere --- so lying on the edge is lying on the edge, and a body
        that drifts off-centre is gently returned rather than thrown.
        """
        out = np.hypot(pos[:, 0] / BED["x"], pos[:, 2] / BED["z"])
        rim = np.clip(1.0 - (out - 1.0) / SKIRT, 0.0, 1.0)
        # THE BOWL FADES WITH THE RIM.  His find, 2026-09-13: the nest's 5 cm
        # (BASIN) was added everywhere, so outside the cot --- the pen, the open
        # floor --- her ground was 0.05 m above the floor her eye draws at 0.0:
        # she stood and reached on an invisible plane, a ball resting on the
        # floor sat below her hand's stop, and things looked "deeper than the
        # floor".  Outside the skirt the ground is now the room floor, the same
        # 0.0 the render uses (`light._SIDES`).  The nest itself is unchanged:
        # 0.34 at the centre, 0.39 at the rim, then down to 0.
        return rim * (BED["top"] + BASIN * np.clip(out, 0.0, 1.0) ** 2) + radius

    def stop(self, pos: np.ndarray, radius: np.ndarray, pinned: np.ndarray) -> None:
        """Put back anything that has ended up somewhere it cannot be.

        Position correction only, every solver pass --- bones and muscles can
        push a joint back through a surface just as they can push one back below
        the floor.  Nothing here measures an impact or a friction: those are
        measured once per substep in the ragdoll, because this runs eight times a
        substep and a resting joint nudged a millimetre would otherwise register
        a fresh landing every single pass.

        THE BED IS A SOLID BOX, NOT A HEIGHT.  It was a height field first, the
        way the previous project had it, and that put a 0.34 m CLIFF at the
        mattress edge: a joint drifting across it was teleported a third of a
        metre upward in one pass, which is an enormous injection of energy.
        MEASURED, with every muscle at zero: her pelvis went 0.000 -> -0.152 ->
        0.085 -> -0.177 -> 0.288 and she had walked into the wall by tick 30.  A
        limp newborn bouncing herself off her own bed is not physics.  As a box
        she is pushed out along whichever face she is least far through, which
        is continuous everywhere, so lying on the edge is lying on the edge.
        """
        free = ~pinned
        was = pos.copy()

        # A WALL IS AS THICK-AWARE AS THE FLOOR IS, and for a long time it was
        # not.  These three lines clamped a joint's CENTRE to the plane and
        # ignored `radius` entirely, while `under()` twelve lines down has
        # always ended `+ radius` --- so the four walls and the ceiling were
        # the only surfaces in her room a body could sink into by its own
        # thickness.  The owner, watching her live: *"why she stuck in wall"*.
        #
        # MEASURED off her own `/pose` at the moment he asked, and every
        # overhang was exactly that joint's own radius: her left hand's centre
        # sat at x = -10.0000 against a wall at 10.0 with r 0.024, her elbow
        # 0.0186 through with r 0.022, her toes 0.0211 with r 0.022.  Four
        # joints half buried in plaster.  Nothing raises, nothing scores it,
        # and it looks from the page exactly like a baby inside a wall ---
        # because that is what it was.
        #
        # Her skin stops where the surface is, on all six faces now.  This does
        # not touch WHY she is at a wall, which is a different thing entirely:
        # a drift with no preferred direction in a closed box ends at the
        # boundary and stays, and nothing in here returns her.
        edge = radius[free]
        pos[free, 0] = np.clip(pos[free, 0], -ROOM["x"] + edge, ROOM["x"] - edge)
        pos[free, 2] = np.clip(pos[free, 2], -ROOM["z"] + edge, ROOM["z"] - edge)
        pos[free, 1] = np.minimum(pos[free, 1], ROOM["y"] - edge)

        # ...AND NEITHER DOES THE GROUND YIELD TO A HAND.  `free` here was the
        # other half of the same hole: with the bars closed, a hand simply
        # pulled her DOWN THROUGH THE MATTRESS and out from under them.  A
        # floor that only pushes UP cannot interfere with lifting her --- a
        # lifted baby is above it and this never fires --- so nothing about
        # picking her up changes, and pushing her through a solid stops being
        # possible.  Measured, dragging her sideways at a hand's speed: she
        # ended 0.80 m outside with the bars alone, and stays in with this.
        ground = self.under(pos, radius)
        below = pos[:, 1] < ground
        if np.any(below):
            pos[below, 1] = ground[below]

        # THE PEN'S TWO FENCES --- thin walls that END at PEN["h"]: above
        # it nothing (a carried body passes, a climb is possible), below it
        # her skin stops at the plane from either side, exactly as at any
        # wall.  The doorway in the south fence stays open.
        lowJ = free & (pos[:, 1] - radius < PEN["h"])
        if np.any(lowJ):
            r9 = radius
            # west fence: the plane x = x0, spanning z0..z1
            m = lowJ & (pos[:, 2] > PEN["z0"]) & (pos[:, 2] < PEN["z1"]) \
                & (np.abs(pos[:, 0] - PEN["x0"]) < r9)
            if np.any(m):
                inside = pos[m, 0] >= PEN["x0"]
                pos[m, 0] = np.where(inside, PEN["x0"] + r9[m],
                                     PEN["x0"] - r9[m])
            # south fence: the plane z = z0, spanning x0..x1, no door
            m = lowJ & (pos[:, 0] > PEN["x0"]) & (pos[:, 0] < PEN["x1"]) \
                & (np.abs(pos[:, 2] - PEN["z0"]) < r9)
            if np.any(m):
                inside = pos[m, 2] >= PEN["z0"]
                pos[m, 2] = np.where(inside, PEN["z0"] + r9[m],
                                     PEN["z0"] - r9[m])

        # HER COT STOPS A HAND TOO, AND ONLY HER COT DOES.
        #
        # `free = ~pinned` is right for the floor and the walls: a person may
        # lift a baby, and a lifted baby is not standing on anything.  It was
        # wrong here, because a bar is not a floor --- YOU CANNOT PULL A BABY
        # SIDEWAYS THROUGH THE BARS OF A COT.  You lift her over the rail, and
        # above the rail this returns her untouched, so that still works.
        #
        # MEASURED, and it is the whole reason she was not in her cot:
        # `measured/cot.md` recorded that a hand may put any joint of her
        # anywhere in twenty metres of room, instantaneously, because the page's
        # `toRoom` clamps to the ROOM and `stop()` began with `free = ~pinned`
        # --- so one sideways drag ended her containment for the rest of the
        # life, "and nothing ever puts her back."  Found again on 2026-08-18, in
        # a life three hours old: her pelvis at (9.37, 0.10, -9.92), 13.65 m
        # from her cot, 0.08 m off a wall, lying on bare floor.  Every complaint
        # about that life --- *"she everytime going to wall"*, *"she don't see
        # tv anymore"* --- was about a baby who was not in the room the
        # measurements assumed.  She was at the wall because she was AT the
        # wall, and 17 m from the television.
        self.cot(pos, radius, np.ones(len(pos), bool))

        for thing in self.things.values():
            away = pos - thing.at
            gap = np.linalg.norm(away, axis=1)
            inside = (gap < thing.size + radius) & free
            if not np.any(inside):
                continue
            if thing.size <= YIELDS:
                # A SMALL THING YIELDS TO HER BODY, never the other way round.
                #
                # It was a teleport the other way: any joint inside the sphere
                # SNAPPED to its surface --- position moved, the velocity that
                # fabricates left in place --- eight passes a substep, ninety
                # substeps a tick.  The bed had this exact disease (the 0.34 m
                # cliff above) and was cured; things never were.  MEASURED
                # (audit_fast, 2026-08-11): a bottle held at her lips made her
                # face travel 0.25 m A TICK, max 0.81 m, on a body that is
                # 0.0000 m/tick still without it.  Every emergency feeding of
                # her solitude was a beating, and the milk mostly missed
                # because her mouth was being hurled away from it.
                deep = np.where(inside, (thing.size + radius) - gap, -1.0)
                j = int(np.argmax(deep))
                dir_ = thing.at - pos[j]
                span = float(np.linalg.norm(dir_))
                dir_ = (dir_ / span if span > 1e-6
                        else np.array([0.0, 1.0, 0.0], np.float32))
                thing.at = (pos[j] + dir_ * (thing.size + radius[j])).astype(
                    np.float32)
                thing.at[0] = float(np.clip(thing.at[0], -ROOM["x"], ROOM["x"]))
                thing.at[2] = float(np.clip(thing.at[2], -ROOM["z"], ROOM["z"]))
                thing.at[1] = float(np.clip(thing.at[1], thing.size, ROOM["y"]))
                continue
            # ...anything big enough to bear her weight stays solid: pushed
            # straight out along the line from its centre.  A joint exactly at
            # the centre gets pushed up, because it has to go somewhere.
            out = np.where(gap[:, None] > 1e-6, away / np.maximum(gap, 1e-6)[:, None],
                           np.array([0.0, 1.0, 0.0], np.float32))
            pos[inside] = thing.at + out[inside] * (thing.size + radius[inside])[:, None]

        # HOW HARD THE ROOM HAD TO PUSH HER BACK --- in metres, per joint.
        #
        # It is friction's answer and touch's answer at once, and they are the
        # same fact: a joint the room did not have to move is not in contact
        # with anything, and one it had to shove is being pressed.  This is the
        # same shape as a muscle's `force` --- what she ordered against what she
        # got --- which is not a coincidence: both are the world refusing.
        #
        # Friction was restricted to joints below the floor line first, and
        # everything resting against the bed's SIDE then had none at all:
        # MEASURED, her pelvis slid a steady 0.018 m/tick in one direction and
        # never slowed until it hit the wall.  Contact is contact.
        return np.linalg.norm(pos - was, axis=1).astype(np.float32)

    def cot(self, pos: np.ndarray, radius: np.ndarray, free: np.ndarray) -> None:
        """Her cot's four sides: uprights, and a rail across the top of them.

        THE SAME PATH AS THE BED AND THE WALLS.  Called from `stop()`, it moves
        positions and nothing else, and what it costs her comes back the way the
        floor's refusal does --- as the distance the room had to move her, which
        her skin reads as `pressed`.  No new sense, no new line, no new
        fingerprint: a slat is a place her body cannot be, and that is all any
        of this has ever been.

        EVERY BAR IS ROUND, AND THAT IS THE WHOLE TRICK.  A square post has to
        be answered with "which face is she least far through" --- three
        distances, an argmax, a three-way branch per side, and a step the width
        of a bar wherever the answer changes.  A dowel has ONE answer at every
        point: straight out from its axis, by however much of her is inside it.
        Smooth everywhere, it is what a cot bar is anyway, and MEASURED it is
        what makes this affordable at all: as boxes it cost +77.8 us on every one
        of the 720 calls a tick, as dowels +56.3 with her lying against a side
        and +12.9 with her clear of them (`measured/cot.md`).

        SO THERE ARE ONLY EVER TWO DISTANCES.  Across the side, always; and then
        either ALONG it --- down among the uprights, where the fold turns all
        eleven of them into one question --- or UP IT, in the band at the top
        where the rail lies across them as one long horizontal dowel.  A hand
        that comes down on the rail is lifted by the same line that pushes a
        knee back out of a bar.

        AND IT OPENS WITH THE CHEAPEST QUESTION IT HAS: is any part of her even
        within a bar's reach of a face.  A baby lying near the middle answers no
        and pays four lines; one who has settled against a side answers yes on
        every pass, so that is a floor on the cost, not a description of it.
        """
        r = radius[:, None, None]
        xz = pos[:, ::2]                          # her (x, z), a plain view
        across = xz[:, :, None] - _PLANE          # how far out from each face
        if not np.any(np.abs(across) - r < COT["slat"]):
            return

        a = xz[:, ::-1, None]                     # ...and how far along it
        y = pos[:, 1, None, None]
        # WHICH OF THE TWO SHE IS BESIDE.  Below the rail the second distance is
        # her offset from the nearest upright --- folding the side's length into
        # one spacing asks about every one of them at once.  In the rail's own
        # band it is her height above the rail's axis instead.
        grip = y > self._band
        beside = np.where(grip, y - self._axle,
                          np.mod(a + _AOFF, _STEP) - _HALFSTEP)
        along = np.abs(a) - r < _ENDS
        skin = COT["slat"] + r                    # how near an axis she may come
        # A GAP TOO NARROW FOR A PART OF HER IS A WALL TO THAT PART, and no
        # single bar can say so.
        #
        # A round bar answers everything radially, which is the whole trick
        # below --- and radially from ONE axis is straight ALONG the side at
        # exactly the place a body is really trying to get through: the middle
        # of a gap, equidistant from both uprights.  There is no outward
        # component left there at all.  So she was pushed sideways, the fold
        # flipped to the other upright, she was pushed back, and whatever was
        # pushing walked her out between the two.  Both uprights at once does
        # not fix it: their sideways pushes cancel and neither has any outward
        # part to contribute (measured --- 0.0342 before, 0.0342 after).
        #
        # MEASURED (`measured/cot.md`), one joint shoved at a side 2 mm a pass:
        # the ends passed EVERY radius up to 0.0342 and the long sides up to
        # 0.0352, against a geometric limit of 0.0290 and 0.0299.  10.4 and
        # 10.6 mm of diameter that are not there --- and HER NECK IS 0.030
        # ACROSS, the middle of that band, so her neck went through a gap her
        # head cannot follow through, which is the one thing a cot's spacing
        # exists to prevent.
        #
        # So a joint whose skin does not fit between two axes meets the side as
        # a SURFACE instead: scalloped, `sqrt(skin^2 - beside^2)` off the face,
        # which is the real shape of a row of dowels and is the same number the
        # bar rule was already demanding --- only with a direction that leads
        # somewhere.  It is continuous along the side and across it, it is the
        # bed's own least-far-through rule, and it is not a second answer to
        # "where is a bar": below the rail these two are the same surface, so
        # exactly one of them speaks.  Nothing she can already fit through a
        # gap is touched by any of it.
        tight = (skin > _HALFSTEP) & ~grip
        gap = np.hypot(across, beside) - r
        inside = (gap < COT["slat"]) & along & ~tight
        need = np.sqrt(np.maximum(skin ** 2 - beside ** 2, 0.0))
        depth = across * _OUT                     # + is out of the cot, - is in
        short = (np.abs(depth) < need) & along & tight
        if not np.any(inside) and not np.any(short):
            return

        # ...and out along the line from the bar's axis to her, which is the
        # bed's own rule (least far through, continuous everywhere) arriving for
        # nothing, because a circle has no faces to choose between.  A pinned
        # joint --- one a hand is holding where it put it --- is left alone, on
        # `stop()`'s own `free`, for `stop()`'s own reason.
        span = gap + r                            # her distance from each axis
        act = inside & free[:, None, None]
        push = np.where(act, skin / np.maximum(span, 1e-9) - 1.0, 0.0)
        # A JOINT EXACTLY ON AN AXIS HAS NO DIRECTION TO BE PUSHED IN, and a
        # scale of anything times zero is zero --- so dead centre of a bar was
        # the one place her hand went straight through it.  MEASURED before this
        # line: a hand-sized ball laid on the rail's own axis was refused at
        # every height around it and passed at that one.  It goes out of the
        # cot, because it has to go somewhere; the bed's centre does the same.
        #
        # ...on a row that is answering at all, which `act` now says: without
        # it a joint this rule is NOT speaking for --- a pinned one, or one the
        # surface below has --- was thrown a whole bar's width out of the cot
        # for standing on an axis it was allowed to stand on.
        adrift = span > 1e-6
        out = np.where(adrift | ~act, across * push, skin * _OUT)
        up = np.where(adrift, beside * push, 0.0)
        lift = np.where(grip, up, 0.0)
        slide = up - lift
        # ...and the scalloped surface, for the parts of her a gap is a wall to
        out += np.where(short & free[:, None, None],
                        np.copysign(need, depth) - depth, 0.0) * _OUT
        # her (x, z) take each pair's ACROSS from their own axis and the other
        # pair's ALONG, which is the same swap `a` was read with
        pos[:, 0] += out[:, 0].sum(1) + slide[:, 1].sum(1)
        pos[:, 2] += out[:, 1].sum(1) + slide[:, 0].sum(1)
        pos[:, 1] += lift.sum((1, 2))

    # --- what is in it ---------------------------------------------------------

    def put(self, what: str, at, size: float = 0.08,
            sounds: bool = False, feeds: float = 0.0,
            left: float = 0.0, shape: str = "ball") -> Thing:
        thing = Thing(what, at, size, sounds, feeds, left, shape)
        self.things[what] = thing
        return thing

    def take(self, what: str) -> bool:
        return self.things.pop(what, None) is not None

    def reach(self, was: np.ndarray, want: np.ndarray, radius: float,
              trunk: np.ndarray | None = None,
              arm: float = 0.0) -> np.ndarray:
        """Where a HAND may actually put that joint, given where it is now.

        **A HAND CANNOT PULL A BABY SIDEWAYS THROUGH THE BARS OF A COT.**  It
        lifts her over the rail, and this returns anything above the rail
        untouched, so that still works exactly as it did.

        WHY THIS IS HERE AND NOT IN THE SOLVER.  `stop()` moves a joint OUT of
        a bar it is inside, which is the right answer for a body arriving at a
        bar under its own speed.  A hold does not arrive: `advance()` writes
        `her.pos[joint] = at` outright, so a grip dragged toward a point three
        metres away TUNNELS --- one write and she is on the far side, with no
        bar between her and anywhere for the solver to find.  Measured: closing
        the bars against held joints changed the drag test by nothing at all,
        to three decimal places, because the bars were never touched.  The only
        place a passage can be refused is where it is ASKED FOR.

        MEASURED, and it is why this exists (`measured/cot.md`, and again on
        2026-08-18): one sideways drag ended her containment for a whole life
        --- her pelvis 13.65 m from her cot, 0.08 m off a wall, on bare floor,
        three hours into a fresh life --- and *"nothing ever puts her back."*
        """
        want = np.asarray(want, np.float32).reshape(3).copy()
        was = np.asarray(was, np.float32).reshape(3)
        here = was if trunk is None else np.asarray(trunk, np.float32).reshape(3)
        # LIFTING HER OUT OVER THE RAIL IS ALLOWED.  TELEPORTING HER IS NOT ---
        # and for a long time this line could not tell the two apart: it was
        # `return want`, UNCONDITIONAL, so any target one millimetre above the
        # rail was granted anywhere in twenty metres.
        #
        # THIS IS THE ONE THE WALK HELPER GOES THROUGH, and it is why she kept
        # arriving at a wall with `held: None` and nobody dragging her.
        # `walkHolds` asks her head for `ground + 0.92*leg + ...`; measured off
        # her real rest pose that is 0.8048 with her chest at x = +0.09 (just
        # under the 0.8150 ceiling, so clamped) and 0.8202 with her chest at
        # x = +0.25 --- still WELL INSIDE HER OWN COT --- at which point it
        # cleared this line and was granted verbatim, and `ahead()` aims at
        # 9.88 m and re-posts every 400 ms.  Her pelvis was found at +9.869.
        # `ahead()`'s clamp is 9.88; the wall's is 9.95.  That number was the
        # helper's signature and never a wall's, and two fixes to the clamps
        # below could not touch it because this returns before them.
        above_rail = float(want[1]) >= (float(self._axle.max())
                                        + COT["slat"] + radius)
        # WHETHER SHE IS IN HER COT IS ABOUT HER MIDDLE, NOT ABOUT THE PART YOU
        # ARE HOLDING, and reading it off the held joint was a hole big enough
        # to lose her through.  The page offers five joints to grab --- both
        # hands, both feet and her head --- and MEASURED, her foot rests at
        # x = -0.449 against a footprint of 0.42: it is ALWAYS outside.  So the
        # one exemption meant for a baby already out of her cot was permanently
        # open on a joint you can grab, and dragging her by that foot took her
        # out every time while the same drag on her hands, her head and her
        # chest was refused.  Found 2026-08-18 with her at x = +9.95, hand
        # against the wall at exactly 10.000, for the second time in a day.
        # AND BEING OUT USED TO SWITCH THE WHOLE RULE OFF, which is the third
        # time this hole has been found and the first time it has been closed.
        # It read `return want` --- unclamped, anywhere in twenty metres --- so
        # the moment her middle crossed the footprint a drag could put her on
        # the far wall, and being on the far wall kept the exemption open.  It
        # fed itself: out, so no limit; no limit, so further out.  *"Nothing
        # ever puts her back."*  Measured 2026-08-18 in a live hour: pelvis
        # (-3.803, 0.100, +9.869), ~10 m from her cot, nobody holding her --- and
        # she cannot have walked, because her own muscles at FULL effort for 600
        # ticks never take her past 0.568 m, and limp she does not move at all.
        # A hand was the only thing that could have, and this was the line that
        # let it.
        #
        # OUT IS STILL A PERSON'S BUSINESS --- what it may not be is a teleport.
        # A baby on the floor can be picked up and moved; that is what arms do.
        # So the FOOTPRINT limit stays a cot rule (below), and the REACH limit
        # below it now applies wherever she is: the part you hold may not be
        # taken further from her middle than it already is.  One grab moves her
        # by her own reach and no more, in the cot or out of it, and putting her
        # back is as many grabs as taking her away was.
        out_of_cot = float(max(abs(here[0]) / (BED["x"] + 1e-9),
                               abs(here[2]) / (BED["z"] + 1e-9))) > 1.0
        # A HAND MAY MOVE THE PART IT HOLDS AND MAY NOT RELOCATE HER.  The
        # target stays within the distance that joint ALREADY is from her
        # middle, so a foot through the bars can be taken and moved about and
        # her bones are never stretched --- and it is her bones, pulled taut by
        # a target metres away, that drag the rest of her after it.  No constant
        # and no footprint arithmetic: her own reach is the limit, whatever pose
        # she is in.
        # TWO LIMITS, AND BOTH ARE NEEDED.  The sphere alone is degenerate when
        # the part you are holding IS her middle --- `span` is zero, nothing
        # clamps, and dragging her by the chest took her 5 m out while the same
        # drag on a foot was refused.  The footprint alone was the hole above.
        # ...AND HER OWN REACH IS THE FLOOR UNDER IT.  Holding her CHEST makes
        # `was` and `here` the same point, so this is exactly 0.0, the sphere
        # below cannot fire, and a chest hold moved her 19 m when asked for 19.
        #
        # THE FIRST FIX FOR THAT USED `radius` AND IT MADE THE GRAB FEEL DEAD:
        # her chest is 0.052 m thick, so a hold on her middle followed your hand
        # five centimetres at a time and dragging her stopped working.  The
        # owner, immediately: *"your grabfix is shit."*  He is right --- the
        # limit is here to stop her being RELOCATED, not to throttle how fast
        # the part you are holding follows you.
        #
        # `arm` is how far her furthest joint is from her middle --- HER OWN
        # BODY, measured off her pose by the caller each time, not a constant
        # and not a guess.  A hand hold keeps its own arm's length as before; a
        # hold on her middle now moves by her whole reach, which is what one
        # grab of a baby is.  Repeating the grab still carries her, one reach at
        # a time, and that is a person walking her rather than a teleport.
        span = max(float(np.linalg.norm(was - here)), float(arm))
        # ...she cannot leave her cot, however far out the part you hold is ---
        # and this one stays a COT rule, because applying it to a baby already
        # across the room would snap her back to the cot in a single frame,
        # which is a teleport in the other direction and no more honest.
        # ...OR WHEN THE PART HELD IS HER MIDDLE, wherever she is.  Making this
        # a cot-only rule re-opened the degeneracy the paragraph above names:
        # hold her CHEST and `was` IS `here`, so `span` is exactly 0.0, the
        # sphere below cannot fire (`span > 1e-6` is false) and NOTHING clamps.
        # Measured on the file as I first left it: chest held out of the cot,
        # asked +19 m, moved +19.000 m.  The comment predicted it and the fix
        # walked into it anyway, which is why the check is run and not reasoned
        # about.  Her middle is the one part that cannot be moved without
        # moving HER, so it keeps the footprint limit in every case.
        if not (out_of_cot or above_rail):
            want[0] = float(np.clip(want[0], -(BED["x"] + span), BED["x"] + span))
            want[2] = float(np.clip(want[2], -(BED["z"] + span), BED["z"] + span))
        # ...and that part cannot be taken further from her middle than it
        # already is, so her bones are never pulled taut and never drag her
        off = want - here
        far = float(np.linalg.norm(off))
        if far > span > 1e-6:
            want = here + off * (span / far)
        return want

    def surfaces(self) -> list[dict]:
        """Everything a ray can hit, for when she has an eye.

        Axis-aligned planes and spheres, which is all a room like this is ---
        and all of it analytic, so her eye is a few thousand dot products rather
        than a renderer, and reproduces exactly on a re-run.
        """
        out: list[dict] = [
            {"kind": "plane", "axis": 1, "at": 0.0, "facing": 1, "what": "floor"},
            {"kind": "plane", "axis": 1, "at": ROOM["y"], "facing": -1, "what": "ceiling"},
            {"kind": "plane", "axis": 0, "at": -ROOM["x"], "facing": 1, "what": "wall"},
            {"kind": "plane", "axis": 0, "at": ROOM["x"], "facing": -1, "what": "wall"},
            {"kind": "plane", "axis": 2, "at": -ROOM["z"], "facing": 1, "what": "wall"},
            {"kind": "plane", "axis": 2, "at": ROOM["z"], "facing": -1, "what": "wall"},
            {"kind": "box", "at": [0.0, BED["top"] * 0.5, 0.0],
             "half": [BED["x"], BED["top"] * 0.5, BED["z"]], "what": "bed"},
        ]
        # HER COT'S SIDES, upright by upright, because every one of them is a
        # place `cot()` will really refuse her --- a list of what a ray can hit
        # that left them out would be a second and quieter answer to what is in
        # her room.  Each is the box that contains the dowel, which is what an
        # axis-aligned tracer can say about a cylinder.
        #
        # NOTE THE COST BEFORE WIRING THIS TO AN EYE: 44 boxes against the 7
        # surfaces above.  And `light.one_look` does not read this function at
        # all --- it carries its own copy of the floor, the walls and the bed,
        # so SHE CANNOT PRESENTLY SEE WHAT NOW STOPS HER.  That is the one thing
        # the cot still owes her, and it is a `light.py` change.
        band, axle = self._band.reshape(-1), self._axle.reshape(-1)
        for i, side in enumerate(SIDES):
            axis, other = side["axis"], 2 - side["axis"]
            for centre in side["slats"]:
                at = [0.0, float(band[i]) * 0.5, 0.0]
                half = [0.0, float(band[i]) * 0.5, 0.0]
                at[axis], half[axis] = side["at"], COT["slat"]
                at[other], half[other] = centre, COT["slat"]
                out.append({"kind": "box", "at": at, "half": half, "what": "cot"})
            at = [0.0, float(axle[i]), 0.0]
            half = [0.0, COT["slat"], 0.0]
            at[axis], half[axis] = side["at"], COT["slat"]
            half[other] = side["along"] + COT["slat"]
            out.append({"kind": "box", "at": at, "half": half, "what": "cot"})
        # HER MIRROR --- a quad, so a ray can be told to bounce off it
        out.append({"kind": "mirror", "axis": MIRROR["axis"],
                    "at": MIRROR["at"], "facing": -1,
                    "centre": [MIRROR["at"], MIRROR["y"], MIRROR["z"]],
                    "half": [MIRROR["half_w"], MIRROR["half_h"]],
                    "what": "mirror"})
        out += [{"kind": "sphere", "at": t.at.tolist(), "size": float(t.size),
                 "what": t.what} for t in self.things.values()]
        return out

    def as_json(self) -> dict:
        return {"size": [ROOM["x"] * 2, ROOM["y"], ROOM["z"] * 2],
                "bed": dict(BED),
                # his training pen, the same numbers stop() fences her with
                "pen": dict(PEN),
                # ...and where her mirror is, so the page draws the one she
                # actually looks into rather than one of its own invention.
                "mirror": dict(MIRROR),
                # HER COT'S SIDES, the same numbers `cot()` stops her against.
                # Every upright's centre is listed rather than described, so
                # the page draws a bar per bar and derives nothing: a drawn slat
                # and a solid slat cannot come apart.  `band` is where an
                # upright ends, `axle` is the rail's own line, and `top` is the
                # top of that rail --- the front side's three are lower than the
                # others exactly when someone has dropped it.
                "cot": {"slat": COT["slat"], "front": round(self.rail, 4),
                        "sides": [dict(s, band=round(float(b), 4),
                                       axle=round(float(x), 4),
                                       top=round(float(t), 4))
                                  for s, b, x, t in zip(
                                      SIDES, self._band.reshape(-1),
                                      self._axle.reshape(-1),
                                      self._top.reshape(-1))]},
                "things": [t.as_json() for t in self.things.values()]}

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Room {len(self.things)} things: {', '.join(self.things) or 'empty'}>"
