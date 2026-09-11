"""HER JOINTS, AS JOINTS: named axes she turns, instead of distances she shortens.

**WHAT IS WRONG WITH THE BODY THIS REPLACES.**  `ragdoll.py` was nineteen point
masses joined by distance constraints, and a muscle was one more distance
constraint with a length she could ask for.  One degree of freedom per muscle,
whose direction was whatever her current geometry happened to give.  Everything
that failed on `body/hydraulic` was that one fault wearing faces: no order
could move a hand toward her own head (316 -> 289 mm on a 160 mm arm), her hip
needed a crossed workaround span for a missing axis, a muscle added to raise
her arm braked her knee, and her arm could not press at all --- 0.0000 in every
arrangement --- because pushing is coordinated rotation, not a span lengthening.

**AND THE COST THAT MATTERED MOST WAS TO LEARNING.**  A distance constraint's
effect depends on where the limb already is.  A rotation about a named axis
means the same thing every time.  She was trying to learn a body whose commands
changed meaning underneath her.

**THE GATE THIS PASSED** (`measure/shoulder3.py`, one shoulder, before twelve
joints were converted on the strength of an argument): ordered lift reads back
1.7 -> 161.1 degrees monotonic; hand-to-face 53 mm where every span
arrangement measured 289; a hand pressing 0.0002 where every span pressed
0.0000; twist -46.5 -> +44.6 degrees, ordered and read; the knee unbraked.

**TWIST IS CARRIED BY THE CHAIN, NOT BY NEW STATE.**  "Twist cannot be seen
from points" is true of one segment alone --- and was proven drivable with a
marker particle first.  But a chain with a bent distal joint exposes twist
through the distal point: rotate an upper arm with the elbow bent and the
forearm sweeps.  Her elbows and knees REST BENT by design, so the full body
needs no new particles --- each twist axis reads and drives through the next
point down the chain.  When the distal joint is perfectly straight the twist
is momentarily unreadable, exactly as it is mechanically meaningless.

**THE RULE THIS IS BUILT UNDER: THE BRAIN CONTROLS EVERYTHING.**  Every axis is
a line in the contract that she orders as `(order, effort)` x 15 slides.  There
is no body-side controller, no stabiliser, no reflex that decides anything.

**AXES LIVE IN THE PARENT'S OWN FRAME**, computed from the parent body's own
points and never through the driven joint, so an order cannot move its own
axes and the map from order to place is fixed for the whole of her life:

    pelvis frame   (pelvis, hiL, hiR)     --- her hips and spine turn in this
    chest frame    (chest, shL, shR)      --- her shoulders and neck turn in this
    a limb frame   the segment above, with the fold plane of ITS parent joint

**STRENGTH CAPS EFFORT AND NEVER THE ORDER.**  The old body capped the order
--- `min(order, can)` --- so the top quarter of every motor line was dead and
ordering 0.75 and 1.00 were the same movement.  Here the order is an ANGLE and
all of it is hers; what her flesh limits is the GAIN with which the solver
insists on it, so a weak baby can aim a full reach and lack the force to hold
it against anything --- which is what `force` reports, and what a newborn is.

HOW AN ORDER BECOMES A MOVEMENT
-------------------------------
Per axis, `order` is 0..1 across that axis's fixed range in degrees.  A ball
joint's `(lift, swing)` name a direction for the child bone in the parent
frame; its `twist` names an azimuth for the grandchild about the child bone.
A hinge's `bend` names an angle whose outer-to-outer distance follows by the
law of cosines.  Every correction is applied pairwise inside the same solver
passes as the bones, so what a limb gains its trunk absorbs, and momentum
stays hers --- she can scoot by thrashing, and does, through friction alone.
"""
from __future__ import annotations

import numpy as np

#: Every joint, its parent frame, what it drives, and its axes with ranges.
#:
#: `(joint, frame, child, grand, axes)` --- `frame` names which parent frame
#: the axes live in; `child` is the point the (lift, swing) or (bend, side)
#: direction drives; `grand` is the point a `twist` azimuth drives, or None.
#: Ranges are degrees, `(from, to)`: 0.0 in her order is the first number,
#: 1.0 the second, forever.
JOINTS3: tuple = (
    ("spine", "pelvis", "chest", "shpair",
     (("bend", -20.0, 60.0), ("side", -40.0, 40.0), ("twist", -40.0, 40.0))),
    ("neck", "chest", "head", "face",
     (("bend", -45.0, 45.0), ("side", -45.0, 45.0), ("twist", -60.0, 60.0))),
    # twist +-90, not +-45: a floppy newborn's forearm falls across her belly,
    # which IS humeral rotation past 45 --- measured, her settled rest read
    # 1.4 ranges outside the narrower book value.  A range must contain her.
    ("shoulder_left", "chest", "elL", "haL",
     (("lift", 0.0, 160.0), ("swing", -60.0, 150.0), ("twist", -90.0, 90.0))),
    ("shoulder_right", "chest", "elR", "haR",
     (("lift", 0.0, 160.0), ("swing", -60.0, 150.0), ("twist", -90.0, 90.0))),
    ("elbow_left", "arm_left", "haL", None, (("bend", 0.0, 145.0),)),
    ("elbow_right", "arm_right", "haR", None, (("bend", 0.0, 145.0),)),
    # HER HIP HAD A SHOULDER'S ABDUCTION.  `swing` ran to 150 degrees --- the
    # shoulder's own upper bound, which is an arm straight up over the head.  A
    # hip does not do that: a newborn's frog-position abduction is about 75-80
    # degrees and that is already the flexible extreme.
    #
    # The owner watched her and said it before anyone measured it: *"she looks
    # like she doesn't have one axis between the legs, she usually doesn't move
    # them, splits too much"*.  Both halves were the same fault:
    #
    #   she SPLIT       the order is 0..1 across the range, so the middle of it
    #                   was +52.5 degrees PER LEG --- a 105 degree split at rest
    #   she DID NOT     each 0.01 of order was 1.95 degrees, so nothing she
    #   MOVE IT         could order was a small adjustment.  An axis that only
    #                   makes coarse splays reads exactly as an axis she has not
    #                   got.
    #
    # -30 .. 80 is a real hip: 30 degrees of adduction, 80 of abduction, span
    # 110 instead of 195, and its middle rests her at +25 rather than +52.5.
    ("hip_left", "pelvis", "knL", "foL",
     (("lift", 0.0, 140.0), ("swing", -30.0, 80.0), ("twist", -40.0, 40.0))),
    ("hip_right", "pelvis", "knR", "foR",
     (("lift", 0.0, 140.0), ("swing", -30.0, 80.0), ("twist", -40.0, 40.0))),
    ("knee_left", "leg_left", "foL", None, (("bend", 0.0, 145.0),)),
    ("knee_right", "leg_right", "foR", None, (("bend", 0.0, 145.0),)),
    # AND HER ANKLE POINTED TO 120 DEGREES, which is a ballerina twice over ---
    # a human ankle plantarflexes about 50.  The middle of that range rested her
    # at 60 degrees: permanently on tiptoe.  Same shape of error as the hip,
    # found in the same pass.
    ("ankle_left", "shin_left", "toL", None,
     (("point", 0.0, 55.0), ("roll", -25.0, 25.0))),
    ("ankle_right", "shin_right", "toR", None,
     (("point", 0.0, 55.0), ("roll", -25.0, 25.0))),
)

#: Flat, in contract order: one name per axis, `joint.axis`.  TWENTY-SIX.
AXES: tuple[str, ...] = tuple(
    f"{joint}.{axis}" for joint, _f, _c, _g, axes in JOINTS3
    for axis, _lo, _hi in axes)

#: `axis line name -> (lo, hi)` degrees, the map that does not move.
RANGE: dict[str, tuple[float, float]] = {
    f"{joint}.{axis}": (lo, hi)
    for joint, _f, _c, _g, axes in JOINTS3 for axis, lo, hi in axes}


# --- the arithmetic her whole body is made of ---------------------------------
#
# HER PHYSICS IS SEVENTEEN POINTS AND IT COST 441 ms A TICK.  Profiled body-only
# --- no brain, no render, so no tape-age variable at all --- at 4.90 ms per
# substep, of which essentially none was arithmetic.  A cross product of two
# 3-vectors is nine multiplies; **`np.cross` charges 13.1 us for it**, because it
# is a dispatched array function that calls `moveaxis` three times and
# `normalize_axis_tuple` six times before it multiplies anything.  Her substep
# asks for 26 of them, her tick for 2,367.  `np.linalg.norm` is the same tax in
# a smaller coat.
#
# These do the IDENTICAL arithmetic in the IDENTICAL order.  numpy's own `cross`
# is literally `cp0 = a1*b2 - a2*b1` after a `promote_types`, and its 2-norm is
# literally `sqrt(x.dot(x))` for a vector and `sqrt(add.reduce(x*x, axis))` for a
# stack.  Bit-identical is not a hope here; it is the same expression with the
# dispatch removed, and it was checked seven ways --- see
# `measured/cheapersubstep.md`.
#
# ONE IMPLEMENTATION OF EVERYTHING: they live here, in the module that owns her
# geometry, and `ragdoll.py` imports them.  Do not grow a second copy.


def _dt2(a: np.ndarray, b: np.ndarray):
    """What `np.cross` would compute in --- `promote_types`, skipped when the
    two already agree, which in her body they nearly always do."""
    return a.dtype if a.dtype == b.dtype else np.promote_types(a.dtype, b.dtype)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`np.cross` for two single 3-vectors: 13.1 us -> 1.4 us."""
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return np.array((a1 * b2 - a2 * b1,
                     a2 * b0 - a0 * b2,
                     a0 * b1 - a1 * b0), _dt2(a, b))


def _crossn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`np.cross` for two `(n, 3)` stacks: 12.6 us -> 5.6 us."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), _dt2(a, b))
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out[..., 0] = a1 * b2 - a2 * b1
    out[..., 1] = a2 * b0 - a0 * b2
    out[..., 2] = a0 * b1 - a1 * b0
    return out


def _len(v: np.ndarray) -> float:
    """`float(np.linalg.norm(v))`, which is `sqrt(v . v)`: 0.98 us -> 0.58 us."""
    return float(np.sqrt(v.dot(v)))


def _lens(x: np.ndarray, keepdims: bool = True) -> np.ndarray:
    """`np.linalg.norm(x, axis=1, keepdims=...)`, the same reduction."""
    return np.sqrt(np.add.reduce(x * x, axis=1, keepdims=keepdims))


def _norm(v: np.ndarray) -> np.ndarray:
    n = _len(v)
    return (v / n).astype(np.float32) if n > 1e-9 else np.array(
        [0.0, 0.0, 1.0], np.float32)


def _spin(v: np.ndarray, axis: np.ndarray, radians: float) -> np.ndarray:
    """Rodrigues: `v` turned about unit `axis`."""
    c, s = float(np.cos(radians)), float(np.sin(radians))
    return (v * c + _cross3(axis, v) * s
            + axis * float(axis @ v) * (1.0 - c)).astype(np.float32)


class Frames:
    """The parent frames, from a body's points.  `J` maps names to indices.

    Each returns `(up, across, fwd)`, orthonormal: `up` along the parent
    body toward her head, `across` toward her right, `fwd` out of her front.
    Left-hand limbs mirror `across` where a lateral direction is needed, so
    "swing forward" means forward on both sides.
    """

    def __init__(self, J: dict[str, int]) -> None:
        self.J = J

    def pelvis(self, pos: np.ndarray):
        """From the pelvis's OWN rigid triangle (pelvis, hiL, hiR) --- the
        first draft used chest-minus-pelvis for `up`, which is the very
        direction the spine drive moves: a joint must not define its own
        frame, or an order chases its own axes."""
        J = self.J
        pelvis = pos[J["pelvis"]]
        fwd = _norm(_cross3(pos[J["hiR"]] - pelvis, pos[J["hiL"]] - pelvis))
        across = pos[J["hiR"]] - pos[J["hiL"]]
        up = _norm(_cross3(across, fwd))
        return up, _norm(_cross3(fwd, up)), fwd

    def chest(self, pos: np.ndarray):
        """From her TRUNK and her shoulder line --- NOT from a triangle.

        A pelvis is a rigid plate, so the normal of (pelvis, hiL, hiR) is a real
        direction of it.  **A shoulder girdle is not a plate.**  The first draft
        of this used the normal of (chest, shL, shR) the same way, and that was
        wrong twice over:

        - **Anatomically.**  That normal is her forward only while her sternum
          sits exactly in the plane of her acromia.  A real mid-sternum is
          ANTERIOR to them, and putting hers where it belongs swung the normal
          42 degrees AT BIRTH, before she had moved.  No rest geometry satisfies
          both: the recipe needed an anatomical falsehood to work at all.
        - **Numerically.**  It was built on a 2.20 cm apex lever --- the
          sternum's offset below the shoulder line --- inside a trunk whose bones
          settle 0.4 to 1.5 cm from rest.  An 11 mm error on a 22 mm lever is 90
          degrees, and that is what it read: **the frame lay 91.09 degrees off
          her own spine**, so `neck.bend` reported +83.23 on a neck bent -7.85,
          and `neck.side` and all six shoulder axes turned in a frame on its
          side.  Not fixable by stiffness (braces to 1.0 -> 88.51 deg) nor by
          convergence (32 passes hold the triangle to 1.9% and it is STILL 61.85
          degrees off).

        So `up` is her trunk, `across` is her shoulder line, and `fwd` follows
        from the two.  Both are long and both are real.  This is also the SAME
        expression `Ragdoll.spine()` has always used for her trunk's forward ---
        which was a second implementation of this very quantity, sitting beside
        it and disagreeing by 91.09 degrees.  There is one now.

        Bit-identical to the triangle AT BIRTH, so `_ax_zero` and `_ax_rest` do
        not move.  Settled limp it reads **-7.66 degrees against a true -7.66**.

        It could not be built until her flesh was damped.  Anchoring the frame to
        her trunk gives six axes a real error to chase instead of one that moves
        with them, and undamped that rang her shoulders 40.06 mm peak-to-peak for
        ever.  With `FLESH_DAMP` it is 0.000 mm.  See `measured/herbody.md`.
        """
        J = self.J
        up = _norm(pos[J["chest"]] - pos[J["pelvis"]])
        across = pos[J["shR"]] - pos[J["shL"]]
        fwd = _norm(_cross3(up, across))
        return up, _norm(_cross3(fwd, up)), fwd

def ball_read(frame, d: np.ndarray, left: bool) -> tuple[float, float]:
    """`(lift, swing)` in degrees, read back from the child direction."""
    up, across, fwd = frame
    lateral = -across if left else across
    lift = float(np.degrees(np.arccos(np.clip(float(d @ (-up)), -1.0, 1.0))))
    swing = float(np.degrees(np.arctan2(float(d @ fwd), float(d @ lateral))))
    return lift, swing


def hinge_read(upper: float, lower: float, span: float) -> float:
    """...and back: the bend a measured span means, in degrees."""
    c = (upper * upper + lower * lower - span * span) / (2.0 * upper * lower)
    return 180.0 - float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
