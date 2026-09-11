"""Her skin: eight places, and how hard each is being pressed.

Pressure comes from the same fact as a muscle's force --- **how hard the room had
to push her back**.  A joint the room never had to move is touching nothing; one
it had to shove is being pressed, and how far it had to shove is how hard.  That
is not an analogy between two mechanisms, it is one mechanism read at two places.

Three things can press her, and none of them is labelled as itself: the surface
she is lying on, a thing you put in her room, and a touch you send directly.  She
gets a number at a place.  What it was is hers to work out.
"""

from __future__ import annotations

import numpy as np

#: WHERE SHE CAN BE TOUCHED.  Feet are here because pushing the floor has to
#: be felt.  It left the contract with everything else: a fact about the thing
#: she is, not a fingerprint.
SKIN = (
    "mouth_left", "mouth_right",
    "hand_left", "hand_right",
    "foot_left", "foot_right",
    "body_left", "body_right",
)

from .ragdoll import JIDX

#: Which joints each patch of skin listens to.  Coarse, because a newborn's is:
#: what matters is mouth, hands, feet, and the broad sides of her body.
WATCHES: dict[str, tuple[str, ...]] = {
    "mouth_left": ("face",), "mouth_right": ("face",),
    "hand_left": ("haL",), "hand_right": ("haR",),
    "foot_left": ("foL",), "foot_right": ("foR",),
    "body_left": ("shL", "hiL"), "body_right": ("shR", "hiR"),
}

#: How far the room has to shove a joint for that to read as full pressure.
#: Small, because a constraint solver corrects in millimetres --- and it is
#: `measured/`-worthy the moment anything depends on the exact value.
FIRM = 0.01
#: How close a thing's SURFACE has to be to count as touching her --- its own
#: size is added on, because what she feels is the outside of a ball against her
#: hand, not the point at its middle.  Measured against her: with the size left
#: out, a ball resting on her palm read 0.0, because the room had already pushed
#: her hand out to the ball's own radius.
REACH = 0.06


def feel(ragdoll, room, sent: dict[str, float] | None = None) -> dict[str, float]:
    """What each patch of skin has to say, 0..1.

    `sent` is a touch you injected through `/world` --- for tests, and for
    holding her without placing an object.  It is the strongest thing at that
    place, not an addition to it: a hand on her cheek does not become firmer
    because she also happens to be lying down.
    """
    pressed = np.asarray(ragdoll.pressed, np.float32)
    out: dict[str, float] = {}

    for patch in SKIN:
        joints = WATCHES[patch]
        idx = [JIDX[j] for j in joints]
        felt = float(np.max(pressed[idx])) / FIRM

        for thing in room.things.values():
            felt = max(felt, ragdoll.contact(thing.at, joints, REACH + thing.size))

        if sent:
            felt = max(felt, float(sent.get(patch, 0.0)))
        out[patch] = float(np.clip(felt, 0.0, 1.0))
    return out
