"""THE AIR --- every sound in her room, as it arrives at each ear.

Copied from the old tree's `body/sound.py`, unchanged in its arithmetic.  Every
constant here is the measured one.

**THEY ADD**, because that is what air does.  Two sounds at once are their sum,
which is why splitting a moment by frequency and summing it back reproduces the
moment exactly.  Her own voice goes through this same air, so there is no path
in this body that treats what she says as a special kind of sound.

WHAT SHE HEARS WITH is anything that can say where her ears are and which way
she faces --- `ear`, and `frame()` giving `(up, right, forward)`.  Her ragdoll
already answers exactly that, so when it comes across nothing here changes.
`Still` below is the stand-in until it does: a head that does not move.  It is
not a second answer to "where is she" --- it is the same question with nobody
asking it yet.

**A MEASURED TRAP THAT COST A DAY.**  From the old tree: *"his voice arrives
thin --- 21 bands at the mic, 3.8 on her tape; distance is applied before
FAINTEST, so past ~2 m no word can reach her."*  The order below is the old
order, deliberately, so the same thing can be measured again rather than quietly
changed: distance first, then the hair cell's floor.
"""

from __future__ import annotations

import numpy as np

from .hearing import BAND_HZ, HIGH_HZ, SOUND_BANDS, SOUND_SLIDES

#: how far apart her ears are, from the middle of her head
EAR_OUT = 0.075
#: how much the far ear loses of a band it cannot bend around
SHADE = 0.85
#: below this a wavelength bends around a head and she cannot place it
BENDS_BELOW_HZ = 700.0
#: ...and a pinna takes a bite out of the treble of anything behind her
NOTCH_LOW_HZ = 3000.0
NOTCH_HIGH_HZ = 6000.0
NOTCH_DEPTH = 0.6
#: HOW SOUND FADES.  `NEAR / (NEAR + far)` --- not inverse-square, because a
#: room is not free space and this is the curve that was measured against her.
NEAR = 10.0
#: A HAIR CELL HAS A THRESHOLD, so a band it can barely feel says nothing at
#: all.  Without it her own babble put a trace in **92% of her ear lines every
#: tick** at a mean level of 0.064 --- bookkeeping, not hearing.
FAINTEST = 0.0002


class Still:
    """A head that does not move.  The ragdoll replaces this and nothing else."""

    def __init__(self, at=(0.0, 0.5, 0.0), looking=(0.0, 0.0, -1.0)) -> None:
        self.ear = np.asarray(at, np.float32)
        f = np.asarray(looking, np.float32)
        self._forward = f / max(float(np.linalg.norm(f)), 1e-6)

    def frame(self):
        up = np.asarray((0.0, 1.0, 0.0), np.float32)
        right = np.cross(self._forward, up)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        return up, right, self._forward


def _shadow(toward: np.ndarray, ear_out: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """What a head does to a sound coming from `toward`, per band.

    The far ear loses treble and keeps bass, because a wavelength longer than a
    head bends around it --- which is why you cannot tell where a bass note is.
    """
    behind_head = float(np.clip(-np.dot(toward, ear_out), 0.0, 1.0))
    blocks = np.clip((BAND_HZ - BENDS_BELOW_HZ) / (HIGH_HZ - BENDS_BELOW_HZ), 0.0, 1.0)
    heard = 1.0 - SHADE * behind_head * blocks

    from_behind = float(np.clip(-np.dot(toward, forward), 0.0, 1.0))
    notch = (BAND_HZ >= NOTCH_LOW_HZ) & (BAND_HZ <= NOTCH_HIGH_HZ)
    heard = heard * np.where(notch, 1.0 - NOTCH_DEPTH * from_behind, 1.0)
    return heard.astype(np.float32)


def hear(sources, head) -> np.ndarray:
    """Every sound in the room, as it arrives at each ear.  `(2, bands, slides)`.

    `sources` is `[(where, picture), ...]` --- a point in the room and a
    `(bands, slides)` spectrogram of what is being made there.
    """
    up, right, forward = head.frame()
    middle = np.asarray(head.ear, np.float32)
    # AS WIDE AS WHAT ARRIVED.  It was pinned to `SOUND_SLIDES`, which is 1 ---
    # so however many placements of her frame her ear was handed, only one could
    # come out the other side and `align` had nothing to choose between.
    wide = max((np.asarray(p).shape[1] for _, p in sources), default=SOUND_SLIDES)
    out = np.zeros((2, SOUND_BANDS, wide), np.float32)
    if not sources:
        return out

    for side, outward in ((0, -right), (1, right)):
        here = middle + outward * EAR_OUT
        for where, picture in sources:
            gap = np.asarray(where, np.float32) - here
            far = float(np.linalg.norm(gap))
            toward = gap / far if far > 1e-6 else forward
            reaches = NEAR / (NEAR + far)
            heard = _shadow(toward, outward, forward) * reaches
            out[side] += np.asarray(picture, np.float32) * heard[:, None]
    out = np.clip(out, 0.0, 1.0)
    out[out < FAINTEST] = 0.0
    return out
