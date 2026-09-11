"""THE ONE CHECK THAT GUARDS `bind` --- run it before and after any visual change.

    python -m measure.screen

**A screen is the only object in her world that changes without moving.**  That
is the thing that defeated her grouping: everything else that changes, moves,
so "what moved together" finds objects --- and a television lights up in place
and pulls every box onto itself.

THE NUMBER THAT MATTERS: **how much of the top box lay ON THE SCREEN.**  It was
**63%** of its area before the boundary rule and **16%** after (2026-08-15, tag
`screen-does-not-win`).  If it climbs back, `bind` is probably not what broke:
look first at what packs a picture, at `parts.find`, and at what carries a track.

HOW THE SCREEN IS FOUND HERE, and it needs no projection maths: her eye is
rendered twice with **only** what is on the screen changed, and nothing else in
the room touched.  What differs between the two pictures IS the screen.  That is
the rule's own definition used as its own instrument, and it cannot drift out of
step with her optics the way a hand-computed rectangle could.

This is `measure/`, so it may use whatever yardstick it likes.  Nothing here is
hers and her brain imports none of it.
"""

from __future__ import annotations

import sys

import numpy as np

from body import bind, light, parts
from body.room import Room
from body.window import Window


def _picture(room, window, her_at, up, right, fwd):
    """One look, as her body has it: `(h, w, cones)`."""
    got = light.see(room, window, [(her_at, up, right, fwd)])
    return np.asarray(light._home(got))[:, :, 0].reshape(
        light.RETINA_H, light.RETINA_W, light.CONES)


def _screen_mask(room, window, at, up, right, fwd) -> np.ndarray:
    """Which cells the screen covers --- found by changing ONLY the screen."""
    h = w = 32
    dark = np.zeros((h, w, 3), np.float32)
    bright = np.ones((h, w, 3), np.float32)
    window.show(dark)
    a = _picture(room, window, at, up, right, fwd)
    window.show(bright)
    b = _picture(room, window, at, up, right, fwd)
    return (np.abs(b - a).max(axis=2) > 1e-4)


def _boxes_of(frames) -> list:
    """What she groups into things across a few looks --- HER live stage.

    `bind.Seen.look` is what `run.py` calls once a tick, and it carries
    identity from her own `alike` index rather than from where a box landed.
    `bind.bind` underneath it takes tracks with velocities, which only exist
    once something has been followed.
    """
    seen = bind.Seen()
    things = None
    for k, f in enumerate(frames):
        rows = np.asarray(parts.find(f))
        pic = frames[k]
        was = frames[k - 1] if k >= 1 else None
        older = frames[k - 2] if k >= 2 else None
        things = seen.look(rows, k, (0.0, 0.0, 0.0), pic, was, older)
    return np.asarray(things) if things is not None else np.zeros((0, 0))


def main() -> int:
    room, window = Room(), Window()
    up = np.asarray((0.0, 1.0, 0.0), np.float32)
    fwd = np.asarray(window.at, np.float32) - np.asarray((0.0, 0.4, 0.0), np.float32)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, up)
    right = (right / max(float(np.linalg.norm(right)), 1e-9)).astype(np.float32)
    up = np.cross(right, fwd).astype(np.float32)
    at = np.asarray((0.0, 0.4, 0.0), np.float32)

    onScreen = _screen_mask(room, window, at, up, right, fwd)
    covers = float(onScreen.mean())
    print(f"the screen covers {covers * 100:6.2f}% of her view "
          f"({int(onScreen.sum()):,} of {onScreen.size:,} cells)")
    if not onScreen.any():
        print("  the screen is not in her view --- nothing to check.")
        return 1

    # ...AND NOW SOMETHING ON IT CHANGES, WHILE NOTHING IN THE ROOM MOVES.
    frames = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        window.show(rng.random((32, 32, 3)).astype(np.float32))
        frames.append(_picture(room, window, at, up, right, fwd))

    got = _boxes_of(frames)
    if not len(got):
        print("  she grouped nothing --- no top box to measure.")
        return 1

    # A BOX'S CENTRE IS -1..1 AND ITS SIZE IS A FRACTION OF THE VIEW.  Two
    # units in one row, and it is `parts.find`'s own convention --- settled
    # 2026-08-15 and not to be re-opened.  Half a box in cells is
    # `w * width * 0.5`.
    top = got[0]
    h, w = onScreen.shape
    cx, cy, bw, bh = float(top[0]), float(top[1]), float(top[3]), float(top[4])
    x0 = int(round((cx * 0.5 + 0.5) * (w - 1) - bw * w * 0.5))
    x1 = int(round((cx * 0.5 + 0.5) * (w - 1) + bw * w * 0.5))
    y0 = int(round((cy * 0.5 + 0.5) * (h - 1) - bh * h * 0.5))
    y1 = int(round((cy * 0.5 + 0.5) * (h - 1) + bh * h * 0.5))
    x0, x1 = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
    box = np.zeros_like(onScreen)
    box[y0:y1 + 1, x0:x1 + 1] = True
    if not box.any():
        print("  the top box has no area.")
        return 1

    lay = float((box & onScreen).sum()) / float(box.sum())
    print(f"things she made   {len(got)}")
    print(f"the top one       x {cx:+.3f}  y {cy:+.3f}  w {bw:.3f}  h {bh:.3f}")
    print()
    print(f"HOW MUCH OF THE TOP BOX LAY ON THE SCREEN   {lay * 100:6.2f}%")
    print(f"  before the boundary rule, 2026-08-15       63.00%")
    print(f"  after it                                   16.00%")
    print()
    if lay > 0.40:
        print("  DRIFTED --- the screen is winning again.  Look first at what")
        print("  packs a picture, at `parts.find`, and at what carries a track.")
        return 1
    print("  HELD --- the screen does not win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
