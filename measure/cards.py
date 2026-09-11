"""THE DOCTOR'S CARDS --- pictures on her screen, by name.

His idea, 2026-09-04: *"add one monitor on the wall and just change pictures on
it"* --- every item of the examination that is about looking, following,
turning and naming is done on the screen she already has (`body/window.py`),
at no cost to her body's physics.  A card is a drawn thing --- a bottle, a
ball, a rattle, a bear --- posted to `/show` with its name; her eye renders the
screen as a quad in her room and takes from it what her optics take, and her
mother, seeing her look at the screen, says the thing's name out loud
(`body/teacher.py THING_WORDS`).  Nothing here names anything to her.

    python -m measure.cards ball                 # show one card and leave it
    python -m measure.cards --sequence 20        # bottle, ball, rattle, bear, 20 s each, for ever
    python -m measure.cards --blank              # the screen shows nothing
"""
from __future__ import annotations

import sys
import time
import urllib.request

import numpy as np

AT = "http://127.0.0.1:8090"
SIZE = 96
CARDS = ("bottle", "ball", "rattle", "bear")


def _disc(img, cx, cy, r, colour):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = colour


def _box(img, x0, y0, x1, y1, colour):
    img[int(y0):int(y1), int(x0):int(x1)] = colour


def draw(name: str) -> np.ndarray:
    """A card, (SIZE, SIZE, 3) uint8, on a pale ground."""
    img = np.full((SIZE, SIZE, 3), 235, np.uint8)
    if name == "ball":
        _disc(img, 48, 50, 30, (220, 40, 40))
        _disc(img, 38, 40, 8, (250, 120, 120))
    elif name == "bottle":
        _box(img, 34, 30, 62, 84, (250, 250, 250))
        _box(img, 40, 14, 56, 30, (240, 200, 90))
        _box(img, 34, 60, 62, 84, (250, 250, 250))
        _box(img, 36, 62, 60, 82, (245, 245, 235))
    elif name == "rattle":
        _box(img, 44, 46, 52, 88, (120, 80, 40))
        _disc(img, 48, 34, 22, (60, 120, 230))
        _disc(img, 42, 28, 5, (240, 240, 100))
    elif name == "bear":
        _disc(img, 30, 30, 12, (130, 85, 40))
        _disc(img, 66, 30, 12, (130, 85, 40))
        _disc(img, 48, 52, 30, (150, 100, 50))
        _disc(img, 38, 46, 4, (20, 20, 20))
        _disc(img, 58, 46, 4, (20, 20, 20))
        _disc(img, 48, 60, 6, (40, 20, 10))
    elif name == "blank":
        pass
    else:
        raise ValueError("no such card: " + name)
    return img


def show(name: str, at: str = AT) -> None:
    img = draw(name)
    req = urllib.request.Request(at + "/show", data=img.tobytes(), method="POST",
                                 headers={"X-Width": str(SIZE), "X-Height": str(SIZE),
                                          "X-Name": name,      # "blank" clears the board
                                          "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def main() -> int:
    args = sys.argv[1:]
    if "--blank" in args:
        show("blank"); print("the screen shows nothing"); return 0
    if "--sequence" in args:
        hold = float(args[args.index("--sequence") + 1]) if len(args) > args.index("--sequence") + 1 else 20.0
        k = 0
        while True:
            name = CARDS[k % len(CARDS)]
            show(name); print("%s  %s on the screen" % (time.strftime("%H:%M:%S"), name), flush=True)
            time.sleep(hold); k += 1
    name = args[0] if args else "ball"
    show(name); print("%s on the screen" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
