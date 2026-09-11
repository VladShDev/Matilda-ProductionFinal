"""HOW GOOD HER GROUPING IS --- scored against the ruler, on her own retina.

    python -m measure.grouping            the constants, A/B on identical rows
    python -m measure.grouping --live     her against the ruler, off the server

This is what the ruler is FOR, beyond putting a name on the panel.  His words,
2026-08-15, when the hand-built grouping had spent a day bordering the painting
instead of him: a trained detector is the yardstick, and her own vision stays
hand-built.  So this asks the only two questions that matter about a thing she
made:

    COVERED   how much of what is really there did her thing cover
    LEAKED    how much of her thing was NOT the thing at all

Both are needed.  One box round the whole view covers everything and leaks
everything; sixty tiny boxes leak nothing and cover nothing.  `bind`'s `NEAR`
and `SCALE` were chosen on exactly this pair --- 86% covered / 54% leaked, then
85 / 48.

IT IS `measure/`.  Her brain does not import it, her body does not import it,
and no name or score it computes ever reaches her.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOOKS = 30


# --- the constants, A/B on identical rows ---------------------------------------
def frames(light, Room, Window):
    """Her eye on her room, with the screen alive --- the one thing that changes
    without moving, which is what her grouping has to survive."""
    room, window = Room(), Window()
    up = np.asarray((0.0, 1.0, 0.0), np.float32)
    fwd = np.asarray(window.at, np.float32) - np.asarray((0.0, 0.4, 0.0), np.float32)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, up)
    right = (right / max(float(np.linalg.norm(right)), 1e-9)).astype(np.float32)
    up = np.cross(right, fwd).astype(np.float32)
    at = np.asarray((0.0, 0.4, 0.0), np.float32)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(LOOKS):
        window.show(rng.random((32, 32, 3)).astype(np.float32))
        got = light.see(room, window, [(at, up, right, fwd)])
        out.append(np.asarray(light._home(got))[:, :, 0].reshape(
            light.RETINA_H, light.RETINA_W, light.CONES))
    return out


def run(bind, rowsets, pics, keeps, forgets):
    """One pass with the two remembering constants forced.

    Monkeypatched on purpose: the point is that both settings see IDENTICAL
    rows, so nothing but the constants can differ.  A probe may do this; her
    body may not.
    """
    was, old = bind.KEEPS, bind.FORGETS
    bind.KEEPS, bind.FORGETS = keeps, forgets
    seen = bind.Seen()
    kept, counts, biggest, cover = [], [], [], []
    before: set = set()
    try:
        for k, (rows, pic) in enumerate(zip(rowsets, pics)):
            things = np.asarray(seen.look(
                rows, k, (0.0, 0.0, 0.0), pic,
                pics[k - 1] if k >= 1 else None,
                pics[k - 2] if k >= 2 else None))
            if not len(things):
                continue
            ids = {int(r[bind.LOOKS]) for r in things}
            areas = sorted((float(r[bind.AREA]) for r in things), reverse=True)
            counts.append(len(things))
            biggest.append(areas[0])
            cover.append(sum(areas))
            if before:
                kept.append(len(ids & before) / len(before))
            before = ids
    finally:
        bind.KEEPS, bind.FORGETS = was, old
    n = max(1, len(counts))
    return (sum(kept) / max(1, len(kept)), sum(counts) / n,
            sum(biggest) / n, sum(cover) / n)


def constants() -> int:
    from body import bind, light, parts
    from body.room import Room
    from body.window import Window
    pics = frames(light, Room, Window)
    rowsets = [np.asarray(parts.find(p)) for p in pics]
    print(f"{LOOKS} of her looks, {len(rowsets[0])} surfaces in the first\n")
    print(f"{'':36}{'a name survives':>16}{'things':>9}{'biggest':>10}{'cover':>9}")
    for label, k, f in (
            ("per LOOK, as the old 1.5 s tree had", 0.45, 0.04),
            ("per SECOND, this tree's clock", bind.KEEPS, bind.FORGETS)):
        s, c, b, v = run(bind, rowsets, pics, k, f)
        print(f"  {label:34}{s * 100:15.1f}%{c:9.1f}{b * 100:9.2f}%{v * 100:8.1f}%")
    return 0


# --- her against the ruler, live ------------------------------------------------
def _box(x, y, w, h):
    """A box in HER units to (x0, y0, x1, y1) in fractions of the view.

    A BOX'S CENTRE IS -1..1 AND ITS SIZE IS A FRACTION OF THE VIEW --- two units
    in one row, settled 2026-08-15, and both producers here already speak it.
    Nothing is compensated anywhere.
    """
    cx, cy = (x + 1) * 0.5, (y + 1) * 0.5
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def _overlap(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def live(at: str, looks: int) -> int:
    scored: dict = {}
    ticks: list = []
    was, quiet = -1, 0
    while len(ticks) < looks and quiet < 400:
        with urllib.request.urlopen(at + "/eye", timeout=10) as a:
            d = json.load(a)
        if d["tick"] == was:
            quiet += 1
            continue
        was, quiet = d["tick"], 0
        if not d.get("named"):
            continue
        ticks.append(d["tick"])
        hers = [_box(t[0], t[1], t[3], t[4]) for t in d["things"]]
        for t in d["named"]:
            there = _box(t["x"], t["y"], t["w"], t["h"])
            if _area(there) <= 0 or not hers:
                continue
            # HER BEST THING FOR IT --- the one that covers most of it.  Not the
            # best ratio: the question is whether ANY thing she made IS this
            # thing, and a speck with a perfect ratio is not.
            best = max(hers, key=lambda h: _overlap(h, there))
            over = _overlap(best, there)
            row = scored.setdefault(t["name"], [0, 0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += over / _area(there)                       # covered
            row[2] += 1.0 - over / max(_area(best), 1e-9)       # leaked
            row[3] += t["score"]
    if not ticks:
        print("the ruler named nothing in any of her looks.")
        print("there has to be something it knows IN HER VIEW --- put your")
        print("camera on the television and run this again.")
        return 1
    print(f"{len(ticks)} of her looks, ticks {ticks[0]} to {ticks[-1]}\n")
    print(f"{'what is really there':<22}{'looks':>7}{'sure':>8}"
          f"{'COVERED':>10}{'LEAKED':>9}")
    for name, (n, cov, leak, sure) in sorted(scored.items()):
        print(f"  {name:<20}{n:7d}{sure / n:8.2f}"
              f"{cov / n * 100:9.1f}%{leak / n * 100:8.1f}%")
    print("\n  covered: how much of the real thing her thing covered")
    print("  leaked:  how much of her thing was not the thing at all")
    print("  what `bind` was tuned to, 2026-08-15:   86% covered / 54% leaked")
    return 0


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--live", action="store_true",
                     help="score her against the ruler off a running server")
    ask.add_argument("--at", default="http://127.0.0.1:8080")
    ask.add_argument("--looks", type=int, default=20)
    got = ask.parse_args()
    return live(got.at, got.looks) if got.live else constants()


if __name__ == "__main__":
    raise SystemExit(main())
