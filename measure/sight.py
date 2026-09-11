"""DOES WHAT SHE SEES REACH THE PART OF HER THAT LEARNS?

    python -m measure.sight        # she must already be living on :8090

Rule 2: a caller count is not enough.  `bind` is called, `input.view` is
written, her mind reads it --- and every one of those can be true while the two
ends still mean different things.  So this asks from the far end: after she has
lived, do the THINGS SHE MADE come back, and do they reach the part of her that
can learn from them?

    does a thing keep its name      a thing seen once and never again cannot
                                    become anything --- "known" never fires
    is sight on her ticks at all    a view line has to survive to her mind
    does it move her                do her connections grow while she sees

MEASURED 2026-08-25, 900 ticks (30 s) in an empty room:

    ticks carrying a view line      900 of 900   (100%)
    distinct things she ever saw    22, and ALL 22 came back
    median times a thing was seen   132
    experiences with sight in them  2 of 3

**THIS PROBE WAS WRONG TWICE BEFORE IT WAS RIGHT, and that is why it is kept.**
It first read `e.why` and `e.reason`, found neither, and was about to report
that sight never reaches her.  Then it tested reason keys for the string
`"view:..."` --- and a reason key is `("view:291", band)`, a NAME AND A BAND ---
and was about to report the same thing again.  Both times the finding would
have been "her sight is wired to nothing", and both times the bug was here.  An
instrument that has to guess the shape of what it measures will eventually
report the guess.

**AND THEN IT WAS WRONG A THIRD TIME, BY DYING.**  It imported `reasonOf` and
`levelsOf` from `brain.brain` --- the IN-PROCESS brain, retired 2026-08-26 when
her mind moved outside --- so it raised ImportError before its first line and
answered nothing at all for a week, while the suite stayed green.  It reads the
LIVING girl now, off her own tape and her own published counters, which is the
only place the answer was ever going to be.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: she has to see a thing at least this often for "it came back" to mean
#: anything --- once is not a statistic, his own law
TWICE = 2


def get(at, path):
    with urllib.request.urlopen(at + path, timeout=25) as h:
        return json.loads(h.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    at = args.at.rstrip("/")
    try:
        first = get(at, "/state")
    except Exception as why:                                # noqa: BLE001
        print("she is not living at %s (%s) --- start her with python her.py"
              % (at, why))
        return 1

    # HER TICKS AS SHE LIVES THEM.  `/ticks` keeps only her last ~900, so a
    # minute read at the end sees a third of it and calls the rest blindness.
    since = int(first["tick"])
    rows: list = []
    till = time.time() + args.seconds
    while time.time() < till:
        got = get(at, "/ticks?from=%d" % since)
        new = got.get("ticks", []) if isinstance(got, dict) else got
        if new:
            rows.extend(new)
            since = int(got.get("next", since)) if isinstance(got, dict) else since
        time.sleep(0.5)
    last = get(at, "/state")

    withView = 0
    seen: collections.Counter = collections.Counter()
    for one in rows:
        life = one.get("life", one)
        view = (life.get("input", {}) or {}).get("view") or []
        if view:
            withView += 1
        for thing in view:
            seen[int(thing.get("id", 0) or 0)] += 1

    again = [i for i, n in seen.items() if n >= TWICE]
    counts = sorted(seen.values(), reverse=True)
    print("%d of her ticks, %.0f s of her life\n" % (len(rows), args.seconds))
    print("  ticks carrying a view line      %d of %d   (%.0f%%)"
          % (withView, len(rows), 100.0 * withView / max(1, len(rows))))
    print("  distinct things she saw         %d" % len(seen))
    print("  ...that came back at least %dx   %d   (%.0f%%)"
          % (TWICE, len(again), 100.0 * len(again) / max(1, len(seen))))
    if counts:
        print("  median times a thing was seen   %d   (the most: %d)"
              % (counts[len(counts) // 2], counts[0]))
    grew = ((last.get("connections") or 0) - (first.get("connections") or 0))
    runs = ((last.get("runs") or 0) - (first.get("runs") or 0))
    print("  her connections grew by         %d   (experiences opened %d)"
          % (grew, runs))

    bad = (withView == 0) or (not again)
    print("\n  %s" % ("HER SIGHT REACHES HER." if not bad else
                      "WHAT SHE SEES IS NOT REACHING HER."))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
