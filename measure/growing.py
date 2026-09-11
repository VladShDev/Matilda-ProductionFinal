"""HOW SHE GROWS --- one line every few minutes, for as long as she lives.

    python -m measure.growing [minutes] [--every SECONDS]

His, 2026-09-02: *"You have to run her and start to watch how she growing and
try to train her like we did before, but in right way."*

Her milestones are 2-to-9-month skills and no run has yet lasted more than a
few hours.  Nothing here touches her --- it reads `/state` and writes a row, so
the record exists to compare against later, which is the one thing a single
sheet can never do.

**IT READS, IT NEVER DRIVES.**  Polling her hard costs her two ticks a second
(measured), so this asks once every few minutes and no more.

The columns, and what each says about her:

    t/s          her clock.  30 is right; less means she is behind
    real         how much of a wall second she LIVES.  100% is right
    runs         experiences opened --- her life cut into acts
    closes       ...and closed.  A run that never closes is a pinned state
    conn         her own outputs found in her inputs.  This is her learning
    guided       memory chose, instead of the ladder guessing
    laddered     ...and the ladder.  guided rising against laddered is the
                 thing to watch: it is her remembering instead of trying
    recog        a known chain met the moment --- comprehension's raw count
    words        her own sounds found in the world
    state        her wellbeing, -1..1
    saw          things in her view this instant
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def get(at: str, path: str):
    with urllib.request.urlopen(at.rstrip("/") + path, timeout=25) as h:
        return json.loads(h.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("minutes", nargs="?", type=float, default=None,
                    help="how long to watch; forever if left out")
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--every", type=float, default=300.0)
    ap.add_argument("--to", default=None, help="also append to this file")
    got = ap.parse_args()

    head = ("%-9s %6s %5s %6s %7s %7s %7s %8s %7s %6s %8s %5s"
            % ("age", "t/s", "real", "runs", "closes", "conn", "guided",
               "laddered", "recog", "words", "state", "saw"))
    out = [head]
    print(head, flush=True)
    began = time.time()
    was = get(got.at, "/state")
    while got.minutes is None or (time.time() - began) < got.minutes * 60.0:
        time.sleep(got.every)
        try:
            now = get(got.at, "/state")
            saw = get(got.at, "/mind").get("things")
        except Exception as why:                            # noqa: BLE001
            print("   she did not answer (%s)" % why, flush=True)
            continue
        gap = max(1e-9, got.every)
        row = ("%-9s %6.1f %4.0f%% %6s %7s %7s %7s %8s %7s %6s %+8.3f %5s"
               % (_age(time.time() - began),
                  (now["tick"] - was["tick"]) / gap,
                  100.0 * ((now.get("seconds") or 0) - (was.get("seconds") or 0)) / gap,
                  now.get("runs"), now.get("closes"), now.get("connections"),
                  now.get("guided"), now.get("laddered"),
                  now.get("recognised"), now.get("words"),
                  float(now.get("state") or 0.0), saw))
        out.append(row)
        print(row, flush=True)
        if got.to:
            with open(got.to, "a", encoding="utf-8") as fh:
                fh.write(row + "\n")
        was = now
    return 0


def _age(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return "%dh%02dm%02ds" % (h, m, s) if h else "%02dm%02ds" % (m, s)


if __name__ == "__main__":
    raise SystemExit(main())
