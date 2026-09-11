"""DOES SHE ACTUALLY WORK?  Run this before saying anything is finished.

    python -m measure.works              she must already be running on :8080
    python -m measure.works --start      start her, check her, leave her running

**THIS EXISTS BECAUSE OF A PATTERN, AND HE NAMED IT:** *"fuck me everyday we
finish on broken app"*.  He is right.  On 2026-08-26 every measurement said she
was healthy --- her clock held 30 a second, her reasons were proven identical,
`measure.screen` held at 4.16% --- and **`/eye` hung for ever.**  A refactor of
mine left her lock taken twice on one thread, `threading.Lock` is not reentrant,
and nothing raised: her clock kept running, `/state` kept answering, and only
asking for her retina fell into a hole.  Not one of the other instruments looks
at whether the page can be USED.

So this asks the dumbest possible question --- does every wire answer --- and it
asks it with a TIMEOUT, because the failure that gets shipped is not the one
that raises.  It is the one that waits.

Nothing here touches her.  Every path is a read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: EVERY WIRE THE PAGE ACTUALLY PULLS, and what it must not be.
WIRES = [
    ("/meta", 500), ("/state", 400), ("/eye", 2000), ("/said", 50),
    ("/mind", 500), ("/tick", 500), ("/frames?from=0", 20), ("/", 100000),
]
#: HOW LONG IS TOO LONG.  Her tick is 33 ms and nothing here computes anything,
#: so a wire that takes seconds is a wire that is waiting on something.
PATIENCE = 15.0


def ask(at: str, path: str):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(at + path, timeout=PATIENCE) as r:
            body = r.read()
        return len(body), (time.perf_counter() - t0) * 1000.0, None
    except Exception as why:                                # noqa: BLE001
        return 0, (time.perf_counter() - t0) * 1000.0, f"{type(why).__name__}: {why}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8080")
    ap.add_argument("--start", action="store_true",
                    help="start her first, and leave her running")
    ap.add_argument("--lives", default="lives")
    got = ap.parse_args()

    if got.start:
        # ...AND IT REFUSES TO START ON TOP OF ANOTHER ONE.
        #
        # `Start` spawned a server and slept.  If the port was already taken the
        # new one died on bind, quietly, and every test below then measured the
        # OLD server --- so this reported 8 of 8 wires and 30 ticks a second
        # about code that was not the code on disk.  He watched a stale girl for
        # an hour because of it and could not hear her babble: *"i still dont
        # her her"*.  An instrument that measures whatever happens to answer is
        # worse than none.
        _, _, why = ask(got.at, "/state")
        if why is None:
            print(f"SOMETHING IS ALREADY LISTENING ON {got.at}.")
            print("Stop it first --- otherwise this measures a server that may")
            print("not be running the code you just changed.")
            return 1
        tape = got.lives           # her lives folder; no tape is written any more
        print(f"starting her on {got.at}   lives {tape}")
        subprocess.Popen(
            [sys.executable, os.path.join("sandbox", "app.py"),
             "--port", got.at.rsplit(":", 1)[-1], "--lives", tape],
            cwd=HERE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(18)

    print(f"\n{'wire':<18}{'':>4}{'bytes':>10}{'ms':>9}")
    bad = 0
    for path, least in WIRES:
        n, ms, why = ask(got.at, path)
        if why is not None:
            print(f"  {path:<16}{'DEAD':>6}{'':>10}{ms:>9.0f}   {why[:60]}")
            bad += 1
        elif n < least:
            print(f"  {path:<16}{'THIN':>6}{n:>10,}{ms:>9.0f}   "
                  f"under {least:,} --- it answered with nothing")
            bad += 1
        else:
            print(f"  {path:<16}{'ok':>6}{n:>10,}{ms:>9.0f}")

    try:
        a = json.loads(urllib.request.urlopen(got.at + "/state", timeout=10).read())
        t0 = time.perf_counter()
        time.sleep(6.0)
        b = json.loads(urllib.request.urlopen(got.at + "/state", timeout=10).read())
        rate = (b["tick"] - a["tick"]) / (time.perf_counter() - t0)
        print(f"\n  {len(WIRES) - bad} of {len(WIRES)} wires answer")
        print(f"  rate {rate:.2f} ticks a second   behind {b['behind']}   "
              f"things {b['things']}   runs {b['runs']}")
        if rate < 25.0:
            print("  SHE IS NOT KEEPING HER CLOCK --- 30 a second is the tick.")
            bad += 1
    except Exception as why:                                # noqa: BLE001
        print(f"\n  could not read her state: {why}")
        bad += 1

    print("\n  " + ("everything answers --- she can be left running."
                    if not bad else f"{bad} FAULTS.  Do not call this finished."))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
