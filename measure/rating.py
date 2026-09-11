"""DOES A WRONG EXPERIENCE EVER LOSE ITS RATING?

His rule, both halves: a try that missed is a bad TRY and she keeps adjusting
the same experience --- but an experience that has been rebuilt and still does
not pay must have its rating fall, until she throws it away.

Her hands are checkpointed at EVERY close (`Life._close` -> `keepHands`), so
every row of the `hands` table is a snapshot of what each experience was worth
at that moment.  Reading them in order is her ratings' whole history.

    python -m measure.rating                   every record in mind/lives/
    python -m measure.rating NAME.duckdb ...   the ones you name

Nothing here runs her.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import shutil
import tempfile
import time

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _open(path: str):
    """Her record, even while she is living in it: copied byte by byte first."""
    try:
        return duckdb.connect(path, read_only=True), None
    except duckdb.IOException:
        tmp = tempfile.mkdtemp()
        for ext in ("", ".wal"):
            if os.path.exists(path + ext):
                with open(path + ext, "rb") as src,                      open(os.path.join(tmp, os.path.basename(path) + ext), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        return duckdb.connect(os.path.join(tmp, os.path.basename(path)),
                              read_only=True), tmp


def worths(path: str) -> dict:
    """What each experience is worth in her newest checkpoint."""
    con, tmp = _open(path)
    try:
        got = con.execute(
            "SELECT packed FROM hands ORDER BY age DESC LIMIT 1").fetchone()
    finally:
        con.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if not got:
        return {}
    his = (json.loads(got[0]) or {}).get("his") or {}
    return {k: float(v) for k, v in (his.get("worth") or {}).items()}


def watch(path: str, minutes: float = 30.0, every: float = 120.0) -> None:
    """HER RATINGS OVER TIME, off a LIVING record --- the only place the history
    is, because her `hands` table keeps the newest row and nothing older."""
    print("watching her ratings in %s, every %.0f s for %.0f min"
          % (os.path.basename(path), every, minutes))
    print("  min | rated |  fell  rose |    lowest | at or below zero")
    was, t0 = worths(path), time.time()
    fellEver, roseEver = set(), set()
    while time.time() - t0 < minutes * 60:
        time.sleep(every)
        now = worths(path)
        fell = rose = 0
        for k, v in now.items():
            if k in was:
                if v < was[k] - 1e-9:
                    fell += 1
                    fellEver.add(k)
                elif v > was[k] + 1e-9:
                    rose += 1
                    roseEver.add(k)
        print("%5.0f | %5d | %5d %5d | %9.5f | %d"
              % ((time.time() - t0) / 60.0, len(now), fell, rose,
                 min(now.values()) if now else -1,
                 sum(1 for v in now.values() if v <= 0.0)), flush=True)
        was = now
    print("  --- ever fell: %d experiences,  ever rose: %d"
          % (len(fellEver), len(roseEver)))


def read(path: str) -> dict:
    con, _tmp = _open(path)
    try:
        rows = con.execute("SELECT age, packed FROM hands ORDER BY age").fetchall()
    finally:
        con.close()
        if _tmp:
            shutil.rmtree(_tmp, ignore_errors=True)
    seen, fell, rose, worst, best = {}, {}, {}, 0.0, 0.0
    snaps = 0
    for _, packed in rows:
        his = (json.loads(packed) or {}).get("his") or {}
        worth = his.get("worth") or {}
        if not worth:
            continue
        snaps += 1
        for k, v in worth.items():
            v = float(v)
            if k in seen:
                d = v - seen[k]
                if d < -1e-9:
                    fell[k] = fell.get(k, 0) + 1
                    worst = min(worst, d)
                elif d > 1e-9:
                    rose[k] = rose.get(k, 0) + 1
                    best = max(best, d)
            seen[k] = v
    return {"snapshots": snaps, "ever": len(seen),
            "fell": len(fell), "rose": len(rose),
            "biggestFall": worst, "biggestRise": best,
            "lowest": min(seen.values()) if seen else None,
            "atOrBelowZero": sum(1 for v in seen.values() if v <= 0.0)}


def report(path: str) -> None:
    try:
        r = read(path)
    except Exception as why:                                     # noqa: BLE001
        print("=== %s\n    could not read: %s\n"
              % (os.path.basename(path), str(why).splitlines()[0][:70]))
        return
    print("=== %s" % os.path.basename(path))
    if not r["snapshots"]:
        print("    no checkpoints in it\n")
        return
    print("    checkpoints read ............. %d" % r["snapshots"])
    print("    experiences ever rated ....... %d" % r["ever"])
    print("    ratings that ever FELL ....... %d   (biggest fall %.4f)"
          % (r["fell"], r["biggestFall"]))
    print("    ratings that ever ROSE ....... %d   (biggest rise %.4f)"
          % (r["rose"], r["biggestRise"]))
    print("    lowest rating she ever held .. %s"
          % (None if r["lowest"] is None else round(r["lowest"], 5)))
    print("    at or below zero (thrown away) %d" % r["atOrBelowZero"])
    print()


def main(argv) -> int:
    if len(argv) > 2 and argv[1] == "--watch":
        watch(argv[2], float(argv[3]) if len(argv) > 3 else 30.0,
              float(argv[4]) if len(argv) > 4 else 120.0)
        return 0
    paths = argv[1:] or sorted(glob.glob(os.path.join(ROOT, "mind", "lives", "*.duckdb")))
    print()
    print("DO HER RATINGS EVER FALL --- read off her own checkpoints")
    print()
    for p in paths:
        report(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
