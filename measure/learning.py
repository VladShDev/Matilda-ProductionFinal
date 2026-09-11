"""DOES SHE LEARN?  --- the first instrument that looks past ten seconds.

His, 2026-09-08, after three months of green on ten doctor items: every test
she has measures a horizon of SECONDS (turning to a chirp, following a card,
calming in a hold).  All of them pass on a mind that forgets everything, so
none of them could ever see whether anything she does accumulates.

Learning is repetition.  So this asks the only question that horizon cannot:

    of everything she closed, how much did she ever use AGAIN?

Nothing here judges her.  It reads a finished record and counts.

    python -m measure.learning                 every record in mind/lives/
    python -m measure.learning NAME.duckdb     one of them

WHAT EACH NUMBER IS, and where it comes from:

  closed        distinct experience ids in her record (`exp` table)
  reachable     the ids her last checkpoint kept in her hands (`hands`)
  replayed      ids that ever appeared as `tick.plan` --- the exp a plan was
                built from, 0 when discovery made it (`structure.Tick`)
  lived again   how many of the closed were replayed at least once
  worth on disk `exp.weight` non-zero: whether her rating survives the process
  composed      `exp.exp.id` non-zero: an experience that became part of
                another --- his 2026-08-27 decision, `structure.Exp`
"""
from __future__ import annotations

import collections
import glob
import os
import shutil
import sys
import tempfile

import duckdb

TICKS_PER_SECOND = 90.0


def read(path: str) -> dict:
    """One record, counted.  A live record is copied first, never touched."""
    tmp = None
    try:
        con = duckdb.connect(path, read_only=True)
    except duckdb.IOException:                      # she is still writing it
        # A LIVE RECORD IS COPIED BYTE BY BYTE AND NEVER OPENED IN PLACE.
        # `shutil.copy2` asks Windows for the file and is refused.  A plain
        # shared read is usually not --- but sometimes it is too, and then the
        # honest answer is "locked", never a traceback out of an instrument.
        tmp = tempfile.mkdtemp()
        try:
            for ext in ("", ".wal"):
                if not os.path.exists(path + ext):
                    continue
                dst9 = os.path.join(tmp, os.path.basename(path) + ext)
                with open(path + ext, "rb") as src, open(dst9, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
            con = duckdb.connect(os.path.join(tmp, os.path.basename(path)),
                                 read_only=True)
        except (PermissionError, OSError, duckdb.IOException) as why:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"locked": str(why).splitlines()[0][:120], "closed": 0,
                    "ticks": 0, "live": True}

    out = {"live": tmp is not None}
    out["ticks"] = con.execute("SELECT count(*) FROM step").fetchone()[0] + \
                   con.execute("SELECT count(*) FROM key").fetchone()[0]
    out["closed"] = con.execute("SELECT count(*) FROM exp").fetchone()[0]
    out["worth_on_disk"] = con.execute(
        "SELECT count(*) FROM exp WHERE weight != 0").fetchone()[0]
    out["composed"] = con.execute(
        "SELECT count(*) FROM exp WHERE struct_extract(exp, 'id') != 0").fetchone()[0]

    plans = collections.Counter()
    for tbl in ("key", "step"):
        for pid, n in con.execute(
                "SELECT struct_extract(plan, 'id') AS p, count(*) FROM %s "
                "WHERE struct_extract(plan, 'id') != 0 GROUP BY p" % tbl).fetchall():
            plans[int(pid)] += int(n)
    out["replayed"] = len(plans)
    out["replay_ticks"] = sum(plans.values())
    out["times"] = collections.Counter(plans.values())

    got = con.execute("SELECT packed FROM hands ORDER BY age DESC LIMIT 1").fetchone()
    if got:
        import json
        h = (json.loads(got[0]) or {}).get("his") or {}
        out["reachable"] = len(h.get("worth") or {})
        lived = [int(v) for v in (h.get("lived") or {}).values()]
        out["lived_again"] = sum(1 for v in lived if v > 1)
        out["lived_max"] = max(lived) if lived else 0
    else:
        out["reachable"] = out["lived_again"] = out["lived_max"] = 0
    con.close()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def report(path: str) -> None:
    r = read(path)
    if r.get("locked"):
        print("=== %s" % os.path.basename(path))
        print("    LOCKED --- she is still holding it: %s" % r["locked"])
        print()
        return
    mins = r["ticks"] / TICKS_PER_SECOND / 60.0
    print("=== %s%s" % (os.path.basename(path), "   (still living)" if r["live"] else ""))
    print("    %.0f minutes of her, %d experiences closed" % (mins, r["closed"]))
    if not r["closed"]:
        print("    nothing closed --- no life to read")
        print()
        return
    used = r["replayed"]
    print("    ever used again ............ %5d of %-5d  (%.1f%%)"
          % (used, r["closed"], 100.0 * used / r["closed"]))
    print("    still reachable at the end .. %5d of %-5d  (%.1f%%)"
          % (r["reachable"], r["closed"], 100.0 * r["reachable"] / r["closed"]))
    print("    worth surviving on disk ..... %5d of %-5d  (%.1f%%)"
          % (r["worth_on_disk"], r["closed"], 100.0 * r["worth_on_disk"] / r["closed"]))
    print("    part of a bigger experience . %5d of %-5d  (%.1f%%)"
          % (r["composed"], r["closed"], 100.0 * r["composed"] / r["closed"]))
    print("    her life spent repeating .... %5.1f%%  (%d of %d ticks came from a memory)"
          % (100.0 * r["replay_ticks"] / max(r["ticks"], 1), r["replay_ticks"], r["ticks"]))
    if r["times"]:
        once = sum(n for t, n in r["times"].items() if t <= 90)
        print("    of the %d she used again, %d were used for under a second" % (used, once))
        top = sorted(r["times"].items())[-1]
        print("    the one she used most ....... %d ticks = %.0f s of replay"
              % (top[0], top[0] / TICKS_PER_SECOND))
    print("    times lived, from her hands . more than once: %d,  most: %d"
          % (r["lived_again"], r["lived_max"]))
    print()


def main(argv) -> int:
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mind", "lives")
    paths = argv[1:] or sorted(glob.glob(os.path.join(here, "*.duckdb")))
    if not paths:
        print("no records in", here)
        return 1
    print()
    print("DOES SHE LEARN --- of everything she closed, how much did she use again")
    print()
    for p in paths:
        try:
            report(p)
        except Exception as e:                                   # noqa: BLE001
            print("=== %s\n    could not read: %s\n" % (os.path.basename(p), e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
