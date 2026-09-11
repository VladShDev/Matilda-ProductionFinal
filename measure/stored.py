"""WHAT SHE HAS STORED, READ OFF A RUNNING GIRL --- she is never stopped.

    python -m measure.stored              one tick, every table
    python -m measure.stored --follow 40  one thing's size over time

His: *"show me stored data and now everithing in live we can read from values
without stopping her?"*  Yes: `/mind` is a read.  Her lock is taken to copy the
numbers out and no field of hers is touched, so nothing here can change what she
does --- which is the only way an instrument is allowed to work.

`Her.mind` had existed since her body was assembled and NOTHING SERVED IT.  It
died with the second server, an hour after the rule about exactly this was
written down.

WHAT `--follow` IS FOR.  She has no depth and does not need any: his,
2026-08-25, *"i mean it would becounted by object size like peapledo"*.  How
much of her view a thing fills IS its apparent size, it is stored on every tick
beside the thing's own id under `similarity`, and it is the one field her reason
reads.  So a thing coming closer is `view:290` climbing --- a sentence her own
lines can already say --- and this prints that climb.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def mind(at: str) -> dict:
    with urllib.request.urlopen(at + "/mind", timeout=10) as a:
        return json.load(a)


def show(d: dict) -> None:
    print(f"tick {d['tick']}   {d['seconds']} s of life   state {d['state']:+.4f}"
          f"   in run {d['in run']}")
    print(f"seeing you {d['seeing you']}   his voice queued {d['queued']} s, "
          f"late {d['late']} s")

    print("\nHER LINES THIS TICK --- every one keyed by its own id")
    kinds: dict = {}
    for k, v in d["lines"].items():
        kinds.setdefault(k.split(":")[0], []).append((k, v))
    for kind in sorted(kinds):
        rows = sorted(kinds[kind], key=lambda r: -abs(r[1]))
        live = [f"{k}={v:+.4f}" for k, v in rows if abs(v) > 1e-9]
        print(f"  {kind:9} {len(rows):3d} lines, {len(live):3d} carrying "
              f"anything   {', '.join(live[:6])}")

    print(f"\nWHAT SHE IS SAYING AND HEARING")
    print(f"  saying  id {d['saying']['id']:4d}  lvl {d['saying']['lvl']}")
    print(f"  echo    id {d['echo']['id']:4d}  lvl {d['echo']['lvl']}"
          f"      <- her own mouth coming back")
    print(f"  hearing id {d['hearing']['id']:4d}  lvl {d['hearing']['lvl']}"
          f"      <- somebody else")
    print(f"  moving  id {d['moving']['id']:4d}  lvl {d['moving']['lvl']}")

    print(f"\nTHE THINGS SHE IS LOOKING AT --- {d['things']} of them, "
          f"{len(d['kinds'])} kinds: {d['kinds']}")
    print("     id        x        y        w        h   pieces   seen")
    for b in d["boxes"]:
        print(f"  {b['id']:5d}  {b['x']:+7.4f}  {b['y']:+7.4f}  "
              f"{b['w']:7.4f}  {b['h']:7.4f}   {b['pieces']:4d}   {b['seen']:.3f}")
    print("  a centre is -1..1, a size is a FRACTION of the view --- two units in"
          "\n  one row, and `w * h` is the apparent size her reason reads")

    print(f"\nWHAT SHE HAS LEARNED --- {len(d['runs'])} runs, "
          f"short {d['short']}, long {d['long']}")
    print("     id   avgState    weight     best   part of")
    for e in d["runs"]:
        print(f"  {e['id']:5d}  {e['avgState']:+9.5f}  {e['weight']:8.5f}  "
              f"{e['best']:7.3f}   {e['partOf']:5d}")


def follow(at: str, n: int) -> None:
    """One thing's apparent size over time --- her whole distance sense."""
    was, rows = -1, []
    while len(rows) < n:
        d = mind(at)
        if d["tick"] == was:
            time.sleep(0.03)
            continue
        was = d["tick"]
        big = {}
        for b in d["boxes"]:
            big[b["id"]] = max(big.get(b["id"], 0.0), b["w"] * b["h"])
        rows.append((d["tick"], big))
    ids = sorted({i for _, b in rows for i in b})
    print("HOW BIG EACH KIND OF THING LOOKS, tick by tick.")
    print("Bigger is nearer, and that is the whole of her distance sense.\n")
    print("   tick  " + "  ".join(f"view:{i:<7d}" for i in ids))
    for tick, big in rows:
        print(f"  {tick:5d}  " + "  ".join(
            f"{big.get(i, 0.0):.5f}     " if i in big else "    .        "
            for i in ids))
    print("\n  a column that climbs is a thing coming closer.  Nothing computes")
    print("  a distance from it: her reason reads the size, and what it means")
    print("  is hers to learn --- a person is not handed metres either.")


def whole(at: str) -> None:
    """ONE WHOLE TICK, every field she stored in it.

    Her own dataclasses, walked by `asdict` on the server --- so if a field is
    added to `View` it turns up here and nothing has to be told about it.
    """
    with urllib.request.urlopen(at + "/tick", timeout=10) as a:
        d = json.load(a)
    life = d["life"]
    i, o = life["input"], life["output"]
    print(f"TICK {d['at']}   at {d['seconds']} s   part of exp {d['exp']['id']}"
          f"   (its own clock: {d['time']})")

    print("")
    print("WHAT SHE DID --- output")
    print(f"  motor    id {o['motor']['id']:6}  lvl {o['motor']['lvl']:+.4f}")
    print(f"  sound    id {o['sound']['id']:6}  lvl {o['sound']['lvl']:+.4f}")

    print("")
    print("WHAT CAME BACK --- input")
    print(f"  state                {i['state']:+.4f}   "
          f"<- the one line with no id, because it is only ever itself")
    print(f"  spindle  id {i['spindle']['id']:6}  lvl {i['spindle']['lvl']:+.4f}"
          f"   <- the muscle she just moved, felt")
    print(f"  echo     id {i['echo']['id']:6}  lvl {i['echo']['lvl']:+.4f}"
          f"   <- her own mouth coming back")
    print(f"  sound    id {i['sound']['id']:6}  lvl {i['sound']['lvl']:+.4f}"
          f"   balance {i['sound'].get('balance')}   <- somebody else")

    print("")
    print(f"  sensor --- {len(i['sensor'])} lines, one id each")
    for one in i["sensor"]:
        mark = "  <-" if abs(one["lvl"]) > 1e-9 else ""
        print(f"    id {one['id']:6}  lvl {one['lvl']:+.4f}{mark}")

    print("")
    print(f"  view --- {len(i['view'])} THINGS, "
          f"and every number each carries")
    print(f"    {'id':>10}{'similarity':>12}{'x':>10}{'y':>10}")
    for one in i["view"]:
        print(f"    {one['id']:>10}{one['similarity']:>12.5f}"
              f"{one['x']:>+10.4f}{one['y']:>+10.4f}")
    print("    id is WHAT IT LOOKS LIKE **AND WHERE IT IS**; similarity is HOW")
    print("    BIG, which IS her distance --- there is no z and there is no gap.")
    print("")
    print("  each of those becomes its own line in her reason:")
    for one in i["view"][:3]:
        k = one["id"]
        print(f"    view:{k}={one['similarity']:.5f}   viewx:{k}={one['x']:+.4f}"
              f"   viewy:{k}={one['y']:+.4f}")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--at", default="http://127.0.0.1:8080")
    ask.add_argument("--tick", action="store_true",
                     help="one whole tick, every field she stored in it")
    ask.add_argument("--follow", type=int, default=0,
                     help="print one thing's apparent size over N looks")
    got = ask.parse_args()
    if got.tick:
        whole(got.at)
    elif got.follow:
        follow(got.at, got.follow)
    else:
        show(mind(got.at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
