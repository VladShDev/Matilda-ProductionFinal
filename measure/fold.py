"""COULD THE TREE EVER HAVE GROWN?  --- the fold, counted offline.

His, 2026-09-03: *"Memory it is the tree"*, and Fable's build the same minute:
*"a parent's chain as references to its children, each tick held once, walked
as a sequence."*  Under that design her hands may drop a child, because the
parent already contains it and the record hands the leaves back.

The fold's rule was: keep the tale of her runs --- what each one REPLAYED, or
itself when discovery made it --- and when the same ordered pair of runs comes
round a SECOND time, those two become one parent.  Its condition, in full:

    elif (self.pairs[pair] == 0 and a1 and b1
          and pair[0] in self.chains and pair[1] in self.chains):

`self.chains` is her 26+26 hands.  So a pair could only fold if BOTH halves
were still in hand when it recurred --- which is to say the tree could only be
built if her memory was already big enough, and the tree is what would have
made it big enough.  This asks whether that condition was the binding one:

    pairs that recur          how many folds were available at all
    ...both still in hand     how many the old rule would actually have taken
    parents in the record     what really happened

Nothing here runs her.  It reads finished records.

    python -m measure.fold                    every record in mind/lives/
    python -m measure.fold NAME.duckdb ...    the ones you name
"""
from __future__ import annotations

import collections
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "mind")):
    if p not in sys.path:
        sys.path.insert(0, p)

from measure.record import frames                                # noqa: E402

HANDS = 52                      # SHORT 26 + LONG 26, her hands' size


def story(path: str):
    """The tale of her runs: per closed run, what it replayed --- or itself.

    A run is a maximal span of ticks sharing one `exp.id`; what it replayed is
    the `plan.id` its ticks carried (0 when discovery made the plan).
    """
    told, cur, plan = [], None, 0
    for t in frames(path):
        e = int(getattr(t.exp, "id", 0) or 0)
        p = int(getattr(t.plan, "id", 0) or 0)
        if cur is None:
            cur, plan = e, p
        elif e != cur:
            told.append(plan or cur)
            cur, plan = e, p
        elif p and not plan:
            plan = p
    if cur is not None:
        told.append(plan or cur)
    return told


def parentsInRecord(path: str) -> int:
    import duckdb
    con = duckdb.connect(path, read_only=True)
    rows = con.execute("SELECT exp FROM exp").fetchall()
    con.close()
    return sum(1 for (e,) in rows if e and int(e["id"]) != 0)


def report(path: str) -> None:
    told = story(path)
    print("=== %s" % os.path.basename(path))
    print("    runs closed .................... %d" % len(told))
    if len(told) < 3:
        print("    too short to pair\n")
        return
    pairs = collections.Counter()
    folds, inhand, seenAt = 0, 0, {}
    recent = collections.deque(maxlen=HANDS)
    for k in range(1, len(told)):
        a, b = told[k - 1], told[k]
        recent.append(told[k - 1])
        pairs[(a, b)] += 1
        if pairs[(a, b)] == 2:                  # once is not a statistic
            folds += 1
            held = set(recent) | {told[k]}
            if a in held and b in held:
                inhand += 1
    print("    distinct ordered pairs ......... %d" % len(pairs))
    print("    pairs that came round twice .... %d   <- folds AVAILABLE" % folds)
    print("    ...both still in her hands ..... %d   <- folds the OLD RULE takes"
          % inhand)
    try:
        got = parentsInRecord(path)
        print("    parents actually in the record . %d" % got)
    except Exception as why:                                     # noqa: BLE001
        print("    parents in the record .......... unreadable (%s)" % str(why)[:50])
    if folds:
        print("    the hands condition costs ...... %d of %d (%.0f%%)"
              % (folds - inhand, folds, 100.0 * (folds - inhand) / folds))
    print()


def main(argv) -> int:
    paths = argv[1:] or sorted(glob.glob(os.path.join(ROOT, "mind", "lives", "*.duckdb")))
    print()
    print("COULD THE TREE HAVE GROWN --- folds available, against folds taken")
    print()
    for p in paths:
        try:
            report(p)
        except Exception as why:                                 # noqa: BLE001
            print("=== %s\n    could not read: %s\n"
                  % (os.path.basename(p), str(why)[:90]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
