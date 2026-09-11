"""HER ALPHABET, LIVED ONCE, BEFORE SHE HAS TO FIND IT.

His, 2026-08-31: *"before she start leaving, we have to record table for
for example, I don't know, thousand ticks and fill all these ticks to
sounds that we need to produce sounds that needed to speech.  And then
grab those ids and pass to our experience if she need to produce them,
if she already gets similarity with them."*

This is the older tree's own harness, missing from this one: `A TAPE CAN
BE BORN WITH HER OWN VOICE`.  Without it her mouth is found by the
ladder, one sound and one loudness step at a time --- 154 sounds x 20
steps is over three thousand rungs, and MEASURED on her live body she
had used 5 of her 154 sounds after twenty minutes of life.  With it,
every sound she can make is in her record within half a minute: she has
SAID it, HEARD her own echo of it, and the moment carries the pair ---
so when the world says something like it, her own experience can reach
for it by similarity instead of waiting for the ladder to arrive.

IT IS A HARNESS, NOT A BIRTH.  It lives in `measure/`, nothing in
`body/` or the mind imports it, and it invents no ability: every sound
it walks is one HER OWN mouth already holds (`/able`), commanded through
the same `POST /act` her own deciding uses, one tick at a time.  Her
tape records it exactly as it records anything else she does.

    python -m measure.babble                 # her whole alphabet, once
    python -m measure.babble --hold 6 --gap 2 --at http://127.0.0.1:8090
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def ask(at, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(at.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"}
                                 if data else {})
    with urllib.request.urlopen(req, timeout=10) as h:
        return json.loads(h.read().decode() or "{}")


def babble(at="http://127.0.0.1:8090", hold=6, gap=2, lvl=0.6, rounds=1):
    """Say every sound she can make, `hold` ticks each, `gap` ticks apart."""
    able = ask(at, "/able")
    sounds = sorted(int(s) for s in able.get("sounds", []))
    began = ask(at, "/state")["tick"]
    print("her alphabet: %d sounds | %d ticks of her life will carry it"
          % (len(sounds), rounds * len(sounds) * (hold + gap)))
    said = 0
    for r in range(rounds):
        for sid in sounds:
            for _ in range(hold):
                ask(at, "/act", {"kind": "sound", "id": int(sid),
                                 "lvl": float(lvl)})
                said += 1
                time.sleep(1.0 / 60.0)      # ...faster than her tick; never slower
            for _ in range(gap):
                ask(at, "/act", {"kind": "sound", "id": 0, "lvl": 0.0})
                time.sleep(1.0 / 60.0)
    end = ask(at, "/state")
    print("commanded %d times across %d of her ticks"
          % (said, end["tick"] - began))
    return began, end["tick"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--lvl", type=float, default=0.6)
    ap.add_argument("--rounds", type=int, default=1)
    got = ap.parse_args()
    babble(got.at, got.hold, got.gap, got.lvl, got.rounds)
