"""TEACHING HER A WORD, FOR AS LONG AS SHE WILL LISTEN.

    python -m measure.teaching [hours]     # she must already be living

His, 2026-09-02: *"try to train her like we did before, but in right way."*

WHAT WAS WRONG BEFORE, and it is why this file exists rather than a change to
`measure/words.py`.  That instrument builds its word out of `Voice.rows` --- the
names her ear can RECOGNISE --- and its own docstring says it uses *"two sounds
her body can actually make"*.  Those are different tables: she recognises 72
names and MAKES 57.  Measured 2026-09-02, the word it kept asking for was
`[275, 350]`, and **350 is one she can hear and cannot say** --- it arrived at
her ear ZERO times in a whole run.  `p = 1.000` was never about her.

THREE THINGS THIS DOES THAT THE OLD ONE DID NOT:

  1. **The word is built from `Voice.says`** --- the pieces that MAKE a sound,
     the side of her mouth's table repaired on 2026-09-02.
  2. **Every candidate is played into her room first and kept only if her own
     ear gives it back.**  Her filing went from 3 of 12 to 11 of 12 that night,
     so about one in twelve still arrives under another name --- and a word
     whose first half she never HEARS cannot be learned by anyone.
  3. **It runs for hours, not minutes.**  A word said thirty times is not an
     education; her milestones are months of infant life.

NOTHING IS TAUGHT INTO HER.  No score is written, no id is planted, nothing in
her mind is touched.  The word goes in through `/say` --- the same door his
microphone uses --- and what comes out is read off her own ticks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body.hearing import SOUND_BANDS, SOUND_SLIDES            # noqa: E402
from body.muscles import Voice                                # noqa: E402

#: how many ticks one sound is held.  A single tick is a click; his own
#: measurement makes a syllable 10-20 ticks, so six is a short clear one.
HELD = 6
#: how many of her own ear's namings of a played sound count as "it came back"
BACK = 3


def get(at, path):
    with urllib.request.urlopen(at + path, timeout=30) as h:
        return json.loads(h.read())


def say(at, pcm):
    r = urllib.request.Request(at + "/say",
                               data=np.asarray(pcm, "<f4").tobytes())
    urllib.request.urlopen(r, timeout=20).read()


def sound(v, one):
    out = []
    for _ in range(HELD):
        v.play(int(one), 1.0)
        out.append(np.asarray(v.pcm, np.float32).copy())
    return np.concatenate(out)


def heardIds(at, since):
    got = get(at, "/ticks?from=%d" % since)
    rows = got.get("ticks", []) if isinstance(got, dict) else got
    heard, made = [], []
    for r in rows:
        li = r.get("life", r).get("input", {}) or {}
        lo = r.get("life", r).get("output", {}) or {}
        s = li.get("sound") or {}
        if float(s.get("lvl") or 0) > 0:
            heard.append(int(s.get("id") or 0))
        m = lo.get("sound") or {}
        if float(m.get("lvl") or 0) > 0:
            made.append(int(m.get("id") or 0))
    nxt = int(got.get("next", since)) if isinstance(got, dict) else since
    return heard, made, nxt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hours", nargs="?", type=float, default=None)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--every", type=float, default=20.0)
    got = ap.parse_args()
    at = got.at.rstrip("/")

    v = Voice(SOUND_BANDS, SOUND_SLIDES, bank=os.path.join("lives", "mouth.npz"))
    able = set(get(at, "/able").get("sounds") or ())
    canSay = sorted(n for n in v.says if n in able)
    print("she recognises %d names and MAKES %d\n" % (len(v.rows), len(v.says)))
    print("trying her own sounds in her own room, keeping what comes back:")
    good = []
    for one in canSay[::3]:
        since = int(get(at, "/state")["tick"])
        say(at, sound(v, one))
        time.sleep(1.4)
        heard, _, _ = heardIds(at, since)
        if heard.count(int(one)) >= BACK:
            good.append(int(one))
            print("   %-5d comes back as itself (%d of %d)"
                  % (one, heard.count(int(one)), len(heard)))
        if len(good) >= 4:
            break
    if len(good) < 2:
        print("   fewer than two of her sounds survive her own room")
        return 1
    a = good[0]
    b = next((n for n in good[1:] if abs(n - a) > 8), good[1])
    word = np.concatenate([sound(v, a), sound(v, b)])
    print("\nTEACHING HER  %d then %d  --- %.2f s, every %.0f s\n"
          % (a, b, len(word) / 16000.0, got.every))

    began = time.time()
    since = int(get(at, "/state")["tick"])
    said = heardA = heardB = madeA = madeB = inOrder = 0
    lastSaid = None
    while got.hours is None or (time.time() - began) < got.hours * 3600.0:
        say(at, word)
        said += 1
        time.sleep(got.every)
        try:
            heard, made, since = heardIds(at, since)
        except Exception:                                   # noqa: BLE001
            continue
        heardA += heard.count(a)
        heardB += heard.count(b)
        madeA += made.count(a)
        madeB += made.count(b)
        for i in range(len(made) - 1):
            if made[i] == a and made[i + 1] == b:
                inOrder += 1
        if said % 15 == 0:
            s = get(at, "/state")
            print("   %5.0f min   said %4d   she heard %4d/%4d   she MADE "
                  "%3d/%3d   IN ORDER %d   |  runs %s conn %s guided %s"
                  % ((time.time() - began) / 60.0, said, heardA, heardB,
                     madeA, madeB, inOrder, s.get("runs"),
                     s.get("connections"), s.get("guided")), flush=True)
    print("\n   said to her %d times; she made %d of %d and %d of %d; "
          "IN ORDER %d" % (said, madeA, a, madeB, b, inOrder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
