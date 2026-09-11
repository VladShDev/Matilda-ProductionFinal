"""HOW WELL DOES SHE HEAR HIM?  Stage by stage, with the losses named.

    python -m measure.hears           # she must already be living on :8090

His voice crosses five stages to become one number on her tick, and every one
of them can throw the word away:

    the microphone      what the page sends, as raw samples
    the queue           her ears drain `TICK_SECONDS` a tick; a queue that
                        fills is a word being DELETED, oldest first
    the air             `NEAR / (NEAR + far)` --- the television is a distance
                        away, and this multiplies every band before anything
                        else looks at it
    the gate            a window with no piece over `GATE` is silence, whole
    her alphabet        her ear names what is left with one of HER sounds ---
                        or silence, if nothing that arrived is makeable

**THIS DROVE A STANDALONE BODY UNTIL 2026-09-01, AND A STANDALONE BODY RUNS
THE RETIRED IN-PROCESS BRAIN.**  It died on its first tick with an IndexError
and had done since her mind moved outside on 2026-08-26 --- so the one
instrument that could say WHERE his voice was lost said nothing at all, and
the suite stayed green.  It asks the LIVING girl now, through the same doors
the page uses: `POST /say`, `/state`, `/ticks`.

Nothing here is hers: the word goes in through the same door his microphone
uses, and every number is read off her own body.
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

from body import speech                                     # noqa: E402
from body.air import FAINTEST, NEAR                         # noqa: E402
from body.alike import SILENCE                              # noqa: E402
from body.hearing import (GATE, SOUND_HOPS,                  # noqa: E402
                          TICK_SECONDS, bands_from_pcm)
from measure.mouth import SOURCES, _load, cut, heardAs       # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORD = os.path.join(HERE, "measured", "matilda_word_original.wav")
#: how many moments of his the deaf-ear test says back to her
NAMES = 16
#: she must name at least this many of them as themselves
NAMES_BAR = 0.8


def get(at, path):
    with urllib.request.urlopen(at + path, timeout=20) as h:
        return json.loads(h.read())


def say(at, pcm):
    req = urllib.request.Request(at + "/say",
                                 data=np.asarray(pcm, "<f4").tobytes())
    with urllib.request.urlopen(req, timeout=15) as h:
        h.read()


def heard(at, since, seconds=3.0):
    """Every id her ear named after `since`, read off her own tape."""
    time.sleep(seconds)
    got = get(at, "/ticks?from=%d" % since)
    rows = got.get("ticks", []) if isinstance(got, dict) else got
    out = []
    for one in rows:
        life = one.get("life", one)
        snd = (life.get("input", {}) or {}).get("sound") or {}
        if float(snd.get("lvl", 0) or 0) > 0.0:
            out.append(int(snd.get("id", 0) or 0))
    return out, len(rows)


def naming(at, pieces, rows, ears) -> int:
    """HOW MANY OF HIS SOUNDS COME BACK AS THE NAME SHE FILED THEM UNDER.

    **THE ONLY TEST THAT CATCHES A DEAF EAR.**  Every stage above can read
    perfectly --- every band arriving, nothing queued, nothing gated --- while
    she names three quarters of what she hears as some other sound, and
    nothing says a word about it.

    MEASURED 2026-08-26, and it is why `SOUND_HOPS` is 1:

        one placement       84 of 93    90.3%
        two placements      32 of 93    34.4%
        four placements     24 of 93    25.8%

    Say her own sound to her, and if it does not come back as itself her ear
    is not naming, it is guessing.
    """
    # HIS SOUNDS, NOT HERS.  Her ears are for what is NOT hers (she hears her
    # own mouth through her echo, never through the air), and her alphabet's
    # names are what her ear gives HIS voice.  So the question that matters is
    # whether a moment of his, played back into her room, comes home as the
    # name her mouth has filed it under --- that is the whole loop: his voice
    # -> her name -> her piece.
    #
    # Asking it with her own PIECES instead measured 6.2%, and it was the
    # wrong question: a piece is her register, and what arrives at her ear
    # after the air has shaped it is not what was filed.
    raw = _load(os.path.join(HERE, *SOURCES[0].split("/")))
    step = int(round(speech.RATE * TICK_SECONDS))
    grab = int(round(step * 1.75))
    want = []
    for i in range(0, len(raw) - grab, step * 7):
        window = raw[i:i + step]
        if float(np.abs(window).max()) < 0.04:
            continue
        name, _shape = heardAs(window)
        if name is not None and name in rows:
            want.append((name, raw[i:i + step * 6]))
        if len(want) >= NAMES:
            break
    right = 0
    names = want
    for one, sound in want:
        say(at, sound)
        since = int(get(at, "/state")["tick"])
        got, _ = heard(at, since, 1.5)
        if one in got:
            right += 1
    print("  HIS sounds, named back as the name her mouth filed them under: "
          "%d of %d   %.1f%%"
          % (right, len(names), 100.0 * right / max(1, len(names))))
    return 0 if right >= len(names) * NAMES_BAR else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    got = ap.parse_args()
    at = got.at.rstrip("/")
    try:
        state = get(at, "/state")
    except Exception as why:                                # noqa: BLE001
        print("she is not living at %s (%s) --- start her with python her.py"
              % (at, why))
        return 1

    word = _load(WORD)
    step = int(round(speech.RATE * TICK_SECONDS))
    ticks = len(word) // step
    pieces, rows, ears = cut()

    win = state.get("window") or {}
    where = np.asarray(win.get("at") or (0, 0, 0), np.float32)
    print("HIS WORD INTO HER ROOM --- %d ticks of it, %.2f s\n"
          % (ticks, len(word) / speech.RATE))

    # 1 --- the microphone
    mine = [bands_from_pcm(word[k * step:(k + 1) * step], speech.RATE,
                           TICK_SECONDS, SOUND_HOPS) for k in range(ticks)]
    lit = sum(1 for b in mine if float(np.asarray(b).max()) > 0.0)
    print("  1. at the microphone      %d of %d ticks carry anything"
          % (lit, ticks))

    # 2 --- the queue
    since = int(state["tick"])
    say(at, word)
    queued = float(get(at, "/state").get("queued", 0.0) or 0.0)
    print("  2. her ear's queue        %.2f s waiting (a queue that never "
          "drains is a word being deleted)" % queued)

    # 3 --- the air, at the television's real distance
    got2, seen = heard(at, since, len(word) / speech.RATE + 2.0)
    print("  3. the air                the screen is %.2f m from the room's "
          "middle; fade NEAR/(NEAR+far) = %.3f"
          % (float(np.linalg.norm(where)),
             NEAR / (NEAR + max(0.0, float(np.linalg.norm(where))))))

    # 4/5 --- the gate and her alphabet
    named = [i for i in got2 if i != SILENCE]
    print("  4. the gate               %d of her %d ticks in that window "
          "carried a sound at all" % (len(got2), seen))
    print("  5. her alphabet           %d of them got a NAME (%d distinct: %s)"
          % (len(named), len(set(named)), sorted(set(named))[:10]))
    print("     her faintest band is %g and her gate %g\n" % (FAINTEST, GATE))

    bad = naming(at, pieces, rows, ears)
    print("\n  %s" % ("SHE HEARS HIM." if not bad and named else
                      "SHE IS NOT NAMING WHAT SHE HEARS."))
    return 0 if (not bad and named) else 1


if __name__ == "__main__":
    raise SystemExit(main())
