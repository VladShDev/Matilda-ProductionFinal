"""EVERYTHING GOES THROUGH THE EXPERIENCE --- checked, not assumed.

    python -m measure.through            # she must already be living on :8090
    python -m measure.through --wait 20  # longer window per channel

His, 2026-08-25: *"all data resoning everithing goes thught exp!!!!"*  That is
the architecture, and it is the kind of claim that stays true in a docstring
long after it has stopped being true in the code.

**THIS FILE USED TO ASK IT OF AN ORGAN THAT NO LONGER RUNS.**  It asked two
questions --- nothing lost between a tick and an experience, and no decision
taken outside one --- through `reasonOf` from `brain.brain.lookup`: the
IN-PROCESS brain, retired 2026-08-26 when her mind moved OUTSIDE (matilda4,
over HTTP).  From that day it raised ImportError before its first line of work.
The one instrument that asked his central question had been checking nothing
for a week, and nothing said so.  Rewritten 2026-08-31 at his ask: *"let's
check one by one, same sound, everything should go through exp.  Should be
chains, should be analysis, should be prediction."*

**IT ASKS THE LIVING GIRL, NOT A FRESH ONE.**  Each channel gets ONE short
stimulus through the same door the page uses, and then her own published
numbers are watched:

    his voice          POST /say            the word he certified by ear
    the television     POST /show           a picture, raw RGB, as the page sends
    a hand on her      POST /world touch    the helpers' own shape
    milk at her lips   POST /world give     the bottle her body already has

and for each, his list, in his order:

    arrived        did the line MOVE --- is the stimulus in her at all
    runs           did she open an experience
    closed         did a run END --- a chain exists only when one does
    joined         did her analysis connect anything new (his pair law)
    expecting      is she holding a prediction
    words          did a sound of the world become one of hers
    state          did she get paid

**ARRIVING IS THE ONLY FLOOR.**  A channel that does not arrive is a dead wire
--- this project's standing failure, and the only thing here that can be called
broken without knowing what she was busy with.  The rest are REPORTED, because
whether a particular twenty seconds closes a run depends on what she was doing,
and a gate that failed on that would train a bypass.

Nothing here is imported by her, and every read is a read: `/state` takes her
lock to copy numbers out and touches no field of hers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the word he certified by ear, 2026-08-29 --- the same stimulus the voice
#: guardian uses, so ONE sound is asked of every stage of her
WORD = os.path.join(HERE, "measured", "matilda_word_original.wav")


def get(at: str, path: str):
    with urllib.request.urlopen(at + path, timeout=10) as r:
        return json.loads(r.read())


def post(at: str, path: str, body, head=None):
    data = (body if isinstance(body, (bytes, bytearray))
            else json.dumps(body).encode())
    req = urllib.request.Request(at + path, data=data, headers=head or {})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def theWord() -> np.ndarray:
    with wave.open(WORD, "rb") as h:
        rate, n = h.getframerate(), h.getnframes()
        y = np.frombuffer(h.readframes(n), "<i2").astype(np.float32) / 32768.0
    if rate != speech.RATE:
        i = np.arange(0.0, len(y) - 1, rate / float(speech.RATE))
        y = np.interp(i, np.arange(len(y)), y).astype(np.float32)
    return y


def sayIt(at: str) -> None:
    post(at, "/say", theWord().astype("<f4").tobytes())


def showIt(at: str) -> None:
    """A PICTURE ON THE TELEVISION --- raw RGB, exactly as the page sends it.

    Two bright bars on a dark ground, wide enough that her optics keep them at
    the window's distance.  Nothing about it is a face; it is a THING that was
    not there a moment ago, which is what an orienting stimulus is.
    """
    w, h = 160, 120
    px = np.zeros((h, w, 3), np.uint8)
    px[20:100, 30:60] = 240
    px[20:100, 100:130] = 240
    post(at, "/show", px.tobytes(),
         {"X-Width": str(w), "X-Height": str(h),
          "Content-Type": "application/octet-stream"})


def touchIt(at: str) -> None:
    post(at, "/world", {"touch": {"body_left": 0.45, "body_right": 0.45}})


def feedIt(at: str) -> None:
    post(at, "/world", {"give": {"milk": 0.6}})


#: (name, how to give it, what its ARRIVAL looks like ON HER OWN LINES)
#:
#: EVERY ONE OF THESE IS HERS, NOT THE ROOM'S.  `tv.shows` would say the
#: picture reached the SCREEN, which proves nothing about the girl; what
#: counts is that her eye's count of things MOVED.  Same for the rest: an id
#: at her ear, her skin reporting a touch, her hunger actually falling.
CHANNELS = [
    ("his voice", sayIt,
     lambda s, was: int(s.get("hearing", 0) or 0) != 0),
    ("the television", showIt,
     lambda s, was: int(s.get("things", 0) or 0)
     != int(was.get("things", 0) or 0)),
    ("a hand on her", touchIt,
     lambda s, was: bool((s.get("held") or {}).get("touch"))
     or int((s.get("held") or {}).get("pins", 0) or 0) > 0),
    ("milk at her lips", feedIt,
     lambda s, was: float(s.get("hunger", 1.0) or 1.0)
     < float(was.get("hunger", 1.0) or 1.0) - 0.0005),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--wait", type=float, default=15.0,
                    help="seconds to watch her after each stimulus")
    got = ap.parse_args()
    at = got.at.rstrip("/")

    try:
        first = get(at, "/state")
    except Exception as why:                                # noqa: BLE001
        print("she is not living at %s (%s).  Start her first:" % (at, why))
        print("  python her.py --port 8090")
        return 1
    if first.get("runs") is None:
        print("her MIND is not attached --- /state carries no runs.  This asks "
              "whether things reach her experience, so it needs the mind.")
        return 1

    print("ONE STIMULUS PER CHANNEL, then her own numbers.  %.0f s each.\n"
          % got.wait)
    print("%-18s %8s %6s %7s %7s %10s %6s %7s"
          % ("channel", "arrived", "runs", "closed", "joined", "expecting",
             "words", "state"))
    dead = []
    for name, give, arrived in CHANNELS:
        was = get(at, "/state")
        seen = False
        give(at)
        till = time.time() + got.wait
        while time.time() < till:
            now = get(at, "/state")
            seen = seen or bool(arrived(now, was))
            time.sleep(0.1)
        now = get(at, "/state")

        def moved(key, a=was, b=None):
            b = now if b is None else b
            x, y = a.get(key), b.get(key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return y - x
            return 0

        if not seen:
            dead.append(name)
        print("%-18s %8s %6s %7s %7s %10s %6s %7s"
              % (name, "yes" if seen else "NO",
                 "%+d" % moved("runs"), "%+d" % moved("closes"),
                 "%+d" % moved("connections"),
                 int(now.get("expecting", 0) or 0),
                 "%+d" % moved("words"),
                 "%+.3f" % moved("state")))

    print()
    if dead:
        print("A DEAD WIRE: %s reached nothing of her.  That is the failure "
              "this file exists to catch." % ", ".join(dead))
        return 1
    print("EVERY CHANNEL REACHES HER --- each stimulus moved her own lines, "
          "and what it did afterwards is above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
