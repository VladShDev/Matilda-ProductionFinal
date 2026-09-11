"""CAN SHE STILL SAY A WORD BACK?  The one test the whole thing is for.

    python -m measure.words [ticks]

A WORD IS SAID INTO HER ROOM AND HER MOUTH IS WATCHED.  The word is built from
HER OWN alphabet --- two sounds her body can actually make --- because a word
she has no mouth for is not a test of her memory, it is a test of her anatomy,
and that question is `body/sounds.json`'s and was answered long ago.

Nothing is taught, nothing is scored into her, nothing is patched: the word goes
in through the same `window.say` his microphone uses, and what comes out is read
off `tick.life.output.sound`.  Then:

    heard      the ids that reached her ears
    said       the ids her mouth made
    IN ORDER   whether the PAIR came back in the order it went in --- which is
               the whole of "she repeated the word" and not "she made a noise"

**IT DROVE A STANDALONE BODY UNTIL 2026-09-01, AND A STANDALONE BODY RUNS THE
RETIRED IN-PROCESS BRAIN** --- it died on its first tick, and had since her mind
moved outside on 2026-08-26.  The one test that asks whether she LEARNS asked
nothing for a week.  It drives the living girl now, through the same doors the
page uses: `POST /say`, `/state`, `/ticks`.

**IT IS THE FIRST TEST IN THIS TREE THAT ASKS WHETHER SHE LEARNS ANYTHING.**
Everything else measures whether a mechanism is correct.  On 2026-08-26 every
one of them was green while `/eye` hung, and her whole life held two
experiences.  A thing can be right in every part and do nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                       # noqa: E402
from body.alike import FLOOR, SILENCE                         # noqa: E402
from body.hearing import SOUND_BANDS, SOUND_SLIDES            # noqa: E402
from body.muscles import Voice                                # noqa: E402
from measure.mouth import cut                                 # noqa: E402

#: How often the word is said to her, in ticks.  Far enough apart that her ears
#: have drained it before the next one --- her ear buffer is real and a word on
#: top of a word is two words at once, not the same word twice.
EVERY = 30

#: HOW LONG ONE SOUND OF THE WORD IS HELD, in ticks.  A piece is one tick, and
#: a single tick of sound is a click, not a syllable --- his own measurement
#: (2026-08-31): a syllable is 10-20 ticks.  Six is a short, clear syllable.
HELD = 6


def wordOf(mouths, ids, bank: str = "lives/mouth.npz") -> np.ndarray:
    """A word, in HER voice, from sounds she can make.

    HER OWN MOUTH SAYS IT --- `Voice.play`, the pieces she actually has ---
    so what enters her room is a thing her mouth could have produced, and
    comparing what she says back to it is a fair question.

    IT USED TO DRIVE ARTICULATORS.  `v.say(shape)` was the synthesized larynx,
    retired 2026-08-30 when her sounds became recorded pieces (his decision,
    certified by his ear).  From that day this raised
    `ValueError: could not broadcast (46,1) into (7,1)` on its first word, and
    it took `measure.words` --- "the first test in this tree that asks whether
    she learns anything" --- and `measure.hears` down with it.  Both had been
    unrunnable ever since, and neither said so.

    `mouths` is `Her.mouths`: her ear's NAMES, each holding the pieces that
    wear it.  A name is held for `HELD` ticks so the word is a word and not a
    click --- the same law her mouth lives by, a piece a tick, walked in the
    order she heard them.
    """
    v = Voice(SOUND_BANDS, SOUND_SLIDES, bank=bank)
    out = []
    for one in ids:
        if int(one) not in v.rows:
            continue
        for _ in range(HELD):
            v.play(int(one), 1.0)
            out.append(np.asarray(v.pcm, np.float32).copy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _get(at, path):
    with urllib.request.urlopen(at + path, timeout=20) as h:
        return json.loads(h.read())


def _say(at, pcm):
    req = urllib.request.Request(at + "/say",
                                 data=np.asarray(pcm, "<f4").tobytes())
    with urllib.request.urlopen(req, timeout=15) as h:
        h.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--times", type=int, default=20,
                    help="how many times the word is said to her")
    ap.add_argument("--every", type=float, default=6.0,
                    help="seconds between sayings")
    args = ap.parse_args()
    at = args.at.rstrip("/")
    try:
        first = _get(at, "/state")
    except Exception as why:                                # noqa: BLE001
        print("she is not living at %s (%s) --- start her with python her.py"
              % (at, why))
        return 1

    pieces, rows, ears = cut()
    # TWO SOUNDS SHE CAN MAKE, taken off her own table rather than chosen ---
    # AND NEITHER OF THEM IS SILENCE.  An id is not a sound just because it is
    # in the table.  Two that are not neighbours, because neighbouring ids are
    # sounds that resemble each other, and a word of two near-identical pieces
    # cannot tell "she said it back" from "she made a noise twice".
    every = [int(k) for k in sorted(rows) if int(k) not in (SILENCE, FLOOR)]
    if len(every) < 3:
        print("she has no alphabet --- lives/mouth.npz is empty.")
        return 1
    ids = [every[len(every) // 3], every[2 * len(every) // 3]]
    word = wordOf(rows, ids)
    if not word.size:
        print("her mouth made nothing for those sounds.")
        return 1
    print("the word said to her: %s   (%d samples, %.2f s)\n"
          % (ids, word.size, word.size / speech.RATE))

    # HER OWN TAPE, GATHERED AS SHE LIVES IT.  `/ticks` keeps only her last
    # ~900 ticks (30 s), so a minute-long test read at the END sees half of
    # it and calls the missing half silence --- the same trap that had the
    # scale scoring every motor item on no frames at all.
    since = int(first["tick"])
    rowsGot: list = []
    saidTo = 0
    for _ in range(args.times):
        _say(at, word)
        saidTo += 1
        till = time.time() + args.every
        while time.time() < till:
            got = _get(at, "/ticks?from=%d" % since)
            new = got.get("ticks", []) if isinstance(got, dict) else got
            if new:
                rowsGot.extend(new)
                since = int(got.get("next", since)) if isinstance(got, dict) else since
            time.sleep(0.5)
    heard: list = []
    said: list = []
    for n, one in enumerate(rowsGot):
        life = one.get("life", one)
        i = (life.get("input", {}) or {}).get("sound") or {}
        o = (life.get("output", {}) or {}).get("sound") or {}
        if float(i.get("lvl", 0) or 0) > 0.0:
            heard.append((n, int(i.get("id", 0) or 0)))
        if float(o.get("lvl", 0) or 0) > 0.0:
            said.append((n, int(o.get("id", 0) or 0)))

    print("said to her            %d times" % saidTo)
    print("ids that reached her   %d   %s"
          % (len(heard), collections.Counter(i for _, i in heard).most_common(6)))
    print("ids her mouth made     %d   %s"
          % (len(said), collections.Counter(i for _, i in said).most_common(6)))

    everHeard = {i for _, i in heard}
    order = [i for _, i in said]
    half = len(said) // 2

    def hits(rows2):
        return sum(1 for _, i in rows2 if i in everHeard), len(rows2)

    a, an = hits(said[:half])
    z, zn = hits(said[half:])
    print("\n  of the sounds she MADE, how many were sounds she has HEARD")
    print("    first half    %3d of %3d%s"
          % (a, an, "   %5.1f%%" % (a / an * 100) if an else ""))
    print("    second half   %3d of %3d%s"
          % (z, zn, "   %5.1f%%" % (z / zn * 100) if zn else ""))
    print("    distinct ones she managed   %d of %d she has heard"
          % (len(everHeard & set(order)), len(everHeard)))

    # ...AND THE WORD ITSELF, WHICH IS A CHAIN AND NOT A SOUND ---
    # **AGAINST WHAT CHANCE ALONE WOULD GIVE.**
    #
    # A COUNT IS NOT AN ANSWER.  On 2026-08-26 she made the word 3 times and
    # I was one sentence from reporting that she had said it.  Shuffling her
    # OWN sounds into a random order gives it 1.71 times on average and 5 at
    # best --- 42 of 200 shuffles did as well or better, p = 0.210.  She had
    # not said it; she had made enough sounds that the pair turned up.
    #
    # The shuffle keeps exactly what she made and destroys only the ORDER, so
    # it is the null this test needs and not a model of anything.
    got = sum(1 for k in range(len(order) - 1)
              if order[k] == ids[0] and order[k + 1] == ids[1])
    rng = random.Random(0)
    chance = []
    for _ in range(200):
        mix = list(order)
        rng.shuffle(mix)
        chance.append(sum(1 for k in range(len(mix) - 1)
                          if mix[k] == ids[0] and mix[k + 1] == ids[1]))
    mean = sum(chance) / len(chance)
    over = sum(1 for x in chance if x >= got)
    print(f"\n  the WORD --- {ids[0]} then {ids[1]}, in that order:   {got} times")
    print(f"    the same sounds SHUFFLED     mean {mean:.2f}, best {max(chance)}")
    print(f"    shuffles that did as well    {over} of 200   p = {over / 200:.3f}")
    print("    ...she said it only if p is small.  A count on its own is not")
    print("       an answer: enough sounds and the pair turns up by itself.")
    # ...and what her MIND made of it, read off her own published counters
    # (the in-process brain that used to answer this was retired 2026-08-26)
    last = _get(at, "/state")
    print("  experiences opened / closed           %s / %s"
          % (last.get("runs"), last.get("closes")))
    print("  connections she formed                %s   (guided tries %s)"
          % (last.get("connections"), last.get("guided")))
    print("\n  A sound she has heard, coming out of her mouth, is her ALPHABET")
    print("  reaching him.  The WORD is a chain, and needs the order kept.")
    # SHE SAID IT ONLY IF CHANCE CANNOT EXPLAIN IT.
    return 0 if (got and over <= 10) else 1


if __name__ == "__main__":
    raise SystemExit(main())
