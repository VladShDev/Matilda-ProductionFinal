"""WHERE HER LOOK ACTUALLY GOES --- timed on a WARM eye, not a fresh one.

    python -m measure.clock [looks]

**THIS EXISTS BECAUSE I MEASURED A LOOK ON A BODY THAT WAS NOT ALIVE.**  On
2026-08-25 the three stages were timed in a standalone probe --- a fresh
`Seen`, a static room, a few dozen pieces --- and came out


which is where "spread it one stage a tick and no tick is ever over budget"
came from.  Timed inside her real life, holding the 300-600 pieces she
actually accumulates, every single stage was over her 33 ms tick and a look
was twice what had been reported.

**A PROBE THAT STARTS A BODY AND IMMEDIATELY TIMES IT IS MEASURING A BODY WITH
NO MEMORY IN IT.**  Anything that costs more the more she holds --- `parts
be timed after she has been alive for a while.

AND IT HAPPENED AGAIN, 2026-09-01, to me: I timed these three with a fresh
`Seen`, got 35.7 ms for a whole look, and told him her eye was only using 18%
of its budget and the renderer was not worth replacing.  That number was about
a girl with no memory.  This file is the reason the mistake is catchable, so it
now WARMS her eye first and prints both, side by side.

**IT DIED WITH THE BRAIN IT TIMED.**  It wrapped `reason.getReason` and
`reason.levelsOf` --- the IN-PROCESS brain retired 2026-08-26 --- and raised
AttributeError before timing anything from that day.  Those two wrappers are
gone; the three stages that are still hers are timed where they run.

It needs no life and no mind: a look is a pure function of her room, her window
and where she is standing.  Do not run it while she is living on the same card.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import light, parts                              # noqa: E402
from body.hearing import LOOK_SECONDS, TICK_SECONDS         # noqa: E402
from body.room import Room                                  # noqa: E402
from body.window import Window                              # noqa: E402

#: how many looks her eye is given before it is timed.  Her real life holds
#: 300-600 pieces; this is what fills a `Seen` up the way living does.
WARM = 40
#: how many times each stage is timed, after one warm-up for the card
TIMES = 3


def timeit(f, n=TIMES) -> float:
    f()
    began = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - began) / n * 1000.0


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    warm = int(sys.argv[1]) if len(sys.argv) > 1 else WARM
    room, win = Room(), Window()
    # where she actually lies and the way she actually faces --- measured off
    # her own frames 2026-09-01, not a guess
    eye = np.array([0.2, 0.42, -0.34], np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    right = np.array([1.0, 0.0, 0.0], np.float32)
    fwd = np.array([-0.75, -0.56, 0.35], np.float32)
    fwd = fwd / float(np.linalg.norm(fwd))

    def look():
        return light.see(room, [win], [(eye, up, right, fwd)])

    ms1 = timeit(look)
    pic = np.asarray(light._home(look()))[:, :, 0].reshape(
        light.RETINA_H, light.RETINA_W, light.CONES)
    ms2 = timeit(lambda: parts.find(pic))
    rows = np.asarray(parts.find(pic))

    budget = LOOK_SECONDS * 1000.0
    tick = TICK_SECONDS * 1000.0
    print("ONE LOOK --- %dx%d, %d cones = %d rays"
          % (light.RETINA_H, light.RETINA_W, light.CONES,
             light.RETINA_H * light.RETINA_W))
    print("  light.see   ray-trace her room     %6.1f ms" % ms1)
    print("  parts.find  find the pieces        %6.1f ms   (%d pieces)"
          % (ms2, len(rows)))
    print("  ---------------------------------------------")
    whole = ms1 + ms2
    print("  a whole look                       %6.1f ms" % whole)
    print("  her budget                         %6.1f ms   (a look every "
          "%.0f ms, one stage a tick of %.0f ms)" % (budget, budget, tick))
    print("  she uses %.0f%% of it" % (100.0 * whole / budget))
    slow = [n for n, ms in (("light.see", ms1), ("parts.find", ms2))
            if ms > tick]
    if slow:
        print("\n  OVER ONE TICK: %s --- a stage that does not fit in %.0f ms "
              "makes her clock sag, because a look is spread one stage a tick."
              % (", ".join(slow), tick))
        return 1
    print("\n  EVERY STAGE FITS IN A TICK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
