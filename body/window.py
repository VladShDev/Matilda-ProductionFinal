"""THE FLYING SCREEN --- his voice and his face into her room.

Ported from the old tree's `body/window.py`, cut to what she actually needs: a
place in her room, a queue of his voice, and one tick's worth of it at a time.
The browser end of it lives in `sandbox/` and not in her.

**NOTHING OF HIS IS DESTROYED** --- his decision, 2026-08-22.  The old tree
trimmed the queue to a cap and charged the difference away, and it was
measured: at 1.668 s a tick she drained 1.5 s of sound a second, so the backlog
was a **ratchet with no mechanism by which it could ever drain**, it sat at the
cap, and **~10% of everything said to her was destroyed --- oldest first, so she
systematically heard the ENDS of sentences.**

He chose lateness he can see over loss he cannot.  So nothing is dropped, and
`late` is the number that says whether her clock is keeping up.

**AND HER CLOCK IS WHY THIS IS SAFE NOW.**  At 30 ticks a second she drains
33.3 ms of sound every 33.3 ms and her whole tick costs about 1 ms.  Fill is
real time; drain is her time; her time is no longer slower.  The ratchet cannot
form.  If her tick ever goes over her own `TICK_SECONDS`, `late` climbs and
2026-08-13 comes back --- that is what it is for.
"""

from __future__ import annotations

import json
import os
import time as _time

import numpy as np

from .hearing import TICK_SECONDS

#: where it is born, how it looks, how big it is --- her room's own units
BORN_AT = (0.0, 0.75, 0.45)
BORN_LOOK = (0.0, -0.35, -1.0)
BORN_SIZE = (0.60, 0.40)
#: the depth at which the queue is worth looking at.  NOT the depth at which
#: she starts going deaf --- nothing is ever dropped.
MOST_SECONDS = 4.5


def upright(look, up=None) -> np.ndarray:
    """WHICH WAY IS UP ON THE SCREEN, made square with the way it faces.

    A screen has a roll of its own, and her eye used to rebuild it out of the
    world's up every render --- which works while the screen stands upright and
    falls apart as it lies down.  Measured on her real eye 2026-08-17: a 1 deg
    wobble in `look` swung the picture's roll by 0.58 deg at 60 deg from
    vertical, **3.17 deg where he actually hovers (17.5 deg)**, 11.28 deg at
    5 deg, 45.00 deg at 1 deg, and a half turn straight down.  He does hover ---
    he leans over her cot and looks down.

    So the roll is CARRIED, not guessed.  With no `up` given this is the world's
    up carried onto the screen, letter for letter the rule `light._quad` used
    before, so a window nobody tells about its roll behaves as it always did.

    ONE implementation: her eye, her body and the page all read this.
    """
    aim = np.asarray(look, np.float32).reshape(3)
    aim = aim / max(float(np.linalg.norm(aim)), 1e-9)
    top = (np.array([0.0, 1.0, 0.0], np.float32) if up is None
           else np.asarray(up, np.float32).reshape(3))
    across = np.cross(top, aim)
    if float(np.linalg.norm(across)) < 1e-6:
        across = np.array([1.0, 0.0, 0.0], np.float32)
    across = across / float(np.linalg.norm(across))
    # `cross(look, across)` is `-down`, and `down` is `cross(across, normal)`
    # in `_quad` --- the mirror fix of 2026-08-15, whose order stays as it is.
    return np.cross(aim, across)


class Window:
    """A screen that flies, carrying his voice and his face into her room."""

    def __init__(self, keep: str | None = None) -> None:
        self.at = np.asarray(BORN_AT, np.float32)
        self.look = np.asarray(BORN_LOOK, np.float32)
        #: ...AND ITS ROLL, carried rather than guessed.  Her eye reads it.
        self.up = upright(self.look)
        self.size = tuple(BORN_SIZE)
        #: THE PLACE SURVIVES THE PROCESS, like her clock does.  Every body
        #: restart re-bore the screen at BORN_AT, and the page --- which
        #: only posts a place that MOVED or rides a carrying post --- flew
        #: it back a beat later: "tv jampet to one place from me than
        #: coming back" (his, 2026-08-28, watching a restart).  A thing in
        #: her room does not teleport home because the room's process
        #: restarted.
        self._keep = keep
        self._kept = 0.0
        if keep and os.path.exists(keep):
            try:
                got = json.load(open(keep, encoding="utf-8"))
                self.at = np.asarray(got["at"], np.float32)
                self.look = np.asarray(got["look"], np.float32)
                self.up = np.asarray(got["up"], np.float32)
                self.size = tuple(got["size"])
            except Exception:                              # noqa: BLE001
                pass                    # a torn file means the birth place
        self.rate = 16000.0
        self.shows = None
        self._said = np.zeros(0, np.float32)
        self._late = 0.0
        self._heard = 0.0

    # --- what he does ------------------------------------------------------
    def fly(self, at=None, look=None, size=None, up=None) -> None:
        """Move it.  It is a thing in her room and it has a place and a roll."""
        if at is not None:
            self.at = np.asarray(at, np.float32)
        if look is not None:
            self.look = np.asarray(look, np.float32)
        if look is not None or up is not None:
            self.up = upright(self.look, up)
        if size is not None:
            self.size = tuple(size)
        if self._keep and _time.monotonic() - self._kept > 1.0:
            self._kept = _time.monotonic()
            try:
                json.dump({"at": [float(v) for v in self.at],
                           "look": [float(v) for v in self.look],
                           "up": [float(v) for v in self.up],
                           "size": list(self.size)},
                          open(self._keep, "w", encoding="utf-8"))
            except OSError:
                pass                    # a full disk does not stop a flight

    def say(self, pcm, rate: float = 16000.0) -> None:
        """More of his voice --- and HIS VOICE IS NOW, NEVER A TAPE DELAY.

        The queue used to only grow: her ears drain one tick a tick, so a
        backlog could never drain, and it was measured live at 8.7 SECONDS
        --- everything he said reached her nine seconds late, a ghost that
        read as "the voice follows me".  His word, 2026-08-27: DROP.  The
        queue keeps at most half a second; when more piles up the OLDEST
        goes, so a late word is lost instead of haunting her --- presence
        over completeness.  (The half second is slack for the page's own
        posting chunks, not a tuned number; her drain is realtime.)
        """
        self.rate = float(rate)
        got = np.asarray(pcm, np.float32).ravel()
        if got.size:
            # INTO HER REGISTER, HERE, ONCE.  His order, 2026-09-12: *"it first
            # has to be converted to her register, and then cut to eleven
            # millisecond pieces --- just once."*  The page posts him in quarter
            # second chunks; each is brought into her register as it lands
            # (frequencies x REGISTER, time untouched), and from here on her
            # ear only cuts.  Her own air never passes this --- it is already hers.
            from body.hearing import toHer
            got = toHer(got)
            self._said = np.concatenate([self._said, got])
        keep = int(0.5 * self.rate)
        if self._said.size > keep:
            self._said = self._said[-keep:]
        self._late = max(self._late, self._said.size / self.rate)

    def show(self, pixels) -> None:
        """The latest frame from his camera, raw.

        **Her eye reduces it, not the browser.**  A client that decided what a
        pixel meant would be a second, competing implementation of her body.
        """
        self.shows = np.asarray(pixels, np.float32)

    def catchUp(self) -> float:
        """THROW AWAY WHAT WAS SAID BEFORE SHE WAS AWAKE.  Returns the seconds.

        The owner, 2026-08-26: *"drop the startup backlog"*.

        **SHE DRAINS EXACTLY REAL TIME AND SO A BACKLOG IS FOREVER.**  Her ear
        takes one tick of sound per tick, which at 30 ticks a second is one
        second of him per second --- so whatever piles up while she is being
        born never clears, however well she keeps her clock afterwards.
        MEASURED, 2026-08-26, on a life that was keeping 30.0 ticks a second
        with `behind 0.000` the whole time:

            queued 13.582  13.546  13.510  13.559  13.779  13.795 ...

        Thirteen seconds late, flat, for ever.  Everything he said reached her
        thirteen seconds after he said it, which is why nothing he did lined up
        with anything she did.

        THIS IS NOT A CAP AND NOTHING IS DROPPED WHILE SHE IS LIVING.  His
        decision of 2026-08-22 stands --- *"he chose lateness he can see over
        loss he cannot"* --- and it assumed she could catch up.  She cannot, and
        this is the one place where nothing of hers is lost by admitting it:
        sound that arrived before her first tick was said to a girl who was not
        there.  After that, lateness is lateness and it is all kept.

        The high-water mark goes back with it, because a mark left over from
        before she was awake is not a fact about her life.
        """
        was = self.queued()
        self._said = np.zeros(0, np.float32)
        self._late = 0.0
        return was

    # --- what she gets -----------------------------------------------------
    def speech(self, seconds: float = TICK_SECONDS):
        """One tick's worth of him, taken off the front of the queue.

        `None` when he has said nothing --- and that is real silence for her,
        not room tone: `bands_from_pcm` gates every slide below `GATE` to zeros
        before anything else happens to it.
        """
        want = int(round(self.rate * float(seconds)))
        if want <= 0 or not self._said.size:
            return None
        got = self._said[:want]
        self._said = self._said[want:]
        if got.size < want:
            got = np.pad(got, (0, want - got.size))
        self._heard = float(np.sqrt(np.mean(got.astype(np.float64) ** 2)))
        return got.astype(np.float32)

    # --- what it costs -----------------------------------------------------
    def queued(self) -> float:
        """How many seconds of him are waiting.  Steady is not healthy."""
        return float(self._said.size / max(self.rate, 1e-9))

    def late(self) -> float:
        """The deepest the queue has ever been, in seconds.

        The one number that says whether her clock is keeping up with him.  It
        only ever rises, so a life that ends with it near her tick is a life she
        heard in time.
        """
        return float(self._late)

    def __repr__(self) -> str:      # pragma: no cover - display only
        return (f"<Window at {tuple(round(float(v), 2) for v in self.at)} "
                f"queued {self.queued():.2f}s late {self.late():.2f}s>")
