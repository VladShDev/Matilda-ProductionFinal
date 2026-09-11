"""HER LAST FEW SECONDS, IN HAND --- the window `/ticks` serves.

ONE RECORD OF HER LIFE, AND IT IS HER MIND'S.  Her body used to pickle every
tick it lived into `lives/*.tape` (699 MB on 2026-09-02 alone) and nothing
ever read a byte of it back: `/ticks` serves this RAM window, her mind keeps
every tick and every experience in its DuckDB (his four tables), and the
pictures for the page are `sandbox/film.py`.  Two records of one life is rule
4 broken.  His word, 2026-09-02, when asked whether the body's tape could go:
*"you mean we alredy store everithing we need in exp?"* --- yes.  This is the
half that stayed.

THE COUNT CONTINUES ACROSS RESTARTS (`lives/clock`), because the room's clock
outlives any one body: her mother's slots, the belt, the card on the board and
her own pacing all count in it.  A fresh count told a woken body it was
thousands of ticks ahead of schedule and it slept 333 s before its second
tick (see `Her.live`).

WHAT IS KEPT is what her mind can still ask for: `keep` ticks, oldest dropped
first.  Her mind reads every tick since its cursor, so a mind more than `keep`
ticks behind loses time --- `/ticks` answers from the oldest it has and says
so in `from`.
"""
from __future__ import annotations

import io
import os


class Ticks:
    """The newest `keep` finished ticks, numbered from the room's clock."""

    def __init__(self, keep: int = 1024, clock: str | None = None) -> None:
        self.keep = max(64, int(keep))
        self._ram: dict = {}
        self._clock = clock
        start = 0
        if clock and os.path.exists(clock):
            try:
                start = int(io.open(clock).read().strip() or 0)
            except (OSError, ValueError):
                start = 0
        self._n = start          # the next tick's number --- also len()
        self._oldest = start     # the oldest number still in hand

    def append(self, tick) -> None:
        """One finished tick.  Nothing touches it once the next has begun."""
        k = self._n
        self._ram[k] = tick
        self._n += 1
        while self._n - self._oldest > self.keep:
            self._ram.pop(self._oldest, None)
            self._oldest += 1
        if self._clock and k % 30 == 0:
            self._mark()

    def flush(self) -> None:
        """The clock, written --- so the next body continues the count."""
        self._mark()

    def _mark(self) -> None:
        if not self._clock:
            return
        try:
            io.open(self._clock, "w").write(str(self._n))
        except OSError:
            pass

    def __len__(self) -> int:
        return self._n
