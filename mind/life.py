"""HER MIND --- his architecture, confirmed 2026-09-11, and nothing else.

    *"Yes.  I am finally happy to hear really good understanding, and we are
    on the same page."*

THE ROW
    One row of levels every tick.  What arrives and what she sends, side by
    side.  Nothing is named to her: the fixed slots are the app's bookkeeping
    so her body fills the same place each tick, and she only ever sees a
    number in a position.  A thing seen is what it is and where --- similarity,
    x, y.  A sound is the same three: how alike, how loud, which side.  Never
    one line per id.  Her hormones are lines like any other, and so are her
    own orders.

AN EXPERIENCE
    TWO STATES: its start whole, and its end kept as only the lines that
    differ.  Nothing between them is stored --- a tick where zero became zero
    carries no information.  The next begins where the last ended, so it holds
    the last inside itself; the nesting is not a structure that gets built, it
    falls out of starting where you stopped.  Novelty makes it one: felt first
    time, it is an experience.  A step that brings nothing makes no experience
    --- she simply keeps discovering until something changes.  Nothing is
    thrown away and nothing is forgotten; what is old is merely many turns
    away.

WHAT MOVES HER
    ONE VALUE: her state above the floor she has learned she can feel.  His,
    2026-09-11: *"we spelled just one value ... dopamine boost improves her
    state to close experience, and that is her state when it is increasing,
    and the difference became to this state floor, and we already had it in
    our application."*  Value is not computed and not stored: her hormone
    lines are already in the row, so what an experience did to her is already
    in the record.  Effort is how far a level moves and how often; her
    hormones set her speed and her effort, never her decisions.

RECREATING ONE
    She picks the experience with the BEST PROFIT --- the state she felt the
    moment it switched.  His word, 2026-09-12: *"we just pick experience with
    best profit, best state that we feel that moment when experience was
    switched ... so we don't need any closeness, any song rules."*  Retrying it
    is ONE STEP: she posts the stored change, from where she is now, and her
    flesh walks the way.  Novelty is why the second time pays nothing, and
    boredom is what then moves her on.  If a change came from outside and
    she has never made that output herself, she does not have that piece ---
    it is a level she has seen and cannot produce, until discovery gives her
    the output that makes it, and then it is a piece like any other.

HER VOICE
    Her ears are the spindle for her mouth.  There is no echo: her own air
    arrives at her ears with everyone else's, and what makes it hers is her
    own `loud` muscle sitting in the same row.  Her voice is muscles and she
    practises it exactly like her body, one muscle to its end and then the
    next --- and the shape is a chain, not a held pose.  No prepared table, no
    naming a sound, no aiming, and no ruler.

The body owns time, physics and senses; this reads finished ticks over HTTP
and posts decisions (`sandbox.py`).  The record is DuckDB (`store.py`), the
shapes are `structure.py`, her chemistry is `hormones.py`, her ladder is
`discover.py`.
"""

from __future__ import annotations

import random
import time as clock
from dataclasses import replace

from discover import discover
from hormones import DULL, HORMONE_IDS, Hormones
from sandbox import TICK_SECONDS, act, felt, getAble, getTicks
from store import Store
from structure import Exp, ExpRef, Memory

#: how often her hands are written to the record (~5 min of her own seconds)
HANDS_EVERY = int(round(300.0 / TICK_SECONDS))

#: the three lines a thing seen has, and the three a sound has.  Tags, not
#: names: her mind never branches on which of them a level came from.
VIEW = "view"
HEARD = "heard"


def lines(life) -> dict:
    """ONE ROW OF LEVELS --- everything she is, this tick.

    What arrives and what she sends, together: her spindles carry what her
    flesh did, her motors carry what she ordered, and the tick after, one
    answers the other.  A thing seen is its similarity and where it is; a
    sound is how alike it was, how loud, and which side it came from.
    """
    i = life.input
    out = {("sensor", int(s.id)): float(s.lvl) for s in i.sensor}
    for s in i.spindle:
        out[("spindle", int(s.id))] = float(s.lvl)
    # A SOUND IS ITS LEVEL, HOW LOUD, AND WHICH SIDE.
    out[(HEARD, 0)] = float(getattr(i.sound, "similarity", 0.0) or 0.0)
    out[(HEARD, 1)] = float(i.sound.lvl)
    out[(HEARD, 2)] = float(getattr(i.sound, "balance", 0) or 0)
    v = i.view[0] if i.view else None
    out[(VIEW, 0)] = float(v.similarity) if v is not None else 0.0
    out[(VIEW, 1)] = float(getattr(v, "x", 0.0)) if v is not None else 0.0
    out[(VIEW, 2)] = float(getattr(v, "y", 0.0)) if v is not None else 0.0
    for m in life.output.motor:
        out[("motor", int(m.id))] = float(m.lvl)
    return out


def _hormones(levels: dict) -> dict:
    return {k: v for k, v in levels.items()
            if k[0] == "sensor" and k[1] in HORMONE_IDS}


def _key(k) -> str:
    return "%s:%d" % k


def _unkey(s: str):
    kind, sid = s.rsplit(":", 1)
    return (kind, int(sid))


class Life:
    """The loop.  She is always inside an experience; the first is id 1."""

    def __init__(self, tape: str, at: str = "http://127.0.0.1:8090") -> None:
        self.at = at
        self.hormones = Hormones()
        self.able = getAble(at)
        # her body says what it is made of; nothing here counts her parts by
        # hand.  `voiced` is how many of her motors are her mouth, so her chest
        # knows which muscle is the one that blows her air away.
        self.hormones.voiced = int(self.able.get("voiced", 0) or 0)
        self.hormones.motors = int(self.able.get("motors", 1))
        self.res = float(self.able["resolution"])
        try:
            self.store = Store(tape, resolution=self.res)
        except Exception:                                      # noqa: BLE001
            self.store = Store(tape, resume=True, resolution=self.res)

        # WHAT EVERY LINE HAS SHOWN HER --- her habituation, and her sense of
        # what a jump on a line usually is.  Only ever added to.
        self.seen: dict = {}            # line -> the buckets it has shown
        self.usual: dict = {}           # line -> its usual jump

        # HER MEMORY.  An experience is two states: its start whole, and at its
        # end THE CHANGE on each line that moved --- 0.1 to 0.5 is kept as 0.4,
        # and a line that did not move is not written at all.  `parts` is the
        # one it began from, so each holds the last inside itself.  Nothing is
        # capped and nothing is dropped --- a state and a difference weigh
        # almost nothing, and what is far off is far because it is far.
        self.starts: dict = {}          # exp id -> its first state, whole
        self.ends: dict = {}            # exp id -> the change on each line that moved
        self.parts: dict = {}           # exp id -> the experience it began from
        self.profit: dict = {}          # exp id -> the state she felt when it switched
        self.picked: int | None = None  # the one she last put back, judged at her next choice
        self.sank: float = 0.0          # ...and the lowest her state fell while she performed it
        self.lastClosed: int = 0
        self.memory = Memory()

        # THE OPEN RUN.  Only the tick she is in is held; the run's own ticks
        # are not kept, because an experience is its two ends.
        self.living: list = []
        self.livingN: int = 0
        self.opening: dict = {}
        self.lift: float = 0.0          # her state above her floor --- the one value
        self.lowest: float = float("inf")   # the run's lowest of it...
        self.highest: float = 0.0           # ...and its highest
        self.paid: float = 0.0          # the rise of it over the run
        self.prev: dict | None = None
        self.was = None                 # her previous output, for the ladder

        # HER SWEEP --- one output at a time, end to end, at her deficit.  The
        # rung is carried here because her body fades it away between trials.
        self.tryLine, self.tryWay, self.trySaid, self.tryAt = 1, 1, 0, 0.0

        # WHAT HAPPENED, COUNTED (the page and the sheet read these)
        self.openId = 1
        self.age = 0
        self.closes = 0
        self.replays = 0
        self.guided = 0                 # chosen from memory
        self.laddered = 0               # trials
        self.closedHow: dict = {}
        self.news = 0
        self.cursor = 0
        self.passes = 0
        self.feelMs = None
        self._remember()

    # ------------------------------------------------------------ the wake
    def _remember(self) -> None:
        """A life continued: what every line has shown her, and the two states
        of every experience she has closed.  Nothing is left behind --- a start
        and a difference are what a memory is, and both are kept."""
        # `hands()` is None on a life from before checkpoints, else the tuple
        # (age, time, packed) --- the dict is its third element.  Reading `.get`
        # straight off the tuple crashed every `--keep` wake of a life that had
        # a checkpoint (a fresh birth got None and was fine, so it hid).
        got = self.store.hands()
        packed = (got[2] if got else {}).get("his") or {}
        self.seen = {_unkey(k): set(v) for k, v in (packed.get("seen") or {}).items()}
        self.usual = {_unkey(k): float(v) for k, v in (packed.get("usual") or {}).items()}
        for k, lv in (packed.get("starts") or {}).items():
            self.starts[int(k)] = {_unkey(a): float(b) for a, b in lv.items()}
        for k, lv in (packed.get("ends") or {}).items():
            self.ends[int(k)] = {_unkey(a): float(b) for a, b in lv.items()}
        for k, v in (packed.get("parts") or {}).items():
            self.parts[int(k)] = int(v)
        for k, v in (packed.get("profit") or {}).items():
            self.profit[int(k)] = float(v)
        closed = self.store.exps() or []
        if closed:
            self.lastClosed = int(max(int(e.id) for e in closed))
            self.openId = self.lastClosed + 1
        self.age = int(packed.get("age", 0) or 0)

    def _hands(self) -> dict:
        """Her memory, whole, for the record.  Every experience she holds ---
        no cap, so a life continued wakes with everything it lived."""
        return {"his": {
            "age": int(self.age),
            "seen": {_key(k): sorted(v) for k, v in self.seen.items()},
            "usual": {_key(k): float(v) for k, v in self.usual.items()},
            "parts": {str(k): int(v) for k, v in self.parts.items()},
            "profit": {str(k): float(v) for k, v in self.profit.items()},
            "starts": {str(k): {_key(a): float(b) for a, b in lv.items()}
                       for k, lv in self.starts.items()},
            "ends": {str(k): {_key(a): float(b) for a, b in lv.items()}
                     for k, lv in self.ends.items()}}}

    # ------------------------------------------------------------ the loop
    def once(self):
        """One pass: every tick she lived since the last, felt in order.  A
        refused connection is a missed beat, never death."""
        try:
            got, self.cursor = getTicks(self.cursor, self.at)
        except OSError:
            return None
        t = None
        for t in got:
            t9 = clock.perf_counter()
            self.feel(t)
            d = (clock.perf_counter() - t9) * 1000.0
            self.feelMs = d if self.feelMs is None else self.feelMs + (d - self.feelMs) / 300.0
        self.passes += 1
        if t is not None and self.passes % 15 == 0:
            try:
                felt({"state": round(float(t.life.input.state), 4),
                      "blood": {str(k): round(float(v), 3)
                                for k, v in self.hormones.level.items()},
                      "hunger": round(float(self.hormones.level.get(7, 0.0)), 4),
                      "floor": round(float(self.hormones.floor()), 4),
                      "lift": round(float(self.lift), 4),
                      "runs": len(self.starts), "exp": self.openId,
                      "feelMs": round(self.feelMs, 2) if self.feelMs is not None else None,
                      "recognised": self.guided, "guided": self.guided,
                      "laddered": self.laddered, "closes": self.closes,
                      "closedHow": dict(self.closedHow),
                      "news": self.news,
                      "recorded": int(self.store.recorded),
                      "chains": len(self.starts),
                      "pending": len(self.store._ticks),
                      "replays": self.replays,
                      "seconds": round(self.age * TICK_SECONDS, 1)}, self.at)
            except OSError:
                pass
        return t

    # ------------------------------------------------------------- one tick
    def feel(self, t) -> None:
        """One tick: every line read, the new found, her chemistry lived, and
        --- when the one value rises and comes back down --- a close and a
        choice."""
        self.age += 1
        now = lines(t.life)

        # EVERY LINE: is this value one it has never shown?  A value a line has
        # never shown bursts by the size of the change, and from the next try
        # it is usual --- that is her habituation, and nothing else is needed
        # for it.  ONE BURST A TICK, the biggest thing that changed for her:
        # summed over lines instead, fifty lines going new at once sat the
        # burst at its ceiling through a whole guide and cut on the pauses.
        burst = 0.0
        closing = None
        for k, v in now.items():
            burst, closing = self._line(k, v, burst, closing)

        self.hormones.live(t, burst, 0.0)
        # WHAT THE ONE SHE PUT BACK IS BRINGING HER --- her state while she
        # performs it, read every tick until her next choice.  A close is a
        # novelty peak, so judging there would miss that she sank first.
        if self.picked is not None:
            self.sank = min(self.sank, float(t.life.input.state))

        # THE ONE VALUE: her state above the floor she has learned she can
        # feel.  Her dopamine still pays into it, with every other line,
        # through `corrections()` --- it is simply not the thing watched.  A
        # floor-relative measure has no ceiling: the floor comes down with her,
        # so a rise is a rise wherever she is, even at the worst of her life.
        lv = float(t.life.input.state) - float(self.hormones.floor())
        self.lift = lv
        if not self.opening:
            self.lowest = self.highest = lv
        self.lowest = min(self.lowest, lv)
        self.highest = max(self.highest, lv)
        self.paid = max(self.paid, lv - self.lowest)

        # HER HORMONES ARE LINES LIKE ANY OTHER, asked the same question ---
        # they are written onto the tick by her chemistry, so they are read
        # now.  Their own new is the burst just felt, so of them only the cut
        # is taken: a line paid for its own rise would climb by itself.
        felt9 = _hormones(lines(t.life))
        for k, v in felt9.items():
            _, closing = self._line(k, v, 0.0, closing)
        now.update(felt9)

        if not self.opening:
            self.opening = dict(now)

        self.living = [t.life]
        self.livingN += 1

        # THE CUT: the one value rose by more than her resolution and has come
        # back down by her resolution.  The run closes at the peak, so the
        # whole rise belongs to the run that earned it.
        if closing is not None and self.livingN > 1:
            self._close(t, now)
            self._choose(t, now, float(t.life.input.state))

        # BOREDOM PUSHES HER TO A TRIAL WHILE SHE DOES NOTHING.  Its level is
        # her chance of a trial, as a rate per second through her clock: at its
        # cap a trial every few seconds of stillness, and any real change
        # empties it.  No number of mine, no switch.
        dull = float(self.hormones.level.get(DULL, 0.0))
        if random.Random(int(self.age)).random() < dull * TICK_SECONDS:
            self._try(t, float(t.life.input.state))

        self.store.tick(t)
        self.prev = now
        self.was = t.life.output
        if self.age % HANDS_EVERY == 0:
            self.store.keepHands(self.age, float(t.time), self._hands())

    # --------------------------------------------------------------- a line
    def _line(self, k, v: float, burst: float, closing):
        """ONE LINE, THIS TICK: is its value one it has never shown, how big
        was the change, and has the one value peaked."""
        prev = self.prev or {}
        b = int(round(v / self.res))
        shown = self.seen.setdefault(k, set())
        if b not in shown:
            shown.add(b)
            self.news += 1
            burst = max(burst, abs(v - float(prev.get(k, 0.0))))
        if k not in prev:
            return burst, closing
        d = abs(v - prev[k])
        # THE CUT IS THE ONE VALUE, not a line with a name on it: the rise over
        # the run's low and the fall from its peak, each by her own resolution,
        # so both edges are ones she can feel.
        if self.paid > self.res and self.highest - self.lift > self.res:
            closing = "state"
        u = self.usual.get(k, 0.0)
        self.usual[k] = u + (TICK_SECONDS / 3.0) * (d - u)
        return burst, closing

    # ------------------------------------------------------------- the close
    def _close(self, t, now: dict) -> None:
        """TWO STATES.  Its start, whole; its end, the change on each line that
        moved.  Nothing between them is kept.  And the next begins here."""
        done = Exp(id=self.openId)
        # the row she ENDED in --- what a thing LED TO, which is what this row
        # has been documented as since it was written
        last = self.living[-1].input
        done.sensor = [replace(x) for x in last.sensor]
        done.spindle = [replace(x) for x in last.spindle]
        done.echo = replace(last.echo)
        done.sound = replace(last.sound)
        done.view = [replace(x) for x in last.view]
        done.exp = ExpRef(id=int(self.lastClosed))
        self.store.exp(done)

        start = dict(self.opening)
        self.starts[int(done.id)] = start
        # THE CHANGE, NOT THE VALUE.  His word, 2026-09-12: *"output ID one was
        # 0.1, next step it became 0.5, so the step saves 0.4.  The rest stay
        # zero, and we store nothing there."*  A line that did not move by her
        # grain is not written at all.
        self.ends[int(done.id)] = {
            k: v - start.get(k, 0.0) for k, v in now.items()
            if k not in start or abs(v - start[k]) > self.res}
        # HER PROFIT: the state she felt the moment it switched --- what she
        # picks by, and nothing else.
        self.profit[int(done.id)] = float(t.life.input.state)
        self.parts[int(done.id)] = int(self.lastClosed)
        self.lastClosed = int(done.id)

        self.closes += 1
        self.closedHow["state"] = self.closedHow.get("state", 0) + 1
        # No short and long memory.  His word, 2026-09-12: *"our new structure
        # even doesn't need it ... the tree keeps her from unnecessary turns."*
        # The `memory` row was a list of every closed id, rewritten whole at
        # every close and read by nothing that decides.

        # the id grows before she decides, and the next run opens in the row
        # this one closed in --- so it holds this one inside itself
        self.openId = int(done.id) + 1
        t.exp.id = self.openId
        self.living = [t.life]
        self.livingN = 1
        self.opening = dict(now)
        self.paid = 0.0
        self.lowest = self.highest = self.lift

    # ------------------------------------------------------------ the choice
    def _choose(self, t, now: dict, s: float) -> None:
        """The experience with the best profit, put back in one step --- or a trial."""
        # WHAT SHE PUT BACK LAST TIME IS JUDGED BY WHAT IT BROUGHT.  His word,
        # 2026-09-12: *"the state of this experience would be fallen each time
        # she didn't get expected profit from that experience.  That's it."*
        # So its profit becomes what she actually feels now, when that is less
        # --- a wrong experience sinks, and she does not inspect it again.  No
        # short and long memory, no 26 and 26: the tree keeps her from the
        # wrong turns by itself.
        if self.picked is not None:
            got = min(float(s), float(self.sank))
            if got < self.profit.get(int(self.picked), 0.0):
                self.profit[int(self.picked)] = got
        self.picked = None
        cands = [i for i in self.starts if i != self.openId]
        # BOREDOM IS HOW MUCH SHE EXPLORES: bored, she tries things; not, she
        # repeats what paid best.  A level, seeded by her age, no switch.
        dull = float(self.hormones.level.get(DULL, 0.0))
        if cands and random.Random(int(self.age)).random() >= dull:
            # BY PROFIT, AND NOTHING ELSE.  His word, 2026-09-12: *"we just pick
            # experience with best profit ... so we don't need any closeness,
            # any song rules."*  The first time a line moves she is paid once;
            # retrying it pays nothing (novelty), and boredom moves her on ---
            # so she tries everything herself and builds bigger experiences
            # out of the smallest, keeping the ones that paid.
            best = max(cands, key=lambda i: (self.profit.get(i, 0.0), i))
            # RETRYING IS ONE STEP: her flesh caps force and not destination,
            # so ordering the CHANGE she made, from where she is now, IS doing
            # it again --- the physics walks the way there.
            end = self.ends.get(int(best)) or {}
            plan = {"motor": [{"id": int(k[1]),
                               "lvl": max(0.0, min(1.0, float(now.get(k, 0.0)) + float(v)))}
                              for k, v in end.items() if k[0] == "motor"]}
            # CHOSEN IS CHOSEN, whether or not there was anything of hers to
            # post.  An experience where the world moved and she did not ---
            # her mother's voice, the room --- can sit at the top with nothing
            # to replay (measured 2026-09-12: profit 0.992, 21 lines, no motor;
            # picked at 703 of 728 closes, every one a fall-through).  She
            # chose it and it brought her nothing, so it sinks by the same rule
            # as any other, and the next one gets its turn.
            self.picked = int(best)
            self.sank = float(s)
            if plan["motor"]:
                self.replays += 1
                self.guided += 1
                try:
                    act(plan, self.at, tick=int(self.age))
                except OSError:
                    pass
                return
        self._try(t, s)

    def _try(self, t, s: float) -> None:
        """One output, one rung further along it, at her effort.  Her deficit
        is the step and never reaches zero --- a half of her at neutral, all of
        her at her worst, which is his addiction law of 2026-09-02."""
        want = (1.0 - float(s)) / 2.0
        try:
            (got, self.tryLine, self.tryWay, self.trySaid,
             self.tryAt) = discover(
                self.was if self.was is not None else t.life.output,
                self.able, want, self.age, self.tryLine, self.tryWay,
                self.trySaid, self.tryAt)
            act(got, self.at)
        except OSError:
            pass
        self.laddered += 1

    # --------------------------------------------------------------- living
    def run(self, seconds: float | None = None) -> None:
        """Live.  Nothing is passed in and nothing comes back; stopping is the
        caller's business.  The watchdog stays: a mind that stalls says so."""
        import faulthandler
        import sys as _sys
        import threading
        began = clock.monotonic()

        def _watch():
            last = -1
            while True:
                clock.sleep(20.0)
                if self.age == last:
                    print("MIND STALLED: no tick felt for 20 s at age %d --- stacks:"
                          % self.age, file=_sys.stderr, flush=True)
                    faulthandler.dump_traceback(file=_sys.stderr, all_threads=True)
                waiting = len(self.store._ticks)
                if waiting > 1800:
                    print("KEEPER STALLED: %d ticks unwritten at age %d, recorded %d"
                          % (waiting, self.age, self.store.recorded),
                          file=_sys.stderr, flush=True)
                last = self.age
        threading.Thread(target=_watch, daemon=True).start()
        while seconds is None or clock.monotonic() - began < seconds:
            self.once()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    at = "http://127.0.0.1:8090"
    if "--at" in args:
        k = args.index("--at")
        at = args[k + 1]
        args = args[:k] + args[k + 2:]
    Life(args[0] if args else "lives/her.duckdb", at).run(
        float(args[1]) if len(args) > 1 else None)
