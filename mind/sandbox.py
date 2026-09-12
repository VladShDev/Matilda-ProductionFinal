
# ============================================================ #
# HIS RULE, HIGHLY IMPORTANT, MARKED EVERYWHERE HE ORDERED IT:  #
# "RUN HER ONLY ON GPU."  (2026-08-28, twice, the second time    #
# watching her wake grind a CPU core while she lay like a doll.) #
# What it means in this tree: NO BULK PYTHON PASSES, EVER.  A    #
# tick's work is a lookup or a set op (O(1), microseconds).      #
# Anything that touches MANY ticks --- the wake above all ---    #
# must be columnar (DuckDB's own vectorized engine) or on the    #
# GPU beside her body.  A python for-loop over her life is a     #
# violation of this rule BY DEFINITION, whatever it computes.    #
# The wake's columnar rewrite is the standing next stone.        #
# ============================================================ #
"""Her body, asked over HTTP --- data from the sandbox APP, never the browser.

His, 2026-08-26: *"http is created like watcher but self app wors on gpu"* ---
and *"sandbox is data in app not in browser"*.  The sandbox (matilda2
`sandbox/app.py`) runs her body on the GPU and serves finished numbers; this
side asks for them and computes none of them.
"""

from __future__ import annotations

import os as _os

import http.client
import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

#: ONE KEPT CONNECTION PER BODY --- profiled 2026-08-30: 73% of every
#: tick's cost was urlopen opening a fresh TCP connection (~10 ms) that
#: HTTP/1.0 then closed.  The body speaks HTTP/1.1 now; this line is
#: opened once and reused every tick.  A broken line reconnects once;
#: past that it raises OSError, exactly the missed-beat the callers
#: already forgive.
_LINES: dict = {}


def _line(at: str):
    key = at.rstrip("/")
    c = _LINES.get(key)
    if c is None:
        u = urlsplit(key)
        c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        _LINES[key] = c
    return c


def _ask(at: str, method: str, path: str, payload=None, raw: bytes | None = None) -> bytes:
    body = raw if raw is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
    for attempt in (0, 1):
        c = _line(at)
        try:
            c.request(method, path, body=body,
                      headers=({"Content-Type": "application/octet-stream"} if raw is not None
                               else {"Content-Type": "application/json"} if body else {}))
            r = c.getresponse()
            got = r.read()
            if r.status != 200:
                raise OSError(f"body said {r.status} for {path}")
            return got
        except (OSError, http.client.HTTPException):
            try:
                c.close()
            except Exception:
                pass
            _LINES.pop(at.rstrip("/"), None)
            if attempt:
                raise
    raise OSError("unreachable")

from structure import (Echo, ExpRef, Heard, Input, Life, Motor, Output,
                       OutSound, Sensor, Spindle, Tick, View)

#: Her clock.  The sandbox lives at 30 ticks a second (`/meta` `tick_seconds`);
#: every per-second rate is multiplied by this once, where it is used.
#: HER CLOCK, AND IT IS SETTABLE FROM OUTSIDE --- his standing ask, repeated
#: 2026-09-01: *"as you remember i ask you make t/s like env var to avoid this
#: problems"*.  `MATILDA_FPS` sets it once for her body AND her mind: her body
#: starts her mind as a child process, so one export reaches all three.
#:
#: IT IS NOT A FRAME RATE.  Everything of hers is derived from this --- her
#: hormones, her look-back, the length of an experience, and her physics
#: timestep (`ragdoll.DT`) --- so a life lived at one value cannot be compared
#: with a life lived at another, and a tape must be fresh when it changes.
#: NINETY IS HERS --- his 11 ms sound tick, 2026-09-03 ("tick has to be 11ms
#: but it is for sound so its like blueprint for resc channels"); thirty was
#: hers until then, and every tape from before is of another clock.
TICK_SECONDS = 1.0 / float(_os.environ.get("MATILDA_FPS", "90") or "90")


def _rows(got, one) -> list:
    """Rows of a thing.  The app may say one row where the structure says a
    list of them (matilda2's old single-motor shape); one row is rows of one.
    """
    if got is None:
        return []
    if isinstance(got, dict):
        got = [got]
    return [one(**r) for r in got]


def _tick(got: dict) -> Tick:
    """One tick of the app's JSON as the structure --- inventing nothing."""
    l = got.get("life") or {}
    o = l.get("output") or {}
    i = l.get("input") or {}
    return Tick(
        time=got.get("time"),
        life=Life(
            output=Output(
                motor=_rows(o.get("motor"), Motor),
                sound=OutSound(**(o.get("sound") or {}))),
            input=Input(
                state=float(i.get("state") or 0.0),
                sensor=_rows(i.get("sensor"), Sensor),
                spindle=_rows(i.get("spindle"), Spindle),
                echo=Echo(**(i.get("echo") or {})),
                sound=Heard(**(i.get("sound") or {})),
                view=_rows(i.get("view"), View))),
        exp=ExpRef(**(got.get("exp") or {})))


def getTicks(since: int, at: str = "http://127.0.0.1:8090"):
    """Every finished tick since her tick `since` --- `(ticks, next)`.

    His fix for the loop that saw 15% of her: the app serves the batch OFF
    her lock (a finished tick is immutable), so a slow pass loses TIME,
    never TICKS --- her chemistry advances once per tick LIVED.
    """
    # `wait=1`: the body does not answer until it has lived another one, so
    # this asks once and is told the moment there is something --- instead of
    # asking twice a tick and being told nothing most times.  See `LIVED` in
    # the body's `sandbox/app.py`.
    got = json.loads(_ask(at, "GET",
                          "/ticks?from=%d&wait=1" % int(since)).decode("utf-8"))
    return [_tick(r) for r in got.get("ticks", [])], int(got["next"])




def getAble(at: str = "http://127.0.0.1:8090") -> dict:
    """What she is built able to do --- body facts off the app's `/able`:
    `motors`, `resolution`, `sounds`.  An outside brain never hardcodes her.
    """
    with urlopen(at.rstrip("/") + "/able", timeout=5) as h:
        return json.loads(h.read().decode("utf-8"))


def felt(numbers: dict, at: str = "http://127.0.0.1:8090") -> None:
    """Her finished numbers, handed to the page --- his "fix it" on the panel
    that showed a frozen never-hungry girl.  Display only; nothing reads
    them back, and a missed post is a missed frame of a mirror.
    """
    _ask(at, "POST", "/felt", numbers)


def act(decision, at: str = "http://127.0.0.1:8090", tick: int = -1) -> dict:
    """One decision, posted to her body's `/act`.

    A try is one line, `(kind, id, lvl)`.  A remembered moment is a WHOLE
    PLAN, `{"motor": [{"id", "lvl"}...], "sound": {"id", "lvl"}}` --- every
    muscle and her voice, as the stored tick had them (his, 2026-09-02: the
    pose she held to reach a thing *"already would be in her
    exp/preddictionn chanin"*).  `tick` is the moment the plan answers, so
    the body can measure how late it lands.

    It lands on her next tick exactly where her own act would have written
    it; the body executes it the tick after.
    """
    if isinstance(decision, dict):
        return json.loads(_ask(at, "POST", "/act",
                               {"plan": decision, "tick": int(tick)})
                          .decode("utf-8"))
    kind, tid, lvl = decision
    return json.loads(_ask(at, "POST", "/act",
                           {"kind": str(kind), "id": int(tid),
                            "lvl": float(lvl)}).decode("utf-8"))


