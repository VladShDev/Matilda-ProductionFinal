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
# ============================================================ #
"""The record --- one life, one DuckDB file, shaped like `structure.py`.

HIS, 2026-09-06, on a record that grew 19 MB a minute of her life because
every tick was written whole (2,741 bytes a tick, 80% of ticks with nothing
moved by her resolution): *"save only changes: take a start point with all
existing tick information, and then during her experience only the
changes."*  So:

  key     the START POINT of every experience: its first tick, whole
          (`life`, the full struct), with its time, exp and plan.
  step    every other tick: time, exp, plan, and DELTA --- only the lines
          whose value differs from the last one written.  Her inputs are
          written at her body's resolution (the grain her mind reads them
          at: `life._line` buckets by it, closeness matches within it), her
          outputs exactly, so a replay after a wake posts exactly what she
          did.  A tick in which nothing moved is a row with an empty delta.
  echo    the pose that made each sound she made: her motor lines on every
          tick her echo line sounded (`posesOf`, the wake).
  exp, memory, hands   as they were: her experiences' openings, her hands'
          ids, her checkpoint.

Reading is one walk: start at a key, apply steps in time order, and every
reader gets whole ticks back (`walk`, `chainLives`, `openRun`).  Nothing in
her mind reads the record while she lives; it is read at the wake and by the
instruments.  A record of the old shape (a `tick` table, every tick whole)
still reads through the same methods.
"""
from __future__ import annotations

import atexit
import json
import os
import stat
import threading
import time
from dataclasses import asdict

import duckdb

from structure import (Echo, Exp, ExpRef, Heard, Input, Life, Memory, Motor,
                       Output, OutSound, Sensor, Spindle, Tick, View)

_ECHO = "STRUCT(id INTEGER, similarity DOUBLE, lvl DOUBLE, balance INTEGER)"
_VIEW = ("STRUCT(id INTEGER, similarity DOUBLE, distance DOUBLE, "
         "x DOUBLE, y DOUBLE, z DOUBLE)")
_LIFE = f"""STRUCT(
    output STRUCT(
        motor STRUCT(id INTEGER, lvl DOUBLE)[],
        sound STRUCT(id DOUBLE, lvl DOUBLE)),
    input STRUCT(
        state DOUBLE,
        sensor STRUCT(id DOUBLE, lvl DOUBLE)[],
        spindle STRUCT(id INTEGER, lvl DOUBLE)[],
        echo {_ECHO},
        sound {_ECHO},
        view {_VIEW}[]))"""
_FLAT = " ".join(_LIFE.split())
#: one change: what kind of line (K_*), which id, and its numbers
_DELTA = "STRUCT(k SMALLINT, id INTEGER, a FLOAT, b FLOAT, c FLOAT, d FLOAT, e FLOAT, f INTEGER)"
_KEY_COLS = ("{time: 'DOUBLE', exp: 'STRUCT(id INTEGER)', plan: 'STRUCT(id INTEGER)', life: '" + _FLAT + "'}")
_STEP_COLS = ("{time: 'DOUBLE', exp: 'STRUCT(id INTEGER)', plan: 'STRUCT(id INTEGER)', delta: '" + _DELTA + "[]'}")
_ECHO_COLS = "{time: 'DOUBLE', id: 'INTEGER', motor: 'STRUCT(id INTEGER, lvl DOUBLE)[]'}"
_EXP_COLS = ("{id: 'INTEGER', sensor: 'STRUCT(id DOUBLE, lvl DOUBLE)[]', "
             "spindle: 'STRUCT(id INTEGER, lvl DOUBLE)[]', "
             "echo: '" + _ECHO + "', sound: '" + _ECHO + "', "
             "view: '" + _VIEW + "[]', "
             "exp: 'STRUCT(id INTEGER)'}")

_TABLES = f"""
CREATE TABLE IF NOT EXISTS key (
    time DOUBLE,
    exp STRUCT(id INTEGER),
    plan STRUCT(id INTEGER),
    life {_LIFE});
CREATE TABLE IF NOT EXISTS step (
    time DOUBLE,
    exp STRUCT(id INTEGER),
    plan STRUCT(id INTEGER),
    delta {_DELTA}[]);
CREATE TABLE IF NOT EXISTS echo (
    time DOUBLE,
    id INTEGER,
    motor STRUCT(id INTEGER, lvl DOUBLE)[]);
CREATE TABLE IF NOT EXISTS exp (
    id INTEGER PRIMARY KEY,
    sensor STRUCT(id DOUBLE, lvl DOUBLE)[],
    spindle STRUCT(id INTEGER, lvl DOUBLE)[],
    echo {_ECHO},
    sound {_ECHO},
    view {_VIEW}[],
    exp STRUCT(id INTEGER));
CREATE TABLE IF NOT EXISTS memory (
    short INTEGER[],
    long INTEGER[]);
CREATE TABLE IF NOT EXISTS hands (
    age BIGINT,
    time DOUBLE,
    packed VARCHAR);
"""

#: the kinds of line a delta can carry
K_SENSOR, K_SPINDLE, K_MOTOR, K_ECHO, K_SOUND, K_VIEW, K_STATE, K_OUTSOUND = range(8)
GONE = -1.0                     # a thing that left her view: its similarity written as -1


def _life(d: dict) -> Life:
    """His `life` block back out of the db, row by row."""
    o, i = d["output"], d["input"]
    return Life(
        output=Output(
            motor=[Motor(**m) for m in o["motor"]],
            sound=OutSound(**o["sound"])),
        input=Input(
            state=i["state"],
            sensor=[Sensor(**s) for s in i["sensor"]],
            spindle=[Spindle(**s) for s in i["spindle"]],
            echo=Echo(**i["echo"]),
            sound=Heard(**i["sound"]),
            view=[View(**v) for v in i["view"]]))


def _q(v: float, res: float) -> float:
    """A level at her body's resolution --- the grain her mind reads it at."""
    return round(round(float(v) / res) * res, 6) if res > 0 else float(v)


class _Writer:
    """What was last written, line by line, so a step carries only changes."""

    def __init__(self, res: float) -> None:
        self.res = float(res)
        self.reset()

    def reset(self) -> None:
        self.last: dict = {}            # (kind, id) -> tuple of numbers
        self.expId = None

    def whole(self, life: Life) -> dict:
        """A key: the tick whole, inputs at her resolution; remembers it."""
        d = asdict(life)
        i = d["input"]; r = self.res
        i["state"] = _q(i["state"], r)
        for s in i["sensor"]:
            s["lvl"] = _q(s["lvl"], r)
        for s in i["spindle"]:
            s["lvl"] = _q(s["lvl"], r)
        for k in ("echo", "sound"):
            i[k]["similarity"] = _q(i[k]["similarity"], r); i[k]["lvl"] = _q(i[k]["lvl"], r)
        for v in i["view"]:
            for k in ("similarity", "distance", "x", "y", "z"):
                v[k] = _q(v[k], r)
        self.last = self._lines(d)
        return d

    def delta(self, life: Life) -> list:
        """A step: the lines whose value differs from the last written."""
        d = asdict(life)
        now = self._lines(d, quant=True)
        out = []
        for key, val in now.items():
            if self.last.get(key) != val:
                k, i = key
                out.append({"k": k, "id": int(i), "a": val[0], "b": val[1], "c": val[2], "d": val[3], "e": val[4], "f": int(val[5])})
        for key in self.last:
            if key not in now and key[0] == K_VIEW:
                out.append({"k": K_VIEW, "id": int(key[1]), "a": GONE, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0})
        self.last = now
        return out

    def _lines(self, d: dict, quant: bool = False) -> dict:
        r = self.res if quant else 0.0
        q = (lambda v: _q(v, r)) if quant else float
        i, o = d["input"], d["output"]
        out = {(K_STATE, 0): (q(i["state"]), 0.0, 0.0, 0.0, 0.0, 0)}
        for s in i["sensor"]:
            out[(K_SENSOR, int(s["id"]))] = (q(s["lvl"]), 0.0, 0.0, 0.0, 0.0, 0)
        for s in i["spindle"]:
            out[(K_SPINDLE, int(s["id"]))] = (q(s["lvl"]), 0.0, 0.0, 0.0, 0.0, 0)
        e, h = i["echo"], i["sound"]
        out[(K_ECHO, int(e["id"]))] = (q(e["similarity"]), q(e["lvl"]), 0.0, 0.0, 0.0, int(e["balance"]))
        out[(K_SOUND, int(h["id"]))] = (q(h["similarity"]), q(h["lvl"]), 0.0, 0.0, 0.0, int(h["balance"]))
        for v in i["view"]:
            out[(K_VIEW, int(v["id"]))] = (q(v["similarity"]), q(v["distance"]), q(v["x"]), q(v["y"]), q(v["z"]), 0)
        for m in o["motor"]:
            out[(K_MOTOR, int(m["id"]))] = (float(m["lvl"]), 0.0, 0.0, 0.0, 0.0, 0)
        out[(K_OUTSOUND, int(o["sound"]["id"]))] = (float(o["sound"]["lvl"]), 0.0, 0.0, 0.0, 0.0, 0)
        return out


def _apply(cur: dict, delta: list) -> None:
    """One step onto the running tick (nested dicts, as `_life` reads them)."""
    i, o = cur["input"], cur["output"]
    for c in delta or []:
        k, sid = int(c["k"]), int(c["id"])
        if k == K_SENSOR:
            for s in i["sensor"]:
                if int(s["id"]) == sid:
                    s["lvl"] = float(c["a"]); break
            else:
                i["sensor"].append({"id": float(sid), "lvl": float(c["a"])})
        elif k == K_SPINDLE:
            for s in i["spindle"]:
                if int(s["id"]) == sid:
                    s["lvl"] = float(c["a"]); break
            else:
                i["spindle"].append({"id": sid, "lvl": float(c["a"])})
        elif k == K_MOTOR:
            for m in o["motor"]:
                if int(m["id"]) == sid:
                    m["lvl"] = float(c["a"]); break
            else:
                o["motor"].append({"id": sid, "lvl": float(c["a"])})
        elif k == K_ECHO:
            i["echo"] = {"id": sid, "similarity": float(c["a"]), "lvl": float(c["b"]), "balance": int(c["f"])}
        elif k == K_SOUND:
            i["sound"] = {"id": sid, "similarity": float(c["a"]), "lvl": float(c["b"]), "balance": int(c["f"])}
        elif k == K_VIEW:
            i["view"] = [v for v in i["view"] if int(v["id"]) != sid]
            if float(c["a"]) != GONE:
                i["view"].append({"id": sid, "similarity": float(c["a"]), "distance": float(c["b"]),
                                  "x": float(c["c"]), "y": float(c["d"]), "z": float(c["e"])})
        elif k == K_STATE:
            i["state"] = float(c["a"])
        elif k == K_OUTSOUND:
            o["sound"] = {"id": float(sid), "lvl": float(c["a"])}


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Store:
    """One life, on disk."""

    def __init__(self, where: str, read: bool = False,
                 resume: bool = False, resolution: float = 0.05) -> None:
        """One life, one file.  HIS GUARD, 2026-08-26: *"now her memory is
        very important so make some guard for her tape"*.

        - ONE LIFE PER FILE, REFUSED LOUDLY: a file already holding ticks is
          not written again unless `resume=True` continues it.
        - A SEALED LIFE: `close()` marks the file read-only; `read=True`
          opens it to remember.
        - NO MOMENT LEFT IN THE AIR: the keeper flushes once a second, and
          `atexit` the rest.
        """
        self.read = bool(read)
        self._closed = False
        self._path = where
        self._writer = _Writer(resolution)
        if self.read:
            self.db = duckdb.connect(where, read_only=True)
            return
        if resume and os.path.exists(where):
            try:
                os.chmod(where, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
        self.db = duckdb.connect(where)
        self.db.execute("PRAGMA memory_limit='512MB'")
        self.db.execute(_TABLES)
        held = self._count()
        if held and not resume:
            self.db.close()
            raise RuntimeError(
                "%s already holds a life (%d ticks).  Her lives are "
                "evidence: pass resume=True to CONTINUE it, or read=True "
                "to remember it." % (where, held))
        self._ticks: list = []
        self.recorded = 0               # ticks the keeper has written, for the page
        self._exps: list = []
        self._handsRow = None
        self._lock = threading.Lock()
        self._dblock = threading.Lock()
        self._stage = where + ".stage.jsonl"
        self._going = True
        self._keeper = threading.Thread(target=self._keep, daemon=True)
        self._keeper.start()
        atexit.register(self.close)

    # ---------------------------------------------------------------- shape
    def _legacy(self) -> bool:
        """A record of the old shape: a `tick` table, every tick whole."""
        try:
            got = self.db.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'tick'").fetchone()
            return bool(got and got[0])
        except duckdb.Error:
            return False

    def _count(self) -> int:
        if self._legacy():
            return int(self.db.execute("SELECT count(*) FROM tick").fetchone()[0])
        k, = self.db.execute("SELECT count(*) FROM key").fetchone()
        s, = self.db.execute("SELECT count(*) FROM step").fetchone()
        return int(k) + int(s)

    # ---------------------------------------------------------------- write
    def tick(self, t: Tick) -> None:
        """One more moment of her --- in RAM now, on disk by the keeper."""
        if self.read:
            raise RuntimeError("this life is sealed --- opened for reading")
        with self._lock:
            self._ticks.append(t)

    def exp(self, e: Exp) -> None:
        if self.read:
            raise RuntimeError("this life is sealed --- opened for reading")
        with self._lock:
            self._exps.append({"id": e.id,
                               "sensor": [asdict(x) for x in e.sensor],
                               "spindle": [asdict(x) for x in e.spindle],
                               "echo": asdict(e.echo) if e.echo else None,
                               "sound": asdict(e.sound) if e.sound else None,
                               "view": [asdict(x) for x in e.view],
                               "exp": asdict(e.exp)})

    def _keep(self) -> None:
        while self._going:
            time.sleep(1.0)
            try:
                self.flush()
            except Exception:                                      # noqa: BLE001
                pass

    def flush(self) -> None:
        """Everything buffered, to disk, in ONE transaction.  The buffers are
        swapped under `_lock` (instant) and written under `_dblock` --- a slow
        write delays the record, never her."""
        if self.read or self._closed:
            return
        with self._lock:
            ticks, self._ticks = self._ticks, []
            exps, self._exps = self._exps, []
            hands, self._handsRow = self._handsRow, None
        keys, steps, echoes = [], [], []
        w = self._writer
        for t in ticks:
            eid = int(t.exp.id)
            head = {"time": t.time, "exp": asdict(t.exp), "plan": asdict(t.plan)}
            if w.expId != eid or not w.last:
                # THE START POINT: the first tick of an experience, whole
                keys.append(dict(head, life=w.whole(t.life)))
                w.expId = eid
            else:
                steps.append(dict(head, delta=w.delta(t.life)))
            if float(t.life.input.echo.lvl) > 0.0:
                echoes.append({"time": t.time, "id": int(t.life.input.echo.id),
                               "motor": [asdict(m) for m in t.life.output.motor]})
        self.recorded += len(ticks)
        if not (keys or steps or echoes or exps or hands):
            return
        with self._dblock:
            self.db.execute("BEGIN")
            for rows, table, cols, how in (
                    (keys, "key", _KEY_COLS, "INSERT"),
                    (steps, "step", _STEP_COLS, "INSERT"),
                    (echoes, "echo", _ECHO_COLS, "INSERT"),
                    (exps, "exp", _EXP_COLS, "INSERT OR REPLACE")):
                if not rows:
                    continue
                with open(self._stage, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r))
                        f.write("\n")
                self.db.execute(
                    how + " INTO " + table + " SELECT * FROM read_json(?, "
                    "columns=" + cols + ")", [self._stage])
            if hands:
                self.db.execute("DELETE FROM hands")
                self.db.execute("INSERT INTO hands VALUES (?, ?, ?)", list(hands))
            self.db.execute("COMMIT")

    def keepHands(self, age: int, time: float, packed: dict) -> None:
        """THE CHECKPOINT OF HER HANDS --- what her mind holds that the tables
        do not (the seen world, the worths, the openings); the keeper writes
        it with its next flush."""
        if self.read or self._closed:
            return
        row = (int(age), float(time), json.dumps(packed))
        with self._lock:
            self._handsRow = row

    def memory(self, m: Memory) -> None:
        """Her memory, replaced whole --- it is one row and it is hers."""
        if self.read:
            raise RuntimeError("this life is sealed --- opened for reading")
        with self._dblock:
            self.db.execute("DELETE FROM memory")
            self.db.execute("INSERT INTO memory VALUES (?, ?)", [m.short, m.long])

    # ----------------------------------------------------------------- read
    def hands(self):
        """The newest checkpoint, or None on a life from before them."""
        got = self.db.execute(
            "SELECT age, time, packed FROM hands ORDER BY age DESC LIMIT 1").fetchone()
        if got is None:
            return None
        return int(got[0]), float(got[1]), json.loads(got[2])

    def _rows(self, where: str = "", args: list | None = None):
        """Keys and steps together, in time order, as (time, exp, plan, life, delta)."""
        args = list(args or [])
        if self._legacy():
            sql = "SELECT time, exp, plan, life, NULL FROM tick " + where + " ORDER BY time"
        else:
            sql = ("SELECT time, exp, plan, life, NULL AS delta FROM key " + where +
                   " UNION ALL SELECT time, exp, plan, NULL, delta FROM step " + where +
                   " ORDER BY time")
            args = args + args
        return self.db.execute(sql, args)

    def walk(self, where: str = "", args: list | None = None):
        """Every tick in time order, whole --- STREAMED, never a list.  Starts
        at the first key it meets; steps before any key are skipped."""
        if not self.read:
            self.flush()
        cur = None
        with (_Nothing() if self.read else self._dblock):
            q = self._rows(where, args)
            while True:
                got = q.fetchmany(4096)
                if not got:
                    return
                for t, x, p, life, delta in got:
                    if life is not None:
                        cur = life
                    elif cur is None:
                        continue
                    else:
                        _apply(cur, delta)
                    yield Tick(time=t, life=_life(cur), exp=ExpRef(**x), plan=ExpRef(**p))

    def exps(self) -> list[Exp]:
        if not self.read:
            self.flush()
        got = self.db.execute("SELECT id, sensor, spindle, echo, sound, view, "
                              "exp FROM exp").fetchall()
        return [Exp(id=i,
                    sensor=[Sensor(**s) for s in sn],
                    spindle=[Spindle(**s) for s in sp],
                    echo=Echo(**e) if e else None,
                    sound=Heard(**so) if so else None,
                    view=[View(**x) for x in v],
                    exp=ExpRef(**x2) if x2 else ExpRef(id=0))
                for i, sn, sp, e, so, v, x2 in got]

    def remember(self) -> Memory:
        got = self.db.execute("SELECT short, long FROM memory").fetchone()
        return Memory(short=got[0], long=got[1]) if got else Memory()

    def count(self) -> int:
        """Ticks in the record."""
        if not self.read:
            self.flush()
        return self._count()

    def close(self, seal: bool = True) -> None:
        """The life ends whole: flushed, closed, and SEALED read-only.
        `seal=False` for a file that will be lived on."""
        if self._closed:
            return
        if self.read:
            self._closed = True
            self.db.close()
            return
        self._going = False
        try:
            self.flush()
        finally:
            self._closed = True
            self.db.close()
            if seal:
                try:
                    os.chmod(self._path, stat.S_IREAD)
                except OSError:
                    pass
            try:
                os.remove(self._stage)
            except OSError:
                pass
