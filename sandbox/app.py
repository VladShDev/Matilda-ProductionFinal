"""THE VIEWER --- her real page, fed from her real body.

`web/index.html` came whole from the old tree: 4,158 lines of WebGL that draw
her as a child, with her onesie, her cot, the room and the flying screen.  It is
not welded to anything --- it asks questions and draws answers --- so it needed
no change at all.

**WHAT COULD NOT COME IS `body/sandbox.py`.**  The old viewer's server drove her
through `contract/` --- manifest, record, wire, territories, the fingerprint ---
which is precisely what this rewrite deleted: her brain speaks `tick.life` now,
one muscle and one sound at a time, and there are no territories to wire.  So
this is a new server for the same page, and it is the only part that had to be
written.

**THE PAGE IS A MIRROR OF FINISHED NUMBERS.**  Everything the browser would
otherwise compute happens here.  It is never given her field of view, never told
what a pixel means, and never asked to work out where anything is.

**ONE CLOCK: HERS.**  Every answer carries the tick it is true at, and one frame
is one tick --- 30 a second, which is what he asked for.

    python sandbox/app.py --port 8080
"""

from __future__ import annotations

import atexit
import base64
import io
import subprocess
import time as _time
from time import perf_counter as _now
import json
import os
import zlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import random as _rnd

PORT = 8080
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import film                                                        # noqa: E402
from body import speech                                            # noqa: E402
from body.room import BED                                          # noqa: E402
from body.muscles import VOICE_PARTS                               # noqa: E402
from body.hearing import SOUND_BANDS                               # noqa: E402
from body.hearing import TICK_SECONDS                              # noqa: E402
from body import bind
from body import light                                             # noqa: E402  (her eye: which device it runs on, for /meta)
from sandbox.heard import Heard                                  # noqa: E402
from body.ragdoll import JIDX                                      # noqa: E402
from body.alive import Her                                         # noqa: E402

#: WHICH JOINTS ARE JOINED, for drawing only.  Her body's `_BONES` holds every
#: constraint it solves; these are the ones that are a limb.
DRAWN_BONES = [
    ("head", "neck"), ("neck", "chest"), ("chest", "pelvis"),
    ("chest", "shR"), ("chest", "shL"),
    ("shR", "elR"), ("elR", "haR"), ("shL", "elL"), ("elL", "haL"),
    ("pelvis", "hiR"), ("pelvis", "hiL"),
    ("hiR", "knR"), ("knR", "foR"), ("hiL", "knL"), ("knL", "foL"),
    # HER FEET.  `toR`/`toL` are the ball of the foot, and they were never
    # drawn --- so the part of her that has to carry her was the one part you
    # could not see her standing on.
    ("foR", "toR"), ("foL", "toL"),
]
SUIT_BONES = ["chest>pelvis", "chest>shR", "chest>shL", "pelvis>hiR", "pelvis>hiL"]

JOINTS = list(JIDX)
WIDE = len(JOINTS) * 3 + film.WINDOW_VALUES


def tunnelStop() -> None:
    p = getattr(HER, "tunnelProc", None)
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(5)
        except Exception:                                      # noqa: BLE001
            p.kill()
    HER.tunnelProc = None
    HER.tunnelUrl = None


def tunnelStart(port: int) -> None:
    """A road to her from anywhere --- his, 2026-08-27: *"add button to run
    tonnel ... to i be able give link to someone to itterack with her"*, and
    *"caludfalre would be good"*.  cloudflared's quick tunnel; the link it
    prints is read off its own mouth and handed to the page."""
    import re
    import threading as _th
    HER.tunnelUrl = "starting..."
    # --protocol http2: the first road died with Error 1033 --- the edge knew
    # the name, the QUIC leg never came up on this network, and cloudflared
    # exited AFTER printing the link.  http2 rides plain TLS and survives.
    # Its whole mouth is kept in lives/tunnel.log now, so a death has a why.
    HER.tunnelProc = subprocess.Popen(
        ["cloudflared", "tunnel", "--protocol", "http2",
         "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def _read():
        keep = io.open(os.path.join("lives", "tunnel.log"), "a",
                       encoding="utf-8")
        proc = HER.tunnelProc
        for line in proc.stderr:
            keep.write(line)
            keep.flush()
            got = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if got:
                HER.tunnelUrl = got.group(0)
        if proc is HER.tunnelProc:
            HER.tunnelUrl = "failed --- lives/tunnel.log says why"
    _th.Thread(target=_read, daemon=True).start()


def mindStop() -> None:
    """Her mind's process, ended.  The keeper thread has flushed within the
    last second and the guards seal what a hard end leaves."""
    p = getattr(HER, "mindProc", None)
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(5)
        except Exception:                                      # noqa: BLE001
            p.kill()


#: WHAT THE PAGE CAN DRAW.  Her clock is 90; a browser draws 30 of them.
PAGE_FPS = 30
PAGE_EVERY = max(1, int(round(1.0 / (TICK_SECONDS * PAGE_FPS))))


#: THE BODY'S KEEPSAKES THAT BELONG TO A LIFE --- her mother's memory of his
#: words, the screen's place, the clock.  Bundled into the life's record when
#: it stops, put back when it loads: ONE FILE per life (his, 2026-09-06).
KEEPSAKES = ("mom.json", "mom.json.npz", "window", "clock")


def _bodyLives() -> str:
    return getattr(HER, "lives", None) or os.path.join(HERE, "..", "lives")


def bundle(db: str) -> None:
    """A stopped life takes the body's keepsakes into its own file."""
    path = os.path.join(HER.mindDir, "lives", os.path.basename(db))
    if not os.path.exists(path):
        return
    try:
        os.chmod(path, 0o666)
        import duckdb as _dk
        con = None
        for _ in range(40):                # the mind just ended holds the file a moment longer
            try:
                con = _dk.connect(path); break
            except Exception:                                      # noqa: BLE001
                _time.sleep(0.25)
        if con is None:
            raise RuntimeError("the record stayed locked for ten seconds")
        try:
            con.execute("CREATE TABLE IF NOT EXISTS keepsake (name VARCHAR, content BLOB)")
            con.execute("DELETE FROM keepsake")
            for k in KEEPSAKES:
                f = os.path.join(_bodyLives(), k)
                if os.path.exists(f):
                    with open(f, "rb") as h:
                        con.execute("INSERT INTO keepsake VALUES (?, ?)", [k, h.read()])
            con.execute("CHECKPOINT")     # fold the log into the file: ONE file, movable
        finally:
            con.close()
        try:
            os.remove(path + ".wal")
        except OSError:
            pass
    except Exception as e:                                          # noqa: BLE001
        print("  the life's keepsakes were not bundled: %s" % e)


def unbundle(db: str) -> None:
    """A loading life puts the body's keepsakes back; a new life clears them."""
    path = os.path.join(HER.mindDir, "lives", os.path.basename(db))
    for k in KEEPSAKES:
        try:
            os.remove(os.path.join(_bodyLives(), k))
        except OSError:
            pass
    if not os.path.exists(path):
        return
    try:
        import duckdb as _dk
        con = _dk.connect(path, read_only=True)
        try:
            rows = con.execute("SELECT name, content FROM keepsake").fetchall()
        except Exception:                                          # noqa: BLE001
            rows = []
        con.close()
        for k, content in rows:
            with open(os.path.join(_bodyLives(), str(k)), "wb") as h:
                h.write(bytes(content))
    except Exception as e:                                          # noqa: BLE001
        print("  the life's keepsakes were not put back: %s" % e)


def mindStart(db: str) -> None:
    """Her mind, woken on `db` --- a new file is a birth, a file that holds
    her life resumes it (the mind's own law)."""
    lives = os.path.join(HER.mindDir, "lives")
    os.makedirs(lives, exist_ok=True)
    name = os.path.basename(db)
    out = open(os.path.join(lives, name + ".log"), "ab")
    # WHICH BODY IS HERS.  This passed only the tape, and `life.py` defaults
    # to port 8090 --- so a body started on any other port got a mind that
    # died on its first call, in a log nobody reads, while the body printed
    # "her mind is kept here too" and went on living without one.  Found
    # 2026-09-01 after three measurements turned out to be of a mindless girl.
    HER.mindProc = subprocess.Popen(
        [sys.executable, "-u", "life.py", os.path.join("lives", name),
         "--at", "http://127.0.0.1:%d" % int(getattr(HER, "port", 8090) or 8090)],
        cwd=HER.mindDir, stdout=out, stderr=out)
    HER.mindDb = name


def _earBars(her) -> dict:
    """WHAT REACHED EACH EAR, one number a band --- the reducer the page cites.

    `sandbox/web/index.html` has said *"Every number is `/state.ears`, reduced
    on the server (`_ear_bars`)"* since it was ported, and **there was no such
    function and `/state` never carried `ears`** --- so the bars drew "silence"
    on every tick of every life.  A sense with an instrument wired to nothing is
    worse than a sense with none: it reports the good news.

    The PEAK since the page last looked when there is one (`Watched`), and this
    tick when there is not (a plain `Her`, which nobody watches).  Loudest over
    the slides, never the mean --- `alike.one_ear`'s rule, not a second one.
    """
    got = getattr(her, "earsPeak", None)
    if got is None:
        got = np.asarray(her.ears, np.float32).max(axis=2)
    # THE AMBER ROW IS WHAT HER MOUTH MADE.  His, 2026-09-11: *"in that line I
    # am able to see what she tried to produce by yellow colour --- sometimes
    # when I muted something I don't understand that she tried to produce or
    # not."*  It used to be fed by her echo, and she has no echo any more (her
    # own air goes to her ears with everything else), so it reads the air itself:
    # `alive.voiced`, written every tick from what `Voice.say` actually produced.
    # Not what she asked for --- what came out.
    mine = np.asarray(getattr(her, "voiced", np.zeros(SOUND_BANDS)), np.float32)
    # HER OWN VOICE IS A THIRD ROW AND NOT A THIRD EAR.  His, 2026-08-26:
    # *"split sound from echo on bars, she drowing her wabling"*.  `sounding`
    # carries only the window, so nothing of hers has been in these two rows
    # since her ears became for what is not hers --- but the panel drew one
    # picture and there was no way to see that from outside.  Two senses, two
    # rows, and a third for the mouth that answers itself.
    return {"left": [round(float(v), 4) for v in got[0]],
            "right": [round(float(v), 4) for v in got[1]],
            "echo": [round(float(v), 4) for v in mine]}


class Watched(Her):
    """Her, with every tick kept as a frame the page can play."""

    def __init__(self, lives=None, keep: int = 900) -> None:
        super().__init__(lives)
        #: WHAT HE SAID TO HER, in her lives folder, named by the time.  The
        #: watcher's, never hers --- see `sandbox/heard.py`.
        self.yours = Heard(os.path.join(lives or "lives",
                                        _time.strftime("%Y-%m-%d_%H%M%S")
                                        + "_you.wav"), int(speech.RATE))
        #: THE LOUDEST EACH BAND HAS REACHED SINCE THE PAGE LAST LOOKED.
        #:
        #: Her ears change every tick --- 30 a second --- and the panel polls
        #: twice a second, so it was drawing ONE TICK IN FIFTEEN and missing
        #: 93% of her hearing.  A word is 10-20 ticks, so it usually landed
        #: between words and caught random moments: *"they do not align with my
        #: voice, sometimes drown like something loud but when i tald some they
        #: do not moves"*.
        #:
        #: A PEAK, NOT AN AVERAGE, AND NO DECAY.  A mean would make a word and a
        #: silence draw the same picture; a decay would be a number nobody
        #: measured.  This holds the loudest reading until it is read, which is
        #: what a peak meter is, and then starts again --- so nothing she heard
        #: between two looks is lost, whatever the poll rate.
        self.earsPeak = np.zeros((2, SOUND_BANDS), np.float32)
        #: ...and her own mouth, held the same way.
        self.echoPeak = np.zeros(SOUND_BANDS, np.float32)
        self.frames: list = []
        self.first = 0
        self.keep = keep
        #: THE FILM HAS ITS OWN LOCK, AND THAT IS THE WHOLE FIX.
        #:
        #: MEASURED: the page asks `/frames` **23 times a second** --- it is
        #: driven by the browser's own frame loop --- and every one of those
        #: took HER body lock to read a list of bytes.  Each call waited
        #: **32.11 ms** for it, because that is how long her tick holds it, and
        #: 821 calls came to **26.4 seconds of waiting in 35 seconds of wall
        #: time.**  Nearly all of them returned a 20-byte header and no frames.
        #:
        #: Her lock is for HER BODY.  The reel is the watcher's, so it gets its
        #: own.  What she DOES for the watcher was measured at 0.099 ms a tick;
        #: this is what the ASKING was costing her.
        self.reel = threading.Lock()
        #: WHAT IS REALLY THERE --- the ruler's answer, and the tick of the
        #: picture it was measured on.  Nothing of hers reads either.
        self.named: list = []
        self.namedAt = -1
        self.ruler = None
        #: WHAT THE TELEVISION HAS ACTUALLY BEEN SENT.  "the monitor is
        #: freezing" has two completely different causes --- the page not
        #: posting, and the panel riding his own camera so it hangs where
        #: he stopped --- and no screenshot can tell them apart.  These
        #: three numbers can.
        self.flew = 0          # posts that arrived at all
        self.shows = 0         # ...of them, ones carrying a picture
        self.showsAt = -1      # the tick the last picture landed on

    def ruling(self) -> None:
        """THE RULER, ON ITS OWN THREAD, OFF HER CLOCK.

        A look costs 39.6 ms and her tick is 33.3, so this can never run inside
        one --- and it must never hold her lock while it does, which is the
        mistake that cost 16.4 ticks a second the first time the page and she
        were made to share her.  The lock is taken to READ the picture and
        dropped before the model is asked anything.

        IT WAITS FOR A NEW PICTURE RATHER THAN A CLOCK.  Her retina is remade
        every `looksEvery` ticks (200 ms) and the ruler takes 40, so a name
        lands well inside the life of the picture it was measured on --- and
        `namedAt` says which one, so nothing has to be assumed.

        It is `measure/`, so it may use a trained model.  Nothing in `brain/`
        imports it, nothing in `body/` imports it, and a name never enters her
        tape.  If the weights or mediapipe are missing it says so once and this
        thread ends; she does not notice and the panel says "nothing named".
        """
        import time as _t
        from measure.ruler import Ruler
        self.ruler = Ruler()
        if not self.ruler.ready:
            print("  the ruler is not plugged in --- " + self.ruler.why)
            return
        print("  the ruler is loaded: what is really there gets a name")
        was = None
        while self.going:
            with self.lock:
                tick, pic = len(self.ticks), self.picture
            if pic is None or pic is was:
                _t.sleep(0.03)
                continue
            got = self.ruler.look(pic)
            was = pic
            self.named, self.namedAt = got, tick

    def earsSeen(self) -> None:
        """The page has looked --- start the peak again."""
        self.earsPeak[:] = 0.0
        self.echoPeak[:] = 0.0

    def oneTick(self) -> None:
        super().oneTick()
        # HER VOICE, WHOLE, FOR WHOEVER LISTENS.  /said used to serve ONE
        # tick's 33 ms and the page sampled it every 700 --- he heard 5%
        # of his daughter's voice as disconnected bips and reported her
        # "just biping ... all her answer is one tick" (2026-08-28).  Her
        # mouth was never the bug; the monitor was.  Three seconds of the
        # real samples ride here; /said?after=N hands over everything
        # since the ear that asks last heard.
        pcm = np.asarray(self.voice.pcm, np.float32)
        ring = getattr(self, "saidRing", None)
        if ring is None:
            from collections import deque
            ring = self.saidRing = deque(maxlen=90)
        ring.append((len(self.ticks), pcm.copy()))
        # ...AND HER MOTHER, ON THE SAME TERMS.  His, 2026-09-01: *"make me
        # able to hear what her mothers tolds her to be able debug at least
        # something from mom."*  Her mother is the one actor in the room whose
        # output nobody has ever heard --- she speaks in band pictures and the
        # samples behind them were dropped.  This is HIS OWN recorded word,
        # the source her frames were made from; the frames themselves have no
        # phase and cannot be turned back into sound.
        said = getattr(self.teacher, "saidPcm", None)
        if said is not None and len(said):
            mom = getattr(self, "momRing", None)
            if mom is None:
                from collections import deque as _dq
                mom = self.momRing = _dq(maxlen=30)
            mom.append((len(self.ticks),
                        np.asarray(said, np.float32).copy()))
            self.teacher.saidPcm = None
        # THE ARMED BOTTLE.  When her felt hunger (the outside brain's own
        # number, posted to /felt) rises past the level HE chose, milk is at
        # her lips again the moment the last mouthful is gone --- his hand,
        # exactly as he would have pressed it, while he is away.
        # THE MIND WATCHDOG.  Her mind is a subprocess started once; a
        # crash at four in the morning used to mean a body living on with
        # nobody home until he woke --- "i whant to be sur thet i will
        # not loose time by night" (his, going to bed, 2026-08-28).
        # Once every ~30 s of ticks: if the mind process has died, it is
        # started again on the same life --- the wake machinery already
        # makes that a resume, not a loss.
        if (len(self.ticks) % 900 == 0
                and getattr(self, "mindDb", None)
                and getattr(self, "mindProc", None) is not None
                and self.mindProc.poll() is not None):
            mindStart(self.mindDb)
        if getattr(self, "bottle", None) is not None:
            self._bottleKeeps()
        want = getattr(self, "autofeed", None)
        if want is not None:
            hungry = (getattr(self, "felt", None) or {}).get("hunger")
            if hungry is not None and float(hungry) >= float(want) \
                    and self.atMouth <= 0.01:
                self.give(1.0)
        # ONE FRAME IS ONE TICK.  Her clock is the only clock: at 30 a second
        # the picture and her life are the same thing, and there is nothing to
        # interpolate and nothing to drift.
        pose = np.asarray(self.her.pos, np.float32)[
            [JIDX[n] for n in JOINTS]].ravel()
        w = self.window
        ride = list(np.asarray(w.at, np.float32)) + \
            list(np.asarray(w.look, np.float32)) + list(w.size)
        # THE PAGE SEES EVERY THIRD TICK (his order, 2026-09-04): she lives at 90 a
        # second and a browser on a 60 Hz screen cannot draw 90, so it stalled;
        # the page gets PAGE_FPS pictures a second, her clock untouched.
        if len(self.ticks) % PAGE_EVERY == 0:
            row = film.encode(list(pose) + [float(v) for v in ride])
            with self.reel:
                self.frames.append(row)
                if len(self.frames) > self.keep:
                    drop = len(self.frames) - self.keep
                    self.frames = self.frames[drop:]
                    self.first += drop

    # --- what the page asks for -------------------------------------------        # ...AND WHAT HER EARS REACHED, HELD FOR THE PAGE.  Hers is the tick;
        # the panel's is whenever it asks, and the two are 15 ticks apart.
        got = np.asarray(self.ears, np.float32)
        if got.size:
            np.maximum(self.earsPeak, got.max(axis=2), out=self.earsPeak)
        mine = np.asarray(getattr(self, "voiced", None), np.float32)
        if mine.size == SOUND_BANDS:
            np.maximum(self.echoPeak, mine, out=self.echoPeak)

    def meta(self) -> dict:
        return {
            "joints": JOINTS,
            #: WHERE HER EYE RUNS --- "gpu" or "cpu" (his, 2026-09-07: "I shouldn't every time [check]
            #: that she has to start from GPU"); her.py prints it at every start
            "eye": "gpu" if light._gpu is not None else "cpu",
            "radius": [float(r) for r in self.her.radius],
            "bones": [list(b) for b in DRAWN_BONES],
            "suit": SUIT_BONES,
            "room": self.room.as_json(),
            # HER CLOCK, AND THE PICTURE'S ARE THE SAME.  One frame a tick.
            "fps": PAGE_FPS, "per_tick": 1,
            "tick_seconds": TICK_SECONDS * PAGE_EVERY,     # one picture spans PAGE_EVERY of her ticks
            "wide": WIDE, "scale": film.SCALE, "window_values": film.WINDOW_VALUES,
            # `live` FALSE ON PURPOSE, and it is not about whether she is
            # alive.  In the page's `live` mode you are meant to be looking
            # THROUGH the television --- the browser window IS the screen ---
            # so it never draws the panel.  Here the screen is a thing in her
            # room that he wants to SEE, with its own placement, and the page
            # draws it from the eight numbers every frame carries: where it is,
            # where it looks, and how big it is.
            #
            # It also gates a buffering path that needs `STATE.driven`, which
            # nothing here sends: one frame is one tick and there is nothing to
            # buffer.
            # HOW FAR BEHIND HER THE PAGE DRAWS, IN SECONDS.
            #
            # `BUFFER_FRAMES = ceil(lead * fps)`, and I wrote 2 here --- which
            # at her 30 a second is **SIXTY FRAMES, two whole seconds**, so
            # every drag of her hand answered two seconds later.  He felt it and
            # named the cause without seeing the code: *"it moves with huge
            # delay and i gues it is old clock againe"*.  At the old 1.5 s tick
            # `lead: 2` was about one frame; at 33 ms it is sixty.  The fourth
            # constant to mean one thing at 1.5 s and another at 33 ms.
            #
            # 0.2 is the page's own documented cushion --- *"lead is untouched
            # at 0.2: it is the app's own cushion and the page's minimum"* ---
            # and its warning about 6 frames freezing applies only to a driven
            # ride with bursts (`META.live && STATE.driven`), which this is not:
            # her frames arrive one a tick, evenly, at 30 a second.
            "lead": 0.2, "live": False, "loop": False, "seed": 0,
            "oldest": self.first, "latest": self.first + len(self.frames),
        }

    def wholeTick(self) -> dict:
        """ONE WHOLE TICK, EVERY FIELD OF IT, exactly as she stored it.

        His: *"show me whole tick with stored objects and rest"*.

        `asdict` walks her own dataclasses --- `Tick`, `Life`, `Input`,
        `Output`, and every `Sensor` and `View` in them --- so this cannot drift
        away from her shape: add a field to `View` and it appears here with no
        change anywhere.  A hand-written list of what to show would be a second
        copy of her structure and would go stale the first time he changed it.

        NOTHING OF HERS IS TOUCHED AND NOTHING IN `brain/` IS IMPORTED FOR IT.
        It is a read under her lock, and `asdict` copies.
        """
        from dataclasses import asdict
        got = asdict(self.tick)
        got["at"] = len(self.ticks)
        got["seconds"] = round(len(self.ticks) * TICK_SECONDS, 3)
        return got

    def stateNow(self) -> dict:
        """The last snapshot, handed over without touching her."""
        return self.shot.get("state") or self.state()

    def state(self) -> dict:
        life = self.tick.life
        got = {
            "tick": len(self.ticks), "life": "matilda",
            "seconds": round(len(self.ticks) * TICK_SECONDS, 1),
            # HER CHEMISTRY IS HER MIND'S.  It arrives by /felt and overlays
            # below; before the first post there is nothing to show.
            "blood": {},
            # WHOSE HANDS ARE ON HER, truthfully --- it was hardcoded None,
            # so "did my swaddle arrive?" had no answer at all.
            "helper": self.helper,
            "floor": self.floorMoves,
            "belt": ([round(float(self._beltDir[0]), 1),
                      round(float(self._beltDir[2]), 1)]
                     if (self.floorMoves is True
                         or (self.floorMoves is None
                             and (self.helper == "walk"
                                  or bool((self.her.pinned
                                           & ~self._helperPins).any()))))
                     else None),
            "teacher": {"on": bool(self.teacher.on),
                        "saying": bool(self.teacher.saying),
                        "names": sorted(self.teacher._namesHis),
                        #: the word she is saying right now, and its sounds
                        #: --- drawing and measuring data, never hers
                        "word": getattr(self.teacher, "word", None),
                        "wordIds": list(getattr(self.teacher, "wordIds", ())),
                        "meter": {"answers": self.teacher.answers,
                                  "shapes": self.teacher.shapes,
                                  "milks": self.teacher.milks,
                                  "named": getattr(self.teacher, "named", 0),
                                  "namedWhat": dict(getattr(self.teacher, "namedWhat", {}) or {}),
                                  "rescues": self.teacher.rescues,
                                  "lifts": getattr(self.teacher, "lifts", 0)},
                        "joints": (self.teacher.shape()
                                   if self.teacher.on else None)},
            #: how many times a loud sound has turned her --- glass, never hers
            "startles": int(getattr(self, "startles", 0)),
            #: WHAT HER GAZE HOLDS --- the room's own geometry (`_looking`), the one
            #: answer to "where is she looking" (rule 4), for the doctor's glass:
            #: a looking-time test reads it the way a doctor reads a baby's eyes.
            #: Nothing of hers reads it.
            "looking": (lambda v: None if v is None else str(v))(self._looking()),
            #: ...and the geometry it is read from --- her head, and her eyes' axis
            #: (`_glimpse`, the one answer), so a doctor can hold a card in her
            #: line of sight instead of where a constant says her face was
            "gaze": (lambda h, g: {"head": [round(float(v), 4) for v in h], "dir": [round(float(v), 4) for v in g]})(
                self.her._glimpse()[0], self.her._glimpse()[3]),
            "turns": int(getattr(self, "turns", 0)),      # her eyes turned toward a sound that rose
            "held": {"pins": int(self.her.pinned.sum()),
                     "carried": int(self.her.carrying.sum()),
                     "touch": self.touch is not None,
                     "ago": (len(self.ticks) - self.heldAt
                             if self.heldAt >= 0 else -1)},
            "room": self.room.as_json(),
            "window": {"at": [float(v) for v in self.window.at],
                       "look": [float(v) for v in self.window.look],
                       "up": [float(v) for v in self.window.up],
                       "size": list(self.window.size),
                       "queued": round(self.window.queued(), 3),
                       "late": round(self.window.late(), 3)},
            "state": round(float(life.input.state), 4),
            # HOW HUNGRY SHE IS, BY NAME, BECAUSE THE PAGE CANNOT KNOW AN ID.
            #
            # `blood` goes out keyed by hormone id --- his structure, *"lets
            # project whole hormons to inputs"* --- and the page was still
            # reading `blood.hunger`, `blood.comfort` and `blood.boredom`, the
            # names of a `blood.py` this rewrite DELETED.  Every one of them
            # came back `undefined`, so **both her gauges have been dead since
            # the rewrite**: the stomach ring sat full and the heart ring sat
            # green whatever she felt.
            #
            # The page is a mirror of finished numbers, so the naming happens
            # here, once, where `NEED` is already known.
            "hunger": round(float((getattr(self, "felt", None) or {}).get("hunger", 0.0)), 4),
            "saying": int(life.output.sound.id),
            "hearing": int(life.input.sound.id),
            # WHAT REACHED EACH EAR, one number a band --- and this is what the
            # bars have been drawing nothing from.
            #
            # The page's own comment says *"Every number is `/state.ears`,
            # reduced on the server (`_ear_bars`)"*.  **There is no `_ear_bars`
            # anywhere in this tree and `/state` has never carried `ears`**, so
            # `L` and `R` came back undefined, `n` was 0, and the panel printed
            # "silence" on every tick since the page was ported --- the same
            # shape as the gauges reading `blood.hunger`, a name from a file
            # this rewrite deleted.  A sense with an instrument that is wired to
            # nothing is worse than one with none: it reports the good news.
            #
            # LOUDEST OVER THE SLIDES, not the mean: a sound is measured by its
            # loudest reading, which is `alike.one_ear`'s rule and not a second
            # one invented here.  Two rows, because left against right IS her
            # sense of direction.
            "ears": _earBars(self),
            # HER MOUTH, BOTH ENDS OF IT --- *"does she hear her self"*.
            #
            # `said` is what she COMMANDED and `echo` is what her mouth ANSWERED
            # WITH: her voice's spindle, the same shape as `motor -> spindle`.
            # They are the same id whenever her body has a mouth for that sound;
            # `echo 0` against a `said` that is not 0 means she asked for a sound
            # her mouth cannot make, and `back 0.0` means she opened her mouth
            # and nothing came out.
            #
            # Both were on this wire as bare ids with no levels and NOTHING IN
            # THE PAGE HAS EVER READ EITHER.  The sense that carries her own
            # voice had no instrument on it at all.
            "voice": {"said": int(life.output.sound.id),
                      "loud": round(float(life.output.sound.lvl), 4),
                      "echo": int(life.input.echo.id),
                      "back": round(float(life.input.echo.lvl), 5),
                      # A COMMAND AT ZERO IS REST, NOT INABILITY.  The
                      # page's "her mouth CANNOT make X" fired on every
                      # fading command (@ 0.00 makes no sound, so echo
                      # never matches) --- the owner's screenshot,
                      # 2026-08-31, caught the accusation on a sound she
                      # voiced an hour earlier.  Silent commands are hers.
                      "mine": (int(life.input.echo.id)
                               == int(life.output.sound.id)
                               or float(life.output.sound.lvl) <= 0.0)},
            # THE STRONGEST MUSCLE SHE IS HOLDING, or 0 at rest.  `motor` is
            # every muscle now (his "all muscles all inputs"); one id kept the
            # panel's shape --- and reading `.id` off the list killed the live
            # thread at its first snap, silently, twice on 2026-08-26.
            "moving": (int(max(life.output.motor, key=lambda m: float(m.lvl)).id)
                       if any(float(m.lvl) > 0.0 for m in life.output.motor)
                       else 0),
            "things": len(life.input.view),
            "runs": int((getattr(self, "felt", None) or {}).get("runs", 0)),
            # THE TELEVISION, AS A COUNT AND NOT AN IMPRESSION.  `flew` is
            # every post that arrived, `shows` the ones that carried a picture,
            # and `ago` how many ticks since the last one landed.  A frozen
            # panel with `ago` climbing is the page not posting; a frozen panel
            # with `ago` at 0 is the panel riding his camera and hanging where
            # he stopped, which is a different bug entirely.
            "tv": {"flew": self.flew, "shows": self.shows,
                   "ago": (len(self.ticks) - self.showsAt) if self.showsAt >= 0
                          else -1},
            # HOW FAR BEHIND HER OWN CLOCK SHE IS.  Zero is keeping up; a
            # number here is lateness he can see, which is always better than a
            # silent speed-up that makes her seconds not seconds.
            "behind": round(self.behind, 4),
            "mindPid": (getattr(getattr(self, "mindProc", None), "pid", None)),
            # WHERE HER TICK GOES, ms, a running mean over ten of her seconds
            "stages": {k: round(float(v), 2)
                       for k, v in sorted(getattr(self, "stageMs", {}).items())},
            # HOW MUCH OF HER A DECISION MOVED, and how late it came.
            # `landed`: muscle and voice lines the last act wrote (one for a
            # try, all of her for a replayed moment).  `late`: mean ticks
            # between the moment a whole plan answered and the tick it
            # landed on --- her mind's lag, measured, never hidden.
            "landed": int(getattr(self, "landed", 0)),
            "late": round(self.lateSum / self.lateN, 2) if self.lateN else None,
            "autofeed": getattr(self, "autofeed", None),
            "bottle": (None if getattr(self, "bottle", None) is None else
                       {"high": round(float(self.bottle["high"]), 3),
                        "got": int(self.bottle.get("got", 0))}),
            "tunnel": getattr(self, "tunnelUrl", None),
        }
        # THE PANEL MUST NOT LIE.  Her chemistry and her state live in her
        # MIND; the body has none, so the numbers above would show a frozen
        # never-hungry girl --- and that watcher is the owner.  Her mind
        # posts its finished numbers to /felt and they overlay here, a
        # mirror as always.
        if getattr(self, "felt", None):
            got.update({k: v for k, v in self.felt.items()
                        if k in ("blood", "state", "hunger", "runs",
                                 "exp", "seconds",
                                 "guided", "laddered", "closes",
                                 "replays", "recognised", "expecting",
                                 "feelMs", "closedHow", "news", "alarms", "recorded", "pending", "leaves", "reach", "chains")})
        else:
            # HER AGE IS HERS, NEVER THE BODY'S.  While her mind is still
            # remembering, the body's continued clock leaked into
            # `seconds` and the panel showed 34 hours on an 11-hour girl
            # --- "i see 34 hours insted of 11" (his, 2026-08-28).  An
            # age nobody knows yet is None, and the page says "waking".
            got["seconds"] = None
        return got

    def eye(self) -> dict:
        """HER RETINA --- COPIED UNDER HER LOCK, FORMATTED OUTSIDE IT.

        The lock used to be held across the whole handler: the downsample, the
        base64 of 27 KB, the JSON, and the socket write.  A browser that read
        slowly held her body still while it did.  It is held here for the copy
        and for nothing else.
        """
        with self.lock:
            pic = None if self.picture is None else np.asarray(self.picture)
            seen = np.asarray(self.seen) if len(self.seen) else None
            ears = np.asarray(self.ears)
            tick = len(self.ticks)
            named, namedAt = list(self.named), self.namedAt
        return self._eyeOf(pic, seen, ears, tick, named, namedAt)

    def _eyeOf(self, pic, seen, ears, tick, named, namedAt) -> dict:
        """WHAT SHE MADE OF IT --- in the shape the page reads.

        `shot` is base64 RGB at `w x h`, `things` are her bound things with a
        box each, `ears` are her two ears' band levels.  Her body computes all
        of it; the page puts it down.

        A BOX CENTRE IS -1..1 AND ITS SIZE IS A FRACTION OF THE VIEW --- two
        units in one row, settled 2026-08-15, and half a box in pixels is
        `w * width * 0.5`.  Nothing downstream compensates for anything.
        """
        n = 96
        if pic is None:
            return {"tick": tick, "w": 0, "h": 0,
                    "shot": "", "things": [], "found": 0, "gain": 0,
                    "named": [], "namedAt": -1, "ears": [[], []]}
        step = max(1, pic.shape[0] // n)
        small = pic[::step, ::step, :]
        px = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        # BIGGEST FIRST, AND THE CUT KEEPS THE BIGGEST.  `bind` hands them back
        # in whatever order it built them, and this sliced [:64] off that order
        # and sent it on --- so the panel's green box was not her biggest thing
        # and its headline read whatever happened to be first.  Measured
        # 2026-08-25: it printed *biggest 0.2% of field* while her real biggest
        # was 76.3%, and that is the number he has been reading all evening.
        # Sorted HERE, because the page is a mirror of finished numbers.
        rows = sorted((seen if seen is not None else []),
                      key=lambda r: -float(r[bind.AREA]))
        things = [[float(r[bind.X]), float(r[bind.Y]), float(r[bind.AREA]),
                   float(r[3]), float(r[4]), int(r[bind.LOOKS])]
                  for r in rows[:64]]
        return {"tick": len(self.ticks),
                # WHAT IS REALLY THERE, over the picture it was measured on.
                # `namedAt` is that picture's tick: her retina is remade every
                # 200 ms and the ruler answers in 40, so this is the current
                # one --- but it is sent rather than assumed, because the old
                # tree drew camera-coordinate boxes over her retina for weeks.
                "named": self.named, "namedAt": self.namedAt,
                "w": int(px.shape[1]), "h": int(px.shape[0]),
                "shot": base64.b64encode(px.reshape(-1).tobytes()).decode(),
                "things": things,
                "found": 0 if seen is None else len(seen),
                "gain": round(float(pic.max()), 3),
                "ears": [[round(float(v), 4) for v in ears[0].max(axis=1)],
                         [round(float(v), 4) for v in ears[1].max(axis=1)]]}

    def said(self) -> dict:
        """HER MOUTH, OUT LOUD --- the real samples, with the phase.

        The same samples her room carries, not a rendering of a spectrogram:
        `bands_from_pcm` is downstream of this, so what you hear is what her
        ears are about to split, before anything reduced it, and **with the
        phase**, which her tape does not keep and no resynthesis can invent.

        THE PAGE MUST NOT BUILD A VOICE.  It once made 24 sines at her band
        centres, and he heard it and reported it as her: *"she sounds like
        piano, i can't recognise any of my words"*.  Her constants were never
        missing --- `speech.py` is a source-filter mouth and `Voice.pcm` has
        kept its real samples since it was written.  Nobody was asking for them,
        and this server was not either: `/said` was a stub returning nothing,
        which is why he could not hear her at all.

        Packed like her eye and his camera: deflated int16, base64.  Drawing
        data --- she cannot read it, and no part of her changes because somebody
        listened.

        HER LOCK IS TAKEN FOR THE COPY AND NOT FOR THE DEFLATE, the same as her
        eye --- and it IS taken, because her mouth is being written by her tick
        while this reads it.
        """
        with self.lock:
            ring = list(getattr(self, "saidRing", ()))
            tick = len(self.ticks)
        after = int(getattr(self, "_saidAfter", -1))
        got = [c for t2, c in ring if t2 > after] if after >= 0 else             [ring[-1][1]] if ring else []
        pcm = (np.clip(np.concatenate(got), -1.0, 1.0)
               if got else np.zeros(0, np.float32))
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        z = zlib.compressobj(6, zlib.DEFLATED, 15)
        blob = z.compress(raw) + z.flush()
        return {"tick": tick,
                "rate": int(speech.RATE),
                "loudest": round(float(np.abs(pcm).max()) if pcm.size else 0.0, 5),
                "n": int(pcm.size),
                "z": base64.b64encode(blob).decode()}

    def flying(self, body: dict) -> None:
        """YOU, MOVING AND SHOWING AND SPEAKING --- one post, as the page sends it.

        Raw pixels and raw samples: **her body does every reduction**, because
        a browser that decided what a pixel or a direction meant would be a
        second implementation of her body.

        `up` is which way is up on YOUR screen, squared against `look` by
        `window.upright`.  Send none and the window takes its roll from the
        world's up --- fine right up until you lean over her and look down,
        which is 3.17 degrees of roll per degree of wobble where you hover.

        THIS WAS THE WHOLE OF "she still doesn't see me".  The page has posted
        here since it was written; this server answered `{}` and threw it away.
        """
        with self.lock:
            self.flew += 1
            self.window.fly(body.get("at"), body.get("look"),
                            body.get("size"), body.get("up"))
            shows = body.get("shows")
            if isinstance(shows, dict) and shows.get("z"):
                self.shows += 1
                self.showsAt = len(self.ticks)
                raw = zlib.decompress(base64.b64decode(shows["z"]))
                pic = np.frombuffer(raw, np.uint8).astype(np.float32) / 255.0
                self.window.show(pic.reshape(int(shows["h"]), int(shows["w"]), 3))
            says = body.get("says")
            if isinstance(says, dict) and says.get("pcm"):
                pcm = np.asarray(says["pcm"], np.float32)
                rate = float(says.get("rate", speech.RATE))
                self.window.say(pcm, rate)
                # ...AND KEPT, FOR HIM.  One tick later her ears have turned
                # this into a band split and an id, and the waveform is gone.
                self.yours.add(pcm, rate)

    def snap(self) -> None:
        with self.lock:
            self.shot = {"state": self.state()}
        # ...AND WHOEVER IS WAITING FOR HER NEXT MOMENT NOW HAS IT.  This runs
        # once at the end of every tick, outside her lock --- see `LIVED`.
        with LIVED:
            LIVED.notify_all()

    def mom(self) -> dict:
        """HER MOTHER, OUT LOUD --- for his ear, never for hers.

        The same packing as `said`: deflated int16, base64.  What comes back
        is HIS OWN RECORDED WORD, which is what her mother's frames were made
        from --- her mother does not speak in samples and never has, so this
        is the source and not a rendering.  `word` says which one it was.
        """
        with self.lock:
            ring = list(getattr(self, "momRing", ()))
            tick = len(self.ticks)
            word = getattr(self.teacher, "word", None)
        after = int(getattr(self, "_momAfter", -1))
        got = [c for t2, c in ring if t2 > after] if after >= 0 else \
            ([ring[-1][1]] if ring else [])
        pcm = (np.clip(np.concatenate(got), -1.0, 1.0)
               if got else np.zeros(0, np.float32))
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        z = zlib.compressobj(6, zlib.DEFLATED, 15)
        blob = z.compress(raw) + z.flush()
        return {"tick": tick, "rate": int(speech.RATE), "word": word,
                "loudest": round(float(np.abs(pcm).max()) if pcm.size else 0.0, 4),
                "z": base64.b64encode(blob).decode("ascii")}

    #: HIS BOTTLE, IN HER ROOM AND ON A BUTTON --- his, 2026-09-08: *"bottle
    #: has to be able in helpers and if i turn on it has to appear without any
    #: other requirement ... if she didn't reach bottle it lowering till floor
    #: but if she start to grab it fast it start to rise"*.
    #:
    #: It is the world's, not hers: her room holds it, her eye sees it, her own
    #: hand or mouth reaching it is what feeds her, and it climbs or sinks by
    #: how well she is doing.  Nothing here is in her mind and nothing tells her
    #: anything --- she gets milk at her lips, which is a thing her body has
    #: always had.  These numbers are the room's, the way her mother's laws are
    #: hers: her reach, a mouthful's share, and her own resolution as the step.
    BOTTLE_REACH = 0.10          # m: her hand or face this close is a reach
    BOTTLE_MILK = 0.3            # of a mouthful, each time she gets it
    BOTTLE_REST = 3.0            # s before it can feed her again
    BOTTLE_FLOOR = 0.06          # m: as low as it ever sits
    #: HER STOMACH IS THE LADDER, NOT A CLOCK --- his, 2026-09-10: *"if she do
    #: not grab till her 10% of hungry put it again close to her in her view on
    #: floor ... if she start to grab it faster than she filled her stomach put
    #: bottle in same way but higher, and again stomach empty to 10% lower
    #: height"*.  The two clocks that stood here (60 s to rise, 120 s to sink)
    #: were mine and measured nothing about her; a tenth of a stomach is his
    #: number and is read off the hunger she actually feels.
    BOTTLE_LOW = 0.10            # of a stomach left: the bottle comes back

    def _bottlePlace(self) -> None:
        """Somewhere new, at this height, in front of her and never in her bed."""
        b = self.bottle
        chest = np.asarray(self.her.pos[JIDX["chest"]], np.float32)
        try:
            eye, _, _, fwd = self.her._glimpse()
            face = np.array([float(fwd[0]), 0.0, float(fwd[2])], np.float32)
        except Exception:                                       # noqa: BLE001
            face = np.array([1.0, 0.0, 0.0], np.float32)
        n9 = float(np.linalg.norm(face))
        face = face / n9 if n9 > 1e-6 else np.array([1.0, 0.0, 0.0], np.float32)
        last = b.get("at")
        # IN FRONT OF HER FACE --- his, 2026-09-10, after a screenshot of it
        # sitting ON HER SHOULDER.  It used to be swung 20 to 70 degrees OFF
        # her facing before it was placed, so it never was in front of her at
        # any distance: it landed beside her, on the floor, out of her view.
        # Her own looking direction now, and nothing added to it.
        for _ in range(30):
            r = float(b["rng"].uniform(0.30, 0.45))
            at = chest + face * r
            at[1] = float(b["high"])
            if (abs(at[0]) < BED["x"] + 0.06 and abs(at[2]) < BED["z"] + 0.06
                    and at[1] < BED["top"]):
                continue                                        # never inside her cot
            if last is None or float(np.linalg.norm(at - np.asarray(last))) >= 0.25:
                break
        b["at"] = [float(v) for v in at]
        # ...AND NOT UNDER HER LOCK.  The keeper is called from inside her own
        # tick, which already holds it, and her lock is a plain one --- taking
        # it a second time froze her body dead (2026-09-08).
        self.room.put("bottle", tuple(float(v) for v in at), size=0.06)

    def _bottleKeeps(self) -> None:
        """One tick of the bottle: has she reached it, and where should it be."""
        b = self.bottle
        now = len(self.ticks) * TICK_SECONDS
        if not b.get("at"):
            b["put"] = now
            self._bottlePlace()
            return
        at = np.asarray(b["at"], np.float32)
        near = min(float(np.linalg.norm(np.asarray(self.her.pos[JIDX[j]], np.float32) - at))
                   for j in ("haR", "haL", "face"))
        # HER STOMACH, AS SHE FEELS IT.  1.0 is empty, so a tenth left is
        # `1 - BOTTLE_LOW`.  Her mind is the only thing that knows it and it
        # already reaches her body through `felt`.
        hungry = float((getattr(self, "felt", None) or {}).get("hunger", 0.0))
        empty = hungry >= 1.0 - self.BOTTLE_LOW
        if near <= self.BOTTLE_REACH and now - float(b.get("fed", -99.0)) >= self.BOTTLE_REST:
            # SHE GRABBED IT.  Milk at her lips --- and if she got there before
            # her stomach fell to its tenth, she is finding them faster than
            # she empties, so the next one stands HIGHER (his word).
            self.give(self.BOTTLE_MILK)
            b["fed"], b["got"] = now, int(b.get("got", 0)) + 1
            if not b.get("dropped"):
                b["high"] = float(b["high"]) + float(self.resolution)
            b["dropped"] = False
            b["put"] = now
            self._bottlePlace()
            return
        if empty and not b.get("dropped"):
            # HER STOMACH IS DOWN TO A TENTH AND SHE HAS NOT GRABBED IT.  It
            # comes back in front of her face, a step LOWER (his word) --- and
            # once only, until she feeds again, so it never chases her while
            # she is still working toward it.
            # ...AND IT STAYS
            # WHERE IT IS --- his, 2026-09-08: *"bottle shouldn't move before
            # she catch it"*.  A thing that moves while she is still working
            # toward it is a different thing every time she looks, and nothing
            # she does can bring her nearer to it.  Only reaching it moves it.
            b["high"] = max(self.BOTTLE_FLOOR,
                            float(b["high"]) - float(self.resolution))
            b["dropped"] = True
            b["put"] = now
            self._bottlePlace()

    def framesFrom(self, k: int) -> bytes:
        """The frames since `k`.  **HER LOCK IS NOT INVOLVED** --- see `reel`."""
        with self.reel:
            at = max(0, min(k, self.first + len(self.frames)) - self.first)
            first, last = self.first + at, self.first + len(self.frames)
            rows = self.frames[at:]
        return film.pack(first, last, rows, WIDE)


HER: Watched | None = None


#: SHE HAS LIVED ANOTHER ONE --- notified at the end of every tick, waited on
#: by `/ticks`.  His, 2026-09-01: *"all that information is stored in our tick,
#: and it means all that information can be passed by one request and then
#: parse it"*, and then *"it has the disconnection like chats use ... if I send
#: someone message and she see it in same moment"*.
#:
#: HER MIND USED TO GUESS WHEN TO ASK.  It slept `TICK_SECONDS / 2` and asked
#: again --- twice for every tick that existed, so more than half its asks came
#: back empty, and **every one of them took the interpreter away from her
#: body's thread**.  Measured 2026-09-01: her body alone 19.66 ms a tick, and
#: 23.50 with her server asked at that rate --- 16% of her, none of which the
#: server's own `/asked` could see, because it counts time INSIDE the handler
#: and the cost is her thread stopping outside it.
#:
#: A sleep is a guess at a rate and there is no right value: too fast wastes,
#: too slow makes her act late.  So the body simply does not answer until there
#: is something to say.  **A thread waiting on a condition holds no
#: interpreter**, so waiting is free in a way polling can never be, and her
#: mind gets the tick the instant it exists rather than up to half a tick
#: later.
LIVED = threading.Condition()
#: ...and how long a wait may last before answering with nothing anyway, so a
#: body that has stopped ticking cannot hang her mind.
WAIT_SECONDS = 1.0

#: WHAT THE PAGE ACTUALLY ASKS FOR, and what answering costs.  "the watcher is
#: expensive" is not a finding until it says WHICH part --- three guesses died
#: before this existed: her preparing data (0.099 ms a tick), the lock across
#: `_send` (worth 1.5 ticks a second), and the ruler (worth 3).
ASKED: dict = {}


class Page(BaseHTTPRequestHandler):
    # HTTP/1.1, SO A CONNECTION IS KEPT --- profiled 2026-08-30: the mind
    # spent 73% of every tick opening a fresh TCP connection to this
    # server (5,314 connects in 150 s, ~10 ms each), because HTTP/1.0
    # closes after every request.  Every response already carries
    # Content-Length (the one write path is `_send`), which is all 1.1
    # needs.  The page's own fetches get the same free ride.
    protocol_version = "HTTP/1.1"

    def _send(self, obj, kind="application/json"):
        row = ASKED.setdefault(self.path.split("?")[0], [0, 0.0, 0])
        row[0] += 1
        row[2] += len(obj) if isinstance(obj, (bytes, str)) else 0
        if isinstance(obj, (bytes, bytearray)):
            body = bytes(obj)
        else:
            body = (json.dumps(obj) if kind.endswith("json") else obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        # NEVER CACHED.  Twice in one night "I don't see the new button" was
        # the browser serving yesterday's page over today's server.  This
        # page is a mirror of a living girl; a stale mirror is a lie.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        began = _now()
        where = urlparse(self.path)
        q = parse_qs(where.query)
        p = where.path
        # NOTHING IS SENT WITH HER LOCK IN HAND.  Each of these takes it for as
        # long as it takes to COPY what it needs and gives it straight back; the
        # JSON, the base64, the deflate and the socket write all happen outside.
        # It used to wrap this whole block --- so a slow browser held her body
        # still while it read, which is the same fault as the tick loop holding
        # the lock for a whole tick, and that one cost 13 ticks a second.
        #
        # AND IT DEADLOCKED HER PAGE.  Once `eye` and `said` took the lock for
        # their own copy, this outer one made it twice on one thread --- and
        # `threading.Lock` is not reentrant, so `/eye` hung for ever with no
        # error anywhere.  Her clock kept running, `/state` kept answering, and
        # only asking for her retina fell into a hole.
        #
        # `/state` takes NO lock at all: `stateNow` is the snapshot her own loop
        # left at the end of its last tick.
        with _nolock():
            if p == "/meta":
                with HER.lock:
                    got = HER.meta()
                # A ROAD NEEDS A CUSHION.  Through the tunnel the frame
                # buffer starved and she BLINKED between poses --- his
                # diagnosis, watching: "she is dont move jus tonel raframe
                # is wrong".  A viewer arriving through cloudflared gets a
                # full second of lead: one second later, and smooth.
                if self.headers.get("Cf-Connecting-Ip"):
                    got["lead"] = 1.0
                self._send(got)
            elif p == "/state":
                got = HER.stateNow()
                self._send(got)
                # ...AND THE PEAK STARTS AGAIN.  It holds the loudest her ears
                # reached since this was last asked, so a panel polling every
                # 500 ms misses none of a 33 ms tick.  Reset on the READ and
                # never on the tick: resetting per tick is what made it show one
                # tick in fifteen.
                seen = getattr(HER, "earsSeen", None)
                if seen is not None:
                    seen()
            elif p == "/frames":
                self._send(HER.framesFrom(int(q.get("from", ["0"])[0])),
                           "application/octet-stream")
            elif p == "/eye":
                self._send(HER.eye())
            elif p == "/tick":
                with HER.lock:
                    got = HER.wholeTick()
                self._send(got)
            elif p == "/ticks":
                # SINCE N, OFF HER LOCK.  A finished tick is immutable ---
                # the tape's own contract, "nothing touches a tick once the
                # next one has begun" --- so this thread serializes frozen
                # data while she runs free.  The newest tick is still being
                # lived and is not served; moments older than the RAM window
                # are on the tape file, which is her writer's handle, not
                # this thread's, so the window's oldest is the floor.
                from dataclasses import asdict as _asdict
                t = HER.ticks
                end = len(t) - 1
                ask = int(q.get("from", [str(max(0, end))])[0])
                # ...AND IF THERE IS NOTHING YET, WAIT FOR HER RATHER THAN
                # SAYING SO.  `wait=1` is her mind; the page never sets it and
                # is answered at once as before.  See `LIVED` for why a wait
                # costs her nothing and a poll costs her 16%.
                if q.get("wait") and ask >= end:
                    with LIVED:
                        LIVED.wait(timeout=WAIT_SECONDS)
                    t = HER.ticks
                    end = len(t) - 1
                oldest = max(0, end - len(t._ram) + 1)
                at = max(ask, oldest)
                rows = []
                for k in range(at, end):
                    one = t._ram.get(k)
                    if one is None:
                        at, rows = k + 1, []
                        continue
                    rows.append(_asdict(one))
                self._send({"from": at, "next": at + len(rows),
                            "ticks": rows})
            elif p == "/lives":
                # HER LIVES, FOR THE MENU --- his, 2026-08-27: "add to menue
                # one more menu to save load and new life".  Every life she
                # has ever lived, newest first, and which one is living now.
                rows = []
                mind = getattr(HER, "mindDir", None)
                if mind:
                    lives = os.path.join(mind, "lives")
                    named = [f for f in os.listdir(lives)
                             if f.endswith(".duckdb")]
                    named.sort(key=lambda f: os.stat(
                        os.path.join(lives, f)).st_mtime, reverse=True)
                    for f in named:
                        st = os.stat(os.path.join(lives, f))
                        rows.append({"name": f,
                                     "mb": round(st.st_size / 1e6, 1),
                                     "living": f == getattr(HER, "mindDb",
                                                            None)})
                self._send({"lives": rows[:12],
                            "managed": mind is not None})
            elif p == "/able":
                # WHAT SHE IS BUILT ABLE TO DO --- body facts, data out, so an
                # outside brain never hardcodes her.  His, 2026-08-26:
                # "sandbox is data in app not in browser".
                self._send({"motors": int(HER.motors),
                            "resolution": float(HER.resolution),
                            # HOW MANY OF HER MOTORS ARE HER VOICE --- her body's
                            # own word about herself, so her mind can leave them
                            # out of the sweep without knowing what a muscle IS
                            # (his, 2026-09-08: discovery of sounds goes another way).
                            "voiced": len(VOICE_PARTS),
                            "gap": float(HER.gap),
                            "sounds": [int(s) for s in HER.sounds]})
            elif p == "/mind":
                # EVERYTHING SHE IS THINKING, READ OFF A RUNNING GIRL.  His:
                # *"now everithing in live we can read from values without
                # stopping her?"*  `Her.mind` has existed since her body was
                # assembled and NOTHING SERVED IT --- it died when the second
                # server went, an hour after the rule about exactly this was
                # written down.  It is a read: her lock is taken and no field
                # of hers is touched.
                with HER.lock:
                    got = HER.mind()
                self._send(got)
            elif p == "/mom":
                q = parse_qs(where.query)
                try:
                    HER._momAfter = int(q.get("after", ["-1"])[0])
                except (TypeError, ValueError):
                    HER._momAfter = -1
                self._send(HER.mom())
            elif p == "/said":
                q = parse_qs(where.query)
                try:
                    HER._saidAfter = int(q.get("after", ["-1"])[0])
                except (TypeError, ValueError):
                    HER._saidAfter = -1
                self._send(HER.said())
            elif p == "/window":
                with HER.lock:
                    got = HER.state()["window"]
                self._send(got)
            elif p == "/seat":
                self._send({"seat": "owner", "agreed": True, "token": "her"})
            elif p == "/heard":
                self._send({"seconds": round(HER.yours.seconds, 2),
                            "rate": HER.yours.rate, "file": HER.yours.where})
            elif p == "/asked":
                self._send({k: {"times": v[0], "ms": round(v[1], 1),
                                "bytes": v[2]} for k, v in sorted(ASKED.items())})
            elif p in ("/log", "/agree", "/mute", "/world", "/shown"):
                self._send({})
            else:
                self._send((HERE / "web" / "index.html").read_text("utf-8"),
                           "text/html; charset=utf-8")
        row = ASKED.setdefault(p, [0, 0.0, 0])
        row[1] += (_now() - began) * 1000.0

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        p = urlparse(self.path).path
        if p == "/log":
            # THE AUTOPSY DOOR, WHICH WAS NEVER THERE.  The page has reported
            # every exception it throws and its heap once a minute since
            # 2026-08-28 --- `index.html: tell()` POSTs them to `/log` --- and
            # do_POST had no branch for it, so every one of them fell through
            # and was dropped.  His browser has died "once in few hours" ever
            # since and the reason was posted, faithfully, into nothing:
            # `grep -c heap` over every log she has ever written returns 0.
            # The GET branch answered `/log` with `{}`, which is what made it
            # look wired.  Now what the page says reaches her log with a stamp.
            try:
                said = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                said = {"raw": raw[:400].decode("utf-8", "replace")}
            print("PAGE %s %s" % (_time.strftime("%H:%M:%S"), json.dumps(said)[:800]),
                  flush=True)
            self._send({})
        elif p == "/seat":
            self._send({"seat": "owner", "agreed": True, "token": "her"})
        elif p == "/agree":
            self._send({"agreed": True, "token": "her"})
        elif p == "/say":
            pcm = np.frombuffer(raw, "<f4").astype(np.float32)
            HER.say(pcm, speech.RATE)
            HER.yours.add(pcm, speech.RATE)
            self._send({"queued": round(HER.window.queued(), 3)})
        elif p == "/felt":
            # THE OUTSIDE BRAIN'S FINISHED NUMBERS, for the panel and nothing
            # else.  Her brain never reads them back.
            try:
                HER.felt = json.loads(raw or b"{}")
            except Exception:                              # noqa: BLE001
                pass
            self._send({})
        elif p == "/tunnel":
            # THE DOOR IS THE OWNER'S.  A guest arrives through cloudflared
            # and wears its Cf-Connecting-Ip header; the door itself, and
            # her lives, only open from this machine.
            if self.headers.get("Cf-Connecting-Ip"):
                self._send({"tunnel": None})
            else:
                try:
                    got3 = json.loads(raw or b"{}")
                except Exception:                              # noqa: BLE001
                    got3 = {}
                if got3.get("on"):
                    if getattr(HER, "tunnelProc", None) is None or \
                            HER.tunnelProc.poll() is not None:
                        tunnelStart(PORT)
                else:
                    tunnelStop()
                self._send({"tunnel": getattr(HER, "tunnelUrl", None)})
        elif p == "/life":
            # SAVE / LOAD / NEW --- save is the truth (the keeper writes her
            # every second, always); load wakes her from any life she has
            # lived; new is a birth.  Only when the app keeps her mind.
            got2 = {}
            try:
                got2 = json.loads(raw or b"{}")
            except Exception:                                  # noqa: BLE001
                pass
            if self.headers.get("Cf-Connecting-Ip"):
                # a guest may love her; a guest may not end or swap her
                self._send({"living": getattr(HER, "mindDb", None)})
                return
            if getattr(HER, "mindDir", None):
                # HIS FLOW (2026-09-06): NEW --- the living life stops whole, takes
                # the body's keepsakes into its file, and a new one begins clean;
                # LOAD --- the living one stops the same way, the chosen one puts
                # its keepsakes back and wakes; DELETE --- to lives/trash, never gone.
                if got2.get("new"):
                    mindStop(); bundle(getattr(HER, "mindDb", "") or "")
                    fresh = _time.strftime("%Y-%m-%d_%H%M%S") + "_life.duckdb"
                    unbundle(fresh); mindStart(fresh)
                elif got2.get("load"):
                    name = os.path.basename(str(got2["load"]))
                    if name.endswith(".duckdb") and name != getattr(HER, "mindDb", None):
                        mindStop(); bundle(getattr(HER, "mindDb", "") or "")
                        unbundle(name); mindStart(name)
                elif got2.get("delete"):
                    name = os.path.basename(str(got2["delete"]))
                    lives = os.path.join(HER.mindDir, "lives")
                    if name.endswith(".duckdb") and name != getattr(HER, "mindDb", None):
                        trash = os.path.join(lives, "trash"); os.makedirs(trash, exist_ok=True)
                        for f in (name, name + ".log", name + ".wal"):
                            try:
                                os.replace(os.path.join(lives, f), os.path.join(trash, f))
                            except OSError:
                                pass
            self._send({"living": getattr(HER, "mindDb", None),
                        "saved": True})
        elif p == "/act":
            # HER MIND'S DECISION --- one per post, and it lands on her next
            # tick exactly where her own act would have written it.  Two
            # shapes:
            #
            #   one line     `{kind, id, lvl}`, kind "motor" or "sound" ---
            #                a try: the smallest step on one thing.
            #   a whole plan `{plan: {motor: [{id, lvl}...], sound: {id,
            #                lvl}}, tick: N}` --- a remembered moment of hers,
            #                every muscle and her voice, replayed as it was.
            #                His, 2026-09-02: *"whan she decide to stay and
            #                balance to reach some object or so it already
            #                would be in her exp/preddictionn chanin"* --- and
            #                his 2026-08-26 rage at one muscle a tick.  A pose
            #                is all of her, so a replay carries all of her.
            #
            # `tick` is the moment the plan answers; how late it lands is
            # measured (`late` on /state), never acted on --- a lagging mind
            # is a fact about her, not a thing to hide.
            try:
                got = json.loads(raw or b"{}")
                plan = got.get("plan")
                if isinstance(plan, dict):
                    at = int(got.get("tick", -1))
                    if at >= 0:
                        HER.lateSum += max(0, len(HER.ticks) - at)
                        HER.lateN += 1
                    # WHERE IT WILL LAND: the tick being lived now, plus one
                    # per decision already queued ahead of it (one lands a
                    # tick).  Her mind books the moment AFTER that as the
                    # expected answer --- both ends mean the same tick.
                    lands = len(HER.ticks) + len(HER.outside)
                    HER.outside.append(("plan", plan, at))
                    self._send({"queued": len(HER.outside), "at": lands})
                else:
                    kind = str(got.get("kind", ""))
                    if kind not in ("motor", "sound"):
                        self._send({"outside": True})
                    else:
                        HER.outside.append((kind, int(got["id"]),
                                            float(got["lvl"])))
                        self._send({"queued": len(HER.outside)})
            except Exception:                                  # noqa: BLE001
                self._send({"outside": True})
        elif p == "/window":
            try:
                HER.flying(json.loads(raw or b"{}"))
            except Exception:
                pass
            self._send({"ok": True, "seeing you": HER.window.shows is not None,
                        "queued": round(HER.window.queued(), 3)})
        elif p == "/world":
            # HIS HAND ON HER.  `grabMove` has posted this 14 times a second
            # since the page was written and this server answered `{}`.
            try:
                got = json.loads(raw or b"{}")
            except Exception:                                  # noqa: BLE001
                got = {}
            if "teacher" in got:
                # THE TEACHER, ON HIS BUTTON --- R5's door: she exists in
                # the room only while the owner says so.  Present, she is
                # a thing the baby's eye can find; gone, she is gone.
                with HER.lock:
                    want = bool(got.get("teacher"))
                    HER.teacher.on = want
                    if want:
                        HER.room.put("teacher", tuple(
                            float(v) for v in HER.teacher.at), size=0.35)
                    else:
                        HER.room.take("teacher")
            if "bottle" in got:
                # HIS BUTTON: on, and it is in her room from that moment.
                if got.get("bottle"):
                    if getattr(HER, "bottle", None) is None:
                        HER.bottle = {"high": HER.BOTTLE_FLOOR, "at": None,
                                      "got": 0, "rng": _rnd.Random(len(HER.ticks))}
                else:
                    HER.bottle = None
                    with HER.lock:
                        HER.room.take("bottle")
            if "floor" in got:
                # HIS BUTTON for the walker's treadmill: true forces it on,
                # false parks it, null returns it to the hands (auto)
                want = got.get("floor")
                HER.floorMoves = (None if want is None else bool(want))
            if "helper" in got:
                # ONE PRESS, THE BODY HOLDS.  walk / crawl / cradle / null.
                want = got.get("helper")
                with HER.lock:
                    # EVERY CHANGE OF HANDS STARTS WITH LETTING GO.  The
                    # wrap's pins outlived a switch to walk --- nothing
                    # cleared them, `heldAt` kept refreshing, and she stood
                    # swaddled on her feet.
                    HER.letGo()
                    HER.helper = (str(want) if want in
                                  ("walk", "crawl", "cradle") else None)
            hold = got.get("hold")
            # ONE SHAPE.  A finger on her is `{joint, at}` (or a plain
            # `{jointName: [x,y,z]}` dict).  The helpers stopped posting
            # rows when the body took them over; the list shape died with
            # no reader and is gone --- one implementation of a hand.
            holds = {}
            if isinstance(hold, dict):
                if "joint" in hold and "at" in hold:
                    holds = {hold["joint"]: hold["at"]}   # a finger on her
                else:
                    holds = {k: v for k, v in hold.items()
                             if isinstance(v, (list, tuple)) and len(v) == 3}
            with HER.lock:
                if got.get("let_go"):
                    HER.letGo()
                elif holds:
                    HER.hold(holds)
                # THE MILK CAME BACK.  The region rewrite that joined the
                # helpers swallowed this block for twenty minutes, and the
                # feed button reached nothing --- restored, with the wrap's
                # pressure and the armed bottle beside it.
                give = got.get("give")
                if isinstance(give, dict) and give.get("milk"):
                    HER.give(float(give["milk"]))
                touch = got.get("touch")
                if isinstance(touch, dict):
                    HER.touch = {str(k): float(v) for k, v in touch.items()}
                    HER.heldAt = len(HER.ticks)
                # HIS OWN HAND, AUTOMATED AT HIS ASK --- 2026-08-27: *"add
                # button to fed her by thresshold like i scrooll timer and
                # cjsee what trshold sjould be"*.  He arms it, he picks the
                # level; nothing decides for him.
                if "autofeed" in got:
                    af = got.get("autofeed")
                    try:
                        HER.autofeed = None if not af else float(af.get("at"))
                    except (TypeError, ValueError, AttributeError):
                        pass
            # HIS BOTTLE HUNT (2026-09-07 night, his word: "put bottle in her room ...
            # each time when she reach that bottle she has to get some milk").  A
            # thing put in her room through this door, and taken back through it:
            # `{"put": {"what": "bottle", "at": [x, y, z], "size": 0.06}}`, `{"take": "bottle"}`.
            # The room already knows things (`room.put`); this is only the door.  The
            # milk itself is given by the caregiver outside (`give`), counted there.
            if isinstance(got.get("put"), dict):
                pt = got["put"]
                try:
                    with HER.lock:
                        HER.room.put(str(pt.get("what", "bottle")),
                                     tuple(float(v) for v in pt.get("at", (0.0, 0.0, 0.0))),
                                     size=float(pt.get("size", 0.06)))
                except (TypeError, ValueError):
                    pass
            if got.get("take"):
                with HER.lock:
                    HER.room.take(str(got["take"]))
            self._send({"held": len(holds), "fed": bool(got.get("give"))})
        elif p == "/show":
            HER.show(raw, int(self.headers.get("X-Width", 0)),
                     int(self.headers.get("X-Height", 0)),
                     name=self.headers.get("X-Name") or None)
            self._send({})
        else:
            self._send({})

    def log_message(self, *a):
        pass


class _nolock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> int:
    global HER
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--lives", default="lives",
                    help="where her mouth, her mother and her clock are kept")
    # THE RULER IS THE WATCHER'S, NOT HERS, so it must be possible to take it
    # off her and see what she costs alone.  It is a trained model on its own
    # thread at ~40 ms a look, and Python hands one thread the interpreter at a
    # time --- so "the instrument is free because it is on another thread" is
    # a claim that has to be measured, not assumed.
    ap.add_argument("--no-ruler", action="store_true",
                    help="do not name anything --- measure her without it")
    # WHOSE BRAIN.  "hers" is the tree's own (it dies on its first tick since
    # THE APP KEEPS HER MIND.  `--mind` is the folder her mind lives in
    # (life.py and lives/); given, the app starts it, restarts it for the
    # lives menu, and ends it when the app ends --- one process to start her
    # whole.  `--mind-db` picks the life to wake; absent, a new one is born.
    ap.add_argument("--mind", default=None)
    ap.add_argument("--mind-db", default=None)
    ap.add_argument("--sprint", action="store_true",
                    help="the fast-living harness: she ticks as fast as the "
                         "machine gives --- never while he watches")
    args = ap.parse_args()
    global PORT
    PORT = args.port
    HER = Watched(args.lives)
    #: WHICH PORT SHE IS ON, so her mind can be told --- set here because
    #: `mindStart` runs below and the module's `PORT` is set later still.
    HER.port = int(args.port)
    HER.sprint = bool(args.sprint)
    threading.Thread(target=HER.live, daemon=True).start()
    if not args.no_ruler:
        threading.Thread(target=HER.ruling, daemon=True).start()
    else:
        print("  the ruler is OFF --- nothing will be named")
    if args.mind:
        HER.mindDir = os.path.abspath(args.mind)
        mindStart(args.mind_db or (_time.strftime("%Y-%m-%d_%H%M%S")
                                   + "_life.duckdb"))
        atexit.register(mindStop)
        print(f"  her mind is kept here too: {HER.mindDb}")
    atexit.register(tunnelStop)     # no more orphan roads piling up
    print(f"she is living.  watch her at http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Page).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
