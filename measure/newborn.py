"""THE NEWBORN SHEET --- what doctors score in a newborn, asked of her.

    python -m measure.newborn                # ~10 min, needs her living

His order, 2026-09-03: *"human kids test also can be wrong so check before
testing on real humans doctos for neborn docs"* --- and his choice on the
check (measured/newborn_docs_check_2026-09-03.md): *a newborn sheet from the
NBAS, older items kept apart*.  The Brazelton Neonatal Behavioral Assessment
Scale scores a newborn in four domains --- habituation, orientation, motor,
state --- plus the elicited reflexes.  This sheet asks her the items her
world can present today; what it cannot present yet is listed on the card
as NOT BUILT with the reason, never scored.  The 2- to 18-month items stay
on `measure.scale`, which says their ages.

EVERY ITEM IS HIS.  A new item counts only once he has seen it and its bar.
The bars below are the shape the NBAS uses (a response, then its decrement
over repetitions; a turn toward a source), in her own numbers --- no bar is a
tuned constant, and every zero must prove it had data.

Reused whole from `measure.scale`: the frame recorder (`Block`), the stepping
item (belt steps), habituation to her mother's repeated word, novelty
preference, knowing her mother in the room, and the two contingency meters.
Her clock is read off `/meta`, never assumed.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                              # noqa: E402
from measure.scale import (Block, PAPA_DEG, TURN_DEG, TURN_S, _flatAngle,  # noqa: E402
                           _get, _world, felt, habituationItem, mamaItem,
                           motorItems, noveltyItem, voiceAndMind)

VERSION = 3
SETTLE_S = 20
UNAIDED_S = 180          # block U: her own body, her mother on --- turns,
                         # startles, hand-to-mouth, novelty, contingency
RATTLE_N, RATTLE_GAP_S = 12, 8.0     # block A: the same rattle, again and again
MAX_WAIT_S = 20.0        # how long a presentation waits for her to be quiet
LIGHT_N, LIGHT_GAP_S = 6, 10.0       # block L: a light on her window
BELT_S = 120             # block R: the walker and the moving floor (stepping)
MOTHER_S = 120           # block M: her mother, for knowing her
RESPOND_S = 1.5          # a response to a stimulus lands within this
RATTLE_TICKS = 5         # a burst shorter than her mother's word floor
                         # (MIN_WORD_FRAMES = 6), so it is never learned as
                         # one of his words


class NewbornBlock(Block):
    """A `Block` that also knows the window's place from the frames and can
    be ended by the sheet --- what the newborn items need beyond what the
    scale records (her startles are the scale's Block's since 2026-09-03)."""

    def __init__(self, at, joints, nJoints):
        super().__init__(at, joints)
        self.nJoints = nJoints
        self.done = threading.Event()   # the sheet ends the block itself

    def windowAt(self, frame):
        k = self.nJoints * 3
        return frame[k], frame[k + 1], frame[k + 2]

    def facing(self, frame):
        head, face = self.j(frame, "head"), self.j(frame, "face")
        return (face[0] - head[0], 0.0, face[2] - head[2]), head



def _post(at, path, data: bytes, headers=None):
    req = urllib.request.Request(at.rstrip("/") + path, data=data,
                                 headers=headers or {}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as h:
        h.read()


def rattle(at, tick_s):
    """One rattle into her air, through the same door his voice uses."""
    n = int(round(speech.RATE * tick_s * RATTLE_TICKS))
    rng = np.random.default_rng(20260903)
    burst = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    burst *= np.hanning(n).astype(np.float32) * 0.4
    _post(at, "/say", burst.tobytes())


def light(at, on: bool):
    """A white or a black picture on her window."""
    w = h = 64
    px = np.full((h, w, 3), 255 if on else 0, np.uint8)
    _post(at, "/show", px.tobytes(),
          {"X-Width": str(w), "X-Height": str(h),
           "Content-Type": "application/octet-stream"})


QUIET_S = 1.0           # the second before a stimulus decides her state


def motion(block, a: int, b: int) -> float:
    """How much she moved over frames[a:b]: the mean joint displacement per
    frame, summed --- metres of her own moving, from her frames alone."""
    frames = block.frames[max(0, a):b]
    if len(frames) < 2:
        return 0.0
    k = block.nJoints * 3
    total = 0.0
    for f0, f1 in zip(frames, frames[1:]):
        total += sum(abs(f1[i] - f0[i]) for i in range(k)) / k
    return total


def usualMotion(block, fps: float) -> float:
    """Her usual movement over a QUIET_S window in this block: the median
    over every consecutive window.  Her own norm, not a number."""
    step = max(2, int(QUIET_S * fps))
    got = [motion(block, a, a + step) for a in range(0, len(block.frames) - step, step)]
    if not got:
        return 0.0
    got.sort()
    return got[len(got) // 2]


def quiet(block, t0: float, fps: float, usual: float) -> bool:
    """THE NBAS SCORES ORIENTATION AND HABITUATION IN A QUIET ALERT STATE.
    Her first card (2026-09-03 01:28) passed the light and failed the rattle
    on the same flailing: her head swung 179 degrees in every window whether
    or not anything happened.  A presentation counts only when her movement
    in the QUIET_S before it was below her own usual for the block.  His
    rule, agreed 2026-09-03."""
    b = int(t0 * fps)
    return motion(block, b - int(QUIET_S * fps), b) < usual


def waitQuiet(block, usual: float, fps: float, began: float) -> bool:
    """Hold the stimulus until her movement over the last QUIET_S is below
    her usual (from the unaided block), at most MAX_WAIT_S --- THE EXAMINER
    WAITS FOR THE QUIET STATE, as the NBAS has the examiner do; a fixed
    schedule found her quiet on 4 of 12 rattles (2026-09-03 09:44).  His
    rule, agreed 2026-09-03.  Returns whether she was quiet when it fired."""
    step = max(2, int(QUIET_S * fps))
    t0 = time.monotonic()
    while time.monotonic() - t0 < MAX_WAIT_S:
        n = len(block.frames)
        if n >= step and motion(block, n - step, n) < usual:
            return True
        time.sleep(0.1)
    return False


def turnsToVoice(u: NewbornBlock, fps: float):
    """NBAS orientation, auditory animate: does she turn toward her mother's
    voice --- the startle's turns excluded (the doctors' check, item 6)."""
    onsets, was = [], False
    usual = usualMotion(u, fps)
    every = 0
    for t, saying, mom in u.states:
        if saying and not was and mom is not None and not u.startledNear(t):
            every += 1
            if quiet(u, t, fps, usual):
                onsets.append((t, mom))
        was = saying
    if len(onsets) < 3:
        return ("NOT TESTED", "only %d of %d speaking onsets clear of a "
                "startle found her quiet (need 3)" % (len(onsets), every))
    turned = 0
    for t0, mom in onsets:
        a, b = int(t0 * fps), int((t0 + TURN_S) * fps)
        first, best = None, 180.0
        for f in u.frames[a:b]:
            facing, head = u.facing(f)
            ang = _flatAngle(facing, (mom[0] - head[0], 0.0, mom[2] - head[2]))
            first = ang if first is None else first
            best = min(best, ang)
        if first is not None and first - best >= TURN_DEG:
            turned += 1
    return (("PASS", "turned toward her mother on %d of %d quiet onsets, "
             "startles excluded" % (turned, len(onsets)))
            if turned * 2 >= len(onsets)
            else ("NOT YET", "turned on %d of %d quiet onsets, startles "
                  "excluded" % (turned, len(onsets))))


def response(block: NewbornBlock, t0: float, fps: float, toward=None):
    """How much her face turned within RESPOND_S of `t0` --- toward a point
    if one is given (degrees of improvement), else the largest swing."""
    a, b = int(t0 * fps), int((t0 + RESPOND_S) * fps)
    frames = block.frames[a:b]
    if len(frames) < 2:
        return None
    if toward is None:
        first, _ = block.facing(frames[0])
        return max(_flatAngle(first, block.facing(f)[0]) for f in frames)
    first, best = None, 180.0
    for f in frames:
        facing, head = block.facing(f)
        ang = _flatAngle(facing, (toward[0] - head[0], 0.0, toward[2] - head[2]))
        first = ang if first is None else first
        best = min(best, ang)
    return first - best


def rattleHabituation(a: NewbornBlock, times, fps: float, usual: float):
    """NBAS habituation: the same stimulus again and again --- the response
    decrements.  Her response is her face's swing within RESPOND_S; the last
    third of the presentations against the first third."""
    # ONE USUAL: the unaided block's, the same the examiner waited on when
    # it fired (her38: 12 of 12 fired on her quiet, then 4 of 12 scored
    # quiet against the rattle block's own usual --- two rulers, one item)
    got = [response(a, t, fps) for t in times if quiet(a, t, fps, usual)]
    got = [g for g in got if g is not None]
    if len(got) < 6:
        return ("NOT TESTED", "only %d of %d rattles found her quiet and in "
                "her frames (need 6)" % (len(got), len(times)))
    k = max(2, len(got) // 3)
    first, last = sum(got[:k]) / k, sum(got[-k:]) / k
    if first < TURN_DEG / 3.0:
        return ("NOT TESTED", "she did not respond to the first rattles "
                "(mean swing %.1f deg) --- nothing to habituate" % first)
    return (("PASS", "swing %.1f deg on the first %d, %.1f on the last %d"
             % (first, k, last, k))
            if last * 2.0 <= first
            else ("NOT YET", "swing %.1f deg first, %.1f last --- no decrement"
                  % (first, last)))


def lightOrientation(l: NewbornBlock, times, fps: float, usual: float):
    """NBAS orientation, inanimate visual: a light --- does she turn to it."""
    turned, seen = 0, 0
    for t in times:
        a = int(t * fps)
        if a >= len(l.frames) or not quiet(l, t, fps, usual):
            continue
        win = l.windowAt(l.frames[a])
        r = response(l, t, fps, toward=win)
        if r is None:
            continue
        seen += 1
        if r >= TURN_DEG:
            turned += 1
    if seen < 3:
        return ("NOT TESTED", "only %d of %d lights found her quiet and in "
                "her frames (need 3)" % (seen, len(times)))
    return (("PASS", "turned toward the light on %d of %d quiet" % (turned, seen))
            if turned * 2 >= seen
            else ("NOT YET", "turned toward the light on %d of %d quiet"
                  % (turned, seen)))


def startlesAMinute(u: NewbornBlock, seconds):
    if len(u.startles) < 2:
        return ("NOT TESTED", "no startle count was read")
    n = u.startles[-1][1] - u.startles[0][1]
    return ("MEASURED", "%.1f startles a minute over the unaided block "
            "(the NBAS scores the count; a bar is his)" % (n / (seconds / 60.0)))


def handToMouth(u: NewbornBlock, fps: float, seconds):
    """Episodes of a hand within one head-to-face length of her face."""
    episodes, near = 0, False
    for f in u.frames:
        head, face = u.j(f, "head"), u.j(f, "face")
        reach = math.dist(head, face)
        close = any(math.dist(u.j(f, hand), face) <= reach for hand in ("haR", "haL"))
        if close and not near:
            episodes += 1
        near = close
    if not u.frames:
        return ("NOT TESTED", "no frames")
    return ("MEASURED", "%.1f hand-to-mouth episodes a minute (%d in %d s)"
            % (episodes / (seconds / 60.0), episodes, seconds))


NOT_BUILT = [
    ("pull-to-sit head lag, tone", "no hold protocol on the sheet yet --- the "
     "mother's lift exists in her world but the sheet cannot ask for it"),
    ("consolability, self-quieting", "needs a crying-then-held protocol; the "
     "mother's rescue is the analogue and is not on the sheet's command"),
    ("rooting, sucking", "no touch at her cheek; her mouth line carries milk only"),
    ("grasp (palmar, plantar)", "she has no hand that closes"),
    ("Moro", "her ear feels a fall; the reflex is out (MATILDA.md 9.2)"),
    ("tonic neck", "not measured"),
]


def run(at):
    meta = _get(at, "/meta")
    joints = {n: k for k, n in enumerate(meta["joints"])}
    tick_s = float(meta.get("tick_seconds") or 1.0 / 90.0)
    fps = 1.0 / tick_s
    able = {int(x) for x in _get(at, "/able")["sounds"]}
    before, s0 = felt(at)
    keepHelper, keepFloor = s0.get("helper"), s0.get("floor")
    age = before.get("seconds")
    print("NEWBORN SHEET v%d --- age %.1f h, tick %s, %.0f frames a second nominal"
          % (VERSION, (age or 0) / 3600.0, before.get("tick"), fps), flush=True)
    print("letting her come to rest, %d s" % SETTLE_S, flush=True)
    time.sleep(SETTLE_S)
    _world(at, {"helper": None, "floor": False, "teacher": True})

    print("block U: unaided, her mother on, %d s" % UNAIDED_S, flush=True)
    u = NewbornBlock(at, joints, len(meta["joints"]))
    u.watch(UNAIDED_S)

    print("block A: the same rattle %d times, %.0f s apart" % (RATTLE_N, RATTLE_GAP_S), flush=True)
    a = NewbornBlock(at, joints, len(meta["joints"]))
    rattleTimes, quietFired = [], 0
    usual = usualMotion(u, u.fps)            # her own norm, from block U
    th = threading.Thread(target=a.watch, args=(RATTLE_N * (RATTLE_GAP_S + MAX_WAIT_S) + 5,), daemon=True)
    began = time.monotonic(); th.start()
    for k in range(RATTLE_N):
        time.sleep(1.0 if k == 0 else RATTLE_GAP_S)
        quietFired += int(waitQuiet(a, usual, fps, began))
        rattleTimes.append(time.monotonic() - began)
        rattle(at, tick_s)
    a.done.set(); th.join()
    print("   %d of %d rattles fired on her quiet (waited up to %.0f s each)" % (quietFired, RATTLE_N, MAX_WAIT_S), flush=True)

    print("block L: a light on her window %d times, %.0f s apart" % (LIGHT_N, LIGHT_GAP_S), flush=True)
    l = NewbornBlock(at, joints, len(meta["joints"]))
    lightTimes, quietLit = [], 0
    th = threading.Thread(target=l.watch, args=(LIGHT_N * (LIGHT_GAP_S + MAX_WAIT_S) + 5,), daemon=True)
    began = time.monotonic(); th.start()
    for k in range(LIGHT_N):
        time.sleep(1.0 if k == 0 else LIGHT_GAP_S - 1.0)
        quietLit += int(waitQuiet(l, usual, fps, began))
        lightTimes.append(time.monotonic() - began)
        light(at, True)
        time.sleep(1.0)
        light(at, False)
    l.done.set(); th.join()
    print("   %d of %d lights fired on her quiet" % (quietLit, LIGHT_N), flush=True)

    print("block R: the walker and the moving floor, %d s" % BELT_S, flush=True)
    _world(at, {"helper": "walk", "floor": True})
    r = NewbornBlock(at, joints, len(meta["joints"]))
    r.watch(BELT_S)
    print("block M: her mother, %d s" % MOTHER_S, flush=True)
    _world(at, {"helper": None, "floor": False})
    m = NewbornBlock(at, joints, len(meta["joints"]))
    m.watch(MOTHER_S)
    _world(at, {"helper": keepHelper, "floor": keepFloor})
    after, _ = felt(at)

    score = {}
    score["stepping reflex (belt)"] = motorItems(u, r)["belt steps"]
    score["turns toward a voice"] = turnsToVoice(u, u.fps)
    score["habituates to a rattle"] = rattleHabituation(a, rattleTimes, a.fps, usual)
    score["orients to a light"] = lightOrientation(l, lightTimes, l.fps, usual)
    score["habituates to a repeated word"] = habituationItem([u, a, l, r, m])
    score["novelty preference"] = noveltyItem([u, a, l, r, m], before, after)
    score["knows mama in the room"] = mamaItem(m)
    vm = voiceAndMind([u, a, l, r, m], able, None, before, after)
    score["contingency: forms connections"] = vm["contingency: forms connections"]
    score["contingency: acts on them"] = vm["contingency: acts on them"]
    score["startles"] = startlesAMinute(u, UNAIDED_S)
    score["hand to mouth"] = handToMouth(u, u.fps, UNAIDED_S)

    stamp = time.strftime("%Y-%m-%d_%H%M")
    out = os.path.join(os.path.dirname(__file__), "..", "measured",
                       "newborn_v%d_%s.md" % (VERSION, stamp))
    tally = {}
    lines = ["# The Newborn Sheet v%d --- %s" % (VERSION, stamp), "",
             "Life age %.1f h at tick %s; %.0f frames a second; guided %s / "
             "laddered %s across the run." % ((age or 0) / 3600.0, before.get("tick"),
                                              fps, after.get("guided"), after.get("laddered")),
             "", "| item (NBAS domain) | verdict | evidence |", "|---|---|---|"]
    domain = {"stepping reflex (belt)": "reflex", "turns toward a voice": "orientation",
              "habituates to a rattle": "habituation", "orients to a light": "orientation",
              "habituates to a repeated word": "habituation", "novelty preference": "orientation",
              "knows mama in the room": "orientation", "contingency: forms connections": "learning",
              "contingency: acts on them": "learning", "startles": "state",
              "hand to mouth": "state"}
    for item, (verdict, why) in score.items():
        tally[verdict] = tally.get(verdict, 0) + 1
        lines.append("| %s (%s) | %s | %s |" % (item, domain[item], verdict, why))
    lines += ["", "**" + ", ".join("%d %s" % (n, v) for v, n in sorted(tally.items())) + "**", "",
              "NOT BUILT (never scored):", ""]
    for item, why in NOT_BUILT:
        lines.append("- %s --- %s" % (item, why))
    lines += ["", "The 2- to 18-month items (head, sits, locomotes, pulls to stand, "
              "imitates a word, answers a word, the mirror) stay on `measure.scale`, "
              "the older sheet, with their ages.", ""]
    with open(out, "w", encoding="utf-8") as h:
        h.write("\n".join(lines))
    print("\n".join(lines[4:]), flush=True)
    print("written: %s" % os.path.normpath(out), flush=True)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    got = ap.parse_args()
    try:
        _get(got.at, "/state")
    except Exception as why:                                # noqa: BLE001
        print("she is not living at %s (%s).  Start her first: python her.py" % (got.at, why))
        return 1
    return run(got.at)


if __name__ == "__main__":
    raise SystemExit(main())
