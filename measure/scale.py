"""THE MATILDA SCALE, v1 --- her milestones, scored the way babies are.
**v8, 2026-09-06, his "agree with everything":** five items dropped ---
imitates a heard word, answers a named word, learned association (the
twice-law), contingency: forms connections, novelty preference.  They
measured answering on cue, pairs and salience, none of which his mind
has; the doctor's visit (measured/scripts/doctor.py) is the sheet for
what it does have.  Everything else identical to v7.

His, 2026-08-30: *"we don't able to try her test on existing model tests. So
now I would like to think with you how we can test her ... we have to pick
some tests for people maybe for young people also. And after passing it or
not ... I have to create some one new thing called this project"* --- and,
on the proposal: *"Implement it"*.

Model benchmarks are useless on a girl with no text; human infants already
have the right instrument.  This battery adapts infant developmental scales
(Bayley, Denver, CDC milestones) to the lines her body actually has.  Each
item is either PASSIVELY OBSERVED or a scripted episode of her ORDINARY
world --- the teacher speaks, the helpers change hands, exactly the buttons
the owner presses himself --- so she never knows she is tested and nothing
in `brain/` is touched (rule 1) or handed a conclusion (rule 2).

Scoring, per item:

    PASS        she did it, in this run or --- for the word items --- in
                this life (a milestone once reached stands, the way a
                paediatrician credits a caregiver's report)
    NOT YET     the ability is testable and was not shown
    CAN'T       her body or our instruments lack the line --- this indicts
                the design, not her; the red mark is the point
    NOT TESTED  the stimulus was absent (papa not on the TV, too few
                teacher bouts)

**EACH VERSION IS FROZEN ON HIS WORD, like the certified mouth.**  The
items, the bars and the block durations below are the protocol; change any
of them and it is a new version --- results across versions do not
compare.  Every run writes `measured/scale_v<N>_*.md`, and runs at
different ages of the same life, or of two lives, are the comparison he
asked the backup repo for --- for her abilities instead of her code.

**v2, his "Implement", 2026-08-30 (same evening):** two items changed,
everything else identical to v1.  *Novelty preference* is no longer a
hardcoded CAN'T: since the salience law (his "Agree"), the unusual
steers her choosing, and the item now measures it --- the median rarity
of the matches that win her replays during the run.  *Contingency: forms
connections* gets lifetime credit like the word items --- a per-run
delta on a girl who has already connected 82% of her world naturally
falls to zero, and a milestone once reached stands.

**v7, 2026-09-01 --- a step was measured against the room, not against her.**
`STEP_UP`/`STEP_DOWN` were absolute metres and her foot RESTS at 0.078 in the
walker, above the 0.07 re-arm bar, so the counter latched after her first lift
and reported "R 1 / L 1" while she lifted her feet 15 cm over and over.  A step
is now clearing HER OWN resting foot (that foot's tenth percentile in the
block) by STEP_LIFT, re-arming at a third of it.  Her walker is untouched and
was never the problem: `MatildaFinal` measured squat + rails + moving floor at
173 stepping events against a shoulder-carry's 2, and matilda2 carries the same
WALK_SQUAT / WALK_RAIL / WALK_DRIFT.

**v6, 2026-09-01 --- the motor items credited her with being dropped.**
Block U takes the helper off and starts watching immediately, so the first
seconds of it are a girl still falling out of a cradle.  Measured: 60 s unaided
with EVERY MUSCLE AT ZERO moved her pelvis 5.71 m of path and 0.85 m from where
she started, and her pelvis sits at cradle height for a moment after the hand
lets go.  That scored "locomotes 12.69 m under her own power" and "pulls to
stand" on a newborn who did nothing.

So:

    SETTLE_S     after the helper is taken off, she is left to come to rest
                 before anything is watched
    held         every block records whether a hand or a helper is on her
                 (`/state.held`: pins, carried), and the motor items score
                 only frames where NOTHING was holding her

A milestone has to be hers.  Being put down is not locomotion, and the moment
before you hit the floor is not standing.

**v5, 2026-09-01 --- the two word items were not about her at all.**  They
read `answered` and `learned` out of `herwatch_state.json`, a file in a scratch
folder written by a DIFFERENT session about a DIFFERENT life (2026-08-31
11:18).  They reported the same eleven words for a six-minute-old girl and an
eight-hour-old one, because they were never about either.  Now they are asked
of the girl in front of us:

    her mother says a NAMED word, and now says WHICH  (`teacher.word`, and
    `teacher.wordIds`, the sounds it is made of --- she already knew both)
    answered  = the baby voiced one of that word's own sounds within
                IMITATE_S of it
    learned   = she did it for the same word on two separate occasions ---
                the twice-law, his: once is not a statistic

Nothing is credited across lives any more: the girl being tested is the girl
being scored.

**v4, his word, 2026-09-01 --- *"I already go to bed, so add mama instead of
papa for now for testing"*:** the same block is watched whether or not he is
on the television, and a second item scores it against HER MOTHER'S place in
the room instead of the window.  Everything else is v3 exactly.  His camera
needs him awake; her mother does not, so the attention block stops being
NOT TESTED on every run he is not sitting at the laptop.

The block is his own `papaItem` question asked of a different face: how much
of it she spent facing the thing.  One bar (`PAPA_DEG`, `PAPA_DWELL`), one
piece of arithmetic, two targets --- not a second opinion about what looking
at something means.

**v3, his "I agree with all your suggestion. You can implement them one
by one" (2026-08-30, night):** one item changed --- *habituation* is no
longer a CAN'T: it is scored behaviorally, the way a clinic scores it ---
her replies to the single most-repeated word of the run, first half of
its hearings against the second; fading is habituation (Sokolov), and no
instrument reaches inside her to do it.

Run (pure HTTP + file reads, no GPU, safe beside her live body):

    .venv\\Scripts\\python.exe -m measure.scale                # full battery
    .venv\\Scripts\\python.exe -m measure.scale --papa         # he is on the TV
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import time
import urllib.request

VERSION = 7

#: The protocol --- frozen with the version.
#: AFTER THE HELPER COMES OFF, she is left to come to rest before anything is
#: watched.  A girl falling out of a cradle travels metres and passes through
#: standing height on the way down, and neither is a thing she did.
SETTLE_S = 20
UNAIDED_S = 360         # block U: no helper, belt parked --- her own body
RAIL_S = 240            # block R: walk helper + belt --- stepping
PAPA_S = 120            # block P (only with --papa): he is live on the TV
POLL_S = 0.30           # frame fetch cadence (frames arrive at her 30 fps)
STATE_S = 0.50          # state fetch cadence (teacher.saying edges)
AFTER_STARTLE_S = 1.0   # a turn beginning within this of a startle is the
                        # reflex's, not hers (doctors' check, item 6)

#: The bars --- adapted from infant milestone criteria to her geometry,
#: frozen with the version.  Every one is written beside WHY.
HEAD_LEVEL_DEG = 30.0   # "holds head steady": median tilt from vertical
                        # while the trunk is upright (adapted 45-deg
                        # head-lag criterion, tightened: she measured 6
                        # deg held, so 30 leaves room to fail honestly)
# HER BARS ARE HER OWN BONES, NOT METRES (2026-09-04, his 'fix rest').  Her
# torso is 0.095 m from pelvis to chest, and the old bar asked for the chest
# 0.12 m above the pelvis --- 126% of her torso, while a full spine bend
# already lifts it 95%, unseen.  Sitting asked her pelvis below 0.30 m on a
# mattress that holds it at 0.40; standing asked 0.38 m above the surface
# on legs 0.211 m long.  Measured with `ragdoll._rest_len`.
TRUNK_UP = 0.8          # trunk upright: (chest - pelvis) vertical share of its length
                        # --- within 37 degrees of vertical
SIT_RISE = 0.5          # sitting: the pelvis stays on the surface --- risen less
                        # than this fraction of her torso above the block's lowest
SIT_HOLD_S = 5.0        # ... head>chest>pelvis sustained this long
STAND_PELVIS = 0.8      # pull-to-stand: pelvis risen this fraction of her LEGS
                        # (hip-knee + knee-foot) above the surface she lies on
#: (was 0.38 m --- a bar in metres, 180% of her legs) --- pelvis ABOVE THE SURFACE SHE
                        # LIES ON (the block's lowest pelvis), not above the
                        # floor: her cot's mattress is 0.34 m up and she lay
                        # at 0.39-0.449 m on 1800 of 1800 ticks --- a false
                        # PASS (2026-09-02; his step 8, 2026-09-03)
STAND_HOLD_S = 2.0      # ... sustained this long, unaided
MOVE_M = 1.0            # locomotes: own-power displacement (m)
#: A STEP IS MEASURED AGAINST HER OWN RESTING FOOT, NOT AGAINST THE ROOM.
#: These were absolute metres --- rise above 0.10, fall below 0.07 to re-arm
#: --- and her foot RESTS at 0.078 in the walker, above the re-arm bar.  So
#: after her first lift the counter latched "up" for ever and could never
#: count a second: measured 2026-09-01, "R 1 / L 1 lifts" on a girl lifting
#: her feet 15 cm (0.078 resting, 0.227 peak) again and again for a minute.
#:
#: Her own floor is the tenth percentile of that foot's height in the block,
#: so a cot, a mat or a moving belt cannot move the bar.  A step is clearing
#: it by STEP_LIFT; it re-arms at a third of that, which is a foot back down
#: rather than a foot merely lower.
STEP_LIFT = 0.04        # a real lift clear of her own resting foot (m)
STEP_BACK = 0.33        # ...and re-armed once it falls back this fraction
STEPS_EACH = 2          # both feet at least this many rises
TURN_DEG = 15.0         # turns-to-voice: facing improves toward mom by
                        # this much within TURN_S of a speaking onset
TURN_S = 2.5
TURN_BOUTS = 3          # fewer onsets than this = NOT TESTED
REPERTOIRE = 5          # distinct own sounds voiced during the run
IMITATE_S = 4.0         # heard word -> her same id within this (guardian's
                        # own answer window)
PAPA_DEG = 35.0         # facing within this of the window counts as looking
PAPA_DWELL = 0.20       # ... for at least this fraction of block P
NOVELTY_SAL = 0.5       # novelty preference: the matches that win her
                        # replays carry at least this median rarity ---
                        # closer to the rare than to the ambient, which
                        # reads ~0 within minutes of watching
HABIT_MIN = 8           # habituation: the most-repeated word must be
                        # heard at least this often in a run before
                        # fading can be told from chance

GUARDIAN = (r"C:\Users\tscen\AppData\Local\Temp\claude"
            r"\C--Users-tscen-Documents-antropic-practice-project-Matilda"
            r"\16e995bd-2a8a-44a1-bcd7-baf3436ae793\scratchpad"
            r"\herwatch_state.json")

#: HER VOICE AND HER EARS ARE READ OFF /mind, NEVER OFF THE STAGE FILE.
#: The first run of this scale read `*.stage.jsonl` and scored her voice
#: on an empty window: that file is DuckDB's bulk-load SCRATCH, rewritten
#: by every table batch each second --- a lottery, not a log.  /mind is
#: the body's live mirror of the same lines, sampled while the blocks
#: watch; a bout of hers lasts ~1.5 s, so 2 Hz misses nothing that
#: matters.

HEAD = struct.Struct("<4sIIIHH")


def _get(at, path):
    with urllib.request.urlopen(at.rstrip("/") + path, timeout=10) as h:
        return json.loads(h.read().decode("utf-8"))


def _world(at, payload):
    req = urllib.request.Request(
        at.rstrip("/") + "/world", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as h:
        h.read()


def _frames(at, since):
    """Decoded frames since `since` --- `(next, [values...])`, metres."""
    req = urllib.request.Request(at.rstrip("/") + "/frames?from=%d" % since)
    with urllib.request.urlopen(req, timeout=10) as h:
        raw = h.read()
    if len(raw) < HEAD.size:
        return since, []
    _, first, count, latest, wide, scale = HEAD.unpack(raw[:HEAD.size])
    rows = []
    body = raw[HEAD.size:]
    for k in range(count):
        chunk = body[k * wide * 2:(k + 1) * wide * 2]
        if len(chunk) == wide * 2:
            rows.append([v / scale for v in
                         struct.unpack("<%dh" % wide, chunk)])
    # THE CURSOR MUST LEAVE THE SENTINEL.  `Block.watch` starts at 1<<31 to
    # mean "the newest", the server answers with a header and no rows (it has
    # no frame that far ahead), and `max(since, latest + 1)` then kept 1<<31
    # --- so every later ask was also from 1<<31 and the block collected ZERO
    # frames, for its whole life.  Found 2026-09-01: 25 s of watching caught
    # 37 of 37 state samples and 0 frames, which means every motor item and
    # the papa item has been scored on NO DATA --- "moved 0.00 m", "faced the
    # window 0%", "R 0 / L 0 lifts" were empty, not measured.
    #
    # Past the newest: snap to the newest.  Otherwise: on from what was
    # actually served.
    return (latest + 1 if since > latest else first + count), rows


class Block:
    """One observed stretch: her frames and the room's state, in order."""

    def __init__(self, at, joints):
        self.at = at
        self.joints = joints            # joint name -> index into a frame
        self.frames = []                # [values], one per body tick
        self.seconds = 0.0              # how long this block watched
        self.states = []                # (t, saying, momHead) at ~2 Hz
        self.startles = []              # (t, her startle count) at ~2 Hz
        self.ears = []                  # (t, sayId|None, hearId|None) ---
                                        # voiced own id / heard world id
        self.held = []                  # (t, held) --- was a hand or a helper
                                        # on her, sampled with the states
        self.words = []                 # (t, word, ids) --- what her MOTHER
                                        # said and the sounds it is made of,
                                        # recorded when it changes
        self.panel = []                 # (replays, salience) samples ---
                                        # the winning match's rarity per
                                        # replay, for the novelty item

    def startledNear(self, t: float) -> bool:
        """Did her startle count rise within AFTER_STARTLE_S before `t`?  A
        turn that begins there is the startle reflex's (her neck is written
        for ten ticks by a loud sound), not her choosing --- the doctors'
        check, item 6, 2026-09-03.  One implementation for both sheets."""
        was = None
        for tt, n in self.startles:
            if tt > t:
                break
            if was is not None and n > was and t - tt <= AFTER_STARTLE_S:
                return True
            was = n
        return False

    @property
    def fps(self) -> float:
        """Frames a second AS RECEIVED --- her clock is 90 since
        2026-09-03 and was 30; this was `30.0` in two places and every
        timed window was a third of its length at 90.  Measured, not
        assumed (his word, 2026-09-03)."""
        return len(self.frames) / self.seconds if self.seconds > 0 else 0.0

    def watch(self, seconds):
        cursor, _ = _frames(self.at, 1 << 31)   # start at the newest
        began = time.monotonic()
        nextState = 0.0
        while time.monotonic() - began < seconds:
            cursor, rows = _frames(self.at, cursor)
            self.frames.extend(rows)
            now = time.monotonic() - began
            self.seconds = now
            done = getattr(self, "done", None)      # a sheet may end a block
            if done is not None and done.is_set():   # when its presentations are
                break                                # over, not on a clock
            if now >= nextState:
                nextState = now + STATE_S
                try:
                    s = _get(self.at, "/state")
                    tch = s.get("teacher") or {}
                    mom = ((tch.get("joints") or {}).get("head")
                           if tch.get("on") else None)
                    self.states.append((now, bool(tch.get("saying")), mom))
                    self.startles.append((now, int(s.get("startles") or 0)))
                    h = s.get("held") or {}
                    self.held.append((now,
                                      bool(int(h.get("pins", 0) or 0)
                                           or int(h.get("carried", 0) or 0)
                                           or s.get("helper"))))
                    w = tch.get("word")
                    ids = tuple(int(i) for i in (tch.get("wordIds") or ()))
                    if w and ids and (not self.words
                                      or self.words[-1][1] != w
                                      or now - self.words[-1][0] > 2.0):
                        self.words.append((now, str(w), ids))
                    self.panel.append((s.get("replays"), s.get("salience")))
                except OSError:
                    pass
                try:
                    m = _get(self.at, "/mind")
                    e = m.get("echo") or {}
                    h = m.get("hearing") or {}
                    # THE SOUND SHE MADE IS HER ECHO'S NAME.  This read
                    # `saying.id` --- the name she COMMANDED of the recorded
                    # clip player --- and since 2026-09-02 her mouth is her
                    # own (seven articulators; nothing is commanded by name),
                    # so `saying` is 0 for ever and the voice items went
                    # blind: "1 distinct own sounds" against 17 measured on
                    # her echo the same night.  Her ear names what her mouth
                    # made; that is what "her own sound" means.  His to
                    # confirm (a test is his).
                    say = (int(e.get("id") or 0)
                           if float(e.get("lvl") or 0.0) > 0 else None)
                    hear = (int(h.get("id") or 0)
                            if float(h.get("lvl") or 0.0) > 0 else None)
                    self.ears.append((now, say, hear))
                except OSError:
                    pass
            time.sleep(POLL_S)

    def j(self, frame, name):
        k = self.joints[name] * 3
        return frame[k], frame[k + 1], frame[k + 2]

    def window(self, frame):
        k = len(self.joints) * 3
        return frame[k], frame[k + 1], frame[k + 2]


def _angleFromVertical(a, b):
    """Angle of the a->b vector from straight up, degrees."""
    v = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = math.sqrt(sum(x * x for x in v)) or 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, v[1] / n))))


def _flatAngle(v, w):
    """Angle between two vectors on the floor plane, degrees."""
    a = math.atan2(v[2], v[0]) - math.atan2(w[2], w[0])
    return abs(math.degrees(math.atan2(math.sin(a), math.cos(a))))


def _sustained(flags, need):
    run = best = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best >= need


def _freeOnly(b: Block):
    """Her frames with NOTHING holding her --- no helper, no pinned hand.

    A block samples `held` at ~2 Hz and her body at ~30, so a frame is judged
    by the sample covering its moment.  Without this, being carried and being
    dropped both score as things she did: measured 2026-09-01, 60 s unaided
    with every muscle at zero moved her 5.71 m of path.
    """
    if not b.held:
        return list(b.frames)
    n = len(b.frames)
    out = []
    for k, f in enumerate(b.frames):
        j = min(len(b.held) - 1, k * len(b.held) // max(1, n))
        if not b.held[j][1]:
            out.append(f)
    return out


def _namedNull(blocks, times: int):
    """The named-word counts with every naming moved to a random moment of
    its own block, `times` over, seeded (an unseeded shuffle is dice)."""
    import random
    rng = random.Random(20260903)
    answered, learned = [], []
    for _ in range(times):
        hits: dict = {}
        for b in blocks:
            if b is None or b.seconds <= 0:
                continue
            voicedAt = [(tt, say) for tt, say, _ in b.ears if say is not None]
            for _when, word, ids in b.words:
                when = rng.uniform(0.0, b.seconds)
                for tt, one in voicedAt:
                    if 0 < (tt - when) % b.seconds <= IMITATE_S and one in ids:
                        hits.setdefault(word, []).append(when)
                        break
        answered.append(len(hits))
        learned.append(sum(1 for w in hits if len(hits[w]) >= 2))
    return answered, learned


def motorItems(u: Block, r: Block):
    got = {}
    fps = u.fps
    heads, sits, stands = [], [], []
    start = None
    far = 0.0
    free = _freeOnly(u)
    lying = min((u.j(f, "pelvis")[1] for f in free), default=0.0)
    for f in free:
        head, neck = u.j(f, "head"), u.j(f, "neck")
        chest, pelvis = u.j(f, "chest"), u.j(f, "pelvis")
        trunk = math.dist(chest, pelvis) or 1e-9
        legs = (math.dist(u.j(f, "hiR"), u.j(f, "knR")) + math.dist(u.j(f, "knR"), u.j(f, "foR"))) or 1e-9
        up = (chest[1] - pelvis[1]) / trunk >= TRUNK_UP
        if up:
            heads.append(_angleFromVertical(neck, head))
        sits.append(up and pelvis[1] - lying < SIT_RISE * trunk
                    and head[1] > chest[1] > pelvis[1])
        stands.append(pelvis[1] - lying >= STAND_PELVIS * legs)
        if start is None:
            start = pelvis
        far = max(far, math.hypot(pelvis[0] - start[0],
                                  pelvis[2] - start[2]))
    heads.sort()
    med = heads[len(heads) // 2] if heads else None
    got["holds head level"] = (
        ("PASS", "median %.0f deg from vertical (bar %d)"
         % (med, HEAD_LEVEL_DEG)) if med is not None and med <= HEAD_LEVEL_DEG
        else ("NOT YET", "median %.0f deg" % med) if med is not None
        else ("NOT TESTED", "trunk never upright unaided"))
    got["sits"] = (
        ("PASS", ">= %.0f s head over chest over low pelvis" % SIT_HOLD_S)
        if _sustained(sits, int(SIT_HOLD_S * fps))
        else ("NOT YET", "no sustained sit"))
    got["locomotes"] = (
        ("PASS", "moved %.2f m under her own power" % far)
        if far >= MOVE_M else ("NOT YET", "moved %.2f m (bar %.1f)"
                               % (far, MOVE_M)))
    got["pulls to stand"] = (
        ("PASS", "pelvis risen %.0f%% of her legs above where she lay, >= %.0f s unaided"
         % (100 * STAND_PELVIS, STAND_HOLD_S))
        if _sustained(stands, int(STAND_HOLD_S * fps))
        else ("NOT YET", "never stood unaided"))
    rises = {"foR": 0, "foL": 0}
    upNow = {"foR": False, "foL": False}
    # her own resting foot, this block --- the tenth percentile, so one long
    # sag does not become the floor and one high kick does not either
    floor = {}
    for foot in rises:
        ys = sorted(r.j(f, foot)[1] for f in r.frames) or [0.0]
        floor[foot] = ys[len(ys) // 10]
    for f in r.frames:      # the rail block IS held on purpose --- that is
                            # what a belt is: her feet, her lifts, his hands
        for foot in rises:
            y = r.j(f, foot)[1] - floor[foot]
            if not upNow[foot] and y > STEP_LIFT:
                upNow[foot] = True
                rises[foot] += 1
            elif upNow[foot] and y < STEP_LIFT * STEP_BACK:
                upNow[foot] = False
    got["belt steps"] = (
        ("PASS", "R %d / L %d lifts on the moving floor"
         % (rises["foR"], rises["foL"]))
        if min(rises.values()) >= STEPS_EACH
        else ("NOT YET", "R %d / L %d lifts" % (rises["foR"], rises["foL"])))
    return got


def turnItem(u: Block):
    onsets = []
    was = False
    for t, saying, mom in u.states:
        if saying and not was and mom is not None and not u.startledNear(t):
            onsets.append((t, mom))
        was = saying
    if len(onsets) < TURN_BOUTS:
        return ("NOT TESTED",
                "only %d speaking onsets clear of a startle (need %d)"
                % (len(onsets), TURN_BOUTS))
    fps = u.fps
    turned = 0
    for t0, mom in onsets:
        a, b = int(t0 * fps), int((t0 + TURN_S) * fps)
        best0, best = None, 180.0
        for f in u.frames[a:b]:
            head, face = u.j(f, "head"), u.j(f, "face")
            facing = (face[0] - head[0], 0.0, face[2] - head[2])
            toMom = (mom[0] - head[0], 0.0, mom[2] - head[2])
            ang = _flatAngle(facing, toMom)
            if best0 is None:
                best0 = ang
            best = min(best, ang)
        if best0 is not None and best0 - best >= TURN_DEG:
            turned += 1
    return (("PASS", "turned toward mom on %d/%d onsets"
             % (turned, len(onsets)))
            if turned * 2 >= len(onsets)
            else ("NOT YET", "turned on %d/%d onsets"
                  % (turned, len(onsets))))


def voiceAndMind(blocks, able, guardian, before, after):
    got = {}
    voiced = set()
    imitated = False
    for b in blocks:
        if b is None:
            continue
        heardAt = []                    # (time, id) world sounds, per block
        for tt, say, hear in b.ears:
            if say is not None:
                voiced.add(say)
                for ht, hid in heardAt:
                    if hid == say and 0 < tt - ht <= IMITATE_S:
                        imitated = True
            if hear is not None and hear in able:
                heardAt.append((tt, hear))
                heardAt = heardAt[-64:]
    got["voices a repertoire"] = (
        ("PASS", "%d distinct own sounds this run" % len(voiced))
        if len(voiced) >= REPERTOIRE
        else ("NOT YET", "%d distinct own sounds (bar %d)"
              % (len(voiced), REPERTOIRE)))
    # HER MOTHER'S NAMED WORDS, AND WHETHER SHE ANSWERED THEM --- asked of
    # this girl, in this run.  See the v5 note: this used to be read out of a
    # scratch file about another life.
    said = 0
    hits: dict = {}
    for b in blocks:
        if b is None:
            continue
        voicedAt = [(tt, say) for tt, say, _ in b.ears if say is not None]
        for when, word, ids in b.words:
            said += 1
            for tt, one in voicedAt:
                if 0 < tt - when <= IMITATE_S and one in ids:
                    hits.setdefault(word, []).append(when)
                    break
    answered = sorted(hits)
    learned = sorted(w for w, times in hits.items() if len(times) >= 2)
    # AGAINST CHANCE.  A babbling girl answers some named word by luck: her
    # sounds are dense and a word is several ids.  The null model is the
    # same count with every naming moved to a random moment of its block
    # (wrapped), 200 times, seeded --- an answer counts only above the
    # null's 95th percentile.  His step 8, 2026-09-03; a zero must prove it
    # had data, and a PASS must beat dice.
    nullAnswered, nullLearned = _namedNull(blocks, 200)
    barA = sorted(nullAnswered)[int(0.95 * (len(nullAnswered) - 1))] if nullAnswered else 0
    barL = sorted(nullLearned)[int(0.95 * (len(nullLearned) - 1))] if nullLearned else 0
    dGuided = (after.get("guided") or 0) - (before.get("guided") or 0)
    dConn = (after.get("connections") or 0) - (before.get("connections") or 0)
    total = after.get("connections") or 0
    # v2: LIFETIME CREDIT, his "Implement" --- a per-run delta on a girl
    # who already connected 82% of her world naturally falls to zero; a
    # milestone once reached stands, the way a caregiver's report counts.
    got["contingency: acts on them"] = (
        ("PASS", "+%d memory-guided tries this run" % dGuided)
        if dGuided > 0 else ("NOT YET", "no guided tries this run"))
    return got


def habituationItem(blocks):
    """v3: responses to the SAME repeated word fade --- Sokolov, scored
    behaviorally.  Every hearing of the run's most-repeated word is an
    exposure; a reply is her same id back within the answer window.
    Fewer replies in the second half of the exposures than the first is
    habituation.  Whatever carries it inside her, the CARD scores the
    behavior, exactly as a clinic would."""
    events = []                         # (word id, replied) in order heard
    for b in blocks:
        if b is None:
            continue
        run = {}
        heard = []                      # [t, id, replied]
        for tt, say, hear in b.ears:
            if hear is not None:
                run[hear] = run.get(hear, 0) + 1
                if run[hear] == 2:
                    heard.append([tt, hear, False])
            else:
                run = {}
            if say is not None:
                for ev in heard[-8:]:
                    if ev[1] == say and 0 < tt - ev[0] <= IMITATE_S:
                        ev[2] = True
        events += [(h[1], h[2]) for h in heard]
    per = {}
    for wid, rep in events:
        per.setdefault(wid, []).append(rep)
    if not per:
        return ("NOT TESTED", "no repeated word reached her this run")
    wid, seq = max(per.items(), key=lambda kv: len(kv[1]))
    if len(seq) < HABIT_MIN:
        return ("NOT TESTED", "most-repeated word heard %d times "
                "(need %d)" % (len(seq), HABIT_MIN))
    half = len(seq) // 2
    a = sum(seq[:half]) / half
    b2 = sum(seq[half:]) / (len(seq) - half)
    if a > 0 and b2 < a:
        return ("PASS", "replies to the most-repeated word faded "
                "%.0f%% -> %.0f%% across %d hearings"
                % (100 * a, 100 * b2, len(seq)))
    if a == 0 and sum(seq) == 0:
        return ("NOT YET", "she never replied to the most-repeated word "
                "(%d hearings)" % len(seq))
    return ("NOT YET", "replies %.0f%% -> %.0f%% across %d hearings --- "
            "no fading" % (100 * a, 100 * b2, len(seq)))


def noveltyItem(blocks, before, after):
    """v2, his "Implement": the unusual steers her choosing --- measured.

    Since the salience law, each replay's winning match carries the
    rarity of the id that tied its chain to the moment (`salience` on
    the panel).  A baby's novelty preference is looking longer at the
    new; hers is choosing by it: the matches that win her replays sit
    closer to the rare than to the ambient (which reads ~0)."""
    perReplay = {}
    for b in blocks:
        if b is None:
            continue
        for rep, sal in b.panel:
            if rep is not None and sal:
                perReplay[int(rep)] = float(sal)
    ran = (after.get("closes") or 0) - (before.get("closes") or 0)
    if ran <= 0 or not perReplay:
        return ("NOT TESTED", "no experience closed during the run")
    vals = sorted(perReplay.values())
    med = vals[len(vals) // 2]
    declined = ran - ((after.get("replays") or 0)
                      - (before.get("replays") or 0))
    if med >= NOVELTY_SAL:
        return ("PASS", "median winning-match rarity %.2f (bar %.1f), "
                "%d meaningless replays declined" % (med, NOVELTY_SAL,
                                                     declined))
    return ("NOT YET", "median winning-match rarity %.2f (bar %.1f)"
            % (med, NOVELTY_SAL))


def papaItem(p: Block):
    if p is None:
        return ("NOT TESTED", "papa was not on the TV (run with --papa)")
    near = total = 0
    for f in p.frames:
        head, face = p.j(f, "head"), p.j(f, "face")
        win = p.window(f)
        facing = (face[0] - head[0], 0.0, face[2] - head[2])
        toWin = (win[0] - head[0], 0.0, win[2] - head[2])
        if math.hypot(*[toWin[0], toWin[2]]) < 1e-6:
            continue
        total += 1
        if _flatAngle(facing, toWin) <= PAPA_DEG:
            near += 1
    frac = near / total if total else 0.0
    return (("PASS", "faced the window %.0f%% of the block" % (100 * frac))
            if frac >= PAPA_DWELL
            else ("NOT YET", "faced the window %.0f%%" % (100 * frac)))


def mamaItem(p: Block):
    """DOES SHE FACE HER MOTHER --- `papaItem`'s question, asked of the mom.

    Her mother's head rides every block already (`Block.watch` records it at
    ~2 Hz whenever the teacher is on); her own joints ride the frames at ~30.
    A frame is paired with the mother's place at the same point of the block,
    which is as close as two samplings of one stretch can be joined without
    inventing a clock.
    """
    if p is None:
        return ("NOT TESTED", "no attention block was watched")
    moms = [m for _, _, m in p.states if m]
    if not moms or not p.frames:
        return ("NOT TESTED", "her mother was not in the room")
    near = total = 0
    n = len(p.frames)
    for k, f in enumerate(p.frames):
        mom = moms[min(len(moms) - 1, k * len(moms) // max(1, n))]
        head, face = p.j(f, "head"), p.j(f, "face")
        facing = (face[0] - head[0], 0.0, face[2] - head[2])
        toMom = (mom[0] - head[0], 0.0, mom[2] - head[2])
        if math.hypot(toMom[0], toMom[2]) < 1e-6:
            continue
        total += 1
        if _flatAngle(facing, toMom) <= PAPA_DEG:
            near += 1
    frac = near / total if total else 0.0
    return (("PASS", "faced her mother %.0f%% of the block" % (100 * frac))
            if frac >= PAPA_DWELL
            else ("NOT YET", "faced her mother %.0f%%" % (100 * frac)))


def felt(at):
    s = _get(at, "/state")
    return {k: s.get(k) for k in ("guided", "laddered", "closes", "replays",
                                  "connections", "words", "seconds", "tick",
                                  "helper", "floor")}, s


def run(at, papa):
    meta = _get(at, "/meta")
    joints = {n: k for k, n in enumerate(meta["joints"])}
    able = {int(x) for x in _get(at, "/able")["sounds"]}
    before, s0 = felt(at)
    keepHelper = s0.get("helper")
    keepFloor = s0.get("floor")
    age = before.get("seconds")
    print("MATILDA SCALE v%d --- age %.1f h, tick %s"
          % (VERSION, (age or 0) / 3600.0, before.get("tick")), flush=True)

    print("letting her come to rest, %d s" % SETTLE_S, flush=True)
    time.sleep(SETTLE_S)
    print("block U: unaided, %d s" % UNAIDED_S, flush=True)
    _world(at, {"helper": None, "floor": False})
    u = Block(at, joints)
    u.watch(UNAIDED_S)
    print("block R: rail and moving floor, %d s" % RAIL_S, flush=True)
    _world(at, {"helper": "walk", "floor": True})
    r = Block(at, joints)
    r.watch(RAIL_S)
    # THE ATTENTION BLOCK IS WATCHED EITHER WAY.  It used to run only with
    # --papa, so on every run he was not sitting at his camera the whole block
    # was skipped and the item read NOT TESTED --- his, going to bed: *"add
    # mama instead of papa for now for testing"*.  Her mother is in the room
    # whether or not he is awake.
    print("block %s: %d s" % ("P: papa on the TV" if papa else "M: her mother",
                              PAPA_S), flush=True)
    p = Block(at, joints)
    p.watch(PAPA_S)
    _world(at, {"helper": keepHelper, "floor": keepFloor})
    print("her world is restored: helper %r, floor %r"
          % (keepHelper, keepFloor), flush=True)

    after, _ = felt(at)
    # THE GUARDIAN FILE IS NOT SCORED ANY MORE (v5).  It is another session's
    # notes about another life; keeping it as evidence would credit this girl
    # with words she never heard.  Loaded only so the report can say it was
    # ignored, and dropped entirely when nothing reads it.
    guardian = None

    score = {}
    score.update(motorItems(u, r))
    score["turns toward a voice"] = turnItem(u)
    score.update(voiceAndMind([u, r, p], able, guardian, before, after))
    score["habituation"] = habituationItem([u, r, p])
    score["knows papa on the TV"] = (
        papaItem(p) if papa
        else ("NOT TESTED", "papa was not on the TV (run with --papa)"))
    score["knows mama in the room"] = mamaItem(p)
    score["knows herself in the mirror"] = (
        "NOT YET", "the horizon item --- an 18-month skill, kept on the "
        "scale so the road stays visible")

    stamp = time.strftime("%Y-%m-%d_%H%M")
    out = os.path.join(os.path.dirname(__file__), "..", "measured",
                       "scale_v%d_%s.md" % (VERSION, stamp))
    tally = {}
    lines = ["# The Matilda Scale v%d --- %s" % (VERSION, stamp), "",
             "Life age %.1f h at tick %s; guided %s / laddered %s, "
             "closes %s / replays %s across the run." % (
                 (age or 0) / 3600.0, before.get("tick"),
                 after.get("guided"), after.get("laddered"),
                 after.get("closes"), after.get("replays")), "",
             "| item | verdict | evidence |", "|---|---|---|"]
    for item, (verdict, why) in score.items():
        tally[verdict] = tally.get(verdict, 0) + 1
        lines.append("| %s | %s | %s |" % (item, verdict, why))
    lines += ["", "**" + ", ".join("%d %s" % (n, v)
                                   for v, n in sorted(tally.items())) + "**",
              ""]
    with open(out, "w", encoding="utf-8") as h:
        h.write("\n".join(lines))
    print("\n".join(lines[4:]), flush=True)
    print("written: %s" % os.path.normpath(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="http://127.0.0.1:8090")
    ap.add_argument("--papa", action="store_true",
                    help="he is live on the TV: run the papa block")
    got = ap.parse_args()
    run(got.at, got.papa)
