"""HER BODY, ASSEMBLED --- the one place where all of her is put together.

Her room, her flesh, her muscles, her mouth, her ears, her eye and her
chemistry, driven by her brain, one tick at a time.  **There is one of this.**
`brain/run/run.py` drives it headless and `sandbox/` watches it; both use this
and neither has its own copy.

It was written twice for a few hours --- once in her life and once in the thing
that watches her --- which is the duplication this project has been burned by
more than any other.

A LOOK IS SPREAD ACROSS TICKS, and that is what makes 33 ms hold.  Measured:

    light.see        16.3 ms
    parts.find       25.0 ms
                     ------
    one look         63.8 ms      almost TWO ticks

Done in one tick she is 30 ms late five times a second.  Done a stage a tick
the worst tick is 25 ms, a look still finishes every 200 ms, and **no tick is
ever over budget** --- his: *"33 has to be, this is our tick"*.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

#: HOW COARSE "WHERE" IS, across and up --- see the table where it is used.
#: It is her body's, not her brain's: what lines she HAS is a fact about her
#: senses, and `brain/` only ever reads a line's id and its level.
WHERE = 6

#: HOW FAST WHAT IS AT HER LIPS GOES AWAY, a second.  A mouthful is a mouthful
#: and not a tap left running: put milk there and it is gone in about four
#: seconds whether or not she takes it.
FEEDS_FOR = 0.25

from body import light, parts, speech
from body.air import hear
from body.alike import ALIKE_KINDS, FLOOR, SILENCE, _unit, level_of
from body.ears import align
from body.teacher import LIFT_HOLD, LIFT_TO, Teacher
from body.hearing import (LISTENS, LOOK_SECONDS, REGISTER, SOUND_BANDS,  # noqa: E501
                          SOUND_HOPS, SOUND_SLIDES, TICK_SECONDS,
                          bands_from_pcm, door, grabFor, wordFrames)
from body.ticks import Ticks
from time import perf_counter as _perf
from body.joints import AXES
from body.muscles import VOICE_PARTS, Muscles, Voice
from body.ragdoll import _BONES, JIDX, Ragdoll   # `_BONES` is for DRAWING
#: --- her joints joined up, which `pose` hands the page and nothing of
#: hers ever reads.  It was left behind when her body was assembled from
#: one file, and `pose` and `snap` have raised `NameError` ever since ---
#: silently, because `Watched` overrides `snap` and nothing served `/pose`,
#: so the only way to reach them was to run a plain `Her.live()`, which
#: nothing does.  Two of her six body methods were dead and green.
from body.room import ROOM, Room
from body.window import Window
from body import balance, orient, skin
from mind.structure import Motor, Sensor, Spindle, Tick, View


#: WHERE EACH CONE SITS ON HER RETINA, in -1..1, made once.  Her whole
#: picture's x and y are where its light sits, weighted by these.
_RET_X = np.tile(np.linspace(-1.0, 1.0, light.RETINA_W, dtype=np.float32), light.RETINA_H)
_RET_Y = np.repeat(np.linspace(-1.0, 1.0, light.RETINA_H, dtype=np.float32), light.RETINA_W)


def wholePicture(picture) -> tuple:
    """HER WHOLE PICTURE AS ONE THING --- `(level, x, y)`.

    His word, 2026-09-11: *"she sees picture, that picture all has to has ONE
    similarity id, and when all picture moves those directions has to be x and
    y"* --- and, the same day: *"our view already doesn't need any parts.
    Nothing.  She just sees some picture, and this picture is moving."*

    NO CUT.  Until 2026-09-12 this took `parts.find` --- the old per-object
    view, 70 surfaces a look, 20.7 of a 32.6 ms look on his 4060 --- and kept
    the biggest surface's name, throwing 69 away.  The whole picture is named
    directly, by the same `parts._alike` that named a surface: the direction
    of its colour (proportion is the frame's own, constant), in the same 512
    kinds her ear names a sound in, divided by that count so it is a LEVEL.
    Its x and y are where its light sits, in -1..1, and they move when the
    picture moves.  Measured 2026-09-12: 3.3 ms on the host, the same level
    the cut gave (0.5703 against 0.5684) for the same picture.
    """
    pic = np.asarray(picture, np.float32)
    if pic.ndim != 3 or not pic.size:
        return 0.0, 0.0, 0.0
    flat = pic.reshape(-1, pic.shape[-1])
    bright = flat.sum(axis=1)
    total = float(bright.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    row = np.zeros((1, parts.COLUMNS), np.float32)
    row[0, 3] = row[0, 4] = 1.0
    row[0, parts.COLOUR] = flat.mean(axis=0)
    name = int(parts._alike(row)[0])
    n = min(bright.size, _RET_X.size)
    return (float(name) / float(ALIKE_KINDS),
            float(np.dot(_RET_X[:n], bright[:n]) / total),
            float(np.dot(_RET_Y[:n], bright[:n]) / total))


#: THE PICTURE BOARD --- WHERE SHE ACTUALLY LOOKS, which is not where the
#: training floor is.  Placed first at (9.0, 0.45, 9.75), the pen's wall, and
#: measured: her head sits at (0.20, 0.42, -0.34) in her cot at the origin,
#: her face points (-0.75, -0.56, +0.35), and the board was 103 DEGREES off
#: her gaze and 9.7 m away --- outside her mother's naming reach (25 degrees
#: and four metres) and outside her eye.  The pen is where she is TAKEN to
#: train; her cot is where she lives.
#:
#: AND HER SKULL NEVER MOVES.  Measured 2026-09-01 over 1,339 frames: 100% of
#: them lie within 25 degrees of ONE direction, (-0.75, -0.56, +0.35) --- she
#: holds her head still and looks down into the mattress, which is what a
#: newborn with no head control does.  A card along that line is buried in the
#: bed.
#:
#: Her EYES do move (+/-35 degrees, `_aimEyes`), and her mother reads the eye
#: axis, not the skull.  So the card hangs at her own head height along her
#: heading --- 0.4 m away, about 34 degrees above her skull's line and so
#: inside her eyes' reach, and well inside her mother's four metres.
BOARD_AT = (-0.16, 0.42, -0.17)
#: facing back at her
BOARD_LOOK = (0.91, 0.0, -0.42)
#: a nursery card, bigger than his flying screen because it does not fly
BOARD_SIZE = (0.50, 0.36)
#: WHAT IT SHOWS, IN ORDER --- only words her mother actually owns, so the
#: pairing offered is a real one: this look, this word, again and again.  She
#: has nine recorded words; these are the ones that name a thing.
BOARD_CARDS = ("bed", "mirror", "teacher", "papa", "window")
#: how long one card stays up, in ticks.  Habituation needs EIGHT hearings of
#: one word before fading can be told from chance, and her mother names about
#: twice a second when the baby is looking --- so half a minute a card is many
#: hearings and still five cards in three minutes.
#: How long one card stays up, in his seconds.
CARD_SECONDS = 30.0
BOARD_TICKS = int(round(CARD_SECONDS / TICK_SECONDS))
#: (ground, mark) per card --- far apart in her cones, so two cards differ by
#: COLOUR and by SHAPE, never by brightness alone
CARD_COLOURS = {
    "bed":     ((0.75, 0.70, 0.55), (0.20, 0.25, 0.55)),
    "mirror":  ((0.30, 0.35, 0.40), (0.90, 0.92, 0.95)),
    "teacher": ((0.55, 0.30, 0.35), (0.95, 0.85, 0.70)),
    "papa":    ((0.25, 0.45, 0.35), (0.95, 0.80, 0.60)),
    "window":  ((0.20, 0.25, 0.30), (0.85, 0.90, 0.40)),
}


#: HOW MANY MUSCLES MOVE HER EYES --- FOUR, IN OPPOSING PAIRS: left, right,
#: down, up.  They move no bone, so they sit past her skeleton's axes rather
#: than inside `JOINTS3`.
#:
#: NOT A DIAL.  It was two lines, each 0..1 mapped to -1..+1, and a muscle at
#: REST (0, which is what her ladder leaves behind and what the body decays
#: to) then meant HARD LEFT: measured, her eye spindle sat at 0.000 forever
#: and she could neither centre her eyes nor feel them.
#:
#: An eye is moved by opposing muscles --- medial and lateral rectus, superior
#: and inferior --- each PULLING from rest, and rest is the middle because
#: neither is pulling.  That is the biology and it is also her own convention:
#: 0 is a muscle doing nothing.  Her gaze is what the pair disagree by.
EYES = 4
#: WHAT THE WORLD HAS PUT AGAINST HER MOUTH --- the line id, and only the
#: id.  Her mind reads the same number (`mind/hormones.MOUTH`); both ends
#: mean the same thing.
MOUTH = 12

#: WHICH ORDER LINE TURNS HER FACE SIDEWAYS --- `neck.twist`, and only it.
#: Measured why the other two cannot be used (in `orient.turn_head`, never
#: called, deleted 2026-09-03): `neck.bend` rests
#: at 1.421 of its own 0..1 range, outside it entirely, and neither it nor
#: `neck.side` answered an order antisymmetrically.  Looked up by NAME so a
#: reordering of her axes cannot silently point this at a hip.
TWIST = tuple(i for i, a in enumerate(AXES) if a == "neck.twist")


class Her:
    """Her life, in a thread.  Nothing here decides anything about her."""

    def __init__(self, lives: str | None = None) -> None:
        """`lives` is the folder her mouth, her mother's words, his window and
        the room's clock are kept in.  None is a probe's body: it reads her
        mouth from `lives/` and writes nothing anywhere."""
        self.lock = threading.Lock()
        self.room = Room()
        self.her = Ragdoll(self.room)
        self.muscles = Muscles()
        self.voice = Voice(SOUND_BANDS, SOUND_SLIDES)
        #: THE PICTURE BOARD --- a second screen on the training wall, his
        #: 2026-09-01 ask.  It shows one card at a time and CARRIES THAT
        #: CARD'S NAME, so when her gaze lands on it her mother says the word
        #: she already has for it.  Her world held exactly one word before
        #: this ("nowhere", her own name, said because nothing else was ever
        #: within her mother's naming reach): 31 hearings of it in a session
        #: and habituation could not even be asked.
        #:
        #: Her mother owns nine recorded words; the board rotates through the
        #: ones that name a THING, so the pairing she is offered is a real
        #: one --- this look, this word, again and again, which is how a
        #: picture card teaches a child.
        self.board = Window()
        self.board.at = np.asarray(BOARD_AT, np.float32)
        self.board.look = np.asarray(BOARD_LOOK, np.float32)
        self.board.up = np.asarray([0.0, 1.0, 0.0], np.float32)
        self.board.size = tuple(BOARD_SIZE)
        self.board.name = BOARD_CARDS[0]
        self._card = 0
        self._cardAt = 0
        self.window = Window(keep=(os.path.join(lives, "window")
                                   if lives else None))
        self.teacher = Teacher(keep=(os.path.join(lives, "mom.json")
                                     if lives else None))
        self.teacher.convert = self._convertHis
        # ...and her words are cut at THIS body's clock (see Teacher.recut)
        self.teacher.recut()
        #: HER MIND'S DECISIONS, queued by `POST /act`, one landing a tick.
        #: Her body never decides and never feels: it reports its senses and
        #: does what was queued.  His, 2026-08-26: *"http is created like
        #: watcher but self app wors on gpu"*, *"sandbox is data in app not in
        #: browser"*.  The in-process brain that once sat beside this list is
        #: gone (2026-09-02, with `brain/`): one girl, one decider.
        self.outside: list = []
        #: what the last decision moved, and how late whole plans arrive
        #: (see `/act` and `/state` in sandbox/app.py)
        self.landed = 0
        self.lateSum = 0
        self.lateN = 0
        #: WHERE HER TICK GOES, stage by stage --- a running mean in ms
        #: (see `_lap`), published on /state as `stages`.  His, 2026-09-02:
        #: *"do what possible to increase her perfomanse"* --- a cut needs
        #: a number before and after, measured live, warm, with her mind on.
        self.stageMs: dict = {}
        self._t0 = _perf()
        if lives:
            os.makedirs(lives, exist_ok=True)
        #: THE LAST FEW SECONDS OF HER, for `/ticks` --- see `body/ticks.py`.
        #: Her record is her mind's; the body keeps no tape.
        self.ticks = Ticks(keep=1024,
                           clock=os.path.join(lives, "clock") if lives else None)
        #: THE TICK BEING LIVED --- her plan carried forward, her inputs
        #: filled as the tick goes.  Empty before her first tick, so `/state`
        #: asked at birth answers zeros rather than dying (2026-09-02: the
        #: retired brain used to hold this default and its removal took it).
        self.tick = Tick()
        # HER ABLE SOUNDS ARE THE NAMES HER EAR GIVES, NOT THE PIECES.
        # `align` answers in `alike_of` indices and her mind stores and
        # commands the same ones (`expect.near` reads that id arithmetically),
        # so the NAMES are her alphabet --- 57 of them.  The 859 pieces
        # wearing those names are her mouth's business alone: one name, many
        # sounds, and which one she says is `Voice._piece`.  Handing the
        # piece ids here instead would put 859 unrelated integers into a
        # channel that does arithmetic on them, and nothing would raise.
        #: HER EYES ARE MUSCLES SHE COMMANDS.  His, 2026-09-01, the rule the
        #: whole thing rests on: *"All of this app has to build on her
        #: opinion ... She has to decide what she doing and why she has to do
        #: this instead of this"*, and *"she hasn't to have anything that
        #: works next to her.  This is one organism."*
        #:
        #: Her eyes moved by REFLEX until now --- `_aimEyes` wrote `eye_at`
        #: straight into her body, and her mind never saw the choice, never
        #: paid for it and could never learn from it.  It worked BESIDE her,
        #: which is the one thing forbidden.  And it could not be tuned into
        #: place either: measured 2026-09-01, an unconditional pull threw her
        #: head 21 degrees a TICK across a 100 degree field.
        #:
        #: So looking is an act.  Two motor lines past her skeleton's
        #: twenty-six --- her eyes move no bone, so they are not joints ---
        #: and her ladder walks them like every other muscle: she orders one,
        #: her spindle answers, her view changes, `connect` pairs the two, and
        #: "moving my eyes changes what I see" is something she LEARNED.
        #:
        #: The cost, paid once: her motor and spindle territories grow by two,
        #: the contract's fingerprint changes, and every stored life becomes
        #: unreadable.  New lives from zero.
        #: HER MUSCLES: 26 axes, 4 eye muscles, and --- his word, 2026-09-02
        #: (*"recorderd sounds can be thintetic"*, and 2026-09-01: *"we do
        #: not need prelearned sounds anymore"*) --- the seven articulators of
        #: her own mouth, motors 31..37: loud, pitch, open, front, nasal,
        #: close, hiss.  A sound is a pose she commands and feels, like a
        #: hand; nothing plays a recording any more.
        self.motors = len(AXES) + EYES + len(VOICE_PARTS)
        self._eyeWant = np.zeros(EYES, np.float32)
        self._mouthWant = np.zeros(len(VOICE_PARTS), np.float32)
        #: WHAT SHE ASKED TO SAY --- a sound's id and how hard she means it.
        #: that makes each sound her body can make; `Voice.play` walks it at
        #: her effort.  Restored 2026-09-08 on his word: the channel existed
        #: in her voice and nothing in her body had ever called it, so the
        #: only sounds she could make were accidents of her seven articulators
        #: landing in a shape together --- 24 in a quarter of an hour.
        self._sayWant = (0, 0.0)
        #: the startle: ticks of turning left, where to, and how many she has
        #: had --- the last one is glass for him, never hers
        #: how many SOUNDING ticks she has heard --- a usual means nothing
        #: until there are enough of them to have one
        self.startles = 0
        #: THE TURN TOWARD A SOUND --- a pan for her eyes, and the ticks it lasts
        # WHAT SHE CAN ACTUALLY SAY --- and the two RESERVED rows are not it.
        #
        # `0` is her mouth at rest and `1` is the room at rest, and neither is a
        # sound she produces: both carry the rest mouth, so commanding one makes
        # nothing.  Reserving `FLOOR` put it straight into her alphabet and her
        # ladder started ordering it --- measured on the echo line the moment it
        # existed: she commanded id 1 and her mouth answered 0, so 9 tries in 40
        # were her asking for a sound her body cannot make.
        self.sounds = []
        #: HER GRAIN --- her mind's one answer to "did anything change?", on
        #: every line: a value a line never showed, a change worth keeping, the
        #: rise that closes an experience, the step her record is written at.
        #: It is NOT the step her muscles move by (a trial moves by her deficit).
        #:
        #: HIS WORD, 2026-09-12: 0.01.  Measured the same day on her own state,
        #: tick to tick: at 1/512 (0.002) half of all her ticks read as a change
        #: --- an experience every ~100 ms, a fifth of her ticks dropped; at
        #: 0.05, her muscle's step, one tick in 140 --- and ~15 distinguishable
        #: levels of his speech; at 0.01, one tick in 22, and ~75 levels.  The
        #: 512 was the count of the old ear names, which no longer exist: her
        #: similarity is a continuous level.
        self.resolution = 0.01
        # HOW LONG A GAP ENDS A WORD --- the period of her SLOWEST sense.  Her
        # ear names something every tick and her eye every `LOOK_SECONDS`, so
        # anything shorter than a look is not a gap, it is her eye not having
        # reported yet.
        self.gap = LOOK_SECONDS
        self.order = np.zeros(len(AXES), np.float32)
        #: her inner ear, which remembers across ticks because
        #: acceleration is a CHANGE and one tick cannot hold one
        self.balance = balance.Balance()
        #: WHERE SOMEBODY IS PULLING HER, and the tick it last arrived.
        #:
        #: The page has let him drag her by a hand or a foot since it was
        #: written --- `grabMove` posts `/world {hold:{joint, at}}` FOURTEEN
        #: TIMES A SECOND --- and `Ragdoll.handle` has existed to receive it.
        #: The server answered `{}` and threw it away, so nothing ever moved.
        #: The same shape as `POST /window` being swallowed, which was the whole
        #: of *"she still dont see me"*.
        self.heldAt = -1
        #: A TOUCH INJECTED THROUGH /world --- the wrap's own pressure.
        #: `skin.feel` has taken it since it was written (*"a touch you
        #: injected through /world"*, its own words) and nothing ever
        #: passed it.  Joined 2026-08-27: *"swddle also dont works"*.
        self.touch: dict | None = None
        #: THE HELPER THE BODY ITSELF HOLDS --- 'walk', 'crawl',
        #: 'cradle', or None.  Helpers used to live on a stream of page
        #: posts every 400 ms; through the tunnel those arrive seconds
        #: apart and the stale-hand timeout kept releasing her between
        #: them --- "helpers stop to works", from anywhere remote.  One
        #: press sets the mode; every tick the body computes the hands
        #: from HER OWN limb lengths and holds them, smooth and
        #: lag-proof, until told to let go.
        self.helper: str | None = None
        #: the moving floor's own switch --- his button: None follows the
        #: hands (walk helper or his grab), True forces it, False parks it
        self.floorMoves: bool | None = None
        #: the belt's wandering direction --- his design, 2026-08-30: *"floor
        #: has to move in all direction randomly.  Left, right, forward,
        #: back... she will find the pass by her leg.  I don't need just
        #: rolling in front --- she just stuck in front of a wall, and
        #: that's it.  This is not practice."*  A random cardinal, held a
        #: random second or three, then another --- so nothing piles her
        #: anywhere and her legs answer every direction.
        self._beltDir = np.asarray([-1.0, 0.0, 0.0], np.float32)
        self._beltTill = 0
        self._beltAge = 0
        #: THE BELT'S DIRECTIONS ARE A FACT ABOUT HER ROOM, NOT A DICE ROLL
        #: EACH TIME SHE WAKES.  This was `default_rng()` with no seed, so
        #: every run drew a different sequence of directions --- and a
        #: direction is held 7 to 15 seconds, so an 87-second walk saw six to
        #: twelve of them and never the same six.
        #:
        #: MEASURED 2026-09-02: the same girl, the same settings, walked three
        #: times, stepped **25, 16 and 27** times.  Nothing changed between
        #: them.  Her whole walking has therefore never been comparable run to
        #: run --- neither an experiment of ours nor the `belt steps` item on
        #: her own scale, which read R57/L54 one night and R33/L31 the next
        #: and was reported as a change in her.
        #:
        #: Seeded, the directions are still random and still cover every
        #: cardinal --- his design, *"in all direction randomly ... she will
        #: find the pass by her leg"* --- but they are the SAME random, the
        #: way her room's walls are the same walls.  A world she cannot
        #: measure twice is not a world she can learn.
        self._beltRng = np.random.default_rng(20260830)
        #: which pinned joints the HELPER owns (the wrap's), so the
        #: stale-hand timeout can tell a closed browser's forgotten
        #: finger from the body's own standing hands
        self._helperPins = np.zeros_like(self.her.pinned)
        #: WHAT IS AT HER LIPS, and it is a real sensor and not a conclusion:
        #: `MOUTH` is the line her body writes when something is there.  It was
        #: hardcoded to 0.0, so the feed button --- which the page has posted as
        #: `/world {give:{milk}}` since it was written --- reached nothing, and
        #: **nobody could ever feed her.**  His: *"i cant help her at all"*.
        self.atMouth = 0.0
        self.picture = None
        #: HOW MUCH SOUND HER EAR ASKS FOR EACH TICK.  A frame is one tick
        #: wide and may sit in `SOUND_HOPS` places, so the newest placement ends
        #: now and the oldest begins `(SOUND_HOPS-1)/SOUND_HOPS` of a tick
        #: earlier --- she needs that much more than a tick to have the choice.
        self.earSeconds = TICK_SECONDS * (2.0 - 1.0 / SOUND_HOPS)
        #: her LAST look, kept where the reflex reads it --- on the device
        #: when she has one.  One picture, 3 MB, and it saves copying two.
        #: ...AND THE THING HER EYES ARE ON, carried between looks.  `None` is
        #: her holding nothing, which is a real answer and not a failure.
        #: HOW FAR HER VIEW HAS SWUNG SINCE HER LAST LOOK, radians, in her own
        #: frame.  Her canals feel her head turn (`ragdoll.turned`) and her gaze
        #: order says where she asked her eyes to go; `orient.turned` puts the
        #: how far her view moved since her last look.
        #: look-to-look steps on her own retina."*  On her live body, measured
        #: 2026-08-26 before this line existed: **18.5%.**
        #:
        #: IT ACCUMULATES ACROSS THE TICKS BETWEEN LOOKS, because that is what
        #: it says: how far her view moved between the LAST look and this one,
        #: and she looks once every `looksEvery` ticks while her head turns
        #: every one of them.  Measured at 1.48 degrees a tick about her own
        #: axis, which is about 9 degrees between looks.
        self.swungSince = np.zeros(3, np.float32)
        self.wasGaze = (0.0, 0.0)
        #: what the last stage of a look handed on to the next
        self.ears = np.zeros((2, SOUND_BANDS, SOUND_SLIDES), np.float32)
        #: the last `LISTENS` of his air, from which each tick's piece is named
        self._hisAir = np.zeros(0, np.float32)
        self.voiced = np.zeros(SOUND_BANDS, np.float32)
        #: WHAT KEEPS ARRIVING --- so the floor can name itself.
        #:
        #: THE FLOOR IS THE THING THAT IS ALWAYS THERE, AND IT MUST BEAT
        #: SILENCE.  That is what room tone IS, and it needs no number.
        #: MEASURED on 55 s of his real room --- id 283 is 47.9% of every frame
        #: AND 51.9% of the quietest third, so the simple rule and the careful
        #: one give the same answer THERE.  They do not agree in HER room:
        #: 2026-09-02 the most-heard id was a sound taught to her 1,290 times,
        #: and every arrival of it was renamed to room tone (measured live,
        #: 6 of 6).  A simulated room has no continuous hiss, so "most often"
        #: ranks only the ~5% of ticks that carried any sound at all, and the
        #: winner of that ranking is the most REPEATED sound, not the constant
        #: one.  His own recording separates them: room tone was 47.9% of every
        #: frame against silence's 1.1%.  So silence is counted alongside the
        #: names, and the floor has to outnumber it --- still two measured
        #: counts compared, still no constant.
        #:
        #: It is the same habituation her boredom already runs (`hormones._usual`
        #: is a running average of how much moved), asked of a name instead of a
        #: level --- and it replaces asking *is this quiet*, which needs a gate.
        self.heardIds: dict = {}
        self.looksEvery = max(1, int(round(LOOK_SECONDS / TICK_SECONDS)))
        self.going = True
        #: HER LOOK ON ITS OWN THREAD, at its own rate --- his blueprint,
        #: 2026-09-02: *"tick has to be 11ms but it is for sound so its like
        #: blueprint for resc channels so image became for example 20 limes
        #: per second"*.  Measured live the same night: a whole tick was
        #: 20.2 ms and her look 13.6-14.3 of it (68%); the rest of her is
        #: ~6.5 ms.  `live()` starts the thread; a probe body that only
        #: calls `oneTick` looks inline as before (the six wires).
        self.lookThread = None
        self.ticked = threading.Condition()
        self._lookJob = None
        self._eyeBusy = False
        #: HOW FAR BEHIND HER OWN CLOCK SHE IS, in seconds.  Zero is keeping up.
        self.behind = 0.0
        #: WHAT THE PAGE READS.  Built once at the end of a tick and handed out
        #: without a lock: nobody watching her may cost her a tick.
        self.shot: dict = {}

    def snap(self) -> None:
        """One picture of her for whoever is watching.  Costs her nothing."""
        with self.lock:
            self.shot = {"pose": self.pose(), "mind": self.mind()}

    # --- her life ---------------------------------------------------------
    def live(self) -> None:
        """Her clock, and nothing else's.

        SHE IS PACED TO HER OWN TICK.  Running flat out means her seconds are
        not seconds --- a life that runs at whatever the machine manages is a
        life whose hunger, whose look-back and whose six hours all mean
        something different on a faster laptop.

        AND THE PAGE NEVER WAITS FOR HER.  This held her lock for the WHOLE
        tick, so every `/state`, `/frames`, `/eye` and `/window` the browser
        sent queued behind her: measured, 16.4 ticks a second against 30, and
        `/state` taking 79 ms to answer.  She and the page were fighting over
        her.  The lock is held only while her body is actually being touched,
        and what the page reads is a snapshot taken at the end of a tick.
        """
        import time as _t
        began = _t.perf_counter()
        # PACED FROM THIS WAKING, NOT FROM HER WHOLE AGE.  The tick count
        # continues across restarts now (the room's clock outlives the file),
        # and pacing against the absolute count told a freshly woken body it
        # was thousands of ticks AHEAD of schedule --- she slept 333 seconds
        # before her second tick, frozen to anyone watching.
        born = len(self.ticks)
        self.lookThread = threading.Thread(target=self._looks, daemon=True)
        self.lookThread.start()
        while self.going:
            t9 = _perf()
            with self.lock:
                self.oneTick()
            self.stageMs["tick"] = (self.stageMs.get("tick", (_perf() - t9) * 1000.0)
                                    + ((_perf() - t9) * 1000.0 - self.stageMs.get("tick", 0.0)) / 300.0)
            t9 = _perf()
            with self.ticked:
                self.ticked.notify_all()
            self.snap()
            self.stageMs["snap"] = (self.stageMs.get("snap", (_perf() - t9) * 1000.0)
                                    + ((_perf() - t9) * 1000.0 - self.stageMs.get("snap", 0.0)) / 300.0)
            # ...AND SHE WAITS FOR HER OWN CLOCK.  If she is late, she does not
            # sleep and `behind` says so --- lateness she can see, never a
            # silent speed-up.
            if getattr(self, "sprint", False):
                # THE FAST-LIVING HARNESS --- his word, 2026-08-28, after
                # the discussion of teaching her: time-compression of her
                # own experience is the one honest accelerator ("a harness,
                # not a birth" --- the old tree's own bootstrap law).  She
                # ticks as fast as the machine gives; every mechanism ---
                # the mom, the judge, the milk --- runs exactly as at pace.
                # Never with him watching: a sprinting girl is not a
                # watchable one.
                continue
            due = began + (len(self.ticks) - born) * TICK_SECONDS
            gap = due - _t.perf_counter()
            self.behind = max(0.0, -gap)
            # LOST TIME IS LOST.  She once fell 777 seconds behind in a
            # machine stall and then SPRINTED at 52 ticks/s for thirteen
            # minutes of catch-up, choking the machine that had choked her.
            # A live creature slips; she does not time-travel.  More than
            # two seconds late, the clock re-bases and she resumes at her
            # own pace --- the lateness stays visible in `behind` until it
            # is forgiven here.
            if gap < -2.0:
                began = _t.perf_counter() - (len(self.ticks) - born)                     * TICK_SECONDS
            elif gap > 0:
                _t.sleep(gap)

    #: how fast a held joint rises toward the hand, a tick.  0.005 was
    #: eaten whole by the bone passes (measured: +0.000 in 5 s --- the
    #: skeleton argues back about that much a tick); 0.02 a tick = 0.6 m/s,
    #: still a gentle lift, and it wins.
    GENTLE_TICK = 0.02
    #: the walker, 2026-08-30 --- his "draw some lines to guide her legs".
    #: WALK_SQUAT stands her at a slight knee-bend so her feet MUST bear
    #: weight (full extension left them dangling and the support reflex
    #: mute); WALK_RAIL is how far past her own hip half-width a knee or
    #: foot may stray sideways before the frame is a wall; WALK_DRIFT is
    #: the slow treadmill under a loaded foot, per tick.
    WALK_SQUAT = 0.95   # was 0.88 --- at 0.88 the hold left enough slack
                        # for her to fold onto her knees and stay there
                        # (kneeling is the pose her life has practiced);
                        # at 0.95 her shoulders ride too high for knees to
                        # reach the ground while held, and her feet still
                        # press the floor
    WALK_RAIL = 0.05
    WALK_DRIFT = 0.004
    #: the training square --- his design: *"square... in front of mirror"*,
    #: and his correction: *"location has to be in front of mirror, not
    #: under her bed."*  A full 2 m square cannot fit between the mirror
    #: wall and the cot, so the square is the largest that does: 0.9 m a
    #: side, flush to the mirror wall, clear of her bed.  Centre and
    #: half-side, room coordinates.
    #: ...placed toward the mirror's corner, and sized by his last word:
    #: *"It's just a small.  Make it twice bigger."*  1.8 m a side --- as
    #: much of it in the corner as the room allows; its west edge brushes
    #: the cot's ground, where the belt never runs anyway.
    #: ...and its TRUE place, found 2026-08-30: the room is 20 x 20 m
    #: (ROOM x/z = 10) and the mirror lives at the far +x/+z corner ---
    #: every earlier placing sat 12 m from it on a wrong room-size
    #: assumption.  The pen (room.PEN) closes this square: mirror wall,
    #: +z wall, and two low fences with a doorway.
    WALK_ZONE = (9.0, 9.0, 0.95)

    def _helperHold(self) -> None:
        """The body's own hands, every tick, from her own bones.

        A CARRIED JOINT, NEVER A PIN.  The first cut pinned her chest and
        head, and a pinned joint is frozen where physics cannot argue ---
        the moment the giraffe target stopped climbing she became a
        statue: *"before she moves in them but now they like hold her"*
        (his, 2026-08-27).  Her flesh already had the hand with give,
        wired to nothing: `carrying`/`carry_at` in `ragdoll.py`, whose own
        docstring measured why a pin kills her learning.  A carried joint
        is placed at the hand each substep and the solver still argues,
        so she sags, sways, pushes and travels.  The hand fixes HEIGHT;
        x and z are refreshed to HER OWN every tick, so her pushes are
        what move her.  The wrap alone still pins, because a swaddle
        holding her is its truth.
        """
        self._helperPins[:] = False
        if not self.helper:
            return
        p, J = self.her.pos, JIDX
        # BONES ARE MEASURED AT REST, NEVER LIVE.  Measuring from current
        # positions fed back: the pinned head stretched the neck, the
        # stretched neck raised the target, and her head ratcheted to the
        # ceiling 2 cm a tick --- his screenshot, a bead on a three-metre
        # pole.  Her skeleton's rest pose cannot stretch, and `ragdoll`
        # already owns the one implementation of a rest length.
        from body.ragdoll import _rest_len as seg
        # heights stand on the ground the ROOM says is beneath her ---
        # NEVER on her own lowest joint.  min(joints) was the giraffe's
        # second door: tuck her knees and her "ground" rises with her
        # feet, the target rides it, and the bounded hand un-bounds
        # itself at 0.02 a tick.  `room.under` answers from the room's
        # own surfaces and cannot ride her.
        under = self.room.under(p, self.her.radius)

        def carry(name, y, give):
            k = J[name]
            if self.her.pinned[k]:
                # a posted hand owns this joint --- HIS grab wins over the
                # body's helper, and the helper takes back on release
                return
            # THE HAND LIFTS UNTIL SHE IS AT HEIGHT.  Her flesh yields
            # around a carry --- limp, she settles ~0.11 m below the hand
            # (measured: hand 0.680, shoulders 0.565) --- so a hand parked
            # AT standing height stands her low: "walking halper is to
            # low".  Placing the hand higher by a constant would be the
            # forbidden downstream compensation; instead the loop closes
            # on HER: the hand climbs GENTLE_TICK a tick while she is
            # below her height and eases when she is above it, exactly
            # what hands under armpits do.  `give` bounds the hand above
            # the target so the loop cannot giraffe: at the bound she
            # hangs at most that far low, and the bound is her own
            # anatomy, not a tuned number.
            was = (float(self.her.carry_at[k, 1]) if self.her.carrying[k]
                   else float(p[k, 1]))
            lift = y - float(p[k, 1])
            if lift > self.GENTLE_TICK:
                lift = self.GENTLE_TICK
            elif lift < -self.GENTLE_TICK:
                lift = -self.GENTLE_TICK
            hand = was + lift
            if hand > y + give:
                hand = y + give
            self.her.carry_at[k, 0] = float(p[k, 0])
            self.her.carry_at[k, 1] = hand
            self.her.carry_at[k, 2] = float(p[k, 2])
            self.her.carried[k] = 1.0
            self.her.carrying[k] = True

        if self.helper in ("walk", "crawl"):
            # THE TRAINING SPACE IS THE OPEN FLOOR, NEVER THE COT --- his
            # screenshot, 2026-08-30: the walker had stood her up INSIDE
            # the crib, stepping on her own mattress between the bars.  A
            # helper that finds her on furniture first carries her out:
            # chest lifted above the rail, drifted toward the open middle,
            # and only over bare floor does the hold below begin --- the
            # same carrying hand the rescue slide already uses.
            onWhat = min(float(under[J["foL"]]), float(under[J["foR"]]),
                         float(under[J["knL"]]), float(under[J["knR"]]))
            # 0.15: furniture is knee-high and up; the 8 cm rug by the
            # mom's spot is walkable ground, the 37 cm cot is not.
            # ...and WHEREVER the hands find her, the training square is
            # where they take her --- his last piece, 2026-08-30: *"when
            # mom in automatic will press walk and help her, she has to
            # pull her on this walking space because now she just stay
            # outside of this square."*
            cxT, czT, halfT = self.WALK_ZONE
            offSquare = (abs(float(p[J["chest"], 0]) - cxT) > halfT - 0.15
                         or abs(float(p[J["chest"], 2]) - czT) > halfT - 0.15)
            if onWhat > 0.15 or offSquare:
                # TWO HANDS, AND THE HANDS' OWN CLIMB.  The first cut aimed
                # one hand at her chest's current height plus a step --- so
                # a limb snagged on a crib bar stalled the hand for ever
                # (seen live, 2026-08-30, twice: draped on the rail, then
                # hovering at it).  A parent's hands climb regardless and
                # take the body as a unit: chest and pelvis together, the
                # hand target walking from ITS OWN last place to LIFT_TO
                # (the mom's carry height, feet well over any rail), and
                # only at height does the drift to the open middle begin.
                pc = seg("pelvis", "chest")
                # CLEAR MEANS HER LOWEST POINT OVER THE HIGHEST THING UNDER
                # HER --- a fixed height stalled twice: this crib's rail
                # stands at her dangling feet, so "high enough" is not a
                # constant, it is her hanging span over whatever is below.
                lowest = float(np.min(p[:, 1] - self.her.radius))
                top = float(np.max(under))
                clear = lowest > top + 0.08
                span = float(p[J["chest"], 1]) - lowest
                goalY = max(LIFT_TO, top + span + 0.12)
                for name, off in (("chest", 0.0), ("pelvis", -pc)):
                    k = J[name]
                    was = (self.her.carry_at[k].copy() if self.her.carrying[k]
                           else p[k].copy())
                    # ...to the TRAINING SQUARE in front of her mirror ---
                    # never (0,y,0): the middle of this room is where the
                    # cot itself stands, and she hovered over her own crib
                    goal = np.asarray([9.0, goalY + off, 9.0], np.float32)
                    d = goal - was
                    if not clear:
                        d[0] = 0.0
                        d[2] = 0.0               # straight up, THEN across
                    n9 = float(np.linalg.norm(d))
                    at = (was + d * (min(self.GENTLE_TICK, n9) / n9)
                          if n9 > 1e-6 else was)
                    self.her.carry_at[k] = at
                    self.her.carried[k] = 1.0
                    self.her.carrying[k] = True
                return
            # ...over open floor but still high, SET HER DOWN --- the first
            # release fired the moment her feet swung past the cot's edge
            # and dropped her from the lift.  The hands descend to standing
            # height first; only low do they let go.
            if (self.her.carrying[J["chest"]]
                    and float(p[J["chest"], 1]) > 0.55):
                pc = seg("pelvis", "chest")
                for name, off in (("chest", 0.0), ("pelvis", -pc)):
                    k = J[name]
                    was = self.her.carry_at[k].copy()
                    goal = np.asarray([9.0, 0.50 + off, 9.0], np.float32)
                    d = goal - was
                    n9 = float(np.linalg.norm(d))
                    at = (was + d * (min(self.GENTLE_TICK, n9) / n9)
                          if n9 > 1e-6 else was)
                    self.her.carry_at[k] = at
                    self.her.carried[k] = 1.0
                    self.her.carrying[k] = True
                return
            # ...and only once she is low over bare floor do the hands LET
            # GO, or she dangles from them in mid-room for ever
            if self.helper == "walk":
                for name in ("chest", "pelvis"):
                    if not self.her.pinned[J[name]]:
                        self.her.carrying[J[name]] = False
                        self.her.carried[J[name]] = 0.0
        if self.helper == "walk":
            # two hands under her arms --- at a SLIGHT SQUAT, never at full
            # extension.  Measured 2026-08-30: held at full-extension height
            # her feet dangled at y 0.38 and the positive-support reflex ---
            # the one thing that extends a loaded leg --- could never fire,
            # so she hung and flailed ("impossible to teach her", his
            # words).  WALK_SQUAT stands her so her feet MUST press the
            # floor and carry a share of her weight; the reflex does the
            # rest, which is exactly what hands under armpits do.
            leg = 0.5 * (seg("hiL", "knL") + seg("knL", "foL")
                         + seg("hiR", "knR") + seg("knR", "foR"))
            ground = 0.5 * float(under[J["foL"]] + under[J["foR"]])
            shY = ground + self.WALK_SQUAT * leg + seg("pelvis", "chest")
            carry("shL", shY, 0.5 * leg)
            carry("shR", shY, 0.5 * leg)
            # THE UPRIGHT HOLD, measured whole (2026-08-30): shoulder-only
            # hands let her pivot forward into a horizontal hang --- head
            # 110 deg off vertical, the trunk-upness gate keeping the
            # righting reflex OFF.  Three hands make a held child: the
            # pelvis stacked UNDER the shoulders (trunk vertical, upness
            # 0.97), a light head hand above the shoulders' ACTUAL height
            # (never the target's --- stacking on the sag hyperextended
            # her), and the retargeted neck righting underneath.  Result:
            # head 6 deg off vertical.  Her own neck commands still
            # override the reflex; the hands fix only height and stack.
            pc9 = seg("pelvis", "chest")
            neckUp = seg("chest", "neck") + seg("neck", "head")
            mx9 = 0.5 * (float(p[J["shL"], 0]) + float(p[J["shR"], 0]))
            mz9 = 0.5 * (float(p[J["shL"], 2]) + float(p[J["shR"], 2]))
            k9 = J["pelvis"]
            if not self.her.pinned[k9]:
                was9 = (float(self.her.carry_at[k9, 1])
                        if self.her.carrying[k9] else float(p[k9, 1]))
                lift9 = float(np.clip((shY - pc9) - float(p[k9, 1]),
                                      -self.GENTLE_TICK, self.GENTLE_TICK))
                self.her.carry_at[k9] = [mx9, min(was9 + lift9,
                                                  shY - pc9 + 0.3 * pc9), mz9]
                self.her.carried[k9] = 1.0
                self.her.carrying[k9] = True
            k9 = J["head"]
            if not self.her.pinned[k9]:
                shNow = 0.5 * (float(p[J["shL"], 1]) + float(p[J["shR"], 1]))
                was9 = (float(self.her.carry_at[k9, 1])
                        if self.her.carrying[k9] else float(p[k9, 1]))
                lift9 = float(np.clip((shNow + neckUp * 0.9) - float(p[k9, 1]),
                                      -self.GENTLE_TICK, self.GENTLE_TICK))
                self.her.carry_at[k9] = [mx9, min(was9 + lift9,
                                                  shNow + neckUp * 1.1), mz9]
                self.her.carried[k9] = 1.0
                self.her.carrying[k9] = True
        elif self.helper == "crawl":
            # one hand under her chest, one under her hips; head hers
            arm = 0.5 * (seg("shL", "elL") + seg("elL", "haL")
                         + seg("shR", "elR") + seg("elR", "haR"))
            thigh = 0.5 * (seg("hiL", "knL") + seg("hiR", "knR"))
            ground = 0.25 * float(under[J["haL"]] + under[J["haR"]]
                                  + under[J["knL"]] + under[J["knR"]])
            carry("chest", ground + 0.90 * arm, 0.5 * arm)
            carry("pelvis", ground + 0.90 * thigh, 0.5 * thigh)
        else:                                   # cradle: the wrap
            for name, k in J.items():
                if name != "head":
                    self.her.pinned[k] = True
                    self._helperPins[k] = True
            self.touch = {"body_left": 0.45, "body_right": 0.45,
                          "hand_left": 0.45, "hand_right": 0.45,
                          "foot_left": 0.45, "foot_right": 0.45}
        # `heldAt` is NOT refreshed here: it is the POSTS' clock.  Written
        # every tick, it kept every posted pin alive for ever while a
        # helper was on --- the exact girl-left-pinned-by-a-closed-browser
        # the timeout exists for.

    def _walker(self) -> None:
        """The walker's frame and its floor --- his design, 2026-08-30:
        *"we have to draw some lines, some limit to guide a bit her legs
        for walking"*.  Two world facts, nothing in her brain:

        RAILS.  While the walk helper holds her, her knees and feet keep
        only forward-and-back freedom: sideways past the frame they meet a
        wall, exactly as a baby-walker's frame or a parent's ankles.  The
        clamp moves `pos` and `prev` together, so a rail is a wall and not
        a spring --- no energy enters.

        THE MOVING FLOOR.  A foot that presses the ground is carried
        gently backward, the way a slow treadmill drags a loaded foot ---
        the stepping reflex's own trigger: the loaded leg goes back,
        unloads, and swings forward.  She lives real stepping, tick after
        tick, and her spindle records it; nothing is taught.
        """
        # ...AND HIS OWN HANDS COUNT --- his ask, 2026-08-30: *"just let me
        # hold here by myself... floor is not moving like it was.  Just fix
        # this."*  The frame and the moving floor serve whoever holds her
        # upright: the walk helper's hands or HIS grab (posted pins).
        held9 = bool((self.her.pinned & ~self._helperPins).any())
        auto = self.helper == "walk" or held9
        # ...and HIS BUTTON above all --- *"do one more button in helpers,
        # and I will turn on moving floor by my own"*: None follows the
        # hands (auto), True drives the floor whatever holds her or
        # nothing does, False parks it.
        drifting = auto if self.floorMoves is None else bool(self.floorMoves)
        if not (auto or drifting):
            return
        p, J = self.her.pos, JIDX
        under = self.room.under(p, self.her.radius)
        if drifting:
            # THE FLOOR IS A BELT --- his correction, 2026-08-30: *"when I
            # press moving floor and don't do nothing, she has to slide
            # because floor moves."*  The first cut dragged only her feet,
            # in a body-relative direction that means nothing to a lying
            # baby, so nothing visibly moved.  A real belt has one world
            # direction and carries WHATEVER touches it: a lying body
            # slides whole, a loaded foot under a held stand is carried
            # back, a lifted foot escapes it.  pos and prev move together,
            # so the belt is a floor and not a force.
            # THE TRAINING SQUARE, WANDERING --- his design, 2026-08-30:
            # a 2 x 2 m square in front of her mirror, whose floor moves
            # *"in all direction randomly.  Left, right, forward, back...
            # she will find the pass by her leg.  I don't need just rolling
            # in front --- she just stuck in front of a wall, and that's
            # it.  This is not practice."*  A random cardinal, held one to
            # three seconds, then another --- zero mean drift, so nothing
            # piles her anywhere, and her legs answer every direction.
            # Never on furniture (a knee-high ground is not the floor), and
            # never outside the square.
            self._beltAge += 1
            if self._beltAge >= self._beltTill:
                dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[
                    int(self._beltRng.integers(4))]
                self._beltDir = np.asarray([dx, 0.0, dz], np.float32)
                # 7-15 s a direction --- his correction: *"direction
                # changing too fast.  So grip is bad... not enough time to
                # prepare herself."*  One to three seconds gave her legs no
                # time to answer before the question changed.
                self._beltTill = self._beltAge + int(
                    self._beltRng.integers(210, 450))
            belt = self._beltDir * self.WALK_DRIFT
            cx9, cz9, half9 = self.WALK_ZONE
            for k in range(len(p)):
                if self.her.pinned[k] or float(under[k]) > 0.15:
                    continue
                if (abs(float(p[k, 0]) - cx9) > half9
                        or abs(float(p[k, 2]) - cz9) > half9):
                    continue
                if float(p[k, 1]) <= float(under[k]) \
                        + float(self.her.radius[k]) + 0.02:
                    self.her.pos[k] += belt
                    self.her.prev[k] += belt
            # THE RAILS SHE CAN SEE --- his insight, 2026-08-30: *"she
            # don't understand why she moves because everything around her
            # is stay in one place... these rails on floor, she has to see
            # them."*  Three real things riding the belt through her world:
            # her own eye watches them slide, and the sliding explains her
            # feet.  They wrap inside the square's clear strip, never into
            # her cot.
            if not hasattr(self, "_rails"):
                self._rails = [
                    np.asarray([8.60, 0.11, 8.40], np.float32),
                    np.asarray([9.30, 0.11, 9.10], np.float32),
                    np.asarray([8.90, 0.11, 9.60], np.float32)]
            for i9, r9 in enumerate(self._rails):
                r9 += belt
                for a9, lo9, hi9 in ((0, 8.20, 9.85), (2, 8.20, 9.85)):
                    if r9[a9] < lo9:
                        r9[a9] = hi9
                    elif r9[a9] > hi9:
                        r9[a9] = lo9
                self.room.put("rail%d" % i9,
                              (float(r9[0]), float(r9[1]), float(r9[2])),
                              size=0.12)
                # ...AND A RAIL IS SOLID --- his order: *"they has to be
                # fully real for her... maybe she will trying to avoid
                # them."*  A joint inside a rail's ball is pushed out
                # horizontally, pos and prev together: a shove from a
                # slow-moving real thing, not a force from nowhere.
                d9 = p[:, [0, 2]] - r9[[0, 2]][None, :]
                dist9 = np.linalg.norm(d9, axis=1)
                touch9 = (dist9 < 0.12 + self.her.radius) \
                    & (p[:, 1] < r9[1] + 0.14) & ~self.her.pinned
                if np.any(touch9):
                    n9 = d9[touch9] / np.maximum(dist9[touch9], 1e-6)[:, None]
                    push9 = ((0.12 + self.her.radius[touch9])
                             - dist9[touch9])[:, None] * n9
                    self.her.pos[touch9, 0] += push9[:, 0]
                    self.her.pos[touch9, 2] += push9[:, 1]
                    self.her.prev[touch9, 0] += push9[:, 0]
                    self.her.prev[touch9, 2] += push9[:, 1]
        elif hasattr(self, "_rails"):
            for i9 in range(3):
                self.room.take("rail%d" % i9)
            del self._rails
        if not auto:
            return
        hipAx = p[J["hiR"]] - p[J["hiL"]]
        hipAx[1] = 0.0
        n = float(np.linalg.norm(hipAx))
        if n < 1e-6:
            return
        hipAx = hipAx / n
        mid = 0.5 * (p[J["hiR"]] + p[J["hiL"]])
        rail = (0.5 * float(np.linalg.norm(p[J["hiR"]] - p[J["hiL"]]))
                + self.WALK_RAIL)
        for name in ("knL", "knR", "foL", "foR", "toL", "toR"):
            k = J[name]
            d = float(np.dot(p[k] - mid, hipAx))
            if abs(d) > rail:                       # the frame is a wall
                shift = (d - rail if d > 0 else d + rail) * hipAx
                self.her.pos[k] -= shift
                self.her.prev[k] -= shift

    def hold(self, holds: dict | None = None) -> None:
        """HANDS ON HER --- `{joint: [x, y, z]}`, each held where it is put.

        **A GRABBED HAND IS A GRABBED HAND.**  This used to call
        `Ragdoll.handle`, which shoves her CHEST toward the point --- so taking
        hold of one finger dragged all of her.  His: *"when i grab her she
        following by whole body instead of separete part"*.  He is right, and
        her flesh already has the only correct answer: `pinned`.  Her solver's
        own line, `ragdoll.py:1197`: *"a pinned joint goes where it is put
        whatever she does"* --- so the joint follows the hand and the rest of
        her hangs off it, which is what being held by one hand IS.

        Several at once, because that is what the helpers are: walking is two
        hands under her arms, crawling is one under her chest and one under her
        hips.  One shape for one hand and for four.
        """
        # A HAND IS GENTLE OR IT IS A CATAPULT.  Every application used to
        # TELEPORT the joint to the target --- and through the tunnel his
        # posts arrive sparse, so each one flung her a whole stale step:
        # "because of inertia of her helper there is impossible to stop
        # her".  A hand now moves her at most GENTLE per post, toward where
        # it wants her --- firm, and never a throw.
        GENTLE = 0.15

        def toward(cur, want):
            d = np.asarray(want, np.float32) - cur
            n = float(np.linalg.norm(d))
            return (np.asarray(want, np.float32) if n <= GENTLE
                    else cur + d * (GENTLE / n))

        self.letGo()
        for name, at in (holds or {}).items():
            k = JIDX.get(str(name))
            if k is None or at is None:
                continue
            self.her.pos[k] = toward(self.her.pos[k], at)
            self.her.pinned[k] = True
        self.heldAt = len(self.ticks)

    def _convertHis(self, pcm):
        """His word cut into her frames --- AS HER EAR WILL NAME IT LIVE.

        IT USED TO RESAMPLE x1.75 FIRST, and that was the fork.  Her live ear
        names ONE TICK of arriving air RAW (11.1 ms), at his own pitch:
        `window.speech` is a
        draining queue, so her ear cannot advance faster than he speaks.  A
        word captured here and shifted first was filed in a DIFFERENT index
        space from the one she hears him in --- measured 2026-08-31, not one id
        of his word played into her room matched an id computed for it here,
        and nothing raised, because an id nothing matches is simply a word she
        never answers.

        The register shift belongs to her MOUTH, once, in the table
        (`measure.mouth`): a piece is FILED under the sound her ear names and
        PLAYS that same moment x1.75.  One index space in, one translation out.

        Word-sized only: a quarter second to two and a half."""
        n = int(len(pcm))
        if not (speech.RATE * 0.25 <= n <= speech.RATE * 2.5):
            return None
        # THROUGH THE DOOR (his rule, 2026-09-04): the frames the mother keeps
        # are what her ear makes of his raw word --- prepared once, stored
        # ready; the raw pcm stays beside them for his own ears.
        frames = wordFrames(pcm, speech.RATE)
        return frames or None

    def _eyesAsHers(self) -> None:
        """A REFLEX IS HER WANTED OUTPUT --- his, 2026-09-06: *"you impact on her
        eyes directly instead of making it her wanted output, so after it her
        eyes will stay at the same position."*  Until then a reflex wrote her
        eye muscles for its moment and never her output; her output carries
        forward and fades, so the next tick her eyes snapped back, and the
        turn was on her spindles but never in her record as her act.  Now the
        reflex's eye levels are written into her output lines too: they carry
        forward like any act of hers, and an experience that paid replays
        them."""
        out = self.tick.life.output
        want = {len(AXES) + k + 1: float(self._eyeWant[k]) for k in range(EYES)}
        seen9 = set()
        for m in out.motor:
            if int(m.id) in want:
                m.lvl = want[int(m.id)]; seen9.add(int(m.id))
        for k, v in want.items():
            if k not in seen9:
                out.motor.append(Motor(id=int(k), lvl=float(v)))

    def _looking(self):
        """What her gaze holds --- the room's own geometry, for the mom.
        A thing within ~25 degrees of where she is looking and four metres
        away is being looked at.  Candidates are the window and the room's
        things --- the mom's own marker included, so she can name HERSELF.

        HER GAZE IS HER EYES' AXIS, NOT HER SKULL'S.  This read the
        head-to-face line, which was the whole truth while her eyes could
        not move --- they were welded straight ahead.  Now they do move
        (`_aimEyes`), and a mother naming what the baby's SKULL points at
        while the baby looks somewhere else names the wrong thing.
        `_glimpse` is the one answer to where she is looking, and her
        picture is already rendered on it: one implementation (rule 4).
        Measured the night her eyes were wired: her mother named ONE word
        in ninety seconds --- "nowhere", the baby's own name, which is what
        she says when the baby looks at nothing."""
        head = self.her.pos[JIDX["head"]]
        _, _, _, g = self.her._glimpse()
        g = np.asarray(g, np.float32)
        n = float(np.linalg.norm(g))
        if n < 1e-6:
            return None
        g = g / n
        best, bestDot = None, 0.9
        cands = [(str(getattr(self.window, "showing", None) or "window"),
                  np.asarray(self.window.at, np.float32))]
        if self.board is not None:
            # THE BOARD IS WHATEVER IT IS SHOWING.  Its name changes with its
            # card, so her mother says the word that belongs to the picture in
            # front of her --- the pairing, made of two things she already has.
            cands.append((str(getattr(self.board, "name", "window")),
                          np.asarray(self.board.at, np.float32)))
        for name, thing in list(self.room.things.items()):   # see light._scene
            cands.append((name, np.asarray(thing.at, np.float32)))
        for name, at in cands:
            d = at - head
            m = float(np.linalg.norm(d))
            if m < 0.05 or m > 4.0:
                continue
            dot = float(np.dot(d / m, g))
            if dot > bestDot:
                best, bestDot = name, dot
        # THE TV IS A THING; PAPA IS THE PERSON ON IT --- his design,
        # 2026-08-30: *"TV is TV, but when she look exactly on me in the
        # TV, she has to produce papa."*  The body always knows whether his
        # live face is arriving; a lit screen she gazes at IS him, a dead
        # one is furniture.  One thing, one word --- and he is not a thing.
        if best == "window" and time.time() - getattr(self, "_paSeen", 0) < 2.0:
            return "papa"
        return best

    def letGo(self) -> None:
        """Every hand off her.  She falls, which is the truth."""
        self.her.pinned[:] = False
        self.her.carrying[:] = False
        self.her.carried[:] = 0.0
        self.touch = None

    def give(self, milk: float) -> None:
        """SOMETHING AT HER LIPS.  Her body writes the line; she decides
        nothing about it and is told nothing about what it is."""
        self.atMouth = max(0.0, min(1.0, float(milk)))
        # ...and the mom, if she is here, says the food's name --- every
        # feed, the same form: the naming curriculum's first noun
        self.teacher.fed(len(self.ticks))

    def _lap(self, name: str) -> None:
        """The time since the last lap, folded into `stageMs[name]` as a
        running mean over ~300 ticks (ten of her seconds)."""
        now = _perf()
        d = (now - self._t0) * 1000.0
        was = self.stageMs.get(name)
        self.stageMs[name] = d if was is None else was + (d - was) / 300.0
        self._t0 = now

    def _looks(self) -> None:
        """HER LOOK, ON ITS OWN THREAD --- a WORKER on a snapshot her own tick
        takes, never a reader of her state.  The tick is her sound's grain
        (his 11 ms blueprint); her eye is a slower sense and runs beside it.

        THE SNAPSHOT IS TAKEN ON THE TICK THREAD, at exactly the point the
        inline look rendered (stage 0), under her lock: her eyes' axes and
        eye point, her joints, the swing since the last look (zeroed there).
        A first cut had this thread copy those itself right after a look
        tick --- and raced her reflex: the saccade ordered on that tick lands
        The render and the pieces run outside the lock; the finished look is
        published under it in one step.
        """
        while self.going:
            with self.ticked:
                self.ticked.wait(timeout=0.1)
                job = self._lookJob
                self._lookJob = None
            if job is None:
                continue
            self._eyeBusy = True
            n, eye, up, right, fwd, pos, radius, swung = job
            t9 = _perf()
            pic = light.see(self.room, [self.window, self.board],
                            [(eye, up, right, fwd)], body=(pos, radius))
            picture = np.asarray(light._home(pic))[:, :, 0].reshape(
                light.RETINA_H, light.RETINA_W, light.CONES)
            # GRAB THE SIMILARITY AND FORGET THE PICTURE.  Three numbers
            # survive a look --- what it is, and where.  Nothing else of it is
            # kept for her; `self.picture` is the page's mirror alone.
            one = wholePicture(picture)
            with self.lock:
                self.picture = picture
                self.onePicture = one
            self._eyeBusy = False
            d = (_perf() - t9) * 1000.0
            was = self.stageMs.get("look-thread")
            self.stageMs["look-thread"] = d if was is None else was + (d - was) / 50.0

    def oneTick(self) -> None:
        her = self.her
        # ...AND WHAT WAS SAID BEFORE SHE WAS AWAKE IS NOT HERS TO HEAR.
        # She drains exactly one tick of sound a tick, so anything that piled up
        # while she was being born is late for ever --- measured at 13.7 s, flat,
        # on a life that never missed a tick.  Once, on her first.
        if not len(self.ticks):
            self.window.catchUp()
        tick = Tick()
        self._t0 = _perf()
        tick.time = len(self.ticks) * TICK_SECONDS
        lines = {}

        out = self.tick.life.output
        # HER POSE COMES OFF HER TICK, ALL OF IT, AND IS BUILT FRESH EACH TIME.
        #
        # His, 2026-08-26: *"i never tald that her repeat of her full body
        # movement able to conteine only one muscle per tick"*.  It used to
        # write ONE muscle into a standing array that lived here --- so her pose
        # was her body's private state, the tick was not a snapshot of her, and
        # replaying a remembered moment set one muscle out of twenty-six against
        # whatever her body happened to be holding.
        #
        # `self.order` is now a hand-off to `Muscles` and not a memory: nothing
        # survives in it from the tick before, because `live.pose` carries the
        # whole vector forward on the tick where it belongs.
        self.order[:] = 0.0
        for one in out.motor:
            at = int(one.id) - 1
            if 0 <= at < len(AXES):
                self.order[at] = float(one.lvl)
            elif len(AXES) <= at < len(AXES) + EYES:
                # HER EYES, ORDERED LIKE ANY MUSCLE.  0..1 as every axis is,
                # and her eye's own travel is -1..1 of EYE_REACH, so a half is
                # straight ahead --- the same shape `degrees_of` gives a bone.
                self._eyeWant[at - len(AXES)] = float(one.lvl)
            elif len(AXES) + EYES <= at < len(AXES) + EYES + len(VOICE_PARTS):
                # HER MOUTH, ORDERED LIKE ANY MUSCLE --- one articulator each,
                # 0..1; `Voice.say` gives them mass and turns them into air.
                self._mouthWant[at - len(AXES) - EYES] = float(one.lvl)
        # ...AND WHAT SHE ASKED TO SAY, if she asked for a sound by its name.
        self._sayWant = (int(out.sound.id), float(out.sound.lvl))
        # ...AND HER EYES GO WHERE SHE ORDERED THEM.  0..1 as every axis is,
        # her eye's own travel is -1..1 of EYE_REACH, so a half is straight
        # ahead --- the same shape `degrees_of` gives a bone.
        # ...UNLESS A SOUND JUST ROSE: the turn toward it takes her eyes'
        # horizontal pair for its moment, like the startle takes her neck
        # (his: *"never mind what you're thinking, you scare"*).  A positive
        # pan aims at her right (orient.py), and a positive side is her
        # right ear hearing more.
        # ...AND IT FILLS HER SILENCE; IT DOES NOT TAKE HER EYES.  His, 2026-09-10:
        # *"reflexes, i guess they can be a problem for learning, they pull her
        # somewhere but she even dont know about it ... if she able learn how to
        # speak she able learn how to walk and watch to needed direction."*
        #
        # This block ran AFTER her own eye order was unpacked into `_eyeWant`
        # (above) and overwrote both horizontal muscles with no test of what she
        # had asked --- the only mover in her that could beat her own decision.
        # And the damage was worst in her commonest case: her mother stands at
        # one fixed place and answers her constantly, so a sound arriving DEAD
        # AHEAD gives `side9 ~ 0`, `pan9 ~ 0`, and the block wrote 0.0 into both
        # eye lines --- it did not turn her toward anything, it CANCELLED
        # whatever she was doing, and `_eyesAsHers` signed her name to it.  Her
        # record then said "I ordered my eyes to rest" for an act that was not
        # hers and that she could not refuse.
        #
        # Now it takes the same shape her born-in saccade already has
        # (`_reflexEyes`, gated on all four eye muscles at exactly rest): it
        # speaks only when she is not using her eyes, and any order of hers on
        # the same tick locks it out.  Biology has the same order --- the
        # newborn turn to sound is subcortical, dips at 4-6 weeks and returns
        # under cortical control at about four months, so what develops is the
        # ability to SUPPRESS it.  A reflex she cannot suppress is not a floor
        # she learns from; it is a hand on her.
        # ...AND A SOUND THAT IS NOT TO ONE SIDE TURNS HER NOWHERE.  His,
        # 2026-09-10: *"every animal and people also has this measurement by
        # ears to help turn eyes, so we have to keep it, but write it to her
        # decision so she will able to learn this dependency."*  The turn stays,
        # and it stays hers: `_eyesAsHers` writes it into her own output lines,
        # so what her ears differed by and what her eyes then did stand side by
        # side in the same row and the dependency is there to be found.
        #
        # What is gone is the zero.  `side9 = (R - L) / loud`, so a source DEAD
        # AHEAD gives `side9 ~ 0` and `pan9 ~ 0`, and this block wrote 0.0 into
        # both eye lines --- it turned her toward nothing and CANCELLED whatever
        # she was doing, then signed her name to it.  Her mother stands at one
        # fixed place (`teacher.py: SPEAKS`) and answers her constantly, so that
        # was her commonest case, not a corner one.  A real ear does the same
        # thing: a centred sound produces NO TURN, never a reset.  Below her own
        # resolution there is no side to hear, so there is nothing to say.
        # ...AND NOTHING TURNS HER EYES BUT HER.  His, 2026-09-10, and it is the
        # whole of it: *"if her input left rise she has to understand that
        # something is there, but IT IS HER DECISION to turn or not, but her eyes
        # already has this data for analyze, so during the time she will connect
        # those two inputs."*
        #
        # Her two ears are already lines in the same row as her eyes and her eye
        # muscles (`lines[50]`, `lines[51]` above; `mind/life.py: lines()`).  A
        # left ear rising and a thing coming into view when she looks left are
        # two levels in one row, a tick apart --- which is exactly the thing her
        # mind is for.  Turning her for her taught her nothing: it put the answer
        # in her output before she had asked the question, and her record then
        # showed a look she had not chosen.  The measurement stays; the hand
        # comes off.  She learns to look the way she learned to make a sound.
        # WHAT THE PAIRS DISAGREE BY.  Both pulling equally is straight
        # ahead, exactly as it is in a real orbit.
        w = np.asarray(self._eyeWant, np.float32)
        her.eye_at[0] = float(np.clip(w[1] - w[0], -1.0, 1.0))
        her.eye_at[1] = float(np.clip(w[3] - w[2], -1.0, 1.0))
        # ...AND THE REFLEX ONLY SPEAKS WHERE SHE IS SILENT.  His rule:
        # *"when she decide to do some, but she has some call from reflexes,
        # she has to choose reflexes just if it necessary ... everything of it
        # has to be weighted"*, and *"she hasn't to have anything that works
        # next to her"*.
        #
        # HER DECISION ALWAYS WINS.  If she is pulling any eye muscle this
        # tick, the reflex does not touch her eyes at all --- it cannot
        # override her, only fill a silence.  That is the biology too: a
        # newborn's saccade to a peripheral thing is subcortical and present
        # at birth, and what develops over the first months is exactly the
        # ability to suppress and steer it.
        #
        # returns nothing unless the peak of her own eye stands out from the
        # rest of it by `STRENGTH_FLOOR` --- so the reflex is silent in a
        # room where everything looks the same, and speaks when something
        # does not.  Nothing new decides when it is worth it.
        #
        # AND IT IS NOT INVISIBLE.  What it does arrives on her eye spindles
        # like any pull, so she feels where her eyes went and can learn what
        # follows --- which is the only way a reflex is hers rather than
        # something beside her.
        # ...ONCE A LOOK, NOT ONCE A TICK.  A saccade is ballistic and rare:
        # an infant fixates for hundreds of milliseconds and then jumps.  Fed
        # every tick it is a tremor, and measured as one --- her head swung 46
        # degrees a tick across a 100 degree field, worse than either of the
        # pulls this replaced.  Her eye is asked when her LOOK is (5 a second,
        # `LOOKS_PER_SECOND`), which is her own seeing rate and not a new
        # number.
        # ...AND WHAT HER EYES COULD NOT REACH, HER NECK CARRIES.  His own
        # words, 2026-08-15: *"you has to be the part of moving not part of
        # eyes, just simply correct input to eyes when you need it, rest would
        # do head after"*.  A PULL (`ORIENT_BIAS`) laid on top of whatever her
        # mind ordered, never a command --- as her head comes round her eyes
        # reach further, the leftover shrinks and the pull lets go by itself.
        # It is written into her ORDER, so following a thing is something SHE
        # DID, on the same lines every other act of hers is filed on.
        # A STARTLE OVERRIDES HER, AND ONLY A STARTLE.  Written after her own
        # order, so what she decided is simply overwritten on her neck for the
        # third of a second the turn lasts --- "never mind about what you're
        # thinking right now because you scare".
        # HER HEAD HELPS HER EYES --- his, 2026-09-06: *"first go the eyes when
        # the object leaves her frame; her head has to just help her eyes in
        # the same way, putting her input that she has to follow."*  When her
        # eyes are turned aside, her neck twists the same way, a resolution
        # a tick, toward where her eyes aim, so her eyes can come back to the
        # middle and keep the thing; written as her wanted output, so it
        # carries forward and is hers to replay.  Eyes straight, it writes
        # nothing and the carried order fades to rest.  (Until now her neck
        # followed nothing: the startle wrote it directly for a third of a
        # second, and the following was a comment.)
        # ...AND HER NECK IS HERS.  His word, 2026-09-10, on the same rule as her
        # ears: a coupling between two of her own lines is a thing for her to
        # FIND, not a thing to be given.  Where her eyes aim is a level in her
        # row (`her.eye_at`, on her eye motors) and where her neck stands is a
        # level in the same row (her neck.twist spindle), so "my neck follows my
        # eyes" is two numbers a tick apart --- exactly what her mind is for.
        # Writing it for her put the answer in her output before she had asked,
        # and every experience that held a look also held a neck-turn she never
        # ordered.  She learns to carry her head the way she learns to look and
        # the way she learned to make a sound.
        self._helperHold()
        self._lap("order")
        pushing = self.muscles.pull(self.order)
        # A HAND THAT HAS STOPPED SAYING IT IS THERE IS NOT THERE.  The page
        # posts a hold about 14 times a second and the helpers every 400 ms, so
        # a second of silence is a hand that has gone --- and a girl left pinned
        # by a closed browser would hang in the air for ever.
        # ...AND A HAND ON THE FAR SIDE OF A TUNNEL POSTS SLOWLY.  One
        # second of silence used to mean letGo, so his remote grabs kept
        # being dropped mid-hold --- "she frees herself".  Five seconds
        # covers the road's worst gaps; a real release still says let_go
        # and is instant.
        # ...and only what was POSTED expires.  The body's own hands are
        # in-process and never fall silent: helper carries stay, and the
        # wrap's own pins (marked in `_helperPins`) stay.  A forgotten
        # finger pin goes, and so does a lone touch --- posted pressure
        # with no hold used to sit on her skin for ever.
        posted = self.her.pinned & ~self._helperPins
        if ((posted.any()
             or (self.touch is not None and self.helper != "cradle"))
                and len(self.ticks) - self.heldAt > 150):
            self.her.pinned &= self._helperPins
            if self.helper != "cradle":
                self.touch = None
        her.step(np.clip(self.order, 0.0, 1.0), pushing)
        self._walker()
        self.muscles.spend(pushing)
        felt = her.reading(np.clip(self.order, 0.0, 1.0))
        self._lap("physics")
        # ...AND EVERY MUSCLE ANSWERS, EVERY TICK.  One spindle of twenty-six
        # meant twenty-five of her limbs said nothing about themselves, so she
        # could not have known what her own body was doing even in principle.
        # The pair `out.motor[k] -> in.spindle[k]` is her ability to move that
        # part; there are twenty-six of them and she has to be able to find out
        # about all of them.
        spun = np.asarray(felt["spindle"]).reshape(len(AXES), -1)
        tick.life.input.spindle = [Spindle(id=k + 1, lvl=float(spun[k, -1]))
                                   for k in range(len(AXES))]
        # ...AND IF SOMETHING IS HOLDING HER, WHAT IT DID IS HERS (his, 2026-09-08)
        # ...AND WHERE HER EYES ARE POINTED, felt the same way.  A muscle she
        # cannot feel is a muscle she cannot learn: the pair `out.motor[k] ->
        # in.spindle[k]` is her ability to move that part, and her eyes are
        # now two of those parts.
        # ...and she feels each eye muscle by how hard IT is pulling, which
        # is what a spindle reports for every other muscle she has.
        tick.life.input.spindle += [
            Spindle(id=len(AXES) + k + 1, lvl=float(self._eyeWant[k]))
            for k in range(EYES)]
        # ...AND HER MOUTH, felt by where each articulator actually IS ---
        # `Voice._shape`, the shape with mass her larynx carried into this
        # tick --- so a sound is something she did and can find out about,
        # spindle for spindle, like every other part of her.
        tick.life.input.spindle += [
            Spindle(id=len(AXES) + EYES + k + 1,
                    lvl=float(self.voice._shape[k]))
            for k in range(len(VOICE_PARTS))]

        # HER EARS ARE FOR WHAT IS NOT HERS.  His, 2026-08-26: *"her ears do not
        # mix sources at all she has echo for it"*, *"it is her respnse from her
        # voise like spoindel for muscle"*.
        #
        # Her own voice used to be appended here as a source AT HER OWN MOUTH,
        # so everything she said arrived back through her ears summed with
        # everything else --- and then something downstream would have had to
        # subtract her out again to know whose sound it was.  That subtraction is
        # what the old tree spent itself on, and it is not needed: **her mouth
        # already answers her mouth.**  One shape, twice:
        #
        #     out.motor  ->  in.spindle     the muscle answers the muscle
        #     out.sound  ->  in.echo        her voice answers her voice
        #     in.sound                      somebody ELSE, and only somebody else
        #
        # So nothing of hers goes into `sounding`, and `in.sound` is now what its
        # name says it is.
        sounding = []
        look9 = self._looking()
        # ONE TICK OF HIM A TICK.  He is already in her register when he reaches
        # the queue (`window.say`), so the door takes exactly a tick's worth and
        # his pace is his own.  (It took REGISTER ticks a tick and resampled
        # them down --- his voice ran 1.75x fast through her.)
        his = self.window.speech(self.earSeconds)
        self.teacher.hears(his, looking=look9)
        # HER EAR LISTENS `LISTENS` BACK TO NAME THIS TICK'S PIECE (hearing.py):
        # the last four ticks of his air, this tick's on the end; a tick with
        # nothing arriving ages the old air out with silence, so a word ends.
        piece9 = (np.asarray(his, np.float32).ravel() if his is not None
                  else np.zeros(int(round(speech.RATE * TICK_SECONDS)), np.float32))
        self._hisAir = np.concatenate([self._hisAir, piece9])[-int(round(speech.RATE * LISTENS)):]
        # WHAT ARRIVES, ARRIVES.  Her ear has a line for how loud it was
        # (`sound.lvl`), so quiet is a low level and not a thing anybody
        # decides for her.  Her ear's own floor (`hearing.GATE`) is physical
        # and stays; a second bar above it was the old design choosing what
        # she is allowed to hear.
        if his is not None or float(np.abs(self._hisAir).max()) > 0.0:
            sounding.append((self.window.at,
                             door(self._hisAir, speech.RATE, LISTENS,
                                  SOUND_HOPS)))
        # HER MOUTH PLAYS A RECORDED PIECE --- his decision, confirmed
        # 2026-08-30: *"we decide that we able record all needed sounds
        # that she need to produce all human letters and choose them by
        # ticks.  Everything pretty easy."*  The synthesizer's whole range
        # and the palette grows only by his certified recordings.
        # HER OWN MOUTH SPEAKS --- his word, 2026-09-02, over the recorded
        # pieces: *"i need to make her more natural so recorderd sounds can
        # be thintetic"*, and 2026-09-01: *"we do not need prelearned sounds
        # anymore, we recorded them because we can't put sound in tick
        # because of all muscles that we need to produce those sounds"*.
        # The seven articulators she ordered (motors 31..37) go through her
        # larynx with mass, one shape a tick (`SOUND_SLIDES`); what comes
        # out is what she MADE, not what she meant.  The recorded bank
        # stays only as the TARGET her instruments measure her against
        # (rule 10: the recording never enters her).
        # ...OR THE PIECE SHE NAMED, walked at her effort.  Her body knows the
        # pose for every sound it can make; asking for one by name is not a
        # shortcut past her mouth --- `play` drives the same articulators, one
        # moment a tick, and what comes out is still what she MADE.
        # ONE WAY TO MAKE A SOUND, AND IT IS HER MOUTH.  His, 2026-09-11: *"we
        # don't need already prepared table or something like this.  So she would
        # produce any sounds.  Yes, it would take a long time.  I know it.  But it
        # would be natural."*  The branch that stood here let her ASK FOR A PIECE
        # into a table of finished sounds instead of pushing the seven muscles it
        # has.  Measured here 2026-09-11: swept blind, those seven reach her own
        # alphabet at similarity 1.000, and two thirds of their range makes air.
        # measures every sound against, hers and his alike --- it is what a
        # similarity is similar TO.  Only the shortcut is gone.)
        made = np.asarray(self.voice.say(self._mouthWant[:, None]), np.float32)
        self._lap("voice")
        self.voice.spend(made)
        sounded = bool(self.voice.pcm.size) \
            and float(np.abs(self.voice.pcm).max()) > 0.0
        # ...AND HER ECHO IS HER EAR'S NAME FOR IT.  A spindle says what the
        # flesh did; for her voice that is the name her own ear gives the
        # air she made --- `align` over the same rows her ear names the
        # world with, so his sounds and hers compare as integers.  It used
        # to be the id she COMMANDED copied back (a recording plays what it
        # was told); a larynx does not, so she must be told.
        # ...AND HER OWN AIR IS NOT ALIGNED, ONLY MEASURED.  His, 2026-09-11:
        # *"what you aligned in her echo if she already produced by herself, and
        # we have to simply store those similarities as is!"*  The slide search
        # exists to drag SOMEBODY ELSE'S voice into her register and onto her
        # clock; her own sound is already in both, so searching offsets on it is
        # a round trip through a conversion with nothing to convert.  One
        # placement, her own timing, and what comes back is what her flesh
        # actually delivered --- not the id she asked for, because she orders
        # seven articulators and the air is whatever they make, and finding that
        # out is how she learns her own mouth.
        # HER EARS ARE THE SPINDLE FOR HER VOICE.  His word, 2026-09-11:
        # *"we don't need it.  She has ears.  Echo, it was cheat because we have
        # a lot of problem with voicing, and I decide to split those sounds.  But
        # now when we have all in one line, all your inputs, where you can put
        # this fucking echo?"*  Nowhere --- and that is the answer.  The echo's
        # whole job was to say THIS ONE WAS MINE, which is a conclusion handed to
        # her, and she is never handed one.  She does not need telling: when the
        # sound that arrives was hers, her `loud` muscle was pushed in the same
        # row a tick earlier, and when it was somebody else's it was not.  Push,
        # detect, push --- the same pattern as every muscle she has.
        #
        # So her own air stops bypassing her head and goes where a real infant's
        # goes: into her ears, with everything else.  Kept here for the senses
        # stage below, which is where it is mixed in.
        # HER OWN AIR IS ALREADY HERS AND DOES NOT PASS THE DOOR.  His word,
        # 2026-09-12: *"in case our ear broke our sounding ... we have to go not
        # through the ear, we have to go directly to her sounding."*  The door
        # brings the OUTSIDE into her register (pitch, formants and pace by
        # REGISTER); her mouth already speaks there, so sending her own voice
        # through it converts her twice and breaks the very sound she made.
        # Both end up in the SAME space --- his by conversion, hers by birth ---
        # which is what makes them comparable at all.
        # ...and it reaches her ears whenever her ear has anything of hers in
        # it --- including the tail of a sound she has just stopped making,
        # which is still in the last `LISTENS` of her air, as his would be.
        made9 = np.asarray(made, np.float32)
        self._madeAir = made if (made9.size and float(made9.max()) > 0.0) else None
        tick.life.input.echo.id = SILENCE
        tick.life.input.echo.similarity = 0.0
        # THE ECHO IS A SPINDLE, NOT AN EAR --- the owner, 2026-08-29: *"That's
        # why I create echo.  Because she shouldn't hear herself in sound
        # input, and that's it."*  The line used to read `made.max()` --- the
        # BANDS, which carry the ear's floor --- so a quiet mouth reported
        # ZERO to her brain while the flesh audibly moved (measured: pcm
        # 0.046 at full command, bands 0.0 below it).  Every soft try came
        # back "you made nothing", no goal could form, and going silent was
        # the correct conclusion from false data.  The spindle reads the
        # flesh, not the air: what her mouth made, at the level it made it.
        tick.life.input.echo.lvl = (float(np.abs(self.voice.pcm).max())
                                    if sounded and self.voice.pcm.size else 0.0)
        #: WHICH ROW OF HER PRELEARNED ALPHABET IT IS --- his structure,
        #: 2026-08-26, *"prelearned similarity id"*.
        #:
        #: IT CARRIES THE SAME NUMBER AS `id` TODAY AND THAT IS NOT HIDDEN.  Her
        #: ear QUANTISES: `alike_of` turns a band frame into one integer and
        #: there is no second identity a sound could have, so there is nothing
        #: else for `id` to hold.  His structure asks for both fields and both
        #: are filled from the one source rather than one of them being a zero
        #: nothing writes.  When a sound gains an identity of its own --- an
        #: instance, a moment, a source --- it goes in `id` and this stays the
        #: alphabet row.
        tick.life.input.echo.similarity = float(tick.life.input.echo.id)
        #: HER MOUTH'S OWN BANDS, kept whole for whoever is watching.  Her brain
        #: only ever gets the id and the level; this is so HE can see what her
        #: voice actually put into the air, beside what arrived at her ears.
        #: They are two different things and they were drawn as one.
        self.voiced = (np.asarray(made, np.float32).max(axis=1) if made.size
                       else np.zeros(SOUND_BANDS, np.float32))

        # THE TEACHER'S TURN --- after the baby's mouth has spoken and
        # before the ear listens, so her answer is one more source in the
        # room, no different from any voice.
        # THE MOM HEARS HER FLESH, NOT THE FLOORED BANDS --- found
        # 2026-08-30, the spindle disease in a second organ: her quiet
        # human-voice clips band to ZERO, so mom "heard" silence and
        # answered nothing for twenty minutes while the baby spoke.
        spoke = (float(np.abs(self.voice.pcm).max())
                 if sounded and self.voice.pcm.size else 0.0)
        # ...and the FRAMES mom collects carry the sound's FORM at
        # reference loudness --- the third organ of the bands-floor
        # disease (2026-08-30): at her real clip loudness the band
        # pictures floor to all-zero, so every utterance mom gathered was
        # judged empty and silently dropped.  The exemplar is the shape;
        # her true loudness rides `spoke` beside it.
        if spoke > 0.0:
            ref9 = self.voice.pcm / max(spoke, 1e-6) * 0.5
            momMade = bands_from_pcm(ref9, speech.RATE, TICK_SECONDS,
                                     SOUND_HOPS)
        else:
            momMade = made
        told = self.teacher.step(len(self.ticks), momMade,
                                 int(tick.life.input.echo.id),
                                 spoke,
                                 his is not None,
                                 looking=look9)
        if told is not None:
            sounding.append((self.teacher.at, told))
        # ...AND THE MOM'S HANDS, on his three rules (2026-08-28).  The
        # earned milk: a word-shape (the same sound twice, close) brings
        # a third of a mouthful, the word arriving FIRST --- her flesh's
        # own satiety rule decides how much a full baby actually draws.
        # The rescue: sixty seconds of continuous crying and the mom's
        # hand slides her chest gently toward the open middle of the
        # room, three seconds, then lets go --- the same carried hand
        # the helpers use, never while a helper already holds her.
        self.teacher.crying(spoke)
        self._lap("mother")
        # THE NIGHT SHIFT: the mom swaps the baby's situation every five
        # minutes --- helper, free floor, the other helper --- so a night
        # alone is a night of varied practice.  She re-applies only when
        # her own slot CHANGES, so his manual presses stand between swaps.
        # THE LIFT: a minute of chatting while going nowhere, and the
        # mom pulls her vertical by one hand to her whole length, holds a
        # beat, and lets her drop --- moving becomes eventful again, and
        # feeling good stops being free.  Never during a helper's hold or
        # a rescue; hands alternate.
        if (self.helper is None and not getattr(self, "_rescueTill", 0)
                and not getattr(self, "_liftTill", 0)
                and self.teacher.still(
                    len(self.ticks), self.her.pos[JIDX["chest"]],
                    float(made.max()) if made.size else 0.0)):
            self.teacher.lifts = getattr(self.teacher, "lifts", 0) + 1
            self._liftHand = JIDX["haL" if (len(self.ticks) // 9000) % 2 else "haR"]
            self._liftTill = len(self.ticks) + 600      # a generous ceiling
            self._liftHigh = 0
            # THE RAMP IS ANCHORED WHERE THE LIFT BEGINS, and it must be
            # anchored HERE --- this is the branch that starts a lift.  It
            # was set only in the bored-baby branch below, so a lift that
            # began without a bored hold before it read an attribute that
            # did not exist: `AttributeError: no attribute '_liftY'` in
            # `oneTick`, which is a HER TICK THREAD DYING --- her body
            # frozen mid-life while the server went on answering with the
            # last numbers it had.  Measured 2026-08-31: she stopped at
            # tick 11,055,601 and nothing said so.
            self._liftY = float(self.her.pos[self._liftHand, 1])
        elif self.helper is not None and self.teacher._bored:
            # A HELD BABY IS NOT A MOTIONLESS CHATTERER --- found
            # 2026-08-30 by the heartbeat watch: `_bored` set on the free
            # floor could only ever be CLEARED by `still()`, which never
            # runs while a helper holds her --- so the silent treatment
            # froze for ever the moment the walker picked her up, and the
            # mom answered nothing for half an hour while the baby spoke.
            # The mom's own hands on her end the treatment, as they would.
            self.teacher._bored = False
            if getattr(self, "_liftHand", None) is not None:
                self._liftY = float(self.her.pos[self._liftHand, 1])
        if getattr(self, "_liftTill", 0):
            kL = self._liftHand
            if len(self.ticks) >= self._liftTill or self._liftHigh > LIFT_HOLD:
                self._liftTill = 0
                self.her.carrying[kL] = False
                self.her.carried[kL] = 0.0           # the drop
            else:
                # the hand climbs from ITS OWN last height, never from
                # hers --- the walker's ramp lesson, paid once already:
                # ramped off her dragged position the lift stalled at 0.42
                self._liftY = min(self._liftY + 0.02, LIFT_TO)
                atL = self.her.pos[kL].copy()
                atL[1] = self._liftY
                self.her.carry_at[kL] = atL
                self.her.carried[kL] = 1.0
                self.her.carrying[kL] = True
                if self._liftY >= LIFT_TO:
                    self._liftHigh = getattr(self, "_liftHigh", 0) + 1
        slot = self.teacher.helperWanted(len(self.ticks))
        if slot != "off" and getattr(self, "_momSlot", "off") != slot:
            self._momSlot = slot
            self.letGo()
            self.helper = slot
        if self.teacher.on:
            if self.teacher.feedWanted(len(self.ticks)):
                self.give(0.33)
            if self.helper is None and self.teacher.rescueWanted():
                self._rescueTill = len(self.ticks) + 90
        till = getattr(self, "_rescueTill", 0)
        if till and self.helper is None:
            k9 = JIDX["chest"]
            if len(self.ticks) < till:
                at9 = self.her.pos[k9]
                d9 = np.asarray([-at9[0], 0.0, -at9[2]], np.float32)
                n9 = float(np.linalg.norm(d9))
                if n9 > 1e-6:
                    at9 = at9 + d9 * (min(0.02, n9) / n9)
                self.her.carry_at[k9] = at9
                self.her.carried[k9] = 1.0
                self.her.carrying[k9] = True
            else:
                self._rescueTill = 0
                self.her.carrying[k9] = False
                self.her.carried[k9] = 0.0
        ears = self.ears = hear(sounding, her)
        # ONE INPUT, AND EVERYTHING IS IN IT.  His word, 2026-09-12: *"she has
        # to have just one input, her ears, and both our tracks --- every sound
        # of that income gets similarity IDs, so even if we spoke together,
        # anyway in one eleven millisecond we have ONE similarity ID for
        # everything, and that is her level of sound."*
        #
        # So her own air arrives at her ears with everyone else's and mixes
        # there.  Two voices at once are still one ID, because that is what the
        # room sounded like.  Nothing separates hers from his --- what makes a
        # sound hers is her own `loud` muscle standing in the same row a tick
        # earlier, which she has without being told.
        air9 = getattr(self, "_madeAir", None)
        if air9 is not None and getattr(air9, "size", 0) and air9.ndim > 1:
            n9 = min(ears.shape[1], air9.shape[0])
            m9 = min(ears.shape[2], air9.shape[1])
            ears[0, :n9, :m9] += air9[:n9, :m9]
            ears[1, :n9, :m9] += air9[:n9, :m9]
        self._madeAir = None
        loudest = int(np.argmax([e.max() for e in ears]))
        arriving = ears[loudest]
        if arriving.max() > 0.0:
            # AND NOW `align` HAS SOMEWHERE TO PUT THE FRAME --- `SOUND_HOPS`
            # placements instead of one.  The loudness comes from the placement
            # it CHOSE, so the id and the level describe the same piece of
            # sound; the id came from a frame and the level from the whole
            # buffer, which are two different pieces whenever they disagree.
            back, heard, alike = align(arriving, None, SOUND_SLIDES)
            at = max(0, arriving.shape[1] - 1 - int(back))
            # ...AND IF IT IS THE FLOOR, IT IS THE FLOOR.  His, 2026-08-26:
            # *"when we make her prelearn file it has to contain id 1 silent
            # flor"*.  Room tone wore the id of a real sound --- 79.5% of
            # everything her ear named from his room was the floor, and **her
            # mouth can make 283**, so she could babble his room back and be
            # paid for it.  A sound arrived on 38 ticks in 40 and a word could
            # never end, because a word ends on a gap.
            #
            # `SILENCE` is HER mouth at rest; `FLOOR` is the ROOM at rest.  Two
            # different nothings, and she has to be able to tell them apart.
            #
            # ...AND IT HAS TO BEAT SILENCE, OR IT IS NOT ALWAYS THERE.
            # MEASURED 2026-09-02, live, on a word taught to her 1,290 times:
            # sound 172 reached her ear six times of six and was named `1` on
            # every one of them.  The only line that can produce `1` is this
            # one, so 172 WAS her floor --- it had become the most-heard id in
            # her life BY BEING TAUGHT.  Playing it alone, inside the word and
            # with the word reversed all gave the same answer, so it was never
            # about order or level: **the most-repeated sound in her world was
            # being deleted, and repeating it is what deleted it.**  Her
            # mother's most-said word goes the same way.
            #
            # The rule above is right and its proxy was wrong.  `heardIds`
            # counts only the ticks that CARRIED sound, and in her room that is
            # about one tick in twenty --- there is no continuous hiss in a
            # simulated room, so the argmax never finds room tone, it finds
            # whatever was said most.  His own recording is what says how to
            # tell them apart: id 283 was 47.9% of EVERY FRAME and silence was
            # 1.1%.  Room tone outnumbers silence; a word never does.  So
            # silence is counted too and the floor has to beat it --- a
            # comparison between two things she measured, with no constant in
            # it, which is what the rule claimed for itself all along.
            if heard != SILENCE:
                self.heardIds[heard] = self.heardIds.get(heard, 0) + 1
                floor, often = max(
                    ((k, c) for k, c in self.heardIds.items() if k != SILENCE),
                    key=lambda kv: kv[1])
                if heard == floor and often > self.heardIds.get(SILENCE, 0):
                    heard = FLOOR
            else:
                self.heardIds[SILENCE] = self.heardIds.get(SILENCE, 0) + 1
            tick.life.input.sound.id = heard
            # The name as a level, 0..1.  Times `ALIKE_KINDS` it is a name her
            # mouth can reach for again, which is how she says back what she
            # heard.
            tick.life.input.sound.similarity = float(alike)
            tick.life.input.sound.lvl = float(arriving[:, at].max())
            tick.life.input.sound.balance = 1 if loudest else -1
            # ...AND WHICH SOUND OF THAT NAME IT WAS.  Her ear's name is one
            # quantise; the piece is the nearest of those already wearing it,
            # and her mouth keeps them in the order the world made them so
            # she can say the word back the way it was said.  His three
            # steps, 2026-08-31: grab the voice, align it with the pieces,
            # store those pieces --- this is where the storing happens, and
            # it happens in her MOUTH, never in her memory: her mind still
            # holds only names, which is all it ever held.
            #
            # Measured the same night on his certified word: the name alone
            # gives her the sound he actually made on 11 of 39 ticks, the
            # piece on 38 of 40, and the whole word 1.29 from his against
            # 3.47.
            # (her ear used to hand the frame to `Voice.heard` here so the
            # play back; her mouth is her own since 2026-09-02 and nothing
            # plays a piece on the tick, so the feed is gone with it)
        else:
            self.heardIds[SILENCE] = self.heardIds.get(SILENCE, 0) + 1
            tick.life.input.sound.id = SILENCE
            tick.life.input.sound.similarity = float(SILENCE)
            tick.life.input.sound.lvl = 0.0

        # A LOOK, ONE STAGE A TICK.  Her eye, then the surfaces in it, then
        # what one thing IS --- 16.3, 25.0 and 22.6 ms measured on the host.
        # EACH IS LONGER THAN HER 11.1 ms TICK, and that is exactly why a look
        # is not taken in one: one stage lands per tick, a whole look every
        # `looksEvery` ticks (18 of them, 5 looks a second), and her eye is on
        # the GPU.  Measured at HEAD 2026-09-11: she feels a tick in 2.03 ms
        # and `behind` is 0, which is the only proof that matters.
        self._lap("ear")
        # (no cards by the clock --- his rule, no automatic helpers; the board
        #  shows what the doctor shows through /show, 2026-09-06)
        if self.lookThread is None:
            # NO THREAD LOOKS FOR HER (a probe body): look inline, one stage
            # a tick, exactly as a life did before 2026-09-03.
            stage = len(self.ticks) % self.looksEvery
            if stage == 0:
                # HER VIEW IS RENDERED ON HER EYES' AXIS, NOT HER SKULL'S.  This
                # read `her.frame()` --- the skull --- so `eye_at` steered nothing
                # and the reflex above would have written a number no picture ever
                # read.  `_glimpse` IS `frame()` turned by `eye_at`, and it had
                # zero callers in this tree: the actuator and the eye were two
                # halves that were each right and joined to nothing.
                #
                # ONLY THE AXES ARE TAKEN.  `_glimpse` also returns her FACE as
                # the eye point and this body has always rendered from `her.ear`,
                # check holds was measured there.  Moving her eye forward is a
                # real question about her optics and it is HIS, asked on its own,
                # with `measure.screen` run around it.  One change, not two.
                eye, up, right, fwd = her._glimpse()
                # ...AND SHE IS IN HER OWN PICTURE.  `light.one_look` has taken a
                # `body` since it was written --- her joints as skin-coloured
                # spheres, `_scene`'s last block --- and her own render never
                # passed it.  So she had never seen her hands, her feet, or
                # herself in the mirror: not a horizon item, a missing SENSE.  A
                # baby who cannot see her own hand cannot learn that the hand is
                # hers, and the mirror had nothing of her to reflect.
                #
                # `radius` is her own anthropometry (`ragdoll._RADIUS`, measured
                # against real newborns), so nothing here decides how big she is.
                pic = light.see(self.room, [self.window, self.board],
                                [(eye, up, right, fwd)],
                                body=(her.pos, her.radius))
                self.picture = np.asarray(light._home(pic))[:, :, 0].reshape(
                    light.RETINA_H, light.RETINA_W, light.CONES)
            elif stage == 1 and self.picture is not None:
                self.onePicture = wholePicture(self.picture)
                self.swungSince[:] = 0.0
            # SEEING NOTHING IS AN EMPTY LIST, NOT A THING WITH ID 1.  `Input.view`
            # defaults to `[View()]`, so on a tick where she saw nothing she
            # reported one thing, id 1, filling none of her view --- a thing that is
            # not there, and indistinguishable from thing number one.
        else:
            # A THREAD LOOKS FOR HER: at stage 0 --- the same tick the
            # inline look rendered on --- take the snapshot it needs, here,
            # under her lock, and hand it over.  Her eye point is what the
            # glimpse returns: her face joint, her eyes.  (Until 2026-09-07 it
            # was `her.ear`, the centre of her head, and her own face joint sat
            # as a skin disc in the middle of every look --- his find.)
            # HER EYE MUST IDLE, and that is what `looksEvery` is for.  Her
            # eye thread holds the interpreter while it renders; an eye that
            # takes a new look the moment it finishes never lets her HTTP
            # thread answer, and her mind sits blocked on /ticks for ever.
            # Measured 2026-09-11: free-running, a look went 50 -> 278 ms, her
            # mind stopped reporting entirely, and `behind` read 0.0 only
            # because nothing was left to be behind.  Her eye asks on her look
            # rate and only when the last look is done.
            if not self._eyeBusy and len(self.ticks) % self.looksEvery == 0:
                eye, up, right, fwd = her._glimpse()
                self._lookJob = (len(self.ticks),
                                 np.array(eye, np.float32, copy=True),
                                 up, right, fwd,
                                 np.array(her.pos, np.float32, copy=True),
                                 her.radius,
                                 tuple(float(v) for v in self.swungSince))
                self.swungSince[:] = 0.0
        # WHAT SHE SEES, IN THREE LINES: what it is, and where.  Always
        # present, so a line she can learn from never leaves her row.
        lvl, px, py = getattr(self, "onePicture", (0.0, 0.0, 0.0))
        tick.life.input.view = [View(id=0, similarity=float(lvl),
                                     x=float(px), y=float(py))]

        self._lap("look")
        pulled, turning = self.balance.read(her)
        for k, v in enumerate(pulled.values()):
            lines[20 + k] = float(v)
        for k, v in enumerate(turning.values()):
            lines[30 + k] = float(v)
        # HER OWN MOVEMENT, KEPT UNTIL HER NEXT LOOK.  Neither half is the room
        # and neither is a conclusion: one is a copy of what she ordered her
        # eyes to do, the other is an organ.
        gaze = (float(her.eye_at[0]), float(her.eye_at[1]))
        self.swungSince += np.asarray(
            orient.turned(gaze, self.wasGaze, turning), np.float32)
        self.wasGaze = gaze
        for k, v in enumerate(
                skin.feel(her, self.room, self.touch).values()):
            lines[40 + k] = float(v)
        # HER TWO EARS REACH HER MIND AT LAST --- his word, 2026-08-30:
        # *"she has to become human ... what we need to add to complete
        # her, and if we can do that let's continue"*.  `air.hear()` has
        # been binaural since it was copied across --- head shadow, pinna
        # notch, distance, PER EAR --- and the direction half fed nothing:
        # the standing failure, again.  Two lines, one per hair-cell
        # array's TOTAL response this tick --- the MEAN, measured, because
        # the loudest band is bass and bass bends around a head (that is
        # `_shadow`'s own physics): max L/R differed 1.4% for a hard-left
        # source, mean 16%, and both read equal dead ahead.  Physical
        # levels, never a computed direction --- where a sound is stays
        # hers to learn, her neck moving while the difference moves, the
        # same pair law as everything else.
        lines[50] = float(self.ears[0].mean())
        lines[51] = float(self.ears[1].mean())
        # ...AND SOMETHING LOUD TURNS HER.  Not a lean, not a suggestion: her
        # head goes, and what she was doing waits.  It is the one reflex that
        # overrides her, on his word, and it is the one a newborn actually has
        # --- the acoustic startle, present at birth, subcortical, and it
        # habituates (which here is free: a room that is always loud raises
        # her usual, and then it is not loud any more).
        loud = float(lines[50]) + float(lines[51])
        # HER USUAL IS THE USUAL OF SOUNDS, NOT OF SILENCE.  Most of her
        # ticks are quiet, so a usual that counted them settled near zero
        # and EVERY sound was three times usual: measured 2026-09-01, 300
        # startles before a test even began and her head thrown 48 degrees
        # a tick.  A room's usual loudness is what it sounds like when it
        # is sounding.
        # THE TURN TOWARD A SOUND --- his, 2026-09-06: *"rising has to trigger
        # her reflex."*  A newborn turns her eyes toward a sound that rises
        # above what she is used to, and turns less as it becomes usual.  The
        # startle's own law, generalised below its three-times line: the
        # turn is the rise over her usual, as a share of it, toward the ear
        # that got more, for the startle's own moment.  Her usual follows
        # what she hears (STARTLE_SETTLES), so the second chirp rises less
        # and the third less again --- habituation is her ear's, free.  It
        # writes her eye order, so it arrives on her eye spindles like any
        # pull: she feels where her eyes went, and the sound's source enters
        # her picture, a value her view line never showed --- the experience
        # that follows is hers to keep.  General over every sound; names
        # nothing; her mind never sees it as a decision.  (Test 1 of the
        # doctor's, 2026-09-06, before it: no turn at the first chirp.)
        # WHAT IS AT HER LIPS.  It drains as she takes it --- her hormones
        # decide how much she draws, and this is only what is THERE.
        lines[MOUTH] = float(self.atMouth)
        self.atMouth = max(0.0, self.atMouth - FEEDS_FOR * TICK_SECONDS)
        # THE OUTSIDE BRAIN'S TICK.  Her chemistry and her state are its
        # business now --- one implementation of everything --- so the body
        # only reports.  The output CARRIES FORWARD (what she was doing
        # continues, `live.pose`'s own rule) and one queued decision lands
        # on top, exactly what her own `act` would have written.
        tick.life.input.sensor = [Sensor(id=s, lvl=v) for s, v in sorted(lines.items())]
        self._lap("senses")
        self.ticks.append(tick)
        now = tick.life.output
        # A LINE SHE SET STAYS SET UNTIL SHE MOVES IT.  Her output carries
        # forward, whole, and nothing puts it back.  That is what lets the next
        # experience open where the last one closed: she sets one line, then
        # another, and both are still there, so a chain of one-line trials
        # builds a whole pose.
        now.motor = ([Motor(id=int(m.id), lvl=float(m.lvl)) for m in out.motor]
                     or [Motor(id=k, lvl=0.0)
                         for k in range(1, max(1, int(self.motors)) + 1)])
        # A HELD POSE IS A THING SHE IS DOING; A SOUND IS A THING SHE
        # DID.  His rule for the table, 2026-08-31: *"each of
        # prelearned sound has to feel just one tick and has his own
        # ID."*  Her muscles fade above because holding one is a
        # choice she keeps paying for --- but a voice that fades the
        # same way re-commands the SAME piece for twelve ticks, and
        # a piece is exactly one tick long, so it looped twelve
        # times.  Measured on her live room: utterances of 1.5-3.0 s
        # whose insides barely moved (6-13 against 25 for real
        # babble) --- one grain buzzing at 30 Hz.  Her voice does not
        # carry: one command, one tick, and the next tick is hers to
        # choose again.
        now.sound.id = out.sound.id
        now.sound.lvl = 0.0
        # ...AND A DECISION LANDS: one line (a try) or a WHOLE PLAN (a
        # remembered moment, every muscle and her voice --- his rule: a pose
        # is all of her, a replay carries all of her).  Levels the plan
        # names are written; levels it does not name go on fading.
        self.landed = 0
        if self.outside:
            got = self.outside.pop(0)
            if got[0] == "plan":
                _, plan, _at = got
                want = {int(m["id"]): float(m["lvl"])
                        for m in (plan.get("motor") or ())}
                for m in now.motor:
                    if int(m.id) in want:
                        m.lvl = max(0.0, min(1.0, want[int(m.id)]))
                said = plan.get("sound")
                if said is not None:
                    now.sound.id = int(said.get("id", now.sound.id))
                    now.sound.lvl = max(0.0, min(1.0, float(said.get("lvl", 0.0))))
                self.landed = len(want) + (1 if said is not None else 0)
            else:
                kind, tid, lvl = got
                if kind == "motor":
                    for m in now.motor:
                        if int(m.id) == int(tid):
                            m.lvl = float(lvl)
                            break
                else:
                    now.sound.id, now.sound.lvl = int(tid), float(lvl)
                self.landed = 1
        # A HAND ON HER IS NOT HER OWN MOVE --- OUT, 2026-09-08, on his word
        # once it was measured.  `_handsAsHers` wrote her spindle READING into
        # `self.order`, and the two are not the same quantity: a spindle is
        # "where along its range the axis is" (`ragdoll.reading`) while an
        # order is how hard to pull (`muscles.pull`: *"THIS CAPS FORCE, NOT
        # DESTINATION"*).  Both run 0..1, so nothing ever raised.  A joint
        # sitting at 0.9 of its range was ordered to pull at 0.9 effort, all
        # 26 at once, every tick a hand was on her --- measured over her
        # 2.4-hour life of 2026-09-08: her joints reversed direction on 79.4%
        # of ticks while held against 48.7% free (50% is a coin), and crossed
        # a third of a range in one 11 ms tick 2.29 times a tick against 0.05,
        # when her muscles' own ceiling is 6.7 degrees in a tick.  It also
        # wrote those positions into her record as her outputs, so every run
        # he helped her through was banked in the wrong units and could never
        # be replayed.  The eye reflex it was copied from is sound because an
        # eye's order IS a position at both ends (`_eyeWant` -> `eye_at`); a
        # body axis is the opposite.  His rule stands and wants a different
        # implementation: the order would have to be the EFFORT that carries
        # her toward where the hand put her, never the place itself.
        self.tick = tick
        self._lap("seal")

    # --- what the page is handed ------------------------------------------
    def pose(self) -> dict:
        """HER JOINTS, HER ROOM, AND THE WINDOW --- for drawing only.

        Her brain must never read this.  A brain that could would know where
        its own arm is without finding out.
        """
        her = self.her
        return {"tick": len(self.ticks),
                "joints": {n: [round(float(v), 4) for v in her.pos[i]]
                           for n, i in JIDX.items()},
                # HER OWN RADII AND HER OWN BONES.  What reads as "child" is
                # proportion and she already has it --- her head is 0.11 across
                # against shoulders 0.116 apart.  The page is handed both and
                # invents neither.
                "thick": {n: round(float(her.radius[i]), 4) for n, i in JIDX.items()},
                "bones": [[a, b] for a, b, _ in _BONES],
                "room": {k: float(v) for k, v in ROOM.items()},
                "window": {"at": [round(float(v), 4) for v in self.window.at],
                           "look": [round(float(v), 4) for v in self.window.look],
                           "up": [round(float(v), 4) for v in self.window.up],
                           "size": list(self.window.size)},
                "teacher": ({"joints": self.teacher.shape(),
                             "scale": 1.75}
                            if self.teacher.on else None)}

    def mind(self) -> dict:
        """WHAT SHE IS THINKING --- her lines, her runs, what she is saying."""
        life = self.tick.life
        return {"tick": len(self.ticks),
                "seconds": round(len(self.ticks) * TICK_SECONDS, 2),
                "state": round(float(life.input.state), 4),
                "saying": {"id": int(life.output.sound.id),
                           "lvl": round(float(life.output.sound.lvl), 3)},
                "hearing": {"id": int(life.input.sound.id),
                            "lvl": round(float(life.input.sound.lvl), 4)},
                "echo": {"id": int(life.input.echo.id),
                         "lvl": round(float(life.input.echo.lvl), 4)},
                #: WHAT HER MOUTH ACTUALLY PUT INTO THE AIR, beside what
                #: arrived at her ears.  `self.voiced` has been written every
                #: tick since it existed and read by NOTHING --- its own
                #: comment says why it is kept: *"so HE can see what her voice
                #: actually put into the air, beside what arrived at her ears.
                #: They are two different things and they were drawn as one."*
                #: Her ear bands ride `ears` already; this is her MOUTH's, and
                #: it is drawing data --- she cannot read it, and no part of
                #: her changes because somebody listened.
                "voiced": [round(float(v), 4) for v in
                           (self.voiced if getattr(self, "voiced", None)
                            is not None else ())],
                "moving": [{"id": int(m.id), "lvl": round(float(m.lvl), 3)}
                           for m in life.output.motor],
                "things": len(life.input.view),
                "kinds": sorted({int(v.id) for v in life.input.view}),
                # THE OUTLINES, computed here and never in the browser.  A box
                # centre is -1..1 and its size is a FRACTION of the view ---
                # two units in one row, `parts.find`'s own convention, settled
                # 2026-08-15.  Half a box in pixels is `w * width * 0.5`.
                "boxes": [],
                "ears": {"left": round(float(self.ears[0].max()), 4),
                         "right": round(float(self.ears[1].max()), 4),
                         "band": int(np.argmax(self.ears.max(axis=2).max(axis=0)))
                                 if self.ears.max() > 0 else -1,
                         "bands": [round(float(v), 4)
                                   for v in self.ears.max(axis=2).max(axis=0)]},
                "seeing you": self.window.shows is not None,
                "queued": round(self.window.queued(), 3),
                "late": round(self.window.late(), 3)}

    def see(self) -> dict:
        """HER RETINA, as she has it.  Not a redraw --- her own picture."""
        if self.picture is None:
            return {"tick": len(self.ticks), "w": 0, "h": 0, "px": []}
        # ...at the page's scale, because sending 786,432 numbers a look is
        # not watching her, it is moving her eye across a socket.
        n = 64
        step = max(1, light.RETINA_W // n)
        small = np.asarray(self.picture)[::step, ::step, :]
        px = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        return {"tick": len(self.ticks), "w": int(px.shape[1]),
                "h": int(px.shape[0]), "px": px.reshape(-1).tolist()}

    def say(self, pcm: np.ndarray, rate: float) -> None:
        with self.lock:
            self.window.say(pcm, rate)

    def show(self, raw: bytes, w: int, h: int, name: str | None = None) -> None:
        """HIS FACE ONTO THE SCREEN.  Raw RGB, and her eye reduces it.

        The browser sends pixels and nothing else --- it does not decide what a
        pixel means, does not scale to her retina, and is never told her field
        of view.  Her eye renders the screen as a quad in her room and takes
        from it whatever her optics take.
        """
        px = np.frombuffer(raw, np.uint8)
        if px.size != w * h * 3:
            return
        self._paSeen = time.time()      # his live face is on the screen NOW
        with self.lock:
            if name and self.board is not None:
                # THE DOCTOR'S CARD, BY NAME, ONTO THE PICTURE BOARD --- the one
                # 0.4 m from her face, 34 degrees off her skull's line, inside
                # her eyes' reach (BOARD_AT, measured above).  Until 2026-09-06
                # it went onto the flying screen, which hung 62 degrees off
                # her face's line, outside her eyes' 35 and her mother's 25:
                # four doctor's visits, a card up for a minute each, her
                # mother named it 0 times.  Her mother names the board by
                # what it shows (`_looking`).
                self.board.show(px.reshape(h, w, 3).astype(np.float32) / 255.0)
                self.board.name = None if str(name) == "blank" else str(name)
            else:
                self.window.show(px.reshape(h, w, 3).astype(np.float32) / 255.0)
                self.window.showing = (str(name) if name else None)
