# HIS DECREE, 2026-08-28: "we run her on gpu" --- nothing here scans;
# every tick's work is a few comparisons and one list index.
"""THE TEACHER --- a caregiver doll in her room, on the owner's button.

His, 2026-08-28: *"maybe we have to make real teacher with behavior
scenario of some teaching pre school program"* --- corrected in the
discussion to her true grade level: she is a newborn, and a newborn's
whole curriculum is CONTINGENT RESPONSE (Goldstein & Schwade: replies
landing within about a second restructure real babble in minutes;
random-timed attention changes nothing).  Then: *"i gues it has to be
same doll ass she but bidder and different collor"*, *"has to be abele
turn her by button ffrom ui"*, *"to i be able to whtch ehat she is
doing with her"*.

THE SCENARIO (the whole of it --- a written behaviour, worldly and
simple, exactly as discussed):

1. SHE ANSWERS.  When the baby's utterance ENDS, the teacher waits a
   beat (~0.6 s) and says it BACK --- the same band frames the baby's
   own mouth put into the air, replayed from the teacher's place.  A
   perfect imitation costs no synthesis, passes the articulatory
   filter by construction (the baby can only ever hear back what her
   own mouth can make), and arrives on the `hers -> heard` pair the
   mind can now see.
2. SHE LEAVES A TURN.  Never a sound while the baby is sounding; one
   answer per utterance; a refractory beat (~2 s) between answers.
3. SHE NAMES, WHILE THE BABY IS CALM.  After ~10 s of mutual silence
   she says one of the baby's own most-repeated forms --- a small,
   stable vocabulary grown out of the baby's own repertoire, the same
   form said consistently, which is what a name is.
4. THE OWNER'S VOICE ALWAYS WINS THE ROOM.  While he is speaking the
   teacher is silent --- the old mother's own law ("someone real is
   here, so she is not"), kept.

She is a THING in the room (her eye can find it) and a doll on the
page (the same doll as the baby, bigger, her own colour).  Nothing of
her touches the brain: the baby just hears a voice and sees a shape,
and every rule the mind runs is general over sources.
"""

from __future__ import annotations

import numpy as np

from .alike import alike_of
from . import speech
from .hearing import TICK_SECONDS

#: where she stands: beside the TRAINING SQUARE, feet on the floor, close
#: enough that her voice is never thinned away by distance --- his word,
#: 2026-08-30: *"mom is on the center of the training spot... move mom
#: from spot."*
STANDS = (9.0, 0.0, 8.25)
#: her voice comes from her head's height, not her feet.
SPEAKS = (9.0, 0.75, 8.25)   # INSIDE the pen, against its south fence ---
                             # from outside, the far corner sits past the
                             # ~2 m thinning and her words would arrive
                             # too faint; inside, every corner is reached
#: the same doll, bigger: a newborn's skeleton at this scale reads as
#: the grown-up in the room without inventing an anatomy.
SCALE = 1.75

#: HER MOTHER'S LAWS ARE IN SECONDS, NOT IN TICKS --- 2026-09-05, the
#: correction of a real, measured drift.  Every number below is his, dictated
#: across 2026-08-28/29 when a tick was 33 ms.  b650888 made her clock 90 a
#: second (11 ms) on 2026-09-04 and converted NOT ONE tick count anywhere in
#: her body (`git show b650888 | grep '^[-+][A-Z_]* = [0-9]'` returns REGISTER
#: alone), so her mother has run at THREE TIMES the speed he set --- answering
#: in 0.10 s where he said 0.3, resting 1.67 s between rewards where he wrote
#: 5, calling 20 s of crying a minute --- and `SWAPS`'s own comment still read
#: "five minutes of her 33 ms ticks".  Written as SECONDS through her own time
#: base, the clock can never orphan them again.
_T = TICK_SECONDS


def _ticks(seconds) -> int:
    """His seconds, in whatever tick she is living on."""
    return max(1, int(round(float(seconds) / _T)))


ANSWER_AFTER = _ticks(0.3)   # his order 2026-08-29: "turn it as fast as
                        # possible".  Was 18 at 30 fps (the Goldstein
                        # window), then 9; tighter contingency, still a
                        # natural pause.
REFRACTORY = _ticks(0.8)     # between answers --- a quick mom, not a machine
                        # gun; the QUIET_GAP still lets a word end.
NAMES_AFTER = _ticks(4.0)    # of mutual silence before a name
NAME_KEEPS = 2          # her vocabulary: the two forms the baby says most
#: WHAT HER MOTHER CALLS THE THINGS (his rule, 2026-09-04: no soundless word;
#: the doctor's examination as it is done).  A thing he has named in his own
#: voice keeps his word; otherwise her mother says the table's word for it,
#: out loud, through the same door her own ear uses.  She names what the baby
#: is looking at, once per NAME_REST, as a mother does across a table.
THING_WORDS = {"bottle": "milk", "teacher": "mama", "ball": "ball",
               "rattle": "rattle", "bear": "bear"}
NAME_REST = 270         # ~3 s between namings of what she looks at
#: THE WORD RULES, his answers of 2026-08-28 ("agree"):
WORD_GAP = _ticks(1.0 / 6)   # the breath inside a word: ma [gap] ma
CRY_LVL = 0.05          # over half her loudest voice, sustained, is crying
CRY_TICKS = _ticks(60.0)     # 60 seconds of it --- HIS NUMBER --- earns a
                        # rescue.  It had shrunk to 20 s.
FEED_AFTER = _ticks(1.0)     # the word FIRST, the milk a beat later (his own
                        # story: "tell her mommy before feed her")
#: A REST UNTIL SHE IS HUNGRY AGAIN --- his own decision of 2026-09-05
#: (cbffd0f, the guide: *"seven words in two minutes filled her by the second
#: word and the rest earned nothing"*), applied to the half of the law that
#: did not get it.  His 5 s was set on 2026-08-28 when FEEDS was 0.01 and a
#: mouthful was worth ONE SECOND of hunger, so a feed every 5 s balanced her
#: drain.  Measured 2026-09-05 with the 5 s (running as 1.67 s): hunger
#: pinned at 0.005-0.11 for a whole life, never back above ENOUGH 0.35 where
#: his 2026-08-27 law makes a met need silent --- the milk moved her state by
#: nothing, and no word chain could be paid for earning it.  At 110 s her
#: hunger cycles 0.23 <-> 0.31 and her state tracks it.
#: ...BUT THE REST MUST MATCH THE WORTH.  110 s belongs with FEEDS 0.5
#: (`main`, cbffd0f).  ON THIS BRANCH FEEDS IS HIS 0.01 --- a mouthful is
#: worth about one second of hunger --- and 110 s STARVED HER: measured
#: 2026-09-05 on a life from that night's tape, hunger 0.35 -> 0.57 in seven
#: minutes, state -0.44 -> -1.00 (the clamp), relief cut one run.  So here
#: the rest is his 2026-08-28 pair for that worth: 5 s.  (On the good night
#: it ran as 1.67 s, which is what kept her fed; at his true 5 s a mouthful
#: about balances five seconds of her drain.)  His yes the same evening, on
#: the rebuilt mind: the milk is worth two minutes (FEEDS 0.5), so the rest
#: is two minutes --- one law, both knobs together.
FEED_REST = _ticks(110.0)
#: THE MOM GETS BORED OF PARROTING --- his loop find, 2026-08-28: her
#: answers arrive on the world line (a voice is a voice), so the baby
#: heard her own sound "said by the world", learned that saying X brings
#: X, and the two ping-ponged for ever, loud.  A real mother answers a
#: fresh sound fully and tapers on the tenth identical one: each answer
#: settles that form's usual by a quarter; rest drifts it back (~min).
#: Variety is what earns her voice --- the loop starves, the alphabet
#: spreads.
PARROT_SETTLE = 0.35
#: per tick toward fresh again --- written as the 55 s HALF-LIFE it is meant
#: to be, so it does not shorten with her clock (at 0.0004 a tick it had
#: fallen to 19 s: her mother forgot she was bored three times too fast).
PARROT_DRIFT = 1.0 - 0.5 ** (_T / 55.0)
PARROT_FLOOR = 0.25     # below this freshness she simply stays quiet
ANSWER_SOFT = 0.6       # her voice is gentle: the answer's frames scaled
#: A DIP IS NOT AN ENDING --- his, 2026-08-29: "her memory full of
#: broken words ... teach her again on full words".  The baby's fading
#: voice touches zero mid-word and the capture used to close there, so
#: the mom learned FRAGMENTS as if they were wholes and taught them
#: back.  A word ends at a real silence, not a flicker.
QUIET_GAP = _ticks(1.0 / 6)  # true silence that ends an utterance
#: ...and the mom now COMPOSES: once the baby owns two reliable forms,
#: the lexicon holds their doubled words AND the two-sound word A-B ---
#: a full word, every part the baby's own.
LEXICON_KEEPS = 2
#: WHAT COUNTS AS A REAL WORD --- his order, 2026-08-29: "be sure that
#: you really produsd real words for her not repeat her broken one".
#: The mom's tally counts every crumb; her VOCABULARY must not.  A
#: teachable form is at least a syllable's floor of actual voice and
#: clearly phonated on her own scale; and the exemplar kept is the BEST
#: ever heard of that form, never merely the latest.
MIN_WORD_FRAMES = _ticks(0.2)   # 200 ms of voiced sound --- her ear makes one
                        # frame a tick (SOUND_SLIDES 1), so this had fallen
                        # to 67 ms: the mom was teaching back crumbs again,
                        # the very thing his 2026-08-29 order forbade
MIN_WORD_LOUD = 0.03    # clearly phonated (her loudest voice is ~0.09)
#: HIS WORDS, CONVERTED --- his order and his approval, 2026-08-29:
#: "you can gerab my words convert tham and pass to her" ... "its
#: good".  The mom's teaching vocabulary is now the father's own
#: words, shifted x1.75 toward the baby's register (pitch, formants
#: and pace rise together --- motherese by arithmetic).  And naming is
#: father-driven: a word he says while the baby is LOOKING at a thing
#: becomes that thing's name, for ever.
HIS_BAR = 0.02          # his voice, clearly present at the mic
HIS_GAP = _ticks(0.3)   # of quiet ends one of his words
#: (her mother keeps every word of his --- his, 2026-09-06; a rotating nine went)
                        # widened 6 -> 9 on his word (2026-08-30, "I
                        # agree with all your suggestion"): eat, step and
                        # give join the book from his own recordings, and
                        # a keep of 6 would have evicted papa, yes and no
                        # to make room


def _quality(frames):
    voiced = [f for f in frames if f is not None]
    loud = max((float(f.max()) for f in voiced), default=0.0)
    return len(voiced), loud
#: THE NIGHT SHIFT, his 2026-08-28 bedtime order: "swap helpers with
#: he free move once per 5 min".  The mom rotates the baby's situation
#: so a night alone is a night of varied practice: free floor, then the
#: walker, free again, then the crawler --- five minutes each, on the
#: body's own clock, deterministic (no state to lose on a restart).
SWAPS = _ticks(300.0)   # his five minutes, whatever a tick is
#: NO AUTOMATIC HELPERS --- his word, 2026-09-04 (c4ded20 on `main`: *"helpers
#: don't help her at all, they work stupid; I would leave her without them,
#: she will learn how to get up without them"*), and again 2026-09-05 on the
#: rebuilt mind: a clock that puts her in the walker or the crawler every
#: five minutes is not a reflex of hers but a schedule, and it decided where
#: she was when a block of the sheet began.  The night shift's slots are all
#: free; a hand on her is his, from the page.
ROUTINE = (None,)
#: THE LIFT, his rule of 2026-08-28: "ih she doesent move you has to
#: pull her vertical by the had till whole her lenghts and drop on
#: flor".  His diagnosis with it: while speaking she does not move at
#: all --- the mom's answers pay her for chatting motionless, so
#: stillness-with-talk is a paradise.  The trigger is exactly that:
#: a minute of TALKING while her chest went nowhere.  Silent rest is
#: never punished --- the sleep line lives on stillness, and his
#: complaint was the chat, not the rest.
STILL_WINDOW = _ticks(60.0)  # the minute she is judged over
STILL_MOVE = 0.05       # a chest that netted under 5 cm went nowhere
STILL_TALK = STILL_WINDOW // 4   # ...while sounding a quarter of the time
LIFT_TO = 1.40          # the hand's height.  One-handed sag measured
                        # 0.15 and the mattress top is 0.34, so this is
                        # what a dangle above the cot costs against the
                        # one-hand sag (measured 0.41: the whole body
                        # argues against one snap point); the drop is real
                        # either way.
LIFT_HOLD = _ticks(1.0 / 3)  # a beat at the top, then the drop


class Teacher:
    """The scenario, one tick at a time."""

    def __init__(self, keep: str | None = None) -> None:
        #: THE VOCABULARY SURVIVES THE PROCESS --- a name he gives her
        #: world once is given for ever.  His words (converted frames)
        #: and the things they name load here at birth and save as they
        #: grow; the night of 2026-08-29 proved RAM vocabulary dies at
        #: every restart and archaeology in the recordings is misery.
        self._keep = keep
        self.on = False
        self.at = np.asarray(SPEAKS, np.float32)
        self.saying = False
        self._utter: list = []       # the baby's current utterance, frames
        self._key = None             # its opening sound id
        self._counts: dict = {}      # opening id -> how often the baby says it
        self._forms: dict = {}       # opening id -> frames of the last saying
        self._answer = None          # (frames, cursor) being spoken
        self._due = -1               # tick the pending answer starts
        self._quietAt = 0            # last tick anyone sounded
        self._saidAt = -10 ** 9      # last tick she finished speaking
        self._began = 0              # when the current utterance began
        self._lastKey = None         # the previous utterance's opening sound
        self._lastEnd = -10 ** 9     # ...and when it ended
        self._cry = 0                # how long the baby has been crying
        self._feedAt = -1            # when the earned milk arrives
        self._fedAt = -10 ** 9       # last reward, for the rest
        self._parrot: dict = {}      # form -> [usual, last tick]
        self._names: dict = {}       # thing -> its form: assigned once, kept
        self._his: list = []         # his current word, raw pcm chunks
        self._hisQuiet = 0
        self._hisLex: list = []      # his latest words, as her-register frames
        #: ...AND THE SOUND THEY WERE MADE FROM, so he can HEAR what she says.
        #: His, 2026-09-01: *"make me able to hear what her mothers tolds her
        #: to be able debug at least something from mom."*  Her mother speaks
        #: in FRAMES --- `convert` turns his recorded voice into band pictures
        #: and the samples were thrown away, so there has never been anything
        #: to play.  These are HIS OWN RAW WORDS, the source each frame was
        #: made from: not a resynthesis of the frames (they have no phase and
        #: none can be invented), and not shifted, so it is the most
        #: intelligible form of what she just told the baby.
        self._hisPcm: list = []      # parallel to `_hisLex`
        self._pcmHis: dict = {}      # thing -> the sound of its name
        #: what she said THIS tick, for whoever is listening.  Glass: nothing
        #: of the baby's reads it.
        self.saidPcm = None
        self._namedLookAt = -10 ** 9   # when she last named what the baby looked at
        self._namesHis: dict = {}    # thing -> HIS converted word for it
        self.convert = None          # installed by the body: pcm -> frames
        self.named = 0               # the meter's naming line
        self.namedWhat: dict = {}    # ...and by name, for the doctor's sheet (glass, never hers)
        #: WHICH WORD SHE IS SAYING, and the sounds it is made of --- named
        #: words only, so "she answered a NAMED word" can be asked of the girl
        #: in front of us instead of a file.  She already knows both at the
        #: moment she chooses to speak; nothing new is computed for her, and
        #: nothing of hers reads it: it is telemetry, like `Watched.voiced`.
        #:
        #: It exists because `measure.scale` scored "answers a named word" and
        #: "learned association" from a JSON file in a scratch folder written
        #: by a DIFFERENT session about a DIFFERENT life (2026-08-31 11:18) ---
        #: two PASSes that were identical for a six-minute-old girl and an
        #: eight-hour-old one, because they were never about either.
        self.word = None
        self.wordIds: list = []
        self._anchor = None          # where her chest was a minute ago
        self._anchorAt = 0
        self._talk = 0               # sounding ticks inside the window
        #: THE MILESTONE METER --- glass, never implant: how the road to
        #: speech is going, countable from the page.
        self.answers = 0             # imitations she was given
        self.accents = 0             # his word said back to her, fully
        self._loud = ANSWER_SOFT     # how loud the answer in flight is
        self.shapes = 0              # his words she said back (was: doubled shapes)
        self.milks = 0               # milk earned by a word
        self.rescues = 0
        self.lifts = 0
        self._bored = False          # the silent treatment: a motionless
                                     # chatterer loses the mom's answers
                                     # until she moves
        self._shape = None           # her joints, computed once asked
        if keep:
            try:
                import json as _j
                import os as _o
                if _o.path.exists(keep):
                    got = _j.load(open(keep, encoding="utf-8"))
                    self._hisLex = [[np.asarray(f, np.float32) for f in w]
                                    for w in got.get("lex", [])]
                    self._namesHis = {
                        t: [np.asarray(f, np.float32) for f in w]
                        for t, w in (got.get("names") or {}).items()}
                # ...AND THE SOUND, kept beside it since 2026-09-01 so he can
                # HEAR what she says.  A vocabulary saved before that has
                # frames and no samples, and stays silent --- there is nothing
                # to invent from a band picture, which has no phase.
                try:
                    if _o.path.exists(keep + ".npz"):
                        heard = np.load(keep + ".npz")
                        self._pcmHis = {k[2:]: heard[k] for k in heard.files
                                        if k.startswith("n_")}
                        self._hisPcm = [heard[k] for k in sorted(
                            (k for k in heard.files if k.startswith("l_")),
                            key=lambda k: int(k[2:]))]
                        # his rule, 2026-09-04: the frames are what her ear
                        # makes of the raw word through the one door, so a
                        # vocabulary kept before the door is re-made from it
                        from body.hearing import wordFrames as _wf
                        if len(self._hisPcm) == len(self._hisLex):
                            self._hisLex = [(_wf(w) or self._hisLex[k]) for k, w in enumerate(self._hisPcm)]
                        self._namesHis = {k: ((_wf(self._pcmHis[k]) or v) if k in self._pcmHis else v)
                                          for k, v in self._namesHis.items()}
                except Exception:                      # noqa: BLE001
                    self._pcmHis, self._hisPcm = {}, []
            except Exception:                          # noqa: BLE001
                pass                # a torn file is an empty vocabulary

    def _keepVocab(self) -> None:
        if not self._keep:
            return
        try:
            import json as _j
            _j.dump({"lex": [[f.tolist() for f in w] for w in self._hisLex],
                     "names": {t: [f.tolist() for f in w]
                               for t, w in self._namesHis.items()}},
                    open(self._keep, "w", encoding="utf-8"))
            # THE SOUND GOES BESIDE IT, not in it: raw samples as JSON would
            # be a hundred times the size and lose nothing but patience.
            keep = {}
            for t, pcm in self._pcmHis.items():
                keep["n_" + str(t)] = np.asarray(pcm, np.float32)
            for k, pcm in enumerate(self._hisPcm):
                keep["l_%d" % k] = np.asarray(pcm, np.float32)
            if keep:
                np.savez_compressed(self._keep + ".npz", **keep)
        except OSError:
            pass

    def _teachable(self, k) -> bool:
        """Only a real word enters her vocabulary --- length and voice."""
        f = self._forms.get(k)
        if not f:
            return False
        n, loud = _quality(f)
        return n >= MIN_WORD_FRAMES and loud >= MIN_WORD_LOUD

    def hears(self, pcm, looking=None) -> None:
        """His voice at the mic, one tick's worth --- collected into
        words, converted toward her register, kept as the vocabulary.
        A word said while the baby LOOKS at a thing names that thing."""
        if not self.on or self.convert is None:
            return
        loud = float(np.abs(pcm).max()) if pcm is not None and len(pcm) else 0.0
        if loud > HIS_BAR:
            self._his.append(np.asarray(pcm, np.float32).copy())
            self._hisQuiet = 0
            return
        if not self._his:
            return
        self._hisQuiet += 1
        if self._hisQuiet <= HIS_GAP:
            return
        word = np.concatenate(self._his)
        self._his = []
        self._hisQuiet = 0
        frames = self.convert(word)
        if not frames:
            return
        self._hisLex.append(frames)
        # ...and the sound it was made from, kept in step with it
        self._hisPcm.append(np.asarray(word, np.float32).copy())
        if looking and str(looking) not in self._namesHis:
            self._namesHis[str(looking)] = frames
            self._pcmHis[str(looking)] = np.asarray(word, np.float32).copy()
        self._keepVocab()

    def recut(self) -> None:
        """HER MOTHER'S WORDS, CUT AGAIN AT THE CLOCK IN FORCE.  The frames
        kept in `mom.json` are one per tick of the clock that cut them; the
        raw words behind them are kept too (`mom.json.npz`), so a body born
        on another clock (his 11 ms, `MATILDA_FPS`) re-cuts every word from
        its sound rather than playing 33 ms frames three times too fast.
        Called by the body once `convert` is installed; a word with no raw
        sound behind it keeps its old frames."""
        if self.convert is None:
            return
        for thing, pcm in list(self._pcmHis.items()):
            try:
                frames = self.convert(np.asarray(pcm, np.float32))
            except Exception:                              # noqa: BLE001
                continue
            if frames:
                self._namesHis[thing] = frames
        lex = []
        for pcm in self._hisPcm:
            try:
                frames = self.convert(np.asarray(pcm, np.float32))
            except Exception:                              # noqa: BLE001
                continue
            if frames:
                lex.append(frames)
        if lex:
            self._hisLex = lex

    def _tableWord(self, thing: str):
        """The table's word for a thing, as frames through the door, its sound
        kept beside them for his ear --- made once, then it is his word for it."""
        got = self._namesHis.get(thing)
        if got is not None:
            return got
        name = THING_WORDS.get(thing)
        if name is None or self.convert is None or name not in speech.SOUNDS:
            return None
        pcm = np.asarray(speech.word(name), np.float32)
        frames = self.convert(pcm)
        if not frames:
            return None
        self._pcmHis[thing] = pcm
        self._namesHis[thing] = [np.asarray(f, np.float32) for f in frames]
        return self._namesHis[thing]

    def _nameFor(self, thing: str):
        """The thing's own name: the baby's most-said form not yet
        spoken for --- assigned at first naming, kept for ever.  One
        thing, one word; two things never share."""
        got = self._names.get(thing)
        if got is not None:
            return got if got in self._forms else None
        used = set(self._names.values())
        for k in sorted(self._counts, key=self._counts.get, reverse=True):
            if k not in used and self._teachable(k):
                self._names[thing] = k
                return k
        return None

    def _saying(self, name: str, frames) -> None:
        """RECORD WHICH WORD SHE IS ABOUT TO SAY, and its sounds.

        `frames` is the word she already holds --- band pictures, one per tick
        --- so its ids are her own ear's names for it, taken through the one
        quantiser everything else uses.  Nothing here decides anything; it is
        written so an instrument can ask whether the baby answered THAT word.
        """
        self.namedWhat[str(name)] = self.namedWhat.get(str(name), 0) + 1
        self.word = str(name)
        # WHAT IT SOUNDED LIKE, if this word came from one of his.  A word she
        # built out of the baby's own forms has no recording behind it and
        # leaves this None, which is the truth.
        self.saidPcm = self._pcmHis.get(str(name))
        got = []
        for f in frames:
            if f is None:
                continue
            try:
                one = int(alike_of(np.asarray(f, np.float32)))
            except Exception:                              # noqa: BLE001
                continue
            if one and (not got or got[-1] != one):
                got.append(one)
        self.wordIds = got

    def step(self, tick: int, made, soundId: int, herLvl: float,
             hisTalking: bool, looking=None):
        """One tick of her.  Returns band frames for the air, or None.

        `made` is what the baby's mouth actually put out this tick (the
        voice's own band frames), `herLvl` its loudness, `hisTalking`
        whether the owner's voice is in the room this tick.
        """
        self.saying = False
        # SHE NAMES WHAT THE BABY LOOKS AT, out loud, once per NAME_REST
        if (self.on and self._answer is None and looking is not None
                and str(looking) in THING_WORDS and tick - self._namedLookAt >= NAME_REST):
            frames9 = self._tableWord(str(looking))
            if frames9:
                self._answer = (list(frames9), 0)
                self._due = tick
                self._namedLookAt = tick
                self.named += 1
                self._saying(str(looking), frames9)
        if not self.on:
            self._utter = []
            self._answer = None
            return None
        if hisTalking:
            # the owner's voice always wins the room
            self._utter = []
            self._answer = None
            self._quietAt = tick
            return None
        if herLvl > 0.0 and made is not None and made.size:
            # the baby is sounding: listen, never speak over her
            if not self._utter:
                self._key = int(soundId)   # the utterance's opening sound
                self._began = tick         # ...and when it began
            self._utter.append(np.asarray(made, np.float32).copy())
            self._gap = 0
            self._answer = None
            self._quietAt = tick
            return None
        if self._utter and getattr(self, "_gap", 0) < QUIET_GAP:
            # a dip is not an ending: the word continues through a flicker
            self._gap = getattr(self, "_gap", 0) + 1
            self._utter.append(None)
            return None
        if self._utter:
            # her utterance just ended: remember the form, owe an answer
            frames = self._utter
            while frames and frames[-1] is None:
                frames.pop()               # the trailing silence is not hers
            self._utter = []
            self._gap = 0
            k = self._key if self._key is not None else 0
            self._counts[k] = self._counts.get(k, 0) + 1
            # the BEST exemplar of a form is the one worth teaching back
            old = self._forms.get(k)
            if old is None or _quality(frames) >= _quality(old):
                self._forms[k] = frames
            # HIS WORD SAID BACK --- his, 2026-09-06: the twice-law ("the same
            # sound twice, close together") was his own early misunderstanding
            # and is gone; the rule is the core's: is it new, it is an
            # experience, not new, usual.  What her mother pays for is what
            # she teaches: when what the baby just said holds a piece of one
            # of HIS words (whatever has come through the window), her
            # mother says that word back whole, at full voice (his 2026-09-05
            # accent), and the milk follows a beat behind, once per his rest
            # (FEED_REST).  Company relieves her loneliness and the milk her
            # hunger: both are dopamine.  Everything else gets the parrot:
            # thin, and fading on repeats.  No word is named here.
            his9 = self._his9(frames)
            if his9 is not None:
                self._answer = (list(his9), 0)
                self._loud = 1.0
                self._due = tick + ANSWER_AFTER
                self.accents += 1
                self.shapes += 1
                if tick - self._fedAt >= FEED_REST:
                    self._feedAt = tick + ANSWER_AFTER + FEED_AFTER
                    self._fedAt = tick
                    self.milks += 1
            elif tick - self._saidAt >= REFRACTORY:
                got = self._parrot.get(k)
                if got is None:
                    got = [0.0, tick]
                    self._parrot[k] = got
                got[0] *= (1.0 - PARROT_DRIFT) ** max(0, tick - got[1])
                got[1] = tick
                if 1.0 - got[0] >= PARROT_FLOOR:
                    got[0] += (1.0 - got[0]) * PARROT_SETTLE
                    self._answer = (frames, 0)
                    self._loud = ANSWER_SOFT
                    self._due = tick + ANSWER_AFTER
            self._lastKey, self._lastEnd = k, tick
            self._quietAt = tick
            return None
        if self._answer is not None and tick >= self._due:
            frames, k2 = self._answer
            out = frames[k2]           # None inside a word is the breath
            if out is not None:
                out = out * self._loud
            if k2 + 1 >= len(frames):
                self._answer = None
                self._saidAt = tick
                self.answers += 1
            else:
                self._answer = (frames, k2 + 1)
            self.saying = True
            self._quietAt = tick
            return out
        if (True
                and tick - self._quietAt >= NAMES_AFTER
                and tick - self._saidAt >= REFRACTORY
                and (self._counts or self._hisLex)):
            # OBJECT-NAMING, HIS ORDER 2026-08-29: the mom sees what the
            # baby sees.  When the calm gaze holds a thing --- the
            # television, the mom herself, the bottle --- she says THAT
            # THING'S name, its own stable form, every time.  Word and
            # thing stand together in the same moment; the pair machinery
            # does the binding.  Nothing in the baby is told anything.
            if not looking:
                # SHE LOOKS NOWHERE: the mom calls her BY NAME --- his
                # rule, 2026-08-29: "pass mattilda when she looks no
                # where, than mama when she looks aon mom".  The baby's
                # own name is the word for nowhere-in-particular, which
                # is what calling someone is.
                hisW0 = self._namesHis.get("nowhere")
                if hisW0 is not None:
                    self._saying("nowhere", hisW0)
                    self._answer = (list(hisW0), 0)
                    self._due = tick
                    self._quietAt = tick
                    self.named += 1
                    return None
            if looking:
                hisW = self._namesHis.get(str(looking))
                if hisW is not None:
                    self._saying(str(looking), hisW)
                    self._answer = (list(hisW), 0)
                    self._due = tick
                    self._quietAt = tick
                    self.named += 1
                    return None
                k4 = self._nameFor(str(looking))
                if k4 is not None:
                    # naming the thing with the baby's OWN form for it --- a
                    # word all the same, and the one she is likeliest to
                    # answer, so it is recorded like any other
                    self._saying(str(looking), self._word(self._forms[k4]))
                    self._answer = (self._word(self._forms[k4]), 0)
                    self._due = tick
                    self._quietAt = tick
                    self.named += 1
                    return None
            # a long calm: she offers a name --- one of the baby's own
            # most-repeated forms, the same form each time
            if self._hisLex:
                # HIS words are the lessons, in rotation
                at = (tick // NAMES_AFTER) % len(self._hisLex)
                pick = self._hisLex[at]
                # ...and its sound, for his ear only
                self.saidPcm = (self._hisPcm[at]
                                if at < len(self._hisPcm) else None)
                # HIS WORDS IN ROTATION HAVE NO NAME --- `_hisLex` is his
                # recorded speech as frames, kept without a label.  It is
                # still a WORD, said again and again, and the instruments
                # match by its SOUNDS, so it is recorded by its place in the
                # rotation.  Before this, her mother spoke these all day and
                # the sheet said "her mother named no words this run".
                self._saying("his word %d" % at, pick)
                self._answer = (list(pick), 0)
                self._due = tick
                self.named += 1
                return None
            best = [k3 for k3 in sorted(self._counts,
                                        key=self._counts.get, reverse=True)
                    if self._teachable(k3)][:LEXICON_KEEPS]
            if not best:
                return None
            # SHE TEACHES FULL WORDS: each reliable form doubled --- and,
            # once the baby owns TWO forms, the composed word A-B: two
            # different sounds in order, every part the baby's own.  The
            # lexicon is stable and cycles; composition is how "two
            # letters in order" is modelled instead of hoped for.
            lex = [self._word(self._forms[k3]) for k3 in best]
            if len(best) >= 2:
                lex.append(list(self._forms[best[0]]) + [None] * WORD_GAP
                           + list(self._forms[best[1]]))
            pick = lex[(tick // NAMES_AFTER) % len(lex)]
            self._answer = (pick, 0)
            self._due = tick
        return None

    @staticmethod
    def _word(frames: list) -> list:
        """A word out of a form: the form, a breath, the form again."""
        return list(frames) + [None] * WORD_GAP + list(frames)

    @staticmethod
    def _ids(frames) -> list:
        """Her ear's names for a run of frames, repeats folded --- the same
        naming `_saying` records for the instruments."""
        got = []
        for f in frames:
            if f is None:
                continue
            try:
                one = int(alike_of(np.asarray(f, np.float32)))
            except Exception:                              # noqa: BLE001
                continue
            if one and (not got or got[-1] != one):
                got.append(one)
        return got

    def _his9(self, frames):
        """WHICH OF HIS WORDS THE BABY JUST SAID, if any --- the one of
        `_hisLex` whose ids (her ear's) the utterance holds most of, at
        her ear's own resolution: one id is one piece she can tell (his,
        2026-09-06: the pieces are what she has to be rewarded for; "half"
        was the implementation's number).  None when it holds none."""
        u = set(self._ids(frames))
        if not u:
            return None
        best, score = None, 0
        for w in self._hisLex:
            ids = set(self._ids(w))
            if not ids:
                continue
            hit = len(u & ids)
            if hit >= 1 and hit > score:
                best, score = w, hit
        return best

    # --- what the body asks her --------------------------------------------

    def crying(self, herLvl: float) -> None:
        """The mom hears sustained loudness; quiet mends her count fast."""
        if not self.on:
            self._cry = 0
        elif herLvl > CRY_LVL:
            self._cry += 1
        else:
            self._cry = max(0, self._cry - 5)

    def rescueWanted(self) -> bool:
        """60 seconds of continuous crying --- his number --- and she acts
        once, then starts listening afresh."""
        if self._cry >= CRY_TICKS:
            self._cry = 0
            self.rescues += 1
            return True
        return False

    def still(self, tick: int, chest, herLvl: float) -> bool:
        """A minute of talking while going nowhere --- the lift is due.
        Judged on her chest's NET travel against an anchor, so solver
        jitter cannot fake movement and real rolling cannot be missed."""
        if not self.on:
            self._anchor = None
            self._talk = 0
            return False
        if herLvl > 0.0:
            self._talk += 1
        if self._anchor is None:
            self._anchor = [float(v) for v in chest]
            self._anchorAt = tick
            return False
        if tick - self._anchorAt < STILL_WINDOW:
            return False
        moved = sum((float(a2) - b2) ** 2
                    for a2, b2 in zip(chest, self._anchor)) ** 0.5
        due = moved < STILL_MOVE and self._talk >= STILL_TALK
        if due:
            self._bored = True       # the silent treatment begins
        elif moved >= STILL_MOVE:
            self._bored = False      # she moved: the mom is back
        self._anchor = [float(v) for v in chest]
        self._anchorAt = tick
        self._talk = 0
        return due

    def fed(self, tick: int) -> None:
        """THE NAMING CURRICULUM, first noun: milk is at her lips, and
        the mom says the FOOD'S NAME --- one stable form, picked once
        from the baby's own most-said sounds and kept, said at every
        feed from then on.  Word with thing, thing with relief: the pair
        machinery binds them, and the first word ABOUT something is
        earned, never implanted."""
        if not self.on or self._answer is not None:
            return
        hisW = self._namesHis.get("bottle") or self._tableWord("bottle") or (
            self._hisLex[-1] if self._hisLex else None)
        if hisW is not None:
            self._answer = (list(hisW), 0)
            self._due = tick
            self.named += 1
            return
        k = self._nameFor("bottle")
        if k is None:
            return
        self._answer = (self._word(self._forms[k]), 0)
        self._due = tick
        self.named += 1

    def helperWanted(self, tick: int):
        """The night shift's slot for this moment --- None is the free
        floor; 'off' means the mom is away and the helpers are HIS."""
        if not self.on:
            return "off"
        return ROUTINE[(tick // SWAPS) % len(ROUTINE)]

    def feedWanted(self, tick: int) -> bool:
        """The earned milk, the beat AFTER the word --- once."""
        if 0 <= self._feedAt <= tick:
            self._feedAt = -1
            return True
        return False

    # --- her body, for the page and the room -------------------------------

    def shape(self) -> dict:
        """Her joints, cached --- she stands still; nothing recomputes.
        The skeleton import lives here, lazily: the app's module scope
        cannot see `body`, and by the time anyone asks for her shape the
        body is long imported."""
        if self._shape is None:
            from body.ragdoll import _REST, JIDX
            self._shape = self.joints(_REST, JIDX)
        return self._shape

    def joints(self, rest: np.ndarray, jidx: dict) -> dict:
        """The same doll, bigger: the baby's own rest skeleton, stood
        upright at her place and scaled.  Computed once per ask; drawing
        is the page's business, physics is nobody's --- she is a shape,
        not a ragdoll."""
        p = np.asarray(rest, np.float32).copy()
        up = p[jidx["head"]] - p[jidx["pelvis"]]
        n = float(np.linalg.norm(up))
        if n > 1e-6:
            u = up / n
            y = np.asarray([0.0, 1.0, 0.0], np.float32)
            v = np.cross(u, y)
            s = float(np.linalg.norm(v))
            c = float(np.dot(u, y))
            if s > 1e-6:
                vx = np.asarray([[0, -v[2], v[1]],
                                 [v[2], 0, -v[0]],
                                 [-v[1], v[0], 0]], np.float32)
                r = np.eye(3, dtype=np.float32) + vx \
                    + vx @ vx * ((1.0 - c) / (s * s))
                p = (p - p[jidx["pelvis"]]) @ r.T
        p = p * SCALE
        p[:, 1] -= float(p[:, 1].min())
        p[:, 0] += STANDS[0]
        p[:, 2] += STANDS[2]
        return {name: [round(float(v), 4) for v in p[i]]
                for name, i in jidx.items()}
