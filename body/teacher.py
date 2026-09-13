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
from .hearing import GATE, TICK_SECONDS

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
NAME_REST = int(round(3.0 / _T))   # 3 s between namings of what she looks at
#: THE WORD RULES, his answers of 2026-08-28 ("agree"):
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
#: (The parrot --- the mom echoing the baby's own sound back, tapering as it
#: repeated --- is gone: his word, 2026-09-13.  She says words she has the
#: sound of, and nothing the baby said is played back at her.)
#: A DIP IS NOT AN ENDING --- his, 2026-08-29: "her memory full of
#: broken words ... teach her again on full words".  The baby's fading
#: voice touches zero mid-word and the capture used to close there, so
#: the mom learned FRAGMENTS as if they were wholes and taught them
#: back.  A word ends at a real silence, not a flicker.
QUIET_GAP = _ticks(1.0 / 6)  # true silence that ends an utterance
#: HIS WORDS, CONVERTED --- his order and his approval, 2026-08-29:
#: "you can gerab my words convert tham and pass to her" ... "its
#: good".  The mom's teaching vocabulary is now the father's own
#: words, shifted x1.75 toward the baby's register (pitch, formants
#: and pace rise together --- motherese by arithmetic).  And naming is
#: father-driven: a word he says while the baby is LOOKING at a thing
#: becomes that thing's name, for ever.
HIS_GAP = _ticks(0.3)   # of quiet ends one of his words
#: ONE TICK OF HER VOICE, IN SAMPLES --- what `step` hands the body as air
TICK_SAMPLES = int(round(speech.RATE * TICK_SECONDS))
#: HOW MANY OF HIS WORDS SHE KEEPS --- the latest ones.  His word, 2026-09-13:
#: a mic left on beside a TV made a "word" of every loud stretch, and the list
#: had no end (and was rewritten whole to mom.json on each).  His to set.
HIS_WORDS = 32
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

    def hears(self, pcm, looking=None) -> None:
        """His voice at the mic, one tick's worth --- collected into
        words, converted toward her register, kept as the vocabulary.
        A word said while the baby LOOKS at a thing names that thing."""
        if not self.on or self.convert is None:
            return
        loud = float(np.abs(pcm).max()) if pcm is not None and len(pcm) else 0.0
        if loud > 0.0:
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
        # A WORD UNDER HER EAR'S GATE IS NOT A WORD.  His word, 2026-09-13: five
        # near-silent recordings (peak 0.026) sat in her vocabulary; through her
        # door every frame of them was 0.0, so they had no ids and nothing the
        # baby said could ever match them --- word-milk was impossible and
        # nothing raised.  The same gate her ear uses, nothing new.
        if float(np.abs(word).max()) <= GATE:
            return
        frames = self.convert(word)
        if not frames:
            return
        self._hisLex.append(frames)
        # ...and the sound it was made from, kept in step with it
        self._hisPcm.append(np.asarray(word, np.float32).copy())
        if len(self._hisLex) > HIS_WORDS:
            del self._hisLex[:-HIS_WORDS]
            del self._hisPcm[:-HIS_WORDS]
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
        """One tick of her.  Returns ONE TICK OF HER VOICE AS AIR (pcm at
        `speech.RATE`, `TICK_SAMPLES` long), or None when she is silent.

        HER VOICE IS AIR IN THE ROOM, NOT FRAMES IN THE EAR.  His word,
        2026-09-13: *"her mother told everything already converted ... remove
        all of this and make mom just pronouncing in a natural way the word,
        and she would repeat it."*  Until today an answer was a list of band
        frames dropped straight into `sounding` --- a side entrance past her
        ear, a conversion done FOR her.  Now the body carries this air into
        the mother's own last-LISTENS buffer and the same `door` names it,
        exactly as his voice (`alive.py`).  Her word list (`_hisLex`,
        `_namesHis`, band frames) is only her EAR --- how she recognises a
        piece of a word in what the baby says --- never what she speaks.

        `made` is what the baby's mouth put out this tick (band frames, for
        her listening), `herLvl` its loudness, `hisTalking` whether the
        owner's voice is in the room.  What she says, always as her own air:
        the word for what the baby looks at; a word said back WHOLE when the
        baby's sound holds a piece of it, the milk a beat behind; one of his
        words when the room is quiet.  Nothing of the baby's own babble is
        echoed back: the parrot, and the lexicon built of the baby's own
        forms, are gone (his word, 2026-09-13: they taught her to lie still
        and squeak for milk).
        """
        self.saying = False
        # SHE NAMES WHAT THE BABY LOOKS AT, out loud, once per NAME_REST
        if (self.on and self._answer is None and looking is not None
                and str(looking) in THING_WORDS and tick - self._namedLookAt >= NAME_REST):
            frames9 = self._tableWord(str(looking))
            pcm9 = self._pcmHis.get(str(looking))
            if frames9 and pcm9 is not None:
                self._speak(str(looking), frames9, pcm9, tick)
                self._namedLookAt = tick
                self.named += 1
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
            old = self._forms.get(k)
            if old is None or _quality(frames) >= _quality(old):
                self._forms[k] = frames
            # A WORD SAID BACK, AND THE MILK.  When what the baby just said
            # holds a piece of a word she knows --- his, or her own table
            # words --- she says that word back whole, at full voice, and the
            # milk follows a beat behind, once per his rest (FEED_REST).
            # Company relieves her loneliness and the milk her hunger: both
            # are dopamine.  Anything else the baby says gets nothing.
            his9 = self._his9(frames)
            if his9 is not None:
                name9, frames9, pcm9 = his9
                self._speak(name9, frames9, pcm9, tick + ANSWER_AFTER)
                self.accents += 1
                self.shapes += 1
                if tick - self._fedAt >= FEED_REST:
                    self._feedAt = tick + ANSWER_AFTER + FEED_AFTER
                    self._fedAt = tick
                    self.milks += 1
            self._lastKey, self._lastEnd = k, tick
            self._quietAt = tick
            return None
        if self._answer is not None and tick >= self._due:
            # ONE TICK OF THE WORD, AS AIR
            pcm, at = self._answer
            out = np.asarray(pcm[at:at + TICK_SAMPLES], np.float32)
            if at + TICK_SAMPLES >= len(pcm):
                self._answer = None
                self._saidAt = tick
                self.answers += 1
            else:
                self._answer = (pcm, at + TICK_SAMPLES)
            if out.size < TICK_SAMPLES:
                out = np.pad(out, (0, TICK_SAMPLES - out.size))
            self.saying = True
            self._quietAt = tick
            return out
        if (tick - self._quietAt >= NAMES_AFTER
                and tick - self._saidAt >= REFRACTORY
                and (self._namesHis or self._hisLex)):
            # THE ROOM IS QUIET: the word for what the baby looks at (or for
            # nowhere), else one of his words in rotation --- as her air.
            name9 = str(looking) if looking else "nowhere"
            w9, p9 = self._namesHis.get(name9), self._pcmHis.get(name9)
            if w9 is not None and p9 is not None:
                self._speak(name9, w9, p9, tick)
                self._quietAt = tick
                self.named += 1
                return None
            if self._hisLex and self._hisPcm:
                n9 = min(len(self._hisLex), len(self._hisPcm))
                at = (tick // NAMES_AFTER) % n9
                self._speak("his word %d" % at, self._hisLex[at], self._hisPcm[at], tick)
                self._quietAt = tick
                self.named += 1
                return None
        return None

    def _speak(self, name, frames, pcm, due: int) -> None:
        """QUEUE A WORD AS AIR: its sound plays out one tick a step from
        `due`.  `frames` are only what the instruments read (`_saying`)."""
        self._saying(str(name), frames)
        self._answer = (np.asarray(pcm, np.float32).ravel(), 0)
        self._due = int(due)

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
        """WHICH WORD THE BABY JUST SAID A PIECE OF, if any --- his words and
        her own table words alike --- as `(name, frames, pcm)`, the pcm being
        what she will say back as air.  The ids are her ear's (`_ids`), one
        shared id is one piece she can tell (his, 2026-09-06).  A word she has
        no sound for cannot be said back and is not a candidate.  None when
        the utterance holds a piece of nothing."""
        u = set(self._ids(frames))
        if not u:
            return None
        cands = [("his word %d" % k, w, self._hisPcm[k])
                 for k, w in enumerate(self._hisLex) if k < len(self._hisPcm)]
        cands += [(t, w, self._pcmHis.get(t)) for t, w in self._namesHis.items()]
        best, score = None, 0
        for name, w, pcm in cands:
            if pcm is None:
                continue
            ids = set(self._ids(w))
            if not ids:
                continue
            hit = len(u & ids)
            if hit >= 1 and hit > score:
                best, score = (name, w, pcm), hit
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
        the mom says the FOOD'S NAME --- her own word for the bottle
        ("milk", or his word for it if he named it), as air, at every
        feed.  Word with thing, thing with relief: the pair machinery
        binds them.  (It used to pick a form from the baby's own babble
        and say that back; gone, his word 2026-09-13.)"""
        if not self.on or self._answer is not None:
            return
        hisW = self._namesHis.get("bottle") or self._tableWord("bottle")
        pcm = self._pcmHis.get("bottle")
        if hisW is not None and pcm is not None:
            self._speak("bottle", hisW, pcm, tick)
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
