"""EVERYTHING SHE COMMANDS --- and her mouth and her breath are in here.

The owner, 2026-08-24: *"fuuuuck mouth this is muscle, breathe also --- why they
separete again?"*, and 2026-08-25: *"mouth and something else moves to muscles"*.

So there is no `voice.py`.  Her seven articulators --- how hard she BLOWS, how
fast the folds buzz, how open her mouth is, how far forward her tongue is,
whether it goes out through her nose, whether she closes, whether she hisses ---
are muscles, in the file her limbs live in.  Blowing is breathing.

**AND THAT IS WHY HER ALPHABET IS SMALL.**  Measured today, sweeping her whole
articulator space at four levels each (12,288 mouths) and then drawing until
nothing new arrived: **92 distinct sounds of the 513 the index can hold.**  She
reaches 18% of the space, and no amount of trying will find the rest --- a mouth
is a muscle and a muscle has a reach.  It is not a limit on what she can SAY:
one recall replays a part of her LIFE, so a word is a run of these in order and
nothing about it is hardcoded to a single sound.

`Voice` itself is copied unchanged from the old tree's `body/muscles.py`, with
only its imports rebound.  **It is her mother's mouth**: literally
`speech.voiced`, and her room does to it exactly what it does to anyone else's.

HER BRAIN NEVER TOUCHES AN ARTICULATOR.  It commands `output.sound.id` --- a
similarity index --- and `sounds.py` turns that into the seven levels below.
What comes back is `input.echo`, straight into her brain like a spindle:
*"we just know that is our voice"*.
"""

from __future__ import annotations

import numpy as np

from . import speech
from .joints import AXES
from .hearing import LISTENS, SOUND_BANDS, SOUND_SLIDES, TICK_SECONDS, bands_from_pcm

#: what she orders, in order: how hard she blows, how fast the folds buzz, how
#: open her mouth is, how far forward her tongue is, whether it goes out through
#: her nose, whether she closes, whether she hisses
VOICE_PARTS = ("loud", "pitch", "open", "front", "nasal", "close", "hiss")

#: HER PITCH LIVES IN `speech.py` --- it was written here twice, twenty
#: lines apart, and her hearing needs it as well.
from .speech import PITCH_HZ    # noqa: E402,F401
#: A TODDLER'S TRACT, 1.3 x AN ADULT'S --- his choice 2026-09-04 ('make her
#: proper, she has to be able to speak well').  Measured with a separate
#: judge (vosk, an adult-speech recogniser that knows nothing of her): the
#: shared mouth saying his word from her mother's own tracks is heard as
#: "matilda" at 1.00 up to a tract 1.3 x adult with the larynx at 250 Hz, as
#: "banana" at 1.4, and as "your" or nothing at the half-adult tract these
#: ranges gave her (about 1.75 x).  No recogniser hears a word from a
#: newborn's tract, hers or a real one's.  The ranges below are the adult
#: table's spans (F1 270-770, F2 1000-2300) x 1.3; F3 follows the same law.
OPEN_HZ = (350.0, 1000.0)      # F1, a toddler's tract (1.3 x adult)
#: F2, likewise --- BUT THE WHOLE ADULT TABLE, not a slice of it.  The span
#: above (F2 1000-2300) left out every back vowel: Peterson & Barney 1952,
#: men, put /u/ at 870 Hz and /o/ at 840.  x 1.3 = 1100.  Measured 2026-09-12
#: with the slice: the nearest sound her mouth could make to a piece of his
#: speech was 0.75 like it (median, her own ear's bands) --- no lookup can
#: pick a row she does not have.  His word: open her tract's reach.
FRONT_HZ = (1100.0, 3000.0)    # F2, the full adult table x 1.3

MOUTH_GAIN = 2.7
#: born able to be heard, and it is ONE number: she has one mouth
BORN_LOUD = 0.90
#: per second, not per tick
TIRE = 0.05 / 1.5
#: per second, not per tick
RECOVER = 0.02 / 1.5


STRENGTH = 0.75


# --- HER LIMBS --------------------------------------------------------------
#
# What a pull costs her and what it can still give.  Her STRENGTH and her
# FATIGUE never cross the wire: her body reports sensations, never its own
# variables, so she finds her limits only by hitting them.


class Muscles:
    """How strong each of her muscles is, and how tired.

    Not history --- only the latest matters --- so it is saved as one small
    record rather than a row per tick.
    """

    def __init__(self, names: tuple[str, ...] = AXES) -> None:
        self.names = names
        self.can = np.full(len(names), STRENGTH, np.float32)
        self.tired = np.zeros(len(names), np.float32)

    def pull(self, effort) -> np.ndarray:
        """How hard the muscle actually pulls, given how hard it was told to.
        Capped by strength, dragged down by fatigue.  Never the order back.

        **THIS CAPS FORCE, NOT DESTINATION.**  It used to be handed the ORDER,
        so a newborn's strength of 0.15 clipped how far she could even AIM:
        measured, ordering 0.5 and ordering 1.0 moved her arm to the same
        0.1429 m, and the top half of every motor line returned identical
        feedback and could therefore never teach her anything.  Aiming is free;
        getting there against something is what flesh may limit.

        Takes one number per muscle, or a whole movement `(muscles, slides)`.
        """
        effort = np.clip(np.asarray(effort, np.float32), 0.0, 1.0)
        if effort.shape[0] != self.can.size:
            raise ValueError(f"{self.can.size} muscles but {effort.shape[0]} orders")
        flat = effort.ndim == 1
        can = self.can if flat else self.can[:, None]
        tired = self.tired if flat else self.tired[:, None]
        return np.clip(np.minimum(effort, can) * (1.0 - tired), 0.0, 1.0)

    def spend(self, pushing: np.ndarray) -> np.ndarray:
        """Pay for it, and grow from it.  Returns what each pull cost her ---
        squared, because that is what work does.

        **AVERAGED ACROSS THE MOVEMENT, NEVER SUMMED.**  A movement is fifteen
        moments now, and describing one more finely must not make moving
        fifteen times dearer --- the same law `Voice.spend` already states for
        its 1,536 lines, and the previous project measured the failure the
        other way round: eight shape lines at full strength ran a babbling
        life's hunger to 1.0 and starved her.
        """
        pushing = np.clip(np.asarray(pushing, np.float32), 0.0, 1.0)
        held = pushing if pushing.ndim == 1 else pushing.mean(axis=1)
        # her strength does not change; only her tiredness does
        self.tired = np.clip(self.tired * (1.0 - RECOVER) + TIRE * held, 0.0, 1.0)
        # ...AND IT COMES BACK AT THE RATE IT WAS SPENT.  This used to mean
        # the fifteen slides into one number before anyone downstream saw it,
        # so what a movement COST her arrived as a single figure for a whole
        # tick while she had ordered fifteen.  Averaging is still what her
        # tiredness and her blood take (above, and `sandbox` means it for
        # `blood`), because a movement described more finely must not become
        # fifteen times dearer --- but the shape is no longer thrown away on
        # the way to her sense.
        work = pushing ** 2
        return work.astype(np.float32)

    def save(self) -> dict:
        return {"can": self.can.tolist(), "tired": self.tired.tolist()}

    def load(self, state: dict) -> None:
        self.can = np.array(state["can"], np.float32)
        self.tired = np.array(state["tired"], np.float32)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"<Muscles strength={self.can.mean():.3f} "
                f"tired={self.tired.mean():.3f}>")


#: WHAT HER FIVE ARTICULATOR LINES MEAN IN HERTZ.  Nothing in the contract says
#: any of this --- a line is a line and its meaning is the body's business ---
#: but a mouth has to have a real shape somewhere, and this is where.
#:
#: Wider than an adult's on purpose.  `speech.py`'s vowels run F1 280..730 and
#: F2 1090..2250, and a baby's tract is shorter, so her formants sit higher;
#: giving her the full human range and beyond means every sound her mother can
#: make is inside what she can reach, which is the precondition for imitation.
#: **THESE WERE AN ADULT MAN'S THROAT, AND SHE IS A BABY.**
#:
#: 120 Hz is an adult MALE fundamental.  An infant larynx cannot reach it:
#: a newborn's f0 runs about 350-600 Hz, and her vocal tract is roughly
#: half an adult's, so every formant sits far higher than an adult's too.
#: The old numbers described a grown man.
#:
#: IT DID NOT MATTER UNTIL 2026-08-18, and that is why it survived.  Every
#: sound she made lasted 1.2 slides --- 28 ms --- so it was noise and noise
#: has no pitch you can hear.  The moment her mouth was given mass and could
#: SUSTAIN, a baby started producing a held adult-male growl, and the owner's
#: first words on hearing it were *"its horrible"* and *"im scare"*.  He was
#: hearing a body that was never hers.
#:
#: The wider range was justified in this file as letting her reach every
#: sound her mother can make.  That is a superpower and not an ability: a
#: real infant cannot make her mother's pitch either, and imitation in this
#: project is scored on the SHAPE across bands (`utter._unit`, loudness and
#: therefore level divided out), not on matching her absolute f0.  Giving
#: her a throat she does not have bought nothing and cost her her voice.
OPEN_HZ = (350.0, 1000.0)      # F1, a toddler's tract (1.3 x adult; see above)
FRONT_HZ = (1100.0, 3000.0)    # F2, the full adult table x 1.3 (see above)

#: HOW LOUD SHE IS AT FULL EFFORT, as a peak in the microphone.  Her mother's
#: words are normalised to 0.7 and this puts a full-throated open vowel of hers
#: in the same place, so being loud means the same thing for both of them.
#:
#: SHE IS NOT NORMALISED, AND THAT IS THE POINT OF HAVING A NUMBER HERE AT ALL.
#: Her old mouth divided every slide by its own peak, so ordering `loud` 0.05
#: and ordering 1.0 came out at the same level and the line taught her nothing.
#: A word is normalised because a recording is; a mouth is not.
#:
#: MEASURED, not chosen: at 1.0 her loudest vowel peaked at 0.35 and everything
#: below `loud` 0.35 fell under her ear's own noise floor, so the bottom third
#: of the line was silence and could teach her nothing.  This puts her loudest
#: at her mother's 0.70, and makes her audible from about 0.10.
#:
#: It is the ONLY number about her loudness now.  There was briefly a second
#: one hiding in the filter --- a low formant rang about a hundred times louder
#: than a high one, so how loud she was depended mostly on what she was saying.
#: `speech._resonate` divides the peak out exactly, so it does not any more.
MOUTH_GAIN = 2.7



# --- HER MOUTH --------------------------------------------------------------


class Voice:
    """Her mouth: PIECES OF A REAL HUMAN VOICE, in her register.

    HIS DECISION, made 2026-08-29 and CONFIRMED 2026-08-30 over every
    alternative: *"we decide that we able record all needed sounds that she
    need to produce all human letters and choose them by ticks.  Everything
    pretty easy."*  The synthetic larynx could not reach human sounds ---
    its whole measured range was held bell-tones --- so her sound units are
    recorded pieces, cut from his certified recordings, shifted into her
    register.  Speaking is PLAYING a piece: a command (id, lvl) starts its
    clip, a cursor walks it tick by tick, the clip's end is the utterance's
    end, and silence re-arms it.  Loudness, tiring, the echo line, the
    ladder over ids, the judge --- every seam above is untouched.

    The palette GROWS one way only: he records, the pieces are cut, he
    certifies them with his own ears, they join the bank.
    """

    #: A CLICK IS A STEP, AND A STEP IS NOT A SOUND SHE MADE.  Measured
    #: 2026-08-31 on her reproduction of his certified word: 14 of 40 tick
    #: boundaries jumped more than ten times the normal sample-to-sample
    #: step, because a piece begins at its own first sample whatever the
    #: air was doing a moment before.  He heard it as *"like some one put
    #: mic simultineosli with her speach"*.  Two milliseconds of ramp at
    #: each seam is below the shortest sound she can make (one tick, 11.1 ms)
    #: and
    #: removes the step; nothing about WHICH piece she says changes.
    FADE = 0.002

    #: HOW MANY SOUNDS OF ONE NAME HER MOUTH HOLDS.  A word is 10-20 ticks
    #: and the busiest name in his certified word carries five of them, so
    #: this is a word's worth and a little --- long enough to say back what
    #: was just said, short enough that it is imitation and not a recording.
    KEEPS = 12

    def __init__(self, bands: int, slides: int) -> None:
        self.bands, self.slides = bands, slides
        self.parts = len(VOICE_PARTS)   # kept: old probes ask the shape
        self.can = np.float32(BORN_LOUD)
        self.tired = np.float32(0.0)
        self._grain = int(round(speech.RATE * TICK_SECONDS)) // slides
        self._step = self._grain * slides
        self.pcm = np.zeros(0, np.float32)
        #: THE LAST `LISTENS` OF HER OWN AIR --- her ear names each piece of
        #: hers from it, exactly as it names each piece of his (`hearing.door`).
        self._air = np.zeros(0, np.float32)
        #: WHERE HER MOUTH IS RIGHT NOW --- carried across ticks, because a
        #: mouth does not reset between them.  `say` reaches what it is told
        #: from here; see `say`.
        self._shape = np.zeros(self.parts, np.float32)
        #: EVERY PIECE HAS ITS OWN ID, AND THE NAME HER EAR GIVES IT IS ONLY
        #: THE INDEX.  His, 2026-08-31: *"Each of prelearned sound sounds also
        #: has to feel just one tick and has his own ID."*
        #:
        #:     bank   piece id -> one tick of her
        #:     rows   her ear's name -> the pieces wearing it, FRESHEST FIRST
        #:     ears   piece id -> the shape her ear heard it as
        #:
        #: `rows` is what she is COMMANDED in, because that is what her ear
        #: answers and what her mind stores (`expect.near` reads that id
        #: arithmetically --- a piece id in that channel would silently mean
        #: nothing).  One name held two moments with 0.000 in common, and
        #: filing one piece per name made her say the right sequence of wrong
        #: sounds --- 11 of 39 ticks were the sound he actually made, and he
        #: heard it: *"it's again half of my word"*.
        self._ramp = int(round(speech.RATE * self.FADE)) or 1
        self._last = np.float32(0.0)   # where the air was left, for the seam
        #: A HELD COMMAND OUTLIVES ITS PIECE.  His pieces are cut in HER
        #: register --- x1.75 shorter than the word he said --- so a slot
        #: opened by his own pace is longer than the piece that fills it,
        #: and the piece ran out mid-word: measured 15 of 41 ticks silent,
        #: 37% of his certified word torn into holes.  A command still
        #: held means she is still saying it, so the piece begins again.
        #: `False` restores the old law (a finished piece stays finished)
        #: without touching anything else.
        self.rearm = True

    def say(self, ordered) -> np.ndarray:
        """HER OWN MOUTH: `(parts, slides)` in, `(bands, slides)` out --- what
        she actually made, which is not what she meant.  `self.pcm` is the raw
        sound, exactly as a microphone would have it.

        **RESTORED 2026-09-01, and nothing calls it yet.**  It was removed in
        `e235dbc` when her mouth became recorded pieces of his certified words,
        and `body/sounds.py` --- the table of everything her mouth can make ---
        has been calling a method that did not exist ever since, so her
        alphabet could not even be regenerated.

        WHY IT WAS ABANDONED, in his words 2026-08-29: *"we decide that we able
        record all needed sounds that she need to produce all human letters and
        choose them by ticks"*, because the synthetic larynx's *"whole measured
        range was held bell-tones"*.

        **AND WHY THAT IS NO LONGER TRUE.**  His, 2026-09-01: *"we do not need
        prelearned sounds anymore, we recorded them because we can't put sound
        in tick because of all muscles that we need to produce those sounds."*
        A held bell-tone is what ONE articulator setting per tick sounds like
        when a tick is long: `SOUND_SLIDES` is 1, so her mouth is told one
        shape per tick and holds it.  At 1.5 s a tick that is a 1.5-second
        drone and nothing else is reachable.  AT HER 11.1 ms IT IS NINETY
        SHAPES A SECOND, and a formant transition takes 30-50 ms --- three to
        five shapes across one transition, so a syllable is not merely
        expressible but shaped.  A stop burst is 5-20 ms: most of that range
        now fits in a tick, which it did not at 33 ms.  (Her clock is
        `MATILDA_FPS`; 11.1 ms was measured 2026-09-11 to be the best of
        22.2 -> 6.7 ms for naming his voice --- 159 distinct names against 146
        at 8 ms --- so faster is not better here, it is worse.)

        HER MOUTH HAS MASS, THE SAME WAY HER JOINTS DO --- `at + (want - at) *
        rate`, carried in `_shape` across ticks.  The rate is 1.0, which is not
        a number anybody picked: it is her own grain, the same slide her ears,
        `speech.py` and the manifest are all already built on.  `ACT_RATE` is a
        shoulder's and would have made her mouth four to nine times too slow
        for a consonant.
        """
        want = np.clip(np.asarray(ordered, np.float32), 0.0, 1.0)
        if want.ndim != 2 or want.shape[0] != self.parts:
            raise ValueError(
                f"voice has {self.parts} parts, got {want.shape}")
        # HOW MANY STEPS THIS TICK IS BUILT FROM --- his, 2026-09-01: *"if we
        # can put in one fps 9 of parts like path for sound that body has to
        # produce"*.  Her EAR still gets ONE frame a tick (`SOUND_SLIDES`, and
        # that does not move: 64 slides at 33 ms is 8 samples and no band
        # split is possible on that).  Her MOUTH is the other half of the
        # tick, and it can be told a PATH: nine steps is 3.7 ms each, and a
        # stop burst is 5-20 ms, so a plosive fits inside one tick for the
        # first time.  A vowel is the same path with every step alike.
        steps = int(want.shape[1])
        grain = max(1, self._step // steps)
        rate = 1.0
        held = np.empty_like(want)
        at = self._shape
        for k in range(steps):
            at = at + (want[:, k] - at) * rate
            held[:, k] = at
        self._shape = at
        want = held
        loud, pitch, open_, front, nasal, close, hiss = want
        # A MOUTH THAT IS NOT BLOWING MAKES NOTHING, and does not cost her the
        # milliseconds it takes to work that out the long way.
        if not (loud > 0.0).any():
            self._voice = None            # a silent mouth starts afresh
            self._lineEnd = None
            self.pcm = np.zeros(steps * grain, np.float32)
            # ...and the tail of what she just said is still in the air her
            # ear listens to, for as long as it would be in his
            self._air = np.concatenate([self._air, self.pcm])[-int(round(speech.RATE * LISTENS)):]
            return bands_from_pcm(self._air, speech.RATE, LISTENS)
        f0 = PITCH_HZ[0] + (PITCH_HZ[1] - PITCH_HZ[0]) * pitch
        f1 = OPEN_HZ[0] + (OPEN_HZ[1] - OPEN_HZ[0]) * open_
        f2 = FRONT_HZ[0] + (FRONT_HZ[1] - FRONT_HZ[0]) * front
        # the third formant is not commanded: it rides with the second in a
        # real tract, and a line for it would be a dial she cannot hear
        # F3 SITS NEAR ONE PLACE FOR A TRACT OF A GIVEN SIZE.  The old law rose
        # 800 Hz with `front`, above 3,700 Hz at a front vowel; the separate
        # judge heard her mother's own "matilda" through her articulators as
        # "matilda" 0.77 with the table's F3 and as "your" with that law
        # (2026-09-04).  The English table's F3 spans 2,400-2,550 Hz; x 1.3.
        # ...AND THE WHOLE TABLE'S F3 SPANS 2,240-3,010 (Peterson & Barney
        # 1952, men, /u/ to /i/), x 1.3 = 2,900-3,900: it rides with `front`
        # across a child's span, as the second formant does, instead of
        # standing at one place.  His word, 2026-09-12, with the reach above.
        f3 = 2900.0 + 1000.0 * front
        # HOW HARD SHE BLOWS is what her flesh may limit, and it limits the
        # blowing, not the shape --- being tired makes her quiet, not slurred.
        blow = loud * float(min(1.0, self.can)) * (1.0 - float(self.tired))
        last = getattr(self, "_lineEnd", None)          # where each line ended last tick
        def over(per_step, key=None):
            """One value a step -> one a sample, SLIDING rather than stepping.
            A mouth has inertia; a step at a seam is a click she never asked
            for and cannot learn to avoid.  AND IT SLIDES FROM THE LAST TICK:
            with one step a tick the old form was a constant, an 11 ms stair at
            every seam --- the separate judge (2026-09-04) heard the same word
            at 0.48 stepped and 0.85 slid."""
            n = steps * grain
            vals = np.asarray(per_step, np.float64).reshape(-1)
            if key is not None and last is not None and key in last:
                vals = np.concatenate(([last[key]], vals))
                where = (np.arange(n) + 0.5) / grain          # 0 = the last tick's end
                return np.interp(where, np.arange(len(vals)), vals)
            where = (np.arange(n) + 0.5) / grain - 0.5
            return np.interp(where, np.arange(steps), vals)

        # CLIPPED, BECAUSE A MICROPHONE CLIPS --- and because `bands_from_pcm`
        # reads anything past 1.5 as int16 and divides by 32,768, which would
        # turn her loudest sound into silence.
        # HER MOUTH IS CONTINUOUS ACROSS TICKS (2026-09-04): the glottal phase,
        # every resonator, the noise and the lips carry on from the last tick.
        # It used to restart all of them ninety times a second --- the seam
        # rattle the guardian measures, and the buzz that made the same
        # synthesiser unrecognisable through her and recognisable through her
        # mother (one call per word).
        lines = {"f0": f0, "f1": f1, "f2": f2, "f3": f3, "blow": blow, "nasal": nasal, "close": close, "hiss": hiss}
        pcm, self._voice = speech.voiced(
            over(f0, "f0"), over(f1, "f1"), over(f2, "f2"), over(f3, "f3"),
            over(blow, "blow") * MOUTH_GAIN, over(nasal, "nasal"),
            over(close, "close"), over(hiss, "hiss"),
            # the formants retune twice a tick: the judge heard 0.65 once, 0.82 twice
            block=(grain // 2 if grain % 2 == 0 and grain >= 2 else grain),
            state=(getattr(self, "_voice", None) or {}))
        self._lineEnd = {k: float(np.asarray(v_, np.float64).reshape(-1)[-1]) for k, v_ in lines.items()}
        self.pcm = np.clip(pcm, -1.0, 1.0)
        # HER ROOM DOES TO HER WHAT IT DOES TO HER MOTHER.  Same split, same
        # noise floor --- no path through this body treats her own voice as a
        # special kind of sound.  Her piece is named from the last `LISTENS`
        # of her own air, exactly as his is from his (the one gate).
        self._air = np.concatenate([self._air, self.pcm])[-int(round(speech.RATE * LISTENS)):]
        return bands_from_pcm(self._air, speech.RATE, LISTENS)

    def spend(self, made: np.ndarray) -> float:
        """Pay for the sound.  Shouting tires a throat; whispering does not."""
        used = float(np.asarray(made, np.float32).mean())
        self.tired = np.float32(
            min(1.0, max(0.0, float(self.tired) * (1.0 - RECOVER * TICK_SECONDS)
                         + TIRE * TICK_SECONDS * used)))
        # PER TERRITORY, NOT PER LINE.  Her voice is one sound however many
        # numbers it is written down in --- re-describing it more finely must
        # not make sounding more expensive.  The previous project measured the
        # failure the other way round: eight shape lines at full strength alone
        # ran a babbling life's hunger to 1.0 and starved her.
        return float((np.asarray(made, np.float32) ** 2).mean())

    def save(self) -> dict:
        return {"can": float(self.can), "tired": float(self.tired)}

    def load(self, state: dict) -> None:
        # one mouth, one number.  Older saves kept a gain per band.
        self.can = np.float32(np.mean(state["can"]))
        self.tired = np.float32(np.mean(state["tired"]))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Voice loud={float(self.can):.2f} tired={float(self.tired):.3f}>"
