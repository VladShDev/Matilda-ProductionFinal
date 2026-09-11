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
"""HER CHEMISTRY --- levels she feels and cannot control.

Ported whole from matilda2 `body/hormones.py` (the useful thing in the parts
bin), with two changes only: it reads the ALL-MUSCLES tick --- the shape that
killed matilda2 at its first tick, `motor.lvl` on a list --- and it now does
itself what his step asks, 2026-08-26: *"nexst step connect hormons like no
name lines in to sensen but map them to state to the will be able to correct
state"*.  So `live` ends by writing its levels onto the tick as sensor rows
carrying nothing but an id, and their corrections SUM into `state`.

A hormone is not a thing her brain knows about.  It is an input line with an
id and a pull on her state, and nothing anywhere names one.  She feels a line
move, she feels herself get worse, and noticing that those two happened
together is the whole of her learning.

Her state is the SUM of what every hormone is doing to her, and belongs to
none of them.  EVERY CONSTANT HERE IS THE MEASURED ONE, unchanged.
"""

from __future__ import annotations

from sandbox import TICK_SECONDS
from structure import Sensor, Tick

#: HER BODY'S WIRING --- which line, never what it means.  Her brain sees an
#: integer and a level and nothing else.
MOUTH = 12          # what the world has put against her mouth
EASE = 9            # ...and the problem receding
NEED = 7            # ...and what going without does to her
DULL = 8            # ...and what nothing-happening does
NEW = 10            # ...and the world showing her something never seen ---
                    # step 3 of his five: the new itself pulls her, or she
                    # stops exploring the day her own body is mapped
LONE = 11           # ...and nobody being with her --- step 4: the
                    # caregiver's nearness is a need like food, felt as a
                    # line, never named; a voice from anywhere is company
AIR = 14            # ...and what is left in her chest.  His, 2026-09-11:
                    # *"if she pushing hard all her body, it's already mean her
                    # breathing has to recover her, not create some new words or
                    # sounds ... she can try to produce that breeze, but if she
                    # already spend to all her body movements and she need just
                    # one more breeze, it will split her sounding."*  One
                    # resource, two customers: her flesh spends it on work and
                    # her mouth spends it on sound, and they take from the same
                    # chest.  Nothing here is a cycle and nothing switches her
                    # off --- she runs out because of what SHE did, and running
                    # out is the whole behaviour.  (The lungs of 2026-09-04 were
                    # a rhythm imposed on her voice and he threw them out the
                    # same day: *"her voicing became a mess after this update"*.)
PRESS = 13          # ...and being awake itself, accumulating --- the
#: NEW (10) IS GONE FROM HER.  His word, 2026-09-11: the alarm that fed it is
#: out, and novelty --- which is what NEW was named for --- has always travelled
#: on EASE instead, sized by the change and correctly signed.  A line whose only
#: input has been removed is a constant, and a constant is no signal (his rule
#: 8).  The id is left defined above so an older record still reads back.
HORMONE_IDS = (NEED, DULL, EASE, LONE, PRESS, AIR)   # the order of `corrections()`
                    # panel's fifth, on his word 2026-08-28 ("rest 5
                    # mplement"): pressure builds with waking and effort
                    # and drains only in stillness (Borbely's Process S).
                    # NOT a coded sleep state --- rest emerges because
                    # stillness is the one act that pays this line off

#: Every rate is PER SECOND and is multiplied by her tick once, where used.
DRAIN = 0.0012 / 1.5        # what she burns doing nothing at all
WORK_COSTS = 0.010 / 1.5    # what working costs on top, per unit of effort
#: HOW MUCH OF AN OFFERED MOUTHFUL GOES IN, per unit and second.  His yes,
#: twice (2026-09-05 morning, cbffd0f: *"the milk is worth two minutes of
#: hunger"*; and the same evening, on the rebuilt mind, "Yes"): at 0.01 a
#: mouthful was worth one second of hunger --- measured, no run ever changed
#: her state by being fed, and with dopamine as relief the line sat at 0.010
#: for a whole life.  At 0.5 a mouthful is 0.11 of hunger, two minutes of
#: her climb, and relief exists.  The law is unchanged; only its rate.
FEEDS = 0.5
ENOUGH = 0.35               # where her comfort crosses zero
SATED = 0.15                # she stops drawing as she fills
#: HOW FULL "NOTHING IS HAPPENING" CAN GET --- and it KEEPS RISING, because
#: his rule is that it rises: *"it's rising because for us, staying in one
#: state even comfortable, it's like lose any motivation"*.
#:
#: It was 0.05, which x DULLS 2.2 is a -0.11 pull: enough to close ONE run and
#: then nothing, because a saturated boredom is a CONSTANT, and a constant
#: cannot change her state.  Measured 2026-09-01: DULL pinned at 0.050, her
#: state pinned at -0.09, and her runs frozen at 72/69 for a full minute while
#: she laddered 331 times --- acting, but never closing an experience, so
#: never learning from any of it.
#:
#: At 0.45 it can climb all the way to a full -1.0 over a long enough nothing,
#: which is what a bored infant actually does: fusses, then cries.  It still
#: takes minutes to get there (BORES fills 0.0033 a second) and ANY real
#: change empties it, so a girl with something happening never feels it.
#: 0.45 was measured too and it was WRONG THE OTHER WAY: x DULLS it reaches
#: -0.99, which makes being bored exactly as bad as starving to death.  A
#: bored infant fusses and cries; she does not die of it.  0.20 x 2.2 = -0.44:
#: clearly unpleasant, well past the tenth that closes a run, and never
#: mistakable for hunger.
BORING = 0.20
BORES = 0.05 * 0.1 / 1.5    # ...and how fast it fills
DULLS = 2.2                 # how much settled-in nothing drags her down.
                            # 1.4 could NEVER matter: 1.4 x 0.05 (the cap the
                            # old tree measured) = -0.07, under the tenth ---
                            # so the boring whispered below the threshold
                            # forever, full or upset (his, 2026-08-27:
                            # "boring almost not affect on her").  2.2 x 0.05
                            # = -0.11: full boredom can now close a run by
                            # itself, and the cap stays where it was measured.
SETTLES = 0.08 / 1.5        # how fast her sense of the usual moves
#: THE FLOOR --- his, 2026-09-05: *"that state never go back to zero point.
#: Floor of state every time moving slowly by bit ... when she once feel that
#: she can be more good than in this moment, after it she already has to want
#: to arise again."*  Her usual dopamine, followed slowly; dopamine pays its
#: excess over it, and sitting below it drags --- the craving.  The half-life
#: is HIS to set (asked 2026-09-05: a minute, ten, an hour?  "Implement");
#: ten minutes of her life stands until he names it.
FLOOR_HALF_LIFE = 600.0
#: THE BURST CLEARS --- biology's number, on his word (2026-09-06: "decide
#: with biology"): a phasic dopamine transient in the striatum is cleared by
#: reuptake within about a second (half-life of the order of 0.5-1 s; the
#: burst itself lasts 100-500 ms), while the tonic level, the floor, moves
#: over minutes.  Until then the line settled at her senses' own rate, a
#: half-life of 13 s, and bursts and relief piled up faster than they
#: cleared: dopamine sat at 0.55-0.83 through every doctor's visit, a
#: burst at the ceiling could not rise, and a run's worth read the depth of
#: the dip before it instead of the size of the new thing.
BURST_HALF_LIFE = 1.0
#: tiredness's rates, SCALED FROM HUNGER'S OWN so no new timescale is
#: invented.  Honest numbers, measured by the reviewer: this line is an
#: EFFORT INTEGRAL, not clock-time sleep pressure --- it builds only
#: while she is spending (net build needs spent > ~0.1) and drains while
#: she is still.  At full effort it crosses ENOUGH in ~3 minutes of her
#: time and saturates in ~9; at RESTS = DRAIN full stillness clears
#: ENOUGH in ~15 minutes against hunger's 7-minute climb --- autofeed
#: feeds her without her moving, so stillness can carry through a feed.
#: The ratios are FIRST CUTS awaiting his word against her live rhythm.
#: RETUNED AFTER THE FIRST NIGHT, on the observed spiral: state pinned
#: flat at -1 kept the heartbeat firing discover every itch, so she
#: never stopped spending, so pressure never drained, so state stayed
#: pinned --- exhaustion causing the flailing that prevents rest.
#: TIRES at a quarter of work saturated her in minutes of ordinary
#: fidgeting; a sixteenth matches her true duty cycle.  And the drain
#: no longer demands near-total stillness: half-spent fidgeting still
#: rests her at half rate (`quiet` is 1 - spent/2), because a baby
#: winding down fidgets, and winding down must still be possible.
WAKES = DRAIN / 4.0
TIRES = WORK_COSTS / 16.0
RESTS = DRAIN * 1.5


class Hormones:
    """Her levels.  Read her tick, write her lines, pull on her state."""

    def __init__(self) -> None:
        #: her state a tick ago --- boredom is her condition standing still
        self._wasState = 0.0
        self.level = {AIR: 1.0, NEED: 0.30, DULL: 0.0, EASE: 0.0,
                       LONE: 0.30, PRESS: 0.0}
        self._was: dict = {}
        self._usual = 0.0
        self._usualEase = 0.0       # the floor: her usual dopamine, slow
        #: HER ONE VALUE AND ITS FLOOR --- what she has learned she can feel,
        #: followed at the same rate her dopamine's floor is followed.  None
        #: until her first tick has been lived, so it starts where she does.
        self._usualState = None
        self._wasBefore = 0.0       # her state the tick before last

    def live(self, tick: Tick, eased: float = 0.0,
             fresh: float = 0.0) -> None:
        """One tick of being alive --- and the tick leaves changed.

        Reads what her body reported, moves her levels, then connects them:
        one sensor row per hormone (id and level, no name), and `state` set to
        the sum of their corrections.
        """
        life = tick.life
        # WHAT THE WORLD HAS PUT THERE, read off the line it writes.
        at = 0.0
        for one in life.input.sensor:
            if int(one.id) == MOUTH:
                at = float(one.lvl)

        # WHAT SHE SPENT --- her own output, all of it: every held muscle and
        # her voice --- AS A FRACTION OF WHAT HER BODY CAN SPEND.  His word,
        # 2026-09-03: *"read her effort in her body's units for hunger too"*.
        # These rates were measured when an act was one line and `spent` a
        # level or two; since a decision is a whole pose (37 muscles) a
        # newborn's flail summed to ~20 and burned ~0.1 hunger a second
        # against a mouthful's 0.003, so every life pinned at -1 in two
        # minutes and 133 feeds in fifteen did not move it.  `motors` is a
        # body fact off /able (the mind sets it); the timescales measured at
        # rest (a 7-minute climb) are untouched.
        spent = ((sum(float(m.lvl) for m in life.output.motor)
                  + float(life.output.sound.lvl))
                 / max(1, int(getattr(self, "motors", 1))))

        # ...AND HOW MUCH ANYTHING MOVED, read off her own lines.
        now = self._lines(life)
        moved = sum(abs(v - self._was.get(k, v)) for k, v in now.items())
        self._was = now

        # SHE TAKES WHAT SHE WANTS OF WHAT IS THERE, and tails off as she fills.
        wasNeed, wasLone = self.level[NEED], self.level[LONE]   # for the relief below
        drawn = at * min(1.0, self.level[NEED] / SATED)
        self.level[NEED] = min(1.0, max(0.0,
                                        self.level[NEED]
                                        + (DRAIN
                                           + WORK_COSTS * max(0.0, spent)
                                           - FEEDS * drawn) * TICK_SECONDS))

        # ...AND NOTHING HAPPENING FILLS UP.  Only the EXCESS over what has
        # been arriving lately counts.
        # NEWNESS IS A FRACTION OF HER USUAL, NOT A RAW SUM.  `moved` adds up
        # every line she has --- thirty motors, thirty spindles, her sensors,
        # every thing in her view --- so its ordinary jitter is TENS, and
        # `news * 10` saturated on every tick of her life.  Her boredom was
        # wiped before it could fill, and measured 2026-09-01 it sat at
        # 0.000-0.028 for ever while she lay fed and perfectly still: a
        # comfortable Matilda had no reason to do anything, and her laddering
        # froze the moment her hunger was met.
        #
        # His rule, and it is old: *"then goes nothing, we do not hungry and
        # nothing happened, so BORING IS GAME CHANGER our state goes down ...
        # she start exploring, new exp, reward rising her state and decreasing
        # boring --- and she has one more exp: discovering comes to profit"*.
        #
        # The law was already right --- the excess over the usual, the same
        # one her rarity and her reward use --- it was only being read on a
        # scale where every moment looks eventful.  As a FRACTION of her own
        # usual it means what it says: a moment twice as busy as her usual
        # empties her boredom, and a room that is always a little restless is
        # her usual and is not new at all.
        # ...AND WHAT IS BORING IS HER CONDITION STANDING STILL, not her
        # senses jittering.  His, and it is the whole of it: *"if you
        # comfortable and nothing change around you ... staying in one state
        # even comfortable for us, it is like lose any motivation"*, and
        # *"when our past became the same moment, became BORING"*.
        #
        # It read `moved` --- the sum of every line she has, thirty motors,
        # thirty spindles, her sensors, every thing in view --- whose ordinary
        # jitter is TENS, so newness saturated on every tick and her boredom
        # was wiped before it could fill.  Measured 2026-09-01: DULL sat at
        # 0.000-0.028 for ever, and a fed girl froze --- 0 tries in four and a
        # half minutes, her state pinned at +0.001.
        #
        # Her STATE is the thing that is supposed to stop moving.  A state
        # that has not changed is the same moment happening again, however
        # comfortable it is, and that is what has to drag her down.  The law
        # is unchanged --- the excess over her own usual, hers everywhere ---
        # only what it is asked OF.
        #
        # Biology agrees and names it: an affective state provokes an
        # opposing process that grows with repetition (Solomon), and reward
        # is signalled by PREDICTION ERROR, not by level --- a constant
        # reward produces no signal at all.  A fed, warm, unchanging baby is
        # receiving nothing.
        # HER STATE SURVIVES THE TICK.  This read `life.input.state` off a tick
        # rebuilt from the body's JSON, and the body NEVER WRITES THAT FIELD ---
        # a whole-tree grep finds exactly one writer, the last line of `live()`
        # below, which runs AFTER this read and onto the mind's own object.  So
        # `now9` was 0.0 on every tick of her life, `swing` was 0.0 always, and
        # boredom's only input was dead: it would have read zero if her state
        # had swung from +1 to -1 every tick.  Her own last state is what this
        # wants and `_wasState` was already here to hold it.
        now9 = float(self._wasState)
        swing = abs(now9 - self._wasBefore)
        self._wasBefore = now9
        news = max(0.0, (swing - self._usual) / max(self._usual, 1e-6))
        self._usual += SETTLES * TICK_SECONDS * (swing - self._usual)
        self.level[DULL] = min(BORING, max(0.0,
                                           self.level[DULL]
                                           + BORES * TICK_SECONDS
                                           - self.level[DULL]
                                           * min(1.0, news)))

        # ...AND THE PROBLEM RECEDING FEELS GOOD --- the margin WIDENING,
        # never the margin being wide.  AND THE SAME LINE RUNS DOWN.  His,
        # 2026-08-26, on a broken expectation: "same way as up more that you
        # spand withou rewar more you feel bad about" --- so satisfaction is
        # signed: what she spent on a promise that broke pulls her under.
        # DOPAMINE IS RELIEF --- his, 2026-09-05: *"she has to be rewarded by
        # dopamine, not by hungry."*  A need FALLING pays this line: milk while
        # hungry, her mother's voice while alone, by the amount it fell (a
        # fall of ENOUGH --- from hungry to comfortable --- is a full burst).
        # AND THE NEW PAYS IT --- his, 2026-09-06: *"each line can burst
        # dopamine, each change first time ... it depends how big was this
        # increasing or decreasing"*: `eased` is that burst, the biggest
        # first-time change on her lines this tick, found by her mind --- and
        # her own dopamine with it, signed: the ramp along a path, the
        # reward or the DIP where a path ends, the song rule's rise and its
        # fall.  (Until 2026-09-06 a negative `eased` was clamped away here,
        # and his dip --- "decrease her mood and lower experience rating" ---
        # never reached the line.)  The level itself stays within 0..1
        # (`life._line`, one burst a tick, the biggest change).  A burst decays at her usual's own settle.  (The
        # line is finished below, after LONE moves.)

        # ...AND THE NEVER-SEEN LIGHTS A LINE OF ITS OWN.  `fresh` is the
        # fraction of this moment's world that she has never met; the level
        # takes it at once and settles at the same rate her senses settle,
        # so a flash of new lingers just long enough to feel.
        # (the never-seen line is gone with the alarm that fed it; what the
        #  never-seen does to her now is the burst, on EASE, sized by the change)

        # ...AND BEING ALONE GNAWS LIKE HUNGER.  The same rule as food, on
        # another line: loneliness rises at the body's own drain and a voice
        # arriving --- anyone's, the rule is general over sources --- feeds
        # it back down in proportion to how much she wanted company.
        # COMPANY IS PRESENCE, NOT VOLUME.  Feeding loneliness by the
        # voice's raw loudness was tuned for HIS microphone; the mom's
        # gentle answers arrive at ~0.17 and 28% duty, and the drain
        # outran them for ever --- measured: LONE pinned at 1.0 through
        # an afternoon of constant conversation, state clamped at -1
        # while she talked with her mom.  An audible voice is company at
        # any loudness; 0.05 is simply "clearly heard" on her ear's own
        # scale (her loudest voice is 0.09).
        voice = min(1.0, float(life.input.sound.lvl) / 0.05)
        kept = voice * min(1.0, self.level[LONE] / SATED)
        # ...and company-hunger is SLOWER than food-hunger.  At hunger's
        # own refill rate, a mom who speaks every ten seconds could never
        # keep loneliness down (measured overnight: 356 namings and LONE
        # still equilibrated at 0.68, state pinned at -1 in a room full
        # of her father's words).  A quarter of the drain: ten quiet
        # seconds between words is waiting, not abandonment.
        self.level[LONE] = min(1.0, max(0.0,
                                        self.level[LONE]
                                        + (DRAIN / 4.0 - FEEDS * kept)
                                        * TICK_SECONDS))

        # ...AND BEING AWAKE ACCUMULATES, AND STILLNESS PAYS IT OFF.
        # Continuous rates, no gate, no named mode: `quiet` is how little
        # of her output is held, and the drain is proportional to it ---
        # a baby winding down IS the drain beginning.
        quiet = max(0.0, 1.0 - max(0.0, spent) / 2.0)
        self.level[PRESS] = min(1.0, max(0.0,
            self.level[PRESS] + (WAKES + TIRES * max(0.0, spent)
                                 - RESTS * quiet) * TICK_SECONDS))

        # HER CHEST.  Work takes air and so does sound, from the same place.
        # `spent` is what her flesh cost her this tick, already computed above
        # and already her own body's number; `blown` is how hard her mouth is
        # blowing, which is her `loud` muscle and nothing else.  What refills it
        # is being quiet --- the same `quiet` her tiredness already uses, so no
        # new rate is invented here and no clock is introduced.
        #
        # WHY THIS AND NOT A BREATHING RHYTHM: a cycle would switch her voice on
        # and off on a schedule she does not control, which is the designed
        # switch his rules forbid and the thing that ruined the 2026-09-04
        # lungs.  A resource cannot do that.  She simply runs out, and she runs
        # out because of what she just did --- so the gasp always has a cause,
        # and taking the breath BEFORE she speaks is something she can discover.
        # `loud` is the first of her voice muscles, and her body says where they
        # start (`/able: voiced`), so nothing here counts motors by hand.
        blown = 0.0
        mot = list(getattr(life.output, "motor", ()) or ())
        voiced9 = int(getattr(self, "voiced", 0) or 0)
        if mot and voiced9:
            at9 = len(mot) - voiced9                         # the first voice part
            if 0 <= at9 < len(mot):
                blown = float(mot[at9].lvl)
        self.level[AIR] = min(1.0, max(0.0,
            self.level[AIR] + (RESTS * quiet
                               - TIRES * max(0.0, spent)
                               - WORK_COSTS * blown) * TICK_SECONDS))

        # HIS CONNECT.  The levels go onto her tick as lines with only an id
        # --- replacing any row already wearing one of these ids, so a line
        # exists once --- and their pulls sum into her state.
        relief = (max(0.0, wasNeed - self.level[NEED])
                  + max(0.0, wasLone - self.level[LONE])) / max(ENOUGH, 1e-9)
        self.level[EASE] = max(0.0, min(1.0, self.level[EASE] * 0.5 ** (TICK_SECONDS / BURST_HALF_LIFE)
                                             + relief + float(eased)))
        # ...AND THE FLOOR FOLLOWS, SLOWLY --- her usual dopamine.  Its pull on
        # her state (`corrections`) is the excess over this: above it lifts,
        # below it drags, and the drag is the craving that feeds her effort.
        self._usualEase += (1.0 - 0.5 ** (TICK_SECONDS / FLOOR_HALF_LIFE)) * (
            self.level[EASE] - self._usualEase)
        # ...AND THE SAME FLOOR FOLLOWS HER STATE.  His, 2026-09-11: *"we spelled
        # just one value ... dopamine boost improves her state to close
        # experience, and that is her state when it is increasing, and the
        # difference became to this state floor, and we already had it in our
        # application."*  One value and one floor: this is the identical
        # arithmetic eleven lines above, on her state instead of on her dopamine,
        # and it is his addiction law of 2026-09-02 --- the floor rises with what
        # she has learned she can feel, so the same comfort stops paying and she
        # has to reach further for the same lift.
        #
        # AND IT IS WHY THE MEASURE HAS TO BE FLOOR-RELATIVE.  Measured
        # 2026-09-11: she sits 1.297 below the clamp while starving, and EASE
        # spans 0..1 --- so the largest dopamine surge she is physically capable
        # of moved her state by exactly nothing.  A difference from her own floor
        # cannot have that problem: the floor comes down with her, so a rise of
        # a tenth above it is a rise of a tenth whether she is comfortable or at
        # the floor of her life.  She is never out of reach of her own
        # improvement.
        #
        # (Set the floor from her first state rather than from zero, or her whole
        # first minute reads as a fall from a comfort she never had.)
        if self._usualState is None:
            self._usualState = float(life.input.state)
        self._usualState += (1.0 - 0.5 ** (TICK_SECONDS / FLOOR_HALF_LIFE)) * (
            float(life.input.state) - self._usualState)
        self._wasState = float(life.input.state)
        life.input.sensor = ([s for s in life.input.sensor
                              if int(s.id) not in self.level]
                             + [Sensor(id=float(k), lvl=float(v))
                                for k, v in sorted(self.level.items())])
        life.input.state = max(-1.0, min(1.0, sum(self.corrections())))

    def floor(self) -> float:
        """WHAT SHE HAS LEARNED SHE CAN FEEL --- her state's slow-following floor,
        the same shape her dopamine's floor has had since 2026-09-05 and his own
        law of 2026-09-02: *"more that you arise her state, more that you need to
        rise it."*  Her mind watches her state ABOVE this, so the same comfort
        stops paying and she has to reach further for the same lift."""
        return 0.0 if self._usualState is None else float(self._usualState)

    def corrections(self) -> list:
        """WHAT EACH ONE DOES TO HER --- one per hormone, and she sees none of
        it.  Going without pulls her down; so does nothing happening.  Neither
        is a reward and neither is named: they are her condition.

        A NEED PULLS DOWN WHEN UNMET AND ADDS NOTHING WHEN MET --- his,
        2026-08-27, after paradise turned out to be a coma: a full belly and
        company at hand summed +1.2, the clamp held her at +1.0 flat, no
        tenth could ever fire, and boredom's hardest pull (-1.4 x 0.05 =
        -0.07, the old tree's own measured cap) was smaller than the
        ceiling's excess --- boredom whispered, paradise shouted.  Met needs
        are SILENT now: her resting state sits near zero with headroom both
        ways, deprivation drags her under it, and the small pulls --- the
        boring, the new, relief and disappointment --- have their voices.
        """
        # THE NEW IS AN ITCH, NEVER A PAYMENT --- the panel's fourth, on
        # his word: a novelty spike counted as relief would make distress
        # a change-detector (the old tree's own recorded ruling).  Unseen
        # things standing in front of her PULL HER DOWN, and seeing them
        # into the known relieves only by lightening that pull --- so the
        # act that converts fresh to seen is credited by the ordinary
        # machinery, and "try it because it is new" is finally a pull her
        # choosing can feel.  Sleep pressure wears hunger's own shape:
        # silent while low, dragging as it climbs past the same ENOUGH.
        return [min(0.0, (ENOUGH - self.level[NEED]) / max(ENOUGH, 1e-9)),
                -DULLS * self.level[DULL],
                self.level[EASE] - self._usualEase,     # dopamine's excess over her floor
                min(0.0, (ENOUGH - self.level[LONE]) / max(ENOUGH, 1e-9)),
                min(0.0, (ENOUGH - self.level[PRESS]) / max(ENOUGH, 1e-9)),
                # ...AND AN EMPTY CHEST DRAGS, the same shape as any need going
                # unmet: full costs her nothing, running low pulls her under and
                # so raises her deficit, which is her effort.  So when she has
                # spent herself she works harder at getting her breath back ---
                # which is what a body does, and it is why the gasp comes.
                min(0.0, (self.level[AIR] - ENOUGH) / max(ENOUGH, 1e-9))]

    @staticmethod
    def _lines(life) -> dict:
        """Every level on her life, so "did anything move" needs no telling."""
        i = life.input
        # THE SAME THREE LINES HER MIND READS (`life.lines`) --- a sound is
        # what it is, how much of it, and which side, never one line per id.
        out = {("heard", 0): float(i.sound.similarity),
               ("heard", 1): float(i.sound.lvl),
               ("heard", 2): float(getattr(i.sound, "balance", 0) or 0),
               }
        for one in i.spindle:
            out[("spindle", int(one.id))] = float(one.lvl)
        for one in i.sensor:
            out[("sensor", int(one.id))] = float(one.lvl)
        for one in i.view:
            out[("view", int(one.id))] = float(one.similarity)
        return out
