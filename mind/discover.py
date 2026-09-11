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
"""DISCOVER --- nothing known helps, so she does everything her body allows,
all at once, like a newborn.

His, 2026-09-02, closing the audit: *"all at once like a nyeborn"* --- and
the loop: *"if she do not have exp she has to discover her voice, muscles,
and muscles also discovereble by those spindels ... so yes she has to do
everithng at the begining and all moves that would be discovered they stop
to pay her so during the time she would able to chose from profitable exp
only"*.  His theory's first page: a newborn pushes every part of her body
and screams, not knowing what she does; the next ticks carry messy outputs;
what pays gets kept.

It used to be a LADDER --- the same muscle one smallest step harder, then
the next muscle, then every sound one by one (his 2026-08-24 word, when a
step was the smallest her body could make).  One line a tick could never be
a newborn; and its unseeded fourth rung made every experiment dice.

HOW HARD.  His, the same day, when the bottle went and she starved in two
minutes: *"if she do not eat and she bored and so she has to try more with
wore high effort like people do till die so its like scare so hormons has
to push her to discover and find out the way to stay alive"*.  So the size
of the flail is her DEFICIT --- `want = (1 - state) / 2`, exactly the
quantity her reward already pays by (`reward.satisfaction`: more wanted,
more satisfied).  Content, she barely stirs; at the bottom she throws
everything she has.  One law, no new constant, no gate: her chemistry pushes
and the push is continuous.

SEEDED BY HER AGE.  Every draw comes from her tick count, so the same life
flails the same way twice --- an experiment, not dice (the belt taught this
at 25 / 16 / 27 steps).
"""

from __future__ import annotations

import random


def harder(lvl: float, resolution: float) -> float:
    """One step harder --- the smallest change her body can make.  Memory's
    one-line suggestions (`pick.pickTry`) still spend exactly this."""
    return min(1.0, lvl + resolution)


def discover(was, able, want: float, age: int, line: int = 1, way: int = 1,
             said: int = 0, at: float = 0.0):
    """ONE OUTPUT AT A TIME, EACH THROUGH ITS OWN RANGE END TO END.

    His, 2026-09-08: *"discovery also change to each output one by one, has
    to try each own point end to end, so if effort push line up it has to
    reach max than goes back"* --- and *"one by one til she find some output
    that push her closer to expectation inputs"*.

    Everything except the line she is on stays exactly where it was, so what
    changed is ONE thing and what it caused is hers to attribute.  The line is
    carried to its limit, then back to the other limit, and then the next line
    takes its turn.

    HER DEFICIT IS THE STEP, never smaller than the smallest her body can tell
    apart --- his own law, restored 2026-09-08 after I had fixed the step and
    left her deficit driving nothing at all: *"connect deficit back to
    discovery step"*.  It closes the loop he named: her closeness to what she
    is reaching for falls, the song rule takes that off her dopamine, her state
    falls with it, her deficit rises, and her sweep along the line gets longer
    --- she works faster the further she is from the point, and settles as she
    nears it.  Far off she crosses a line in a few steps; close in she inches
    along it at her own grain.

    This replaces the whole-body draw of 2026-09-02 (*"all at once like a
    nyeborn"*), which was measured on 2026-09-08 to be throwing all twenty six
    joints by up to two thirds of their range on every trial, because her
    boredom sits at its ceiling and so her deficit is pinned near 0.65.  A
    newborn who cannot make a small movement can never close on anything.

    AND HER VOICE IS NOT SWEPT --- it is asked for by name.  His, 2026-09-08:
    *"be sure that discover in that case avoid voicing muscles on his own try,
    so she able contain those muscles in her but can't try discover them
    because discover of sounds goes different way"*.  A sound needs loudness
    and a shape at the same instant, so a line-at-a-time sweep of her mouth is
    silence by construction: six of her seven articulators do nothing at all
    while the loudness sits at zero.  Her body already holds the pose for every
    sound it can make, so a trial names ONE sound a tick and her effort is how
    hard she means it --- and it names one while a body line is moving, so her
    voice and her body are tried together, not in turn.

    AND THE RUNG SHE IS ON IS CARRIED IN HER MIND, NOT READ BACK OFF HER
    BODY --- his find, 2026-09-08: *"why 60ms for 4 sec instead of +0.1 than
    -0.1, why you make everything simple into something unreachable"*.  This
    used to step from `was.motor` --- her previous output --- on a docstring of
    mine that said her body carries it forward.  IT DOES NOT: her output fades
    one resolution step a tick toward rest, and she only trials at a close, so
    about four seconds pass and `was` is all zeros by the time the next step
    reads it.  Every trial therefore computed `0 + step` and landed on the
    SAME rung for ever; the line only advanced at all when her deficit hit 1.0
    outright, which makes it a two-position toggle and not a sweep.  That is
    read off the code and not off a sample: `held` is her output, her output
    fades, and a fade to zero makes every step `0 + step`.  (The live number I
    first quoted here came from a sampler that never advanced its cursor ---
    `/ticks?from=` is an absolute tick index --- so it was forty ticks counted
    eight hundred times.  It is out.)  `at` is that rung, carried beside
    `line` and `way` exactly as they already were, so the walk
    is real --- up a rung a trial to the top, turn, down to the bottom, and
    the next line takes its turn.  Her body still fades between trials:
    holding a pose is a choice she keeps paying for, and that is his.

    `was` is her previous output --- what the OTHER lines are re-commanded at,
    so that what changed is one thing.  `able` her
    body's word on what she is --- how many motors, how many of them are her
    voice, her resolution, and the sounds her mouth can make --- `want` her
    deficit 0..1, `age` her tick count, `line` which output she is on and `way`
    +1 or -1, `said` how far along her sounds she has walked.  Returns the plan
    and where all of it stands after.  Nothing here knows what a muscle or a
    sound IS.
    """
    res = float(able["resolution"])
    motors = int(able["motors"])
    voiced = int(able.get("voiced", 0))
    # HER VOICE IS SWEPT LIKE EVERY OTHER MUSCLE.  His, 2026-09-11, on finding
    # her silent: *"before we make your echo, somewhere pointing not to right
    # place."*  It was here.  This line held her seven mouth muscles OUT of the
    # sweep --- she explored 30 of her 37 --- because her voice was reached the
    # other way, by NAMING a sound and letting the body walk a recorded piece.
    # That named path is gone (his word: *"we don't need already prepared table
    # ... she would produce any sounds"*), so with this line her voice was
    # reachable from neither direction: not swept, and the name consumed by
    # nothing.  Measured 2026-09-11: she ordered `loud` 0.729 through the dead
    # name and made 0.0000 of air, silent for ten minutes.  `loud` is one of the
    # seven, so it took her breathing with it.
    swept = max(1, motors)
    step = max(res, float(want))             # her deficit, never less than one step
    held = {int(m.id): float(m.lvl) for m in was.motor}
    line = int(line) if 1 <= int(line) <= swept else 1
    way = 1 if int(way) >= 0 else -1
    now = max(0.0, min(1.0, float(at))) + step * way
    nextLine = line
    if now >= 1.0:                           # the top of its range: turn back
        now, way = 1.0, -1
    elif now <= 0.0:                         # the bottom: this line is done
        now, way = 0.0, 1
        nextLine = line % swept + 1          # ...and the next starts at its own 0
    # her voice is among the motors (her mouth's seven articulators are
    # muscles like any other since 2026-09-02), so the sweep moves it too
    #: ...AND ONE SOUND OF HERS, NAMED, IN THE SAME TICK.  She walks her body's
    #: own list of what it can make, one a tick, at her effort.
    kinds = list(able.get("sounds") or ())
    sid, nextSaid = 0, int(said)
    if kinds:
        nextSaid = int(said) % len(kinds)
        sid = int(kinds[nextSaid])
        nextSaid = (nextSaid + 1) % len(kinds)
    plan = {"motor": [{"id": k, "lvl": (max(0.0, min(1.0, now)) if k == line
                                        else max(0.0, min(1.0, held.get(k, 0.0))))}
                      for k in range(1, motors + 1)],
            # ...and no sound is named.  Her mouth IS the seven muscles in the
            # row above; asking for a piece by name was the shortcut past it.
            "sound": {"id": 0, "lvl": 0.0}}
    return plan, nextLine, way, nextSaid, float(now)
