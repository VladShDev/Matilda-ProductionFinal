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
"""Her structure --- his, posted 2026-08-26, field for field and nothing added.

Four tables: `tick`, `exp`, `prediction`, `memory`.  `store.py` holds them in
DuckDB under exactly these names and shapes.  Every part is a TYPE, not a loose
dictionary: a misspelt field is an error where it is written, not a zero
somewhere later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- what she does ---------------------------------------------------------

@dataclass
class Motor:
    """One muscle, commanded --- one ROW of `Output.motor`.

    ALL OF THEM, EVERY TICK.  His, 2026-08-26: *"i never tald that her repeat
    of her full body movement able to conteine only one muscle per tick"* ---
    *"exp containe all info snapshot of life!!!! all muscles all inputs"*.
    """
    id: int = 1
    lvl: float = 0.0


@dataclass
class OutSound:
    """The sound she means to make.  His note on `id`: *"here is should be
    similarity index"* --- a row of her prelearned alphabet, never a waveform.
    """
    id: float = 0.0
    lvl: float = 0.0


@dataclass
class Output:
    motor: list[Motor] = field(default_factory=list)
    sound: OutSound = field(default_factory=OutSound)


# --- what reaches her ------------------------------------------------------

@dataclass
class Sensor:
    """One line of her body.  His note: *"vestibular one sensor by axis
    (6axes), hungy, pain. Just id every where"* --- one row per line, and the
    id is all she knows of it.
    """
    id: float = 0.0
    lvl: float = 0.0


@dataclass
class Spindle:
    """The answering half of `Output.motor` --- one row per muscle, every
    tick.  She finds out what her body did by being told, never by reading her
    own command back.
    """
    id: int = 1
    lvl: float = 0.0


@dataclass
class Echo:
    """Her own sound, back at her ear.  It says *this one was mine*."""
    id: int = 1
    similarity: float = 0.0
    lvl: float = 0.0
    balance: int = 0


@dataclass
class Heard:
    """Somebody else's sound, in the same index space as hers, so his voice
    and hers compare as integers.  His note on the exp's copy of
    `similarity`: *"prelearned similarity id"*.
    """
    id: int = 1
    similarity: float = 0.0
    lvl: float = 0.0
    balance: int = 0


@dataclass
class View:
    """One thing seen, where it is."""
    id: int = 1
    similarity: float = 0.0
    distance: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Input:
    state: float = 0.0
    sensor: list[Sensor] = field(default_factory=list)
    spindle: list[Spindle] = field(default_factory=list)
    echo: Echo = field(default_factory=Echo)
    sound: Heard = field(default_factory=Heard)
    view: list[View] = field(default_factory=list)


@dataclass
class Life:
    output: Output = field(default_factory=Output)
    input: Input = field(default_factory=Input)


# --- the four tables -------------------------------------------------------

@dataclass
class ExpRef:
    """Which experience a tick belongs to.  `0` is none."""
    id: int = 0


@dataclass
class Tick:
    """One moment of her --- table `tick`.

    A LIFE SNAPSHOT.  His, 2026-08-26: *"our tick is life snapshot so es
    action it is her plan on that tick and input is actualr response so next
    tick alredy have plan exp or discover"*.  `output` is her PLAN, `input`
    the actual response --- and `plan` says where the plan came from: the exp
    being reproduced, or 0 when discover (or nothing) made it, *"to be able
    refference to exp in case shes need to reproduse some"*.
    """
    time: float | None = None
    life: Life = field(default_factory=Life)
    exp: ExpRef = field(default_factory=ExpRef)
    plan: ExpRef = field(default_factory=ExpRef)


@dataclass
class Exp:
    """One experience --- table `exp`.

    THE GOAL IS ALL HER INPUTS AT THE MOMENT HER STATE CHANGED.  His,
    2026-08-26: *"our goual it is all inputs that we reach to moment state was
    changed"*.  Each channel appears once --- *"goal can be only one per
    channel"* --- and carries the rows that channel held at that moment, the
    same shapes as `Input` (his singles are row shapes, as with the muscles).
    `sensor` is here on the same word --- ALL inputs --- and hunger is a
    sensor, so the bottle moment is visible in the goal.
    """
    id: int = 1
    sensor: list[Sensor] = field(default_factory=list)
    spindle: list[Spindle] = field(default_factory=list)
    echo: Echo | None = None
    sound: Heard | None = None
    view: list[View] = field(default_factory=list)
    avgState: float = 0.0
    weight: float = 0.0
    #: WHICH EXPERIENCE THIS ONE IS PART OF --- 0 is none.  His decision,
    #: carried from the old tree and agreed again 2026-08-27: a chain is an
    #: exp with children --- same shape, same fold, at any depth --- and A
    #: RUN OF EXPERIENCES BECOMES ONE WHEN IT REPEATS, because that is when
    #: it has shown itself to be a thing rather than a coincidence.  This is
    #: what lets a small step from a big story outweigh a boring local step.
    exp: ExpRef = field(default_factory=lambda: ExpRef(id=0))


@dataclass
class Prediction:
    """One expected moment --- table `prediction`.

    A REFERENCE, NOT A COPY.  His pick, 2026-08-27 ("a"): a finished tick is
    immutable --- the tape's own contract --- so pointing at the chain equals
    copying it, byte for byte, forever.  `id` is the exp whose chain she
    expects, `at` which step of it, `time` the moment she expects it; the
    expected life is the chain's own tick, one join away.  The copied shape
    cost 2.8 million rows and 5.3 GB in one life, all of it re-writing seven
    immutable chains.
    """
    id: int = 1
    time: float | None = None
    at: int = 0


@dataclass
class Memory:
    """Experience ids, not moments --- table `memory`."""
    short: list[int] = field(default_factory=list)
    long: list[int] = field(default_factory=list)
