"""Her body in space: 17 joints, bones, hinge stops, and 13 muscles that hold a
length.

Ported from the previous project's `skeleton.py`, which is a real
position-based solver and was never the thing that was wrong.  Verlet
integration, distance constraints for bones, one-sided angle stops for knees and
elbows.  Nothing here knows about "sitting" or any other posture --- a shape can
only be the consequence of what her muscles are set to.

**A MUSCLE IS A LENGTH SHE ASKS FOR, AND IT DRIVES BOTH WAYS TO IT.**  The order
is not effort: 0 is the length that span has when she is laid out straight, 1 is
as short as it goes.  Too long and it pulls its ends together; too short and it
pushes them apart.  A cylinder rather than flesh, and a deliberate
simplification --- but it had to be one.  Every one of the thirteen shortens a
span that brings a limb toward her trunk, so pull-only she had no extensor
anywhere and could curl but never straighten, and standing is straightening.
MEASURED: curled to a ball and then ordered flat, she now gets **87% of the way
back**, where before she could not come back at all.

WHAT IT COST AND WHAT IT DID NOT.  Three extensor designs were built and
measured first, all of them pull-only and routed over a standoff like a kneecap,
and none of them worked --- see `measured/joints.md`.  This does, at the price of
two constants that have to be threaded (`MUS_LIMP`, `MUS_GIVE`) and of her
resting posture changing.  What it did NOT cost is `force`: a hand holding her
still still reads, and her own body still reads nothing.  That was the thing
worth protecting, and it is checked by a test rather than hoped for.

TWO THINGS ARE NEW, and both were forced by trying to make her turn her head.

**SHE HAS A FACE JOINT.**  The old body derived her forward as
`cross(right, up)` from the shoulder line and the spine.  Pulling a point-mass
head sideways tilts `up` WITHIN the right-up plane, so the cross product comes
out unchanged --- measured, not guessed: her forward is bit-identical before and
after.  A head made of one point has no direction of its own, and no muscle
anywhere in the old body could yaw her.  So there is a point in front of her
head now, and her gaze is the vector between them.

**HER NECK IS THREE MUSCLES, NOT TWO.**  A muscle can only pull, so one 0..1
line cannot mean "left and right" --- a signed rotation needs an antagonist pair.
`neck_left` and `neck_right` pull her face toward one shoulder or the other, and
`neck_down` tucks her chin.  Looking up is what happens when she stops pulling,
which is also true of a real newborn.

WHAT HER MUSCLES ANSWER WITH.  She orders 0.80; the muscle reaches whatever it
reaches; the gap between the two is reported and IS THE WHOLE POINT.  Hitting a
joint stop, the floor, and a thing in the way are one readable signature: the
order stays high, the length stops changing, and the force rises.  That is the
only way she can find out a limit exists.
"""

from __future__ import annotations

import numpy as np

#: Declared in the contract, not here --- the body implements the names, it does
#: not get to invent them behind the brain's back.
#: ONE CLOCK, and it is hers.  This file used to DEFINE `TICK_SECONDS` from a
#: substep count --- a second answer to how fast she runs, and a per-tick
#: constant besides.  It reads her clock now and works out its own substeps
#: from it, so changing her clock changes her physics and nothing else.
from .hearing import TICK_SECONDS
from .joints import AXES
from .light import EYE_SLIDES

# --- her joints ---------------------------------------------------------------

#: WHICH SIDE IS WHICH.  She faces +z with +y up, and the unambiguous rule is
#: `right = cross(forward, up)` --- so her right is -x and the joint at -x is
#: her RIGHT one.  The pose this is ported from named it `shL`, which made every
#: left and right in her body a mirror of the truth: her left ear was her right
#: ear, `neck_left` turned her right, `hand_left` was the other hand.  It was
#: invisible because `frame()` had the same flip and the two cancelled.
#: `toR`/`toL` are the balls of her feet, and they exist so that an ankle has
#: something to act ACROSS.  A foot that ends at the ankle cannot push off:
#: there is no segment beyond the joint for a muscle to rotate, so the last
#: thing between her and the floor was a point.
JOINTS = ("head", "face", "neck", "chest", "pelvis", "shR", "shL", "elR", "elL",
          "haR", "haL", "hiR", "hiL", "knR", "knL", "foR", "foL", "toR", "toL")
JIDX = {name: i for i, name in enumerate(JOINTS)}

#: HOW FAR HER EYES TURN IN HER HEAD before they give up and snap forward.
#: 35 degrees --- a person's eyes reach about 45 mechanically and live nearer 30
#: in ordinary looking, and past that you turn your head instead.  It is the
#: travel of a real eye, not a number chosen to make a measurement come out.
EYE_REACH = np.radians(35.0)

#: Her proportions, written the way a body is easiest to write: standing up,
#: facing +z.  She is laid down below, and never actually exists like this.
_UPRIGHT = np.array([
    (0.000, 0.615, 0.000),   # head
    (0.000, 0.615, 0.110),   # face  --- in FRONT of her head, so her gaze is a
                             #           real vector and not a derived guess.
                             #   HOW FAR in front is the moment arm her neck
                             #   pulls on, and it is the whole difference
                             #   between a head that turns and one that does
                             #   not: MEASURED, at 0.075 her two neck muscles
                             #   moved her gaze 2-9 degrees and both the SAME
                             #   way; at 0.110 they separate it by 50.
    (0.000, 0.520, 0.000),   # neck
    (0.000, 0.470, 0.000),   # chest
    (0.000, 0.375, 0.000),   # pelvis
    # ...AND THE NAMES BELOW ARE THE ONES IN `JOINTS`, WHICH IS NOT WHAT THEY
    # SAID.  The rule above is `right = cross(forward, up)`, so the joint at -x
    # is her RIGHT one; every side comment in this table named the other one,
    # left over from the pose this was ported from.  The geometry was already
    # correct and only the labels lied, which is the worst way for this
    # particular mistake to sit: it is invisible until someone trusts a comment.
    # Checked row by row against `JOINTS` and the cross product, 2026-08-17.
    (-0.058, 0.492, 0.000),  # shR
    (0.058, 0.492, 0.000),   # shL
    (-0.098, 0.418, 0.010),  # elR
    (0.098, 0.418, 0.010),   # elL
    (-0.120, 0.345, 0.030),  # haR
    (0.120, 0.345, 0.030),   # haL
    (-0.040, 0.358, 0.000),  # hiR
    (0.040, 0.358, 0.000),   # hiL
    # HER KNEES REST BENT, WHICH IS THE WHOLE REASON SHE HAS KNEES.
    #
    # They used to rest almost straight --- hip to foot 211.1 mm against the
    # 211.6 mm of thigh and shin between them, 99.8% extended --- and that one
    # number is why she could not stand.  Her knee muscle spans hip to foot, so
    # its entire working range was the 0.5 mm left over.  Measured on the old
    # pose (`py -m measure.reach`): ordering it 0.00 to 1.00 moved the joint
    # 10.8 mm and 25 degrees, where a human knee has about 140 --- AND THE ANGLE
    # WENT BACK DOWN past half: 177 degrees at 0.50, 165 at 1.00, because the
    # muscle was jamming the leg against its own bones and the solver took it
    # sideways.  There was no map from what she asked to what happened, so there
    # was none for her to learn, on the one joint that has to carry her.
    #
    # A newborn lies with her hips and knees drawn up.  At about 120 degrees her
    # hip sits 186 mm from her foot, which leaves 25 mm of real extension to
    # find.  The knee goes FORWARD of the hip-to-foot line, which is the way a
    # knee bends and the way `_HINGES` already says it must.
    (-0.049, 0.275, 0.064),  # knR
    (0.049, 0.275, 0.064),   # knL
    (-0.060, 0.175, 0.030),  # foR
    (0.060, 0.175, 0.030),   # foL
    # the balls of her feet: below the ankle and IN FRONT of it, which is what
    # makes an ankle a lever rather than a hinge with nothing on the far side
    (-0.060, 0.145, 0.075),  # toR
    (0.060, 0.145, 0.075),   # toL
], dtype=np.float32)

#: AND LAID ON HER BACK ALONG THE MATTRESS, which is where a newborn actually
#: goes: `(x, y, z) -> (y, z, -x)`, so her head points down the bed's LONG axis
#: and her face points at the ceiling.
#:
#: Two things were found here and both were the pose, not the eye.  She was
#: built standing and left to flop, and she landed FACE DOWN --- measured, her
#: forward settled at (0.43, -0.88, 0.22), her nose in the mattress nine
#: centimetres away.  Everything she could see was bedding, and a window thirty
#: centimetres in front of her face was BEHIND the mattress along every ray:
#: her retina came back bit-identical at 1.2 m, 0.6 m and 0.3 m because she was
#: not looking at any of them.  Then, turned face up but laid ACROSS the bed,
#: she was 0.465 m long on a 0.56 m axis and slid off the end by tick 2.
#:
#: A baby who cannot see the room is not a baby with a bad eye.  She was lying
#: wrong, twice.
#:
#: AND IT MUST BE A ROTATION, NOT A REFLECTION.  `(y, z, -x)` looks like the
#: same quarter turn and has determinant -1: it mirrors her left and right, and
#: it flips `spine()`, so her elbows and knees bend BACKWARDS and their stops
#: torque her over on every substep.  `(y, z, x)` is the real rotation.  A sign
#: is not a detail when a cross product depends on it.
_REST = _UPRIGHT[:, [1, 2, 0]].astype(np.float32)

#: HOW THICK SHE IS AT EACH JOINT --- a RADIUS: `_settle` puts her down by
#: `y - _RADIUS` and the floor adds it back, so this is point to skin.
#:
#: HER HEAD WAS AN ADULT'S, AND IT WAS THE ONLY ONE THAT WAS WRONG.  Checked
#: against real newborn anthropometry, one radius at a time:
#:
#:     chest  0.052 -> 10.4 cm across   a newborn thorax is ~10 cm       ok
#:     pelvis 0.050 -> 10.0 cm          bi-iliac breadth 9-10 cm         ok
#:     neck   0.030 ->  6.0 cm          5-6 cm                           ok
#:     knee   0.026 ->  5.2 cm          ~5 cm                            ok
#:     hand   0.024 ->  4.8 cm          ~4.5 cm                          ok
#:     head   0.088 -> 17.6 cm          11.0 cm                          NO
#:
#: A 17.6 cm sphere is 55 cm round, which is a grown man's hat.  A newborn's
#: head measures 34-35 cm round and 11.0-11.5 cm front to back.  **0.055 gives
#: 11.0 cm across and 34.6 cm round** --- the newborn figure itself, not a
#: number chosen to make anything come out.
#:
#: IT IS NOT COSMETIC, AND THE MEASUREMENT THAT SHOWS IT IS HER GAZE.  She lies
#: on her back with the back of her skull on the mattress, so this sphere is the
#: lever her whole head tips over: at 0.088 she settles looking **15.6 degrees
#: off the ceiling**, at 0.070 8.8, at 0.055 **2.8**.  A supine newborn looks
#: straight up.  Her oversized head was aiming her eye at the wall.
#:
#: She still comes to rest (head buzz 0.008 mm against 0.007), slides slightly
#: LESS across the mattress and lies slightly higher on it.  Her head is now
#: 0.110 across against shoulders 0.116 apart --- a shade narrower than her
#: shoulders, which is what a newborn is and what `CLAUDE.md` already says.
_RADIUS = np.array([0.055, 0.020, 0.030, 0.052, 0.050, 0.026, 0.026, 0.022,
                    0.022, 0.024, 0.024, 0.028, 0.028, 0.026, 0.026, 0.028,
                    0.028, 0.022, 0.022], dtype=np.float32)

#: Bones are distance constraints, plus braces that keep her a body and not a
#: noodle.  `head-face` is rigid: her face does not float away from her skull.
_BONES = [
    ("head", "face", 1.0), ("face", "neck", 0.5),
    ("head", "neck", 1.0), ("neck", "chest", 1.0), ("chest", "pelvis", 1.0),
    ("chest", "shL", 1.0), ("chest", "shR", 1.0),
    ("shL", "elL", 1.0), ("elL", "haL", 1.0),
    ("shR", "elR", 1.0), ("elR", "haR", 1.0),
    ("pelvis", "hiL", 1.0), ("pelvis", "hiR", 1.0),
    ("hiL", "knL", 1.0), ("knL", "foL", 1.0), ("foL", "toL", 1.0),
    ("hiR", "knR", 1.0), ("knR", "foR", 1.0), ("foR", "toR", 1.0),
    ("shL", "shR", 1.0), ("hiL", "hiR", 1.0),
    ("shL", "pelvis", 0.6), ("shR", "pelvis", 0.6),
    ("head", "chest", 0.5),
    ("hiL", "knR", 0.15), ("hiR", "knL", 0.15),
]

#: A knee bends one way and an elbow the other, because in a real leg two bones
#: meet.  sign +1: the middle joint stays IN FRONT of the outer-to-outer line.
_HINGES = [
    ("hiL", "knL", "foL", 1), ("hiR", "knR", "foR", 1),
    ("shL", "elL", "haL", -1), ("shR", "elR", "haR", -1),
    # An ankle bends the way a knee does not: her toes stay in FRONT of the
    # shin-to-toe line, so she can point her foot down and cannot fold it up
    # into her own leg.
    ("knL", "foL", "toL", -1), ("knR", "foR", "toR", -1),
]
HINGE_K = 0.5
#: floor grip: how many metres of sideways hold one metre of normal
#: correction buys.  Skin on a mattress is high-friction; and `pushed`
#: carries only the LAST relax pass's normal push, understating the
#: substep's true load, so the coefficient stands in for both.  Measured
#: MEASURED across the triad (glide after a push / free-floor travel
#: under drive / the walker's tuck bound): 3.0 STAPLED her (drive
#: bought 0.004 m --- nothing could ever move); 1.0 let 0.041 m of
#: glide back; 2.0 gives the best of both --- free pushes buy 0.075 m
#: (the most of any setting, all of it earned through grip), glide
#: dies at 0.015 m, the stand holds to a millimetre and the tuck
#: cannot climb.  In-walker travel now comes only from actual stepping,
#: which is what a walker is for.
MU_FLOOR = 2.0
HINGE_SLACK = 0.004

#: HOW FAR HER GAZE MAY GET FROM HER TRUNK'S OWN FORWARD, as a cosine.  You
#: cannot turn your head to look out of your own back: a neck runs out of
#: travel, and a newborn's runs out early.
#:
#: It is not a nicety.  Her face is a point on the end of an 11 cm lever and
#: nothing else resists gravity twisting it: laid carefully on her back she was
#: looking straight up on tick 1 and face down in the mattress by tick 3, every
#: time, and then everything she could see for the rest of her life was bedding.
#: With this she can look up, or to either side, and not through the bed.
NECK_CONE = 0.05
NECK_K = 0.6
#: how far below full extension her head may hang along her own trunk ---
#: at the limit the head's centre still sits this fraction of the neck's
#: rest length ABOVE the chest, which is chin-on-chest.  One-sided, like
#: the cone: muscles move her freely above it.
HEAD_MIN = 0.30

# --- the only things she can actually command ---------------------------------

#: The spans are gone.  What she commands is an AXIS of a JOINT --- the names
#: in `contract.manifest.AXES`, the geometry in `body/joints.py`, the gate
#: that earned the conversion in `measure/shoulder3.py`.  Everything her spans
#: could not do (raise an arm toward her own head, press with a hand, mean
#: the same thing twice) is why.

#: How firmly a drive corrects per solver pass, at effort 1.0.  Proven on the
#: one-shoulder rig before twelve joints trusted it.
#: RECALIBRATED 2026-08-28 against the world that pushes back.  0.30 was
#: measured in the frictionless, self-crossing body --- where 0.45 "threw
#: her toe across the cot" because NOTHING resisted anything.  With real
#: grip, flesh and weight, 0.30 left her helpless: a knee at FULL command
#: moved her foot ONE MILLIMETRE in two seconds ("her moscles now is to
#: week for new body" --- his diagnosis, measured true).  At 1.0 her arm
#: raises 0.15 m and her calm survives; the old warning belongs to the
#: old world.
#:
#: 1.00 -> 0.50, HIS CHOICE 2026-09-03 ON THESE NUMBERS.  At her 1/90 tick
#: a full order at 1.0 made the solver oscillate: joint readings swung
#: +-30% of range tick to tick while the smoothed order crept 4% a tick,
#: peaks of 16 m/s.  His own cases, two-second holds, offline:
#:     gain   arm raise (hand rose)   knee (foot moved)   both legs   peak speed
#:     1.00   0.028 m (flails aside)  0.085 m             0.255 m     10-17 m/s
#:     0.50   0.172 m                 0.058 m             0.081 m     0.8-5.6 m/s
#:     0.35   0.140 m                 0.038 m             0.223 m     jitter 2.3
#: The arm raise he measured at 0.15 m lives at 0.5 now, not at 1.0.  The
#: legs reach less; the belt stepping is measured live before this ships.
AXIS_GAIN = 0.50

#: What the drive's own convergence leaves unfinished in EMPTY AIR, as a
#: fraction of an axis's range --- measured 0.10 at full effort from rest and
#: up to 0.15 from a contorted pose, where gravity and her own bones lean
#: harder on the drive.  `force` must not report any of it: force is the
#: world refusing her, and being an actuator with finite gain is not the
#: world.  The same role `MUS_GIVE` played for the spans; a held limb still
#: reads ~0.7 against this, so the reading she cannot do without survives.
AXIS_SLACK = 0.15

#: HOW FAR OFF ITS OWN BONE THE FAR POINT MUST BE before a twist or a roll is a
#: reading at all, as a sine --- 0.087 is 5 degrees.
#:
#: An azimuth is the direction of the part of the far point that lies OFF the
#: bone it turns about.  `joints.py` says this in words already: *"when the
#: distal joint is perfectly straight the twist is momentarily unreadable,
#: exactly as it is mechanically meaningless."*  The body did not enforce it.
#: The test was `> 1e-6` --- one micrometre --- so a vanished lever still
#: produced an angle, and the angle was whatever the solver's own error pointed
#: at that substep.
#:
#: MEASURED, and it is not hypothetical.  Her ankle straightens in ONE tick of
#: her first life --- knee, ankle and toe come to lie in a straight line, 178.67
#: degrees, and stay there for ever --- so the toe's lever collapses from 5.228
#: cm at birth to **0.124 mm**.  Her two roll axes were then read on that:
#: `ankle_left.roll` settled at **+3.6585** and `ankle_right.roll` at
#: **-2.6584**, 3.7 and 2.7 whole ranges outside 0..1 and the largest range
#: violation in her body.  What her brain was told, every tick of every life:
#: `spindle` pinned at 1.000 and 0.000, and `force` --- the line that means THE
#: WORLD IS REFUSING ME --- pinned at 1.000 on both ankles.
#:
#: 5 degrees, because below it the perpendicular part is under 9% of the segment
#: and its direction is the solver's residual rather than her foot.  Her other
#: azimuths are nowhere near it settled: shoulder twist reads on 31% of the
#: forearm, hip twist on 73% of the shin.  An unreadable azimuth reports the
#: middle of its range, which is what an azimuth whose reference cannot be built
#: has always reported --- one answer, not a second one.
AZIMUTH_SEEN = 0.087

#: HER RESTING TONE --- what a newborn's flexors do when she orders nothing.
#:
#: A drive with no effort corrects nothing, which is honest --- and a body
#: with NO passive tone splays flat and stays there, which is not: a real
#: newborn rests CURLED, held gently toward the fetal pose by flexor tone.
#: So at zero effort every axis is drawn softly toward its birth angle, and
#: her order takes over exactly as fast as she insists on it:

#:     target = order*w + rest*(1-w),  w = effort / (effort + TONE)
#:     gain   = AXIS_GAIN * (effort + TONE*(1-effort))
#:
#: Nothing here decides anything: it is the same tone asleep and awake, and
#: what she does about it is hers.
#:
#: 0.10 -> 0.02, MEASURED 2026-09-03.  At 0.10 her body WRITHED AT REST:
#: body alone, nothing ordered, no sound, no startle, her head moved 3 cm
#: every 11 ms tick (2.75 m/s median, 18 m/s peak) and her head height
#: wandered 0.34-0.67 m in ten seconds --- at 1/90, 1/45, 1/75 and 1/120;
#: still only at 1/60, the old substep rate.  Reproduced offline with
#: `Ragdoll(Room())` and zero orders; switching terms off one at a time:
#: the tone pull was the energy source (AXIS_GAIN*TONE per pass, eight
#: passes, stretching bones faster than 40 bone passes restore them, the
#: residual carried into velocity, growth x1.25 a step from step 4).
#: Max joint move per step at rest, steps 200-600, DT 1/30 1/45 1/60 1/90
#: 1/120: tone 0.10: 4.8 / 9.4 / 0.07 / 10.4 / 10.5 cm; 0.05: 0.9 / 0 /
#: 0.2 / 0 / 0 cm; 0.025: 0 / 0 / 0 / 0.16 / 0.04 cm; 0.02: 0 / 0 / 0 /
#: 0.02 / 0.16 cm.  Her curl still forms, better: the axes' gap to the
#: rest pose after 400 steps 0.32 of range (thrashing) -> 0.14.  Under
#: an order nothing changes: gain at effort 1.0 is AXIS_GAIN as before.
TONE = 0.02
#: ...and her LIGAMENTS: twist and roll have no bony stop the way a hinge
#: does, and with only muscle tone her feet were measured flopping 157
#: degrees of roll while she settled --- anatomically impossible, because a
#: real ankle's passive structures hold it near neutral whatever the leg
#: does.  So azimuth axes carry ligament-strength passive tone toward the
#: birth pose.  Still not a controller: the same pull asleep and awake, and
#: her ordered effort takes over above it exactly as for every other axis.
LIGAMENT = 0.45

#: HER SPINAL REFLEXES --- three more things a newborn's flesh does between
#: decisions, built exactly the way TONE already is: state-dependent passive
#: tone and rest-pose, never a controller.  Her commanded effort overrides
#: every one of them through the same `w = eff / (eff + tone)` blend that
#: already lets an order take over from the fetal curl; a reflex only fills
#: the silence where she is not insisting.  All three are real newborn
#: reflexes: positive support (a loaded leg stiffens toward extension ---
#: why a held newborn "stands"), labyrinthine righting (head and trunk pull
#: toward vertical, and ONLY when she is already held more upright than
#: flat, so her supine day is untouched), and velocity-sensitive tone (flesh
#: resists fast lengthening more than slow --- the catch before a buckle
#: becomes a fall).  Gains are newborn-small and gated in
#: `measure/reflexes.py` before they were allowed near her.
REFLEXES = True
#: how much tone a fully-loaded foot adds to that leg's anti-gravity axes...
SUPPORT_TONE = 0.35
#: ...and how far it shifts their rest toward extension, at full load
SUPPORT_SHIFT = 0.5
#: a foot must genuinely bear weight to fire --- her supine rest grazes the
#: mattress, and a reflex that fired all day lying down would rewrite her
#: settled world
SUPPORT_FLOOR = 0.15
#: the shove (metres of solver correction) that counts as a fully loaded
#: foot --- skin.FIRM's own number, kept equal so skin and spine agree on
#: what "bearing weight" means
PRESS_FULL = 0.01
#: righting: tone and rest-shift toward straight at full engagement ---
#: NEWBORN-WEAK on purpose: at 0.20/0.6 her head stayed steadily up when
#: held by one hand, which is a three-month-old's neck.  A newborn gets
#: moments of righting and then droops; these gains give exactly that.
RIGHT_TONE = 0.04
RIGHT_SHIFT = 0.12
#: ...and the NECK'S OWN SHARE, stronger --- his order, 2026-08-30: her
#: head hung wherever gravity left it ("her head every time in wrong
#: position"), because 0.04 is a droop by design.  The neck rights harder
#: than the trunk; her commanded effort still overrides through the same
#: w = eff/(eff+tone) blend, so orienting at a sound always wins.
NECK_RIGHT_TONE = 0.30
NECK_RIGHT_SHIFT = 0.70
#: ...engaged only past this trunk uprightness (0 flat, 1 vertical), scaled
#: to full by vertical, so lying down it does not exist
RIGHT_FROM = 0.5
#: WHAT A FULL `more` IS WORTH.  The response of standing support to its
#: own gain is a RIDGE, measured (light hold, share 0.25, limp): mod 0.0
#: bears 0.169, 0.5 bears 0.284, 1.0 bears 0.457, 1.3 bears 0.445, 1.6
#: bears 0.203 --- the innate gain already sits at the ridge and past
#: ~1.4x stiff legs bounce and support collapses (hypertonia is real).  So
#: a full `more` asks for 1.4x, the edge of the plateau, and the whole
#: band means something; at 1.0 the top half of it was dead range, the
#: dead-top-quarter disease in new clothes.
GAIN_UP = 0.4

#: velocity tone: how much resisting a fast-moving knee adds, at full speed...
#:
#: 0.30 -> 0.0, MEASURED 2026-09-03.  As built this is not a resistance but
#: a LOOP: a fast joint raises its tone, the tone pulls it toward rest
#: with gain, the pull makes it fast.  Her whole tick offline, no mind:
#: two seconds of random full orders, then six seconds of nothing ---
#: head move per tick in each of those seconds, at 1/90:
#:   VEL_TONE 0.30: 3.3 3.3 2.4 3.2 3.6 3.0 cm  --- she writhes for ever
#:   VEL_TONE 0.0 : 1.8 0.1 0.0 0.0 0.0 0.0 cm  --- she comes to rest
#: (80 bone passes, DAMP 0.90, FLESH_DAMP 1.0 changed nothing; halving
#: AXIS_GAIN also rested her but made her helpless again, his 2026-08-28
#: complaint.)  Every life she lived since 2026-09-02 moved within its
#: first second and writhed from then on --- 2.75 m/s median head speed,
#: 18 m/s peaks, in her cot with nothing ordered.  The catch-before-a-
#: buckle this was for belongs to standing, which she has not reached;
#: when she does, a resistance must oppose velocity, not pull on position.
VEL_TONE = 0.0
#: ...and the knee speed (m/s) that counts as full
VEL_FULL = 0.35

G = -9.8                    # gravity, m/s^2
#: ONE CLOCK, EVERYWHERE.  The owner, 2026-08-25: *"why we make two steps to
#: make a lot of alignment problem in future?  everywhere has to be one
#: clock"*.
#:
#: This was 1/60 with 90 substeps in a 1.5 s tick --- her physics running at
#: one rate inside her thinking at another.  Every alignment question that
#: comes from that (which substep a glimpse was taken at, when a spindle was
#: read, which half of a tick a contact happened in) is a question that does
#: not exist if there is one clock.
#:
#: SO HER PHYSICS STEP IS HER TICK.  One step, one tick, one time.
DT = TICK_SECONDS           # one physics step, and it is her tick
DAMP = 0.985                # air and tissue drag, per substep
#: HOW MUCH OF WHAT HER CONSTRAINTS JUST DID IS DISSIPATED INSTEAD OF BECOMING
#: SPEED.  Flesh is a spring AND a damper; hers was only a spring.
#:
#: This solver is Verlet, so velocity IS `pos - prev`.  A constraint moves `pos`
#: and leaves `prev` where it was, which means **every correction is handed back
#: as momentum on the next substep**.  Her drives run inside all eight passes,
#: ninety substeps a tick, so a muscle that overshoots is paid for its overshoot
#: in kinetic energy 720 times a tick.  That is a positional servo with no
#: velocity term, and it is unstable above a gain her flesh was sitting just
#: under --- by 12%.
#:
#: MEASURED (`measured/herbody.md`).  Limp in the cot, nothing changed but
#: `TONE`, peak-to-peak travel of her HEAD over the last 30 of 120 ticks:
#:
#:     TONE   gain      before        after
#:     0.10   0.030      0.007 mm     0.000 mm     <- hers
#:     0.12   0.036     14.808 mm     0.000 mm
#:     0.13   0.039     61.144 mm     0.000 mm
#:     0.25   0.075    782.828 mm     0.114 mm
#:     0.30   0.090          --       0.002 mm
#:     0.35   0.105          --      44.109 mm     <- the new cliff
#:     0.45   0.135          --     720.261 mm
#:
#: and `AXIS_GAIN` 0.45, which threw her toe 1,205.599 mm across the cot, now
#: moves her head 0.001 mm.
#:
#: **The cliff goes from an effective gain of ~0.034 to ~0.095.**  Her drive gain
#: is `AXIS_GAIN * (eff + TONE*(1-eff))` = `0.27*eff + 0.03`, so the effort she
#: can spend before her own body oscillates goes from **eff 0.015 to eff 0.24**
#: --- from one and a half percent to a quarter.  She had lived her whole life
#: past the old one, because `choose` spends on every axis every tick.
#:
#: AND SHE MOVES BETTER, NOT WORSE.  This is not stability bought with capability:
#: an elbow ordered to 0.8 from rest reached 0.681 and now reaches 0.745, and
#: thirty ticks of full thrashing scooted her 19.34 cm and now scoot her 43.72.
#: Damping the overshoot leaves more of her effort going where she aimed it.
#:
#: 0.75, swept: 0.0 is what shipped, 0.5 still let TONE 0.25 swing her head
#: 169.888 mm, 0.9 was worse than 0.75 (78.983 mm) and 1.0 takes ALL of it ---
#: at which point nothing her constraints do becomes speed, her limp slide falls
#: to 4.10 cm and a hard order reads `force` 0.264 with nothing in her way.  Her
#: momentum is hers and scooting is made of exactly this; 0.75 keeps it.
#:
#: It is not a controller and not a switch: the same dissipation asleep and
#: awake, on every joint, applied once a substep outside the passes, deciding
#: nothing.
FLESH_DAMP = 0.75
#: HOW HARD HER SPINE'S STOP PUSHES BACK, per pass.  The neck's own `NECK_K`
#: shape: a fraction of the overshoot, applied every pass, so eight passes
#: converge on the limit rather than snapping to it --- a stop that teleports
#: is a stop she can feel as a blow.
#:
#: MEASURED, NOT CHOSEN.  A spindle clipped at an end is a constant; only the
#: middle carries anything.  How often her `spine.bend` reads BETWEEN its ends,
#: over 700 ticks:
#:
#:     no stop, as it was      6%
#:     K = 0.25               12%
#:     K = 0.50               21%      <- this
#:     K = 1.00               18%      too stiff: it overshoots the limit
SPINE_K = 0.5
RELAX_PASSES = 8            # constraint-solver iterations per substep
BONE_PASSES = 40            # bones-only tightening after the passes --- see
                            # BONES ARE BONES at the end of the pass loop.
                            # 12 left the hip 34% off under a full push; 40
                            # costs ~1 ms and holds the skeleton (measured
                            # below 10% worst, 2026-08-29)
#: Flesh has inertia; it does not snap to a new length.  Skipping this is not a
#: simplification --- an instantly applied large target-length change is
#: numerically unstable under this solver, the same way asking a real muscle to
#: teleport rather than contract would tear something.
ACT_RATE = 11.0
#: ...AND A MUSCLE HAS A TOP SPEED.  The smoothing above moves the target
#: by a fraction of what is left, so the first tick of a full swing moved
#: it 12% of an axis's whole range at 1/90 --- about 1,300 degrees a second
#: on a 110-degree joint, and her head under random full orders ran at
#: 2.4-3.3 m/s median with 16-19 m/s peaks (measured 2026-09-03, body
#: alone).  A real infant's fastest joint, the kicking knee, peaks near
#: 300-700 degrees a second (Thelen's and Jensen's kicking studies), and
#: a newborn's hand near 1 m/s.  So the ordered angle may move at most
#: this many degrees a second, per axis, whatever the order: reaching
#: hard is still reaching at a baby's speed.  A cap on the ORDER'S travel,
#: not on her strength --- his 2026-08-28 recalibration of AXIS_GAIN stands.
MAX_DEG_PER_S = 600.0
#: 1.5 simulated seconds per tick.
#: ...SO THERE IS EXACTLY ONE.  Not a number: `DT` is her tick, so one step is
#: one tick by construction, and nothing can drift between them.
STEPS_PER_TICK = 1
#: How many of those 90 frames are kept so she can be watched moving rather than
#: teleporting.  The physics computes all 90 either way, so this only decides
#: how many are KEPT --- measured in `preview/app.py`'s docstring at 0.2 ms a
#: tick across the whole range (87.5 ms at 15 against 87.7 ms at 45).
#:
#: **45 IS 30 A SECOND, AND IT IS THE DEFAULT BECAUSE ONE VIEWER NEVER SET IT.**
#: `preview/app.py` turns this dial itself (`FILM = STEPS_PER_TICK // (60 //
#: fps)`), so `--fps 30` has always given 45 there --- but anything NOT launched
#: through preview (`body/serve.py`, the tests, every probe) ran on this default
#: at 10 a second.  Set once on 2026-08-22 and lost the same night when this
#: file was rolled back for an unrelated revert; the owner caught it: *"but we
#: alredy fix this all hour ago"*.
#:
#: Nothing she has reads it --- `/pose` is for drawing only and a test asserts
#: the brain never asks.  It is her appearance, not her physics.
FILM = 45
#: ...and her clock is imported, not defined here.  See above.
#: A real hand carries her at a bounded pace.  Measured in the previous project:
#: a sustained hold applied as plain constant force ran her pelvis up 59 metres
#: in 3 ticks with nothing to stop it.  Capping the SPEED, not the force, leaves
#: a brief nudge exactly as it was.
HANDLE_MAX_SPEED = 0.5

_GRAVITY = np.array([0.0, G, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)

#: ONE IMPLEMENTATION OF A CROSS PRODUCT AND A LENGTH --- they live in
#: `joints.py`, the module that owns her geometry, and every part of her body
#: uses those.  What they are and why they exist is written up there.
from .joints import _cross3, _crossn, _len, _lens          # noqa: E402

#: HER BONES IN C --- `body/native/knit.c: bones`, the same arithmetic as
#: `_solve_bones` + `_relax` below, forty passes in one call instead of
#: forty numpy round trips (his, 2026-09-03: *"do what possible to increase
#: her perfomanse"*; measured: the physics was 70% of a 6-8 ms tick and the
#: solver is interpreter overhead, not arithmetic).  A missing dll or symbol
#: falls back to the numpy passes, which stay the definition.
_BONES_C = None
try:                                                        # pragma: no cover
    import ctypes as _ct
    import os as _os
    _knit = _ct.CDLL(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "native", "knit.dll"))
    # THE C IS HANDED BARE ADDRESSES (performance, 2026-09-06, measured:
    # numpy's per-argument pointer check ran 184 times a physics step, 15%
    # of it).  Every array the C reads is made contiguous and typed ONCE ---
    # the birth constants beside their birth (`_c_bones` ...), the masks in
    # `_hold` when a hand moves --- and `pos` is checked at each call as it
    # always was.  Same C, same memory, same numbers.
    _P = _ct.c_void_p
    _knit.bones.restype = None
    _knit.bones.argtypes = [_P, _ct.c_int, _P, _P, _P, _P, _ct.c_int, _P, _P,
                            _ct.c_int, _P]
    _BONES_C = _knit.bones
    _knit.flesh.restype = None
    _knit.flesh.argtypes = [_P, _P, _P, _P, _ct.c_int, _P]
    _knit.hinges.restype = None
    _knit.hinges.argtypes = [_P, _P, _P, _P, _P, _ct.c_int, _P,
                             _ct.c_float, _ct.c_float, _P, _P, _P]
    _FLESH_C, _HINGES_C = _knit.flesh, _knit.hinges
except Exception:                                           # noqa: BLE001
    _BONES_C = None
    _FLESH_C = _HINGES_C = None


def _norm(v: np.ndarray) -> np.ndarray:
    n = _len(v)
    return (v / n).astype(np.float32) if n > 1e-6 else np.array([0.0, 0.0, 1.0],
                                                                np.float32)


def _spin_rows(v: np.ndarray, axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rodrigues, one rotation per row: `v` and `axis` are `(n, 3)`, `ang` is
    `(n, 1)` radians."""
    c, s = np.cos(ang), np.sin(ang)
    return (v * c + _crossn(axis, v) * s
            + axis * (axis * v).sum(axis=1, keepdims=True) * (1.0 - c))


def _rest_len(a: str, b: str) -> float:
    return float(np.linalg.norm(_REST[JIDX[b]] - _REST[JIDX[a]]))


class Ragdoll:
    """Her, in the room.  Laid down resting rather than built mid-air --- a body
    assembled in the air is crushed into itself by the floor constraint on the
    very first substep and starts life as a puddle."""

    def __init__(self, floor) -> None:
        #: what her body rests on and cannot pass through --- `room.py`'s
        #: surfaces.  The ragdoll asks it where the ground is; it does not know
        #: what a bed is.
        self.floor = floor
        n = len(JOINTS)
        self.pos = _REST.copy()
        self._settle()
        self.prev = self.pos.copy()
        self.radius = _RADIUS.copy()
        # HER FLESH --- no two of her may occupy one place.  His,
        # 2026-08-28: "all her parts cross each other like throught
        # nothing" --- measured under 400 random-driven ticks: her hand
        # sat 0.34 of the way INSIDE her hip, her elbow 0.57 into the
        # other hip, nothing anywhere kept sphere from sphere.  One rule
        # for every non-bonded pair: never closer than their radii ---
        # or than they were BORN, for neighbours the birth pose already
        # packs close (the same one formula covers both, no exclusions).
        bonded = {frozenset((JIDX[a2], JIDX[b2])) for a2, b2, _k in _BONES}
        fa, fb, fm = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((i, j)) in bonded:
                    continue
                rest = float(np.linalg.norm(_REST[i] - _REST[j]))
                fa.append(i)
                fb.append(j)
                fm.append(min(float(_RADIUS[i] + _RADIUS[j]), rest * 0.95))
        self._flesh_a = np.asarray(fa, np.intp)
        self._flesh_b = np.asarray(fb, np.intp)
        self._flesh_min = np.asarray(fm, np.float32)
        self.pinned = np.zeros(n, dtype=bool)

        self._bone_a = np.array([JIDX[a] for a, b, k in _BONES], dtype=np.intp)
        self._bone_b = np.array([JIDX[b] for a, b, k in _BONES], dtype=np.intp)
        self._bone_L = np.array([_rest_len(a, b) for a, b, k in _BONES], np.float32)
        self._bone_k = np.array([k for a, b, k in _BONES], np.float32)
        self._bone_plan = self._relax_plan(self._bone_a, self._bone_b)
        # ...and the same bones as C wants them (int32, contiguous), once
        self._bone_a32 = np.ascontiguousarray(self._bone_a, np.int32)
        self._bone_b32 = np.ascontiguousarray(self._bone_b, np.int32)
        self._bone_safe1 = np.ascontiguousarray(self._bone_plan[5][:, 0], np.float32)

        self._hinge_a = np.array([JIDX[a] for a, b, c, s in _HINGES], dtype=np.intp)
        self._hinge_b = np.array([JIDX[b] for a, b, c, s in _HINGES], dtype=np.intp)
        self._hinge_c = np.array([JIDX[c] for a, b, c, s in _HINGES], dtype=np.intp)
        self._hinge_s = np.array([s for a, b, c, s in _HINGES], np.float32)
        # ...and the flesh pairs and hinges as C wants them (int32, contiguous)
        self._flesh_a32 = np.ascontiguousarray(self._flesh_a, np.int32)
        self._flesh_b32 = np.ascontiguousarray(self._flesh_b, np.int32)
        self._flesh_min32 = np.ascontiguousarray(self._flesh_min, np.float32)
        self._hinge_a32 = np.ascontiguousarray(self._hinge_a, np.int32)
        self._hinge_b32 = np.ascontiguousarray(self._hinge_b, np.int32)
        self._hinge_c32 = np.ascontiguousarray(self._hinge_c, np.int32)
        self._hinge_s32 = np.ascontiguousarray(self._hinge_s, np.float32)

        # HER JOINTS AS AXES.  The tables her solver drives, resolved once
        # from `body/joints.py` --- names from the contract, geometry there.
        from . import joints as J3
        assert J3.AXES == AXES, "the body and the contract disagree about her axes"
        self._frames = J3.Frames(JIDX)
        self._J3 = J3
        at = {name: i for i, name in enumerate(AXES)}
        #: (frame, left, anchor, child, grand, child_len, iLift, iSwing, iTwist)
        self._balls = []
        #: (outer_a, mid, outer_b, upper, lower, iBend)
        self._hinges3 = []
        #: (hip, knee, ankle, toe, iRoll) --- an ankle's second axis
        self._rolls = []
        for joint, frame, child, grand, axes in J3.JOINTS3:
            names = [joint + "." + a for a, _lo, _hi in axes]
            if joint in ("spine", "neck"):
                continue                      # bespoke, below in _solve_axes
            if joint.startswith(("shoulder", "hip")):
                anchor = {"shoulder_left": "shL", "shoulder_right": "shR",
                          "hip_left": "hiL", "hip_right": "hiR"}[joint]
                self._balls.append((
                    frame, joint.endswith("left"), JIDX[anchor], JIDX[child],
                    JIDX[grand], _rest_len(anchor, child),
                    at[names[0]], at[names[1]], at[names[2]]))
            elif joint.startswith(("elbow", "knee")):
                top = {"elbow_left": "shL", "elbow_right": "shR",
                       "knee_left": "hiL", "knee_right": "hiR"}[joint]
                mid = {"elbow_left": "elL", "elbow_right": "elR",
                       "knee_left": "knL", "knee_right": "knR"}[joint]
                self._hinges3.append((
                    JIDX[top], JIDX[mid], JIDX[child],
                    _rest_len(top, mid), _rest_len(mid, child), at[names[0]]))
            elif joint.startswith("ankle"):
                side = "L" if joint.endswith("left") else "R"
                self._hinges3.append((
                    JIDX["kn" + side], JIDX["fo" + side], JIDX["to" + side],
                    _rest_len("kn" + side, "fo" + side),
                    _rest_len("fo" + side, "to" + side), at[names[0]]))
                self._rolls.append((
                    JIDX["hi" + side], JIDX["kn" + side], JIDX["fo" + side],
                    JIDX["to" + side], at[names[1]]))
        self._ax_spine = (at["spine.bend"], at["spine.side"], at["spine.twist"])
        self._ax_neck = (at["neck.bend"], at["neck.side"], at["neck.twist"])
        self._spine_len = _rest_len("pelvis", "chest")
        self._neck_len = _rest_len("neck", "head")

        self._ax_act = np.zeros(len(AXES), np.float32)
        #: ...and how hard she is insisting.  Its own line, because the order
        #: is an ANGLE and this is a FORCE: reaching gently and reaching hard
        #: are different acts with the same destination.
        self._ax_eff = np.zeros(len(AXES), np.float32)
        #: each axis's range, once, so degrees and levels convert in one line
        self._ax_lo = np.array([J3.RANGE[n][0] for n in AXES], np.float32)
        self._ax_span = np.array(
            [J3.RANGE[n][1] - J3.RANGE[n][0] for n in AXES], np.float32)
        # THE SAME TABLES, AS ARRAYS --- the scalar solver measured 651 ms a
        # tick, python overhead on ~500 tiny ops x 720 pass-calls.  Everything
        # below is index arrays for a batched solver that does the identical
        # arithmetic in ~30 vector ops.
        J = JIDX
        dir_rows = [
            # (tip, origin, react, length, frame, kind, left, iA, iB)
            (J["chest"], J["pelvis"], J["pelvis"], self._spine_len, 0, 0, False,
             self._ax_spine[0], self._ax_spine[1]),
            (J["head"], J["neck"], J["chest"], self._neck_len, 1, 0, False,
             self._ax_neck[0], self._ax_neck[1]),
        ]
        az_rows = [
            # (tip, axis_a, axis_b, back_a, back_b, ref_kind, r1, r2, iAx)
            (J["face"], J["neck"], J["head"], J["head"], J["neck"],
             1, 0, 0, self._ax_neck[2]),
        ]
        for frame_name, left, anchor, child, grand, clen, iL, iS2, iT2 in self._balls:
            dir_rows.append((child, anchor, anchor, clen,
                             1 if frame_name == "chest" else 0, 1, left, iL, iS2))
            az_rows.append((grand, anchor, child, child, anchor,
                            1 if frame_name == "chest" else 0, 0, 0, iT2))
        for hip, kn, fo, to, iR in self._rolls:
            az_rows.append((to, kn, fo, fo, kn, 2, hip, kn, iR))

        as_i = lambda k: np.array([r[k] for r in dir_rows], np.intp)
        self._dir_tip, self._dir_origin, self._dir_react = as_i(0), as_i(1), as_i(2)
        self._dir_len = np.array([r[3] for r in dir_rows], np.float32)[:, None]
        self._dir_frame = np.array([r[4] for r in dir_rows], np.intp)
        self._dir_ball = np.array([r[5] == 1 for r in dir_rows], bool)
        self._dir_lat = np.array([-1.0 if r[6] else 1.0 for r in dir_rows],
                                 np.float32)[:, None]
        self._dir_iA = np.array([r[7] for r in dir_rows], np.intp)
        self._dir_iB = np.array([r[8] for r in dir_rows], np.intp)

        as_j = lambda k: np.array([r[k] for r in az_rows], np.intp)
        self._az_tip, self._az_a, self._az_b = as_j(0), as_j(1), as_j(2)
        self._az_backa, self._az_backb = as_j(3), as_j(4)
        self._az_refk = np.array([r[5] for r in az_rows], np.intp)
        self._az_r1, self._az_r2 = as_j(6), as_j(7)
        self._az_i = np.array([r[8] for r in az_rows], np.intp)

        self._h3_a = np.array([h[0] for h in self._hinges3], np.intp)
        self._h3_b = np.array([h[2] for h in self._hinges3], np.intp)
        self._h3_u = np.array([h[3] for h in self._hinges3], np.float32)
        self._h3_l = np.array([h[4] for h in self._hinges3], np.float32)
        self._h3_i = np.array([h[5] for h in self._hinges3], np.intp)
        self._h3_uu = self._h3_u ** 2 + self._h3_l ** 2
        self._h3_ul = 2.0 * self._h3_u * self._h3_l

        # SWING-TWIST TRANSPORT.  A twist reference fixed in the parent
        # frame reads PHANTOM twist under any lift or swing --- measured:
        # settling flopped her arms and the "twist" read 2.0 ranges away with
        # no bone rotated.  So each azimuth row records, at birth, its bone
        # direction and its reference IN PARENT-FRAME COORDS; every pass the
        # reference is re-expressed through the current parent frame and then
        # carried by the minimal rotation that takes the birth bone direction
        # to the current one.  Twist is what remains --- genuine rotation
        # about the bone, zero under any pure swing, which is the whole
        # meaning of the word.
        frames0 = (self._frames.pelvis(self.pos), self._frames.chest(self.pos))
        n_az = len(self._az_tip)
        self._az_a0 = np.zeros((n_az, 3), np.float32)
        self._az_r0 = np.zeros((n_az, 3), np.float32)
        for r in range(n_az):
            kind = int(self._az_refk[r])
            if kind == 2:
                continue                      # rolls keep the leg-plane ref
            M = np.stack(frames0[kind], axis=1)          # columns: up across fwd
            a = self.pos[self._az_b[r]] - self.pos[self._az_a[r]]
            a = a / max(float(np.linalg.norm(a)), 1e-9)
            v = self.pos[self._az_tip[r]] - self.pos[self._az_b[r]]
            v = v - a * float(v @ a)
            nv = float(np.linalg.norm(v))
            v = v / nv if nv > 1e-6 else np.cross(a, frames0[kind][2])
            v = v / max(float(np.linalg.norm(v)), 1e-9)
            self._az_a0[r] = M.T @ a
            self._az_r0[r] = M.T @ v
        # ...and the spine's own twist, the same way, in the pelvis frame
        M = np.stack(frames0[0], axis=1)
        a = self.pos[JIDX["chest"]] - self.pos[JIDX["pelvis"]]
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        v = self.pos[JIDX["shR"]] - self.pos[JIDX["shL"]]
        v = v - a * float(v @ a)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        self._sp_a0 = (M.T @ a).astype(np.float32)
        self._sp_r0 = (M.T @ v).astype(np.float32)

        #: WHICH AZIMUTH ROWS BELONG TO WHICH PARENT FRAME, and their birth
        #: directions gathered --- constants that `_az_world` was rediscovering
        #: with three `flatnonzero`s and four gathers ninety times a tick.
        self._az_kind_rows = tuple(
            (kind, rows, self._az_a0[rows].copy(), self._az_r0[rows].copy())
            for kind in (0, 1)
            for rows in (np.flatnonzero(self._az_refk == kind),)
            if rows.size)
        self._az_rolls = np.flatnonzero(self._az_refk == 2)
        #: HOW LONG A LEVER EACH AZIMUTH NEEDS to be a reading --- `AZIMUTH_SEEN`
        #: of that row's own far bone, taken once off her birth pose because a
        #: bone holds its length to about a percent and a per-substep norm for it
        #: would be paid 90 times a tick for a number that does not move.
        self._az_lever = (AZIMUTH_SEEN * _lens(
            self.pos[self._az_tip] - self.pos[self._az_b], keepdims=False)
        ).astype(np.float32)
        #: the scratch `_relax` scatters into, allocated once instead of 720
        #: times a tick
        self._relax_buf = np.zeros((n, 3), np.float32)
        #: the addresses the C reads, of arrays that never move after birth
        self._c_bones = (self._bone_a32.ctypes.data, self._bone_b32.ctypes.data,
                         self._bone_L.ctypes.data, self._bone_k.ctypes.data,
                         int(len(self._bone_a32)), self._bone_safe1.ctypes.data,
                         self._relax_buf.ctypes.data)
        self._c_flesh = (self._flesh_a32.ctypes.data, self._flesh_b32.ctypes.data,
                         self._flesh_min32.ctypes.data, int(len(self._flesh_a32)))
        self._c_hinges = (self._hinge_a32.ctypes.data, self._hinge_b32.ctypes.data,
                          self._hinge_c32.ctypes.data, self._hinge_s32.ctypes.data,
                          int(len(self._hinge_a32)))
        #: her named points as bare integers, so the pass loop stops paying for
        #: a dict lookup per axis per pass
        (self._i_pelvis, self._i_chest, self._i_shL, self._i_shR,
         self._i_head, self._i_face) = (
            JIDX["pelvis"], JIDX["chest"], JIDX["shL"], JIDX["shR"],
            JIDX["head"], JIDX["face"])
        #: who may move, per constraint row --- see `_hold`
        self._held_was = None
        self._hold()

        #: twist and roll have no anatomical zero about a bone, so theirs is
        #: HER OWN BIRTH POSE: measured once here, subtracted forever after,
        #: which puts every twist's rest at the middle of its range --- always
        #: commandable, never a posture she cannot order herself back to
        self._ax_zero = np.zeros(len(AXES), np.float32)
        for i, raw in self._azimuths().items():
            self._ax_zero[i] = raw
        #: what passive tone each axis has: TONE for muscles at rest,
        #: LIGAMENT for the azimuth axes that have no bony stop
        self._ax_tone = np.full(len(AXES), TONE, np.float32)
        for i, ax_name in enumerate(AXES):
            if ax_name.endswith((".twist", ".roll")):
                self._ax_tone[i] = LIGAMENT
        #: her birth pose as axis levels --- what TONE draws her toward
        self._ax_rest = self._ax_read()
        #: where every axis was when the tick began, so "moving" and "how much
        #: of the try did not happen" are things she can feel
        self._ax_start = self._ax_rest.copy()

        #: her reflex wiring --- which axes each spinal reflex reaches, and the
        #: level it pulls toward.  Extension is level 0.0 on knee.bend and
        #: hip.lift (the range starts at straight); "straight" for righting is
        #: each axis's zero-degree level.  See the REFLEXES block up top.
        ax_at = {ax_name: i for i, ax_name in enumerate(AXES)}
        self._rx_support = (
            (JIDX["foL"], ((ax_at["knee_left.bend"], 0.0),
                           (ax_at["hip_left.lift"], 0.0))),
            (JIDX["foR"], ((ax_at["knee_right.bend"], 0.0),
                           (ax_at["hip_right.lift"], 0.0))),
        )
        self._rx_right = ((ax_at["spine.bend"], 0.25), (ax_at["spine.side"], 0.5))
        #: the neck rights on its own, harder gains --- see NECK_RIGHT_*.
        #: bend's target is 0.80, NOT the anatomical midpoint: the lever was
        #: mapped (2026-08-30, full effort, upright hold) --- order 0.5
        #: leaves the head hanging at 84 deg off vertical, 0.75 raises it
        #: to 45; "level" lives high on this axis because the head's mass
        #: hangs forward of the pivot.
        self._rx_right_neck = ((ax_at["neck.bend"], 0.80),
                               (ax_at["neck.side"], 0.5))
        self._rx_vel = ((ax_at["knee_left.bend"], JIDX["knL"]),
                        (ax_at["hip_left.lift"], JIDX["knL"]),
                        (ax_at["knee_right.bend"], JIDX["knR"]),
                        (ax_at["hip_right.lift"], JIDX["knR"]),
                        (ax_at["spine.bend"], JIDX["chest"]),
                        (ax_at["spine.side"], JIDX["chest"]))
        #: how hard the floor shoved each joint LAST substep --- the only
        #: signal positive support listens to, one substep late the way a
        #: spinal loop is late
        self._rx_push = np.zeros(n, np.float32)
        #: her descending reflex gains, smoothed like an order: 1.0 is
        #: innate, 0 suppressed, 2 doubled --- (supL, supR, righting, catch)
        self._rx_mod = np.ones(4, np.float32)

        #: how firmly a hand draws each joint toward where it is holding it,
        #: 0..1 per substep --- see `_substep`.  Zero is nobody holding her,
        #: which is how she is born.  1.0 is a FIRM hand, not a pin: the
        #: joint is placed at the hand before the passes and the passes
        #: still argue (she sags ~0.11 m limp, and her pushes still land);
        #: a pinned joint is masked out of every pass and cannot be argued
        #: with at all.
        self.carried = np.zeros(n, np.float32)
        self.carrying = np.zeros(n, dtype=bool)
        self.carry_at = self.pos.copy()
        self.disturbance = _ZERO3.copy()
        #: the fastest any joint was moving when it hit something, this tick
        self.impact = 0.0
        #: how hard the room had to push each joint back, this tick --- what
        #: her skin feels.  Reset at the start of every `step()`.
        self.pressed = np.zeros(n, np.float32)
        self._prev_frame = self.frame()
        #: how much her head turned across the whole tick, in her own frame ---
        #: what the canals feel.  Zero until she has lived one.
        self.turned = np.zeros(3, np.float32)
        #: where her face was and which way it pointed, at a few moments DURING
        #: the tick.  Her eye takes a photograph at each --- four copies of the
        #: last one would be a movie in which nothing ever moves.
        #: WHERE SHE HAS ASKED TO LOOK: pan, tilt, -1..1 of `EYE_REACH`.
        #: Written from her `gaze` orders each tick.  See `_glimpse`.
        self.eye_at = np.zeros(2, np.float32)
        self.glimpses: list[tuple] = [self._glimpse()]
        self._felt: list[dict] = []
        #: and where ALL of her was, at a handful of moments during the tick.
        #:
        #: Purely so she can be WATCHED.  Her body already runs at 60 Hz --- 90
        #: substeps of 1/60 s a tick --- and only the last of them was ever
        #: looked at, so she appeared to teleport once every 1.5 seconds.  This
        #: keeps a few of the frames that were being computed and thrown away.
        #: It costs a copy of 17 points; nothing is simulated twice.
        self.film: list[np.ndarray] = [self.pos.copy()]

    def _glimpse(self) -> tuple:
        """Where she is looking --- HER EYES, pointed where she asked.

        **HER GAZE WAS WELDED TO HER SKULL AND THAT MADE HER BLIND** (found
        2026-08-14).  Measured: her gaze swung **80 degrees a tick** across a
        100 degree field, so a mother parked motionless in front of her came out
        as 59-176 flickering blobs and nothing she was shown could be tracked,
        recognised or taught.  It was not her eye and not her neck --- her neck
        angles never moved at all while her CHEST swung 36 deg/tick, and her
        head rode it.

        Eyes are how an animal solves this.  They are light, they turn without
        shoving anything, and they are what a fovea is for: the dense middle of
        a retina is only worth having if it lands on the thing being examined.
        `eye_at` is where she has asked to look, `pan` and `tilt` in -1..1 of
        her travel, and the reflex that finds a thing in her field writes it.
        """
        # POINTING IS BUILT RIGHT NOW, SO IT IS BACK.  This returned the head
        # axis between 2026-08-15 04:32 and today, on the owner's call ("we
        # dont have right pointing by eyes yet so simply align them by
        # existing direction") --- and the reason he made it was that a screen
        # held on her SKULL's axis disagreed with a picture rendered along her
        # EYES'.  That disagreement is fixed and is not fixed here: the hold
        # and `covers` both read `Sandbox._gazing`, which IS this glimpse, so
        # there is one axis whatever her eyes do.
        #
        # What the head axis cost, measured 2026-08-15: asking to look 0.6
        # right moved her gaze 0.03 degrees.  Her orienting reflex writes
        # `gaze.pan`/`gaze.tilt` every tick of her life and the body dropped
        # every one of them, so nothing in her could look from one thing to
        # another --- the owner, watching her: she stopped switching between
        # him on her screen and the rest of her room.  With the rotation the
        # same order turns her 21 degrees, and 21 is what a real eye does.
        up, right, fwd = self.frame()
        pan, tilt = float(self.eye_at[0]), float(self.eye_at[1])
        if abs(pan) < 1e-6 and abs(tilt) < 1e-6:
            return (self.pos[JIDX["face"]].copy(), up, right, fwd)
        aim = (fwd * np.cos(pan * EYE_REACH) * np.cos(tilt * EYE_REACH)
               + right * np.sin(pan * EYE_REACH)
               + up * np.sin(tilt * EYE_REACH))
        aim /= max(_len(aim), 1e-9)
        # her view's own right and up, squared against the new aim --- the
        # same orthonormalisation `frame()` does, for the same reason: a
        # picture rendered on axes that are not at right angles is skewed.
        r = _cross3(aim, up)
        rn = _len(r)
        if rn < 1e-6:                      # looking straight up her own axis
            return (self.pos[JIDX["face"]].copy(), up, right, aim)
        r /= rn
        u = _cross3(r, aim)
        u /= max(_len(u), 1e-9)
        return (self.pos[JIDX["face"]].copy(), u, r, aim)

    def _settle(self) -> None:
        """Put her down in the middle of the bed, resting on it.

        All three axes, not just height.  Laid on her back she is written head-
        first from the origin, so without centring she started with her head
        0.62 m off the end of a mattress 0.28 m long: measured, she rolled onto
        the floor by tick 3 and was against the wall by tick 20.
        """
        middle = (self.pos.max(axis=0) + self.pos.min(axis=0)) * 0.5
        self.pos[:, 0] -= middle[0]
        self.pos[:, 2] -= middle[2]
        lowest = float(np.min(self.pos[:, 1] - _RADIUS))
        self.pos[:, 1] += self.floor.rest_top - lowest

    # --- which way she is facing --------------------------------------------

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Her head's own `(up, right, forward)`.

        Forward is `face - head`: a real gaze between two real points, which is
        what makes turning her head possible at all.  Up is head-over-chest, so
        tipping just her head still changes what she faces.
        """
        up = _norm(self.pos[JIDX["head"]] - self.pos[JIDX["chest"]])
        fwd = _norm(self.pos[JIDX["face"]] - self.pos[JIDX["head"]])
        # HER VIEW'S UP, SQUARED AGAINST HER GAZE.  Head-over-chest and
        # face-out-of-head are body facts and nothing keeps them at right
        # angles: chin tucked they measured 23 degrees apart (fwd.up -0.386),
        # and a ray fan built on a sheared basis spreads asymmetrically, so
        # everything "in front of her" sat 8% below her view through every
        # fix of the evening ("tv position!!!!!!").  Same bodily up, made
        # perpendicular --- a camera never mounts its film crooked.
        up = _norm(up - fwd * float(up @ fwd))
        # `cross(fwd, up)`, and it matters: face north with the sky above you
        # and your right hand points east, which is what this gives.
        # `cross(up, fwd)` gives the other hand.
        return up, _norm(_cross3(fwd, up)), fwd

    def spine(self) -> np.ndarray:
        """Her trunk's forward --- used only to keep the hinge stops meaning the
        same thing whichever way up she is, never for what she is looking at.

        THIS IS HER CHEST FRAME'S `fwd`, AND IT IS NOT A COPY OF IT.  It used to
        be: this expression and `Frames.chest` both answered "which way does her
        chest face", they were written apart, and they drifted **91.09 degrees**
        apart without anything raising --- the frame was a triangle normal on a
        2.20 cm lever and this was her trunk.  This one was right.  The bug lived
        for months because two right-looking lines in two files cannot be
        compared by reading them.  So there is one now, and it is the frame's.
        """
        return self._frames.chest(self.pos)[2]

    @property
    def eye(self) -> np.ndarray:
        """Where she sees from."""
        return self.pos[JIDX["face"]]

    @property
    def ear(self) -> np.ndarray:
        """The centre of her head --- her ears are half a head either side."""
        return self.pos[JIDX["head"]]

    # --- one tick of moving ---------------------------------------------------

    def step(self, achieved, effort=None, *, handled: bool = False,
             glimpses: int = EYE_SLIDES, reflex=None) -> None:
        """Advance real physical time by one tick.

        `achieved` is what her flesh actually delivered --- never the raw order.
        A muscle that could not obey must not move anything either.

        **A MOVEMENT IS A SEQUENCE, NOT A POSTURE.**  `achieved` may be one
        number per muscle, which is held for the whole tick and is what her
        body did for its whole life until now; or `(muscles, slides)`, which is
        a movement in time.  The substep loop below already runs her body at
        60 Hz --- 90 steps of a 1.5 s tick --- and only the brain's inability to
        say anything new inside that made her a statue.  `measure/gait.py`
        measured what the difference is worth on an unchanged body: peak foot
        load 1.1x her weight -> 4.3x, foot movements 5 -> 110.

        `effort` is how hard she insists, the same shape.  `None` means it
        comes from the order itself, which is what the body assumed when there
        was only one line to read it off.
        """
        act = np.clip(np.asarray(achieved, np.float32), 0.0, 1.0)
        if act.ndim == 1:
            act = act[:, None]
        if act.shape[0] != len(AXES):
            raise ValueError(f"{len(AXES)} axes but {act.shape[0]} achieved")
        eff = act if effort is None else np.clip(
            np.asarray(effort, np.float32), 0.0, 1.0)
        if eff.ndim == 1:
            eff = eff[:, None]
        if eff.shape != act.shape:
            raise ValueError(f"effort is {eff.shape}, order is {act.shape}")
        slides = act.shape[1]
        self.impact = 0.0
        self.pressed[:] = 0.0
        self._ax_start = self._ax_read()
        drive = self.disturbance if handled else _ZERO3
        rate = min(1.0, ACT_RATE * DT)
        rx_aim = (np.ones(4, np.float32) if reflex is None else
                  np.clip(np.asarray(reflex, np.float32).ravel()[:4],
                          0.0, 2.0))
        self.glimpses, self.film, self._felt = [], [], []
        slide_from, last_s = self._ax_read(), 0
        every = max(1, STEPS_PER_TICK // max(1, glimpses))
        reel = max(1, STEPS_PER_TICK // FILM)
        for at in range(STEPS_PER_TICK):
            # which moment of the movement this substep belongs to.  With one
            # slide this is always 0 and nothing about her old body changes.
            s = at * slides // STEPS_PER_TICK
            if s != last_s:
                # that slide is over: what her body made of it
                self._felt.append(self._axes_now(
                    np.clip(act[:, last_s], 0.0, 1.0), slide_from))
                slide_from, last_s = self._ax_read(), s
            move = (act[:, s] - self._ax_act) * rate
            lim = MAX_DEG_PER_S * DT / self._ax_span     # her top speed, per axis
            self._ax_act += np.clip(move, -lim, lim)
            self._ax_eff += (eff[:, s] - self._ax_eff) * rate
            self._rx_mod += (rx_aim - self._rx_mod) * rate
            self._substep(drive, self._ax_ctx())
            if len(self.glimpses) < glimpses and (at + 1) % every == 0:
                self.glimpses.append(self._glimpse())
            if (at + 1) % reel == 0:
                self.film.append(self.pos.copy())
        self._felt.append(self._axes_now(
            np.clip(act[:, last_s], 0.0, 1.0), slide_from))
        while len(self._felt) < slides:      # one slide bodies
            self._felt.append(self._felt[-1])
        while len(self.glimpses) < glimpses:
            self.glimpses.append(self._glimpse())
        if not self.film:
            self.film.append(self.pos.copy())

        # Her rate of turn, measured ONCE for the whole tick --- at DT = 1/60 a
        # per-substep difference would be almost entirely numerical noise.
        # Wrap-safe for yaw: the signed angle BETWEEN the two forward vectors,
        # not a subtraction of two wrapped readings, which would spike at the
        # seam on an ordinary turn through it.
        up, right, fwd = self.frame()
        p_up, p_right, p_fwd = self._prev_frame
        self.turned = np.array([
            float(np.dot(np.cross(p_fwd, fwd), p_right)),   # pitch, about right
            float(np.dot(np.cross(p_fwd, fwd), p_up)),      # yaw,   about up
            float(np.dot(np.cross(p_right, right), p_fwd)),  # roll,  about fwd
        ], np.float32)
        self._prev_frame = (up, right, fwd)

    # --- what her muscles have to say about it --------------------------------

    def _az_world(self, pos, frames):
        """Axis and swing-transported reference for every azimuth row, world
        space: `(axis (n,3), ref (n,3), ok (n,))` --- one builder, used by the
        solver and the readings, so they cannot disagree."""
        axis = pos[self._az_b] - pos[self._az_a]
        axis /= np.maximum(_lens(axis), 1e-9)
        ref = np.zeros_like(axis)
        ok = np.ones(len(axis), bool)
        # WHICH ROWS BELONG TO WHICH FRAME NEVER CHANGES, and neither do their
        # birth directions --- so the `flatnonzero` and both gathers are done
        # once at birth (`_az_kind_rows`), not ninety times a tick.
        for kind, rows, a0p, r0p in self._az_kind_rows:
            M = np.stack(frames[kind], axis=1)
            a0 = a0p @ M.T
            r0 = r0p @ M.T
            ax_r = axis[rows]
            k = _crossn(a0, ax_r)
            nk = _lens(k)
            c = (a0 * ax_r).sum(axis=1, keepdims=True)
            ang = np.arctan2(nk, c)
            kn = k / np.maximum(nk, 1e-9)
            carried = _spin_rows(r0, kn, ang)
            ref[rows] = np.where(nk > 1e-6, carried, r0)
            ok[rows] &= (c[:, 0] > -0.999)      # folded fully back: no frame
        rolls = self._az_rolls
        if rolls.size:
            plane = _crossn(pos[self._az_r2[rolls]] - pos[self._az_r1[rolls]],
                            pos[self._az_b[rolls]] - pos[self._az_r2[rolls]])
            ref[rolls] = _crossn(plane, axis[rolls])
        ref = ref - axis * (ref * axis).sum(axis=1, keepdims=True)
        n = _lens(ref, keepdims=False)
        ok &= n > 1e-6
        ref /= np.maximum(n[:, None], 1e-9)
        return axis, ref, ok

    def _sp_ref(self, pos, pelvis):
        """The spine twist's transported reference, same rule as `_az_world`."""
        M = np.stack(pelvis, axis=1)
        a0 = M @ self._sp_a0
        r0 = M @ self._sp_r0
        axis = pos[self._i_chest] - pos[self._i_pelvis]
        axis = axis / max(_len(axis), 1e-9)
        k = _cross3(a0, axis)
        nk = _len(k)
        if nk > 1e-6:
            ref = self._J3._spin(r0, k / nk,
                                 float(np.arctan2(nk, float(a0 @ axis))))
        else:
            ref = r0
        ref = ref - axis * float(ref @ axis)
        n = _len(ref)
        return axis, (ref / n if n > 1e-6 else None)

    def _azimuths(self) -> dict[int, float]:
        """Raw twist/roll azimuths in degrees, per axis index --- through the
        same transported references the solver drives about."""
        J3, pos = self._J3, self.pos
        frames = (self._frames.pelvis(pos), self._frames.chest(pos))
        out: dict[int, float] = {}
        axis, ref, ok = self._az_world(pos, frames)
        v = pos[self._az_tip] - pos[self._az_b]
        v = v - axis * np.sum(v * axis, axis=1, keepdims=True)
        n = np.linalg.norm(v, axis=1)
        # CAN THE DRIVE ACT and IS THIS AN ANGLE are two questions, and only the
        # second one is asked here.  `ok` is the first: whether a reference can
        # be built at all, which is what the solver needs to push about.  The
        # second is whether the far point is far enough OFF the bone for its
        # direction to mean anything --- see `AZIMUTH_SEEN`.  Gating the DRIVE
        # on it as well was tried and measured: a boolean flipping inside the
        # pass loop is a discontinuity, her two ankles crossed it on different
        # substeps, and a body that had always settled mirror-symmetric came to
        # rest at 169.71 degrees on one side and 174.87 on the other.
        good = ok & (n > self._az_lever)
        v = v / np.maximum(n[:, None], 1e-9)
        cosr = np.sum(ref * v, axis=1)
        sinr = np.sum(np.cross(ref, v) * axis, axis=1)
        deg = np.degrees(np.arctan2(sinr, cosr))
        for r in np.flatnonzero(good):
            out[int(self._az_i[r])] = float(deg[r])

        axis1, ref1 = self._sp_ref(pos, frames[0])
        if ref1 is not None:
            v1 = pos[JIDX["shR"]] - pos[JIDX["shL"]]
            v1 = v1 - axis1 * float(v1 @ axis1)
            n1 = float(np.linalg.norm(v1))
            if n1 > 1e-6:
                v1 = v1 / n1
                out[self._ax_spine[2]] = float(np.degrees(np.arctan2(
                    float(np.cross(ref1, v1) @ axis1), float(ref1 @ v1))))
        return out

    def _ax_read(self) -> np.ndarray:
        """Every axis's level, 0..1 across its own range --- NOT clipped, so a
        joint shoved past its range by the world reads past it rather than
        lying.  Twist and roll are measured FROM HER BIRTH POSE (`_ax_zero`),
        because their absolute azimuth about a bone has no anatomical zero.

        READ ONCE PER POSE (performance, 2026-09-06, measured: this reading
        is 35% of a physics step and a step asks it three times, twice on
        the same unmoved joints).  The answer is a pure function of her
        joints' positions, so the last one is kept beside the positions it
        was read from and given back while they have not moved.  Same
        numbers, bit for bit."""
        key = self.pos.tobytes()
        got = getattr(self, "_ax_cache", None)
        if got is not None and got[0] == key:
            return got[1].copy()
        deg = self._ax_read_now()
        self._ax_cache = (key, deg.copy())
        return deg

    def _ax_read_now(self) -> np.ndarray:
        J3, pos, J = self._J3, self.pos, JIDX
        deg = np.zeros(len(AXES), np.float32)
        pelvis = self._frames.pelvis(pos)
        chest = self._frames.chest(pos)

        iB, iS, iT = self._ax_spine
        up, across, fwd = pelvis
        d = J3._norm(pos[J["chest"]] - pos[J["pelvis"]])
        deg[iB] = np.degrees(np.arctan2(float(d @ fwd), float(d @ up)))
        deg[iS] = np.degrees(np.arcsin(np.clip(float(d @ across), -1.0, 1.0)))

        iB, iS, iT = self._ax_neck
        up, across, fwd = chest
        d = J3._norm(pos[J["head"]] - pos[J["neck"]])
        deg[iB] = np.degrees(np.arctan2(float(d @ fwd), float(d @ up)))
        deg[iS] = np.degrees(np.arcsin(np.clip(float(d @ across), -1.0, 1.0)))

        for frame_name, left, anchor, child, _grand, _clen, iL, iS2, _iT in self._balls:
            frame = chest if frame_name == "chest" else pelvis
            lift, swing = J3.ball_read(
                frame, J3._norm(pos[child] - pos[anchor]), left)
            deg[iL], deg[iS2] = lift, swing

        for a, m, b, upper, lower, iB2 in self._hinges3:
            deg[iB2] = J3.hinge_read(
                upper, lower, float(np.linalg.norm(pos[b] - pos[a])))

        for i, raw in self._azimuths().items():
            # nearest-turn difference, so a wrap at 180 does not read as a spin
            gap = raw - self._ax_zero[i]
            deg[i] = (gap + 180.0) % 360.0 - 180.0

        return ((deg - self._ax_lo) / self._ax_span).astype(np.float32)

    def reading(self, achieved) -> dict[str, np.ndarray]:
        """What each axis actually did, in her own units and never a degree.

        `spindle`  where along its range the axis is, 0..1
        `force`    how much of the ordered turn did NOT happen, as a fraction
                   of what she was trying --- fully stopped reads high whether
                   the try was large or small, which is what a wall feels like
        `moving`   turning one way or the other, 0.5 being still

        Against what she ORDERED at the last slide, because a movement is a
        sequence and any earlier slide is somewhere she stopped aiming a tenth
        of a second ago.  Nothing labels whether a refusal is the floor, a
        thing, or her own joint's stop --- that is hers to work out.
        """
        # ONE READING PER SLIDE, TAKEN WHILE SHE WAS MOVING.
        #
        # This used to be `asked[:, -1]` --- the last of her fifteen orders,
        # measured once after every substep had run.  Fourteen fifteenths of
        # what her body did inside the tick was computed by the solver and
        # thrown away on that line, so `OPEN.md` 1g's missing half was never
        # missing data: it was discarded data.  `step` now keeps the reading
        # at the end of each slide, against THAT slide's order and from where
        # that slide began, which is what makes two attempts at a step
        # comparable on what her body DID.
        if self._felt:
            return {k: np.stack([f[k] for f in self._felt], axis=1)
                    for k in ("spindle", "force", "moving")}
        asked = np.asarray(achieved, np.float32)
        asked = asked[:, -1] if asked.ndim > 1 else asked.ravel()
        return self._axes_now(np.clip(asked, 0.0, 1.0), self._ax_start)

    def _axes_now(self, want, start) -> dict[str, np.ndarray]:
        """Her three axis readings right now, against `want` and from `start`."""
        now = self._ax_read()
        # AS A FRACTION OF THE TRY, AND THE TRY IS MEASURED FROM REST --- the
        # axis version of the span body's `span * act`.  Measured from the
        # tick's start instead, a held order reads fully-refused forever: at
        # equilibrium `start ~= now`, so the denominator vanishes while the
        # honest shortfall does not.  From rest, holding 11% short of a full
        # order reads a steady 0.18, and a small try that is completely
        # stopped still reads big --- which is what a wall feels like.
        trying = np.abs(want - self._ax_rest)
        short = np.maximum(np.abs(now - want) - AXIS_SLACK, 0.0)
        missed = np.where(np.abs(want - start) + trying > 1e-3,
                          short / np.maximum(trying, 0.05), 0.0)
        return {
            "spindle": np.clip(now, 0.0, 1.0).astype(np.float32),
            "force": np.clip(missed, 0.0, 1.0).astype(np.float32),
            "moving": np.clip(0.5 + (now - start), 0.0, 1.0
                              ).astype(np.float32),
        }

    def contact(self, point, joints: tuple[str, ...], within: float) -> float:
        """How firmly `point` touches the nearest of the named joints, 0..1.

        Graded, not a bare yes or no --- a thing brushing her hand and one
        pressed into it are not the same touch.
        """
        idx = [JIDX[name] for name in joints]
        d = np.linalg.norm(self.pos[idx] - np.asarray(point, np.float32), axis=1)
        return float(np.clip(1.0 - float(np.min(d)) / within, 0.0, 1.0))

    # --- the solver ------------------------------------------------------------

    def _substep(self, drive: np.ndarray, ctx=None) -> None:
        pin = self.pinned
        if not np.array_equal(pin, self._held_was):
            self._hold()
        vel = (self.pos - self.prev) * DAMP
        # Where she stood at the END of the last substep --- the one stable
        # reference `impact` needs.  `self.prev` cannot serve: a collision
        # correction inside this substep's own passes overwrites it, which made
        # an already-resting joint register a fresh landing every pass.
        before = self.pos.copy()
        new_prev = self.pos.copy()
        new_pos = self.pos + vel + (drive + _GRAVITY) * (DT * DT)
        if np.any(drive != 0.0):
            speed = float(_lens(new_pos - new_prev, keepdims=False).max()) / DT
            if speed > HANDLE_MAX_SPEED:
                new_pos = new_prev + (new_pos - new_prev) * (HANDLE_MAX_SPEED / speed)
        new_prev[pin] = self.pos[pin]
        new_pos[pin] = self.pos[pin]
        # A HAND THAT HOLDS SOME OF HER, AND LEAVES THE REST TO HER.
        #
        # A pinned joint goes where it is put whatever she does, so being
        # upright stops depending on her --- and `credit` marks what CHANGED,
        # so a baby who cannot sink has nothing to learn from staying up.
        # Measured under a pin: an hour of practice moved her weight-bearing
        # 0.0003 -> 0.0003, and her hardest press all hour was 0.0036.
        #
        # So a carried joint is drawn toward where the hand is, by a fraction
        # of the way each substep, and gravity pulls the other way the whole
        # time.  Limp, she settles BELOW the hand --- her own weight against
        # a give that yields.  Pushing, she rises toward it.  The hand never
        # lets go and never dictates: it is help, and what she does with help
        # is hers.
        #
        # A FRACTION OF GRAVITY WAS THE FIRST TRY AND IT WAS WRONG: at 1.5
        # seconds a tick, even a twentieth of g falls metres, so every value
        # below 1.0 simply put her on the floor --- 0.85 and 0.95 both sank
        # her the full 0.30 m in fifteen ticks.  Support is a place, not a
        # discount on physics.
        soft = self.carrying
        if np.any(soft):
            new_pos[soft] += (self.carry_at[soft] - new_pos[soft]) \
                * self.carried[soft, None]
        self.prev, self.pos = new_prev, new_pos

        pushed = np.zeros(len(self.pos), np.float32)
        #: the WHOLE substep's normal load, summed over every pass --- the
        #: friction budget.  `pushed` (the last pass alone) stays for
        #: impact and reflexes, but a RESTING knee's last-pass correction
        #: is ~zero (earlier passes already resolved it), so grip built on
        #: it held feet under a dangling walker and let knees and elbows
        #: skate --- "we still has problem with greep putthis time with
        #: knees end elbows" (his, 2026-08-28 morning).
        load = np.zeros(len(self.pos), np.float32)
        #: where she was when the passes started, so what they move her by can
        #: be told apart from where gravity and her own speed had already put her
        integrated = self.pos.copy()
        for _ in range(RELAX_PASSES):
            self._bones(1)
            self._solve_axes(ctx)
            # HER TRUNK'S FORWARD, ONCE PER PASS rather than once per solver
            # that wants it. Both the joint stops and the neck limit need it,
            # and computing it twice cost 30 ms a tick -- 1,440 cross products
            # and square roots for a number that had not changed in between.
            forward = self.spine()
            self._solve_hinges(forward)
            self._solve_neck(forward)
            self._solve_spine()
            self._solve_flesh()
            # only the LAST pass counts --- it is the one that describes where
            # she actually came to rest, and the earlier ones are the solver
            # arguing with itself on the way there
            pushed = self.floor.stop(self.pos, self.radius, self.pinned)
            load += pushed
        # BONES ARE BONES --- measured 2026-08-29, on the owner's own report
        # ("she has good grip on ends of hand and legs and her tryes just
        # stretched her"): under a sustained full push her pelvis-hip pair sat
        # 213% off rest length, the median bone 26% off, 15 of 26 bones more
        # than 10% out of shape.  A push travelled into ELONGATION instead of
        # into her mass, which is why a helper (an external skeleton) let her
        # crawl and the floor let her do nothing.  More whole-world passes
        # plateau --- 32 of them still left the worst bone 58% off --- so the
        # skeleton simply has to win: after the world has argued, the 26 bone
        # pairs alone are projected until they hold.  They cost microseconds
        # beside one full pass, and the floor gets the last word so the
        # tightening never leaves a joint underground.
        self._bones(BONE_PASSES)
        self.floor.stop(self.pos, self.radius, self.pinned)
        # ...AND WHAT THE SOLVE JUST DID IS NOT ALL GIVEN BACK AS SPEED.  Verlet
        # reads velocity as `pos - prev`, so a correction left here in full is
        # momentum next substep --- see `FLESH_DAMP`.  One subtract and one
        # multiply-add on seventeen points, once a substep, outside the passes.
        self.prev += (self.pos - integrated) * FLESH_DAMP
        # the hardest shove any substep of this tick needed, per joint --- what
        # her skin has to go on
        self.pressed = np.maximum(self.pressed, pushed)
        self._rx_push = pushed
        touching = pushed > 1e-9

        ground = self.floor.under(self.pos, self.radius)

        # FRICTION, once for the whole substep.  Damping inside the pass loop
        # does not dissipate real momentum, it chases the solver's own
        # in-progress corrections --- measured in the previous project: a body
        # at rest with zero pull drifting sideways at a never-decaying
        # 0.055 m/s forever instead of coming to a stop.
        #
        # WHEREVER SHE IS TOUCHING, not only where she is standing.  Restricting
        # this to joints below the floor line left everything resting against
        # the bed's SIDE with no friction at all, and she slid at a steady
        # 0.018 m/tick until she hit the wall.
        rubbing = touching & ~self.pinned
        if np.any(rubbing):
            self.prev[rubbing] += (self.pos[rubbing] - self.prev[rubbing]) * 0.5
        # ...AND A PLANTED FOOT GRIPS.  Half-damping is kinetic friction:
        # it halves sideways speed and the next pass pours more in from the
        # swaying mass above, so a touching foot slides for ever under any
        # steady force --- measured in the walk helper, one 10-tick push
        # glided her 0.103 m over 120 IDLE ticks at a speed that never
        # decayed: "big problem with floor it is like ice she slade when
        # try to push it" (his, 2026-08-27).  Real contact has STATIC
        # friction: a joint touching a surface ends the substep with ZERO
        # sideways speed --- position corrections still move it (bones,
        # hands, a real push translate her), but no momentum survives in
        # the plant.  Grabs are pins and stay out of it.
        if np.any(rubbing):
            self.prev[rubbing, 0] = self.pos[rubbing, 0]
            self.prev[rubbing, 2] = self.pos[rubbing, 2]
        # ...and a joint that has come to rest ON something stops falling
        # through it, rather than keeping the downward speed it arrived with.
        landed_on = (self.pos[:, 1] <= ground + 1e-4) & ~self.pinned
        if np.any(landed_on):
            self.prev[landed_on, 1] = self.pos[landed_on, 1]
        # ...AND POSITION IS WHERE GRIP LIVES, NOT SPEED.  This solver
        # moves her by position projection, so every pass DRAGS a planted
        # foot sideways in position and killing its speed afterwards only
        # wipes the evidence --- the creep stays.  His test, 2026-08-28:
        # held by a pin she fights realistically; on the floor her pushes
        # slide ("i cnow phisics and what i see is not right").  Measured:
        # 200 push ticks moved her chest 0.268 m while her planted foot
        # WANDERED 0.167 m.  Coulomb, in the solver's own currency: the
        # lateral ground a touching joint gained this substep is taken
        # back --- fully while it is within MU_FLOOR x this substep's
        # normal correction (static grip), the excess kept (slip).
        gripped = (load > 1e-9) & ~self.pinned
        if np.any(gripped):
            idx = np.where(gripped)[0]
            lat = self.pos[idx][:, [0, 2]] - before[idx][:, [0, 2]]
            need = np.sqrt((lat * lat).sum(axis=1))
            # the AVERAGE pass's correction, not the sum: `load` piles
            # up all RELAX_PASSES, so multiplying it raw made every
            # resting contact ~8x stickier than MU says --- a knee
            # folding a leg could not drag its own foot an inch
            hold = MU_FLOOR * load[idx] / float(RELAX_PASSES)
            take = np.minimum(1.0, hold / np.maximum(need, 1e-9))
            take[need <= 1e-9] = 0.0
            self.pos[idx, 0] -= lat[:, 0] * take
            self.pos[idx, 2] -= lat[:, 1] * take
            self.prev[idx, 0] = self.pos[idx, 0]
            self.prev[idx, 2] = self.pos[idx, 2]

        # IMPACT, once for the whole substep: a joint clearly clear of a surface
        # at the end of the last one, resting on it by the end of this one.  A
        # fast fall can cross straight through "above" to "resting" inside one
        # substep's integration, before any correction has run.
        landed = (before[:, 1] > ground + 1e-4) & (self.pos[:, 1] <= ground + 1e-4)
        if np.any(landed):
            self.impact = max(self.impact, float(np.max(
                (before[landed, 1] - self.pos[landed, 1]) / DT)))

    def _hold(self) -> None:
        """Who is free to move, gathered per constraint row.

        A hand on her changes at most once a tick --- `sandbox.advance` writes
        `pinned` and then steps her --- but every solver was re-gathering the
        same masks out of it eight times a substep, 5,760 times a tick, for
        arrays of seventeen booleans that had not moved.  This rebuilds them
        the moment `pinned` differs from the copy it was built against, so a
        probe that pins her mid-tick is still answered correctly; the check is
        one 17-byte comparison per substep.
        """
        free = ~self.pinned
        self._free = (
            free[self._dir_tip, None], free[self._dir_react, None],
            free[self._az_tip, None], free[self._az_backa, None],
            free[self._az_backb, None],
            free[self._h3_a, None], free[self._h3_b, None],
            bool(free[self._i_shR]), bool(free[self._i_shL]),
            free[self._hinge_b], free[self._hinge_a], free[self._hinge_c],
            bool(free[self._i_face]), bool(free[self._i_head]),
        )
        self._free_relax = (self._bone_plan[3] & free).astype(np.float32)[:, None]
        # ...and the same masks as the C wants them, kept on self so they live
        self._free_relax32 = np.ascontiguousarray(self._free_relax[:, 0], np.float32)
        self._hinge_free8 = tuple(np.ascontiguousarray(self._free[i], np.uint8)
                                  for i in (9, 10, 11))
        self._pinned8 = np.ascontiguousarray(self.pinned, np.uint8)
        self._c_masks = (self._free_relax32.ctypes.data,
                         tuple(m.ctypes.data for m in self._hinge_free8),
                         self._pinned8.ctypes.data)
        self._held_was = self.pinned.copy()

    def _relax_plan(self, a_idx: np.ndarray, b_idx: np.ndarray):
        """The constant half of `_relax`'s work.  Which joints a batch touches
        never changes, only what they want to do about it --- so this is done
        once rather than 720 times a tick."""
        n = len(self.pos)
        m = len(a_idx)
        idx = np.concatenate([a_idx, b_idx])
        sign = np.concatenate([np.ones(m, np.float32), -np.ones(m, np.float32)])
        count = np.bincount(idx, minlength=n).astype(np.float32)
        hit = count > 0
        # `sign[:, None]` and the divisor were being rebuilt 720 times a tick
        # for values that are the same at every substep of her life.  `safe` is
        # the per-joint count with the joints no bone touches set to 1, so the
        # division needs no mask --- see `_relax`.
        safe = np.where(hit, count, 1.0).astype(np.float32)[:, None]
        return idx, sign, count, hit, sign[:, None], safe

    def _relax(self, plan, corr: np.ndarray) -> None:
        """A batch of pairwise corrections, AVERAGED per joint rather than
        summed when several constraints touch it in one pass --- what keeps a
        vectorised batch comparable to solving each constraint singly."""
        idx, sign, count, hit, sign2, safe = plan
        delta = self._relax_buf
        delta.fill(0.0)
        np.add.at(delta, idx, np.concatenate([corr, corr]) * sign2)
        # AVERAGING AND APPLYING WITHOUT TWO BOOLEAN MASKS.  `delta[hit] /=
        # count[hit]` and `pos[free] += delta[free]` are each a gather, an
        # arithmetic op and a scatter -- 17.8 us a pass for two divisions and
        # an addition.  A joint in no bone has `count` 0 and `delta` 0, so
        # dividing every row by `max(count, 1)` leaves it 0/1 = 0, exactly
        # where the mask left it; and a joint that may not move takes
        # `delta * 0`, which adds nothing.  Same numbers, three ops.
        delta /= safe
        self.pos += delta * self._free_relax

    def _bones(self, passes: int) -> None:
        """`passes` bone passes --- in C when `knit.dll` has them and her
        positions are the contiguous float32 the C expects (they are, from
        birth and from every load); the numpy passes otherwise.  In place:
        the C writes `self.pos` itself, so nothing is copied and lost."""
        pos = self.pos
        if (_BONES_C is not None and pos.dtype == np.float32
                and pos.flags.c_contiguous and self._relax_buf.flags.c_contiguous):
            a32, b32, L, k, nb, safe1, buf = self._c_bones
            _BONES_C(pos.ctypes.data, int(len(pos)), a32, b32, L, k, nb, safe1,
                     self._c_masks[0], int(passes), buf)
            return
        for _ in range(int(passes)):
            self._solve_bones()

    def _solve_bones(self) -> None:
        d = self.pos[self._bone_b] - self.pos[self._bone_a]
        dist = _lens(d, keepdims=False)
        safe = np.where(dist > 1e-6, dist, 1e-6)
        self._relax(self._bone_plan,
                    d * ((dist - self._bone_L) / safe * 0.5 * self._bone_k)[:, None])

    def _reflexes(self, pos, eff):
        """Her spinal reflexes: tone and rest for THIS substep.

        State-dependent versions of the same two arrays TONE already uses ---
        never a controller, and free (a no-op returning the birth arrays)
        whenever nothing has triggered, which is her whole supine day.  Her
        commanded effort overrides all of it through the existing
        `w = eff / (eff + tone)` blend.  See the REFLEXES block up top.
        """
        tone, rest = self._ax_tone, self._ax_rest
        if not REFLEXES:
            return tone, rest
        grew = False

        # POSITIVE SUPPORT: a leg whose foot bears real weight stiffens
        # toward extension, in proportion to the load --- scaled by her
        # own descending gain for that side
        for side, (foot, axes) in enumerate(self._rx_support):
            press = min(1.0, float(self._rx_push[foot]) / PRESS_FULL)
            if press <= SUPPORT_FLOOR:
                continue
            p = ((press - SUPPORT_FLOOR) / (1.0 - SUPPORT_FLOOR)
                 * float(self._rx_mod[side]))
            if p <= 1e-4:
                continue
            if not grew:
                tone, rest, grew = tone.copy(), rest.copy(), True
            s = min(1.0, SUPPORT_SHIFT * p)
            for i, ext in axes:
                rest[i] = rest[i] * (1.0 - s) + ext * s
                tone[i] = tone[i] + SUPPORT_TONE * p

        # RIGHTING: head and trunk toward vertical, engaged only once she is
        # already held more upright than flat --- lying down it does not exist
        trunk = pos[self._i_chest] - pos[self._i_pelvis]
        upness = float(trunk[1]) / max(_len(trunk), 1e-9)
        engage = (max(0.0, (upness - RIGHT_FROM) / (1.0 - RIGHT_FROM))
                  * float(self._rx_mod[2]))
        if engage > 1e-4:
            if not grew:
                tone, rest, grew = tone.copy(), rest.copy(), True
            s = min(1.0, RIGHT_SHIFT * engage)
            for i, straight in self._rx_right:
                rest[i] = rest[i] * (1.0 - s) + straight * s
                tone[i] = tone[i] + RIGHT_TONE * engage
            sN = min(1.0, NECK_RIGHT_SHIFT * engage)
            for i, straight in self._rx_right_neck:
                rest[i] = rest[i] * (1.0 - sN) + straight * sN
                tone[i] = tone[i] + NECK_RIGHT_TONE * engage

        # VELOCITY TONE: flesh resists a fast knee, and only where she is not
        # already insisting --- the catch before a buckle becomes a fall
        for i, joint in self._rx_vel:
            speed = _len(pos[joint] - self.prev[joint]) / DT
            v = min(1.0, speed / VEL_FULL)
            if v <= 0.05:
                continue
            if not grew:
                tone, rest, grew = tone.copy(), rest.copy(), True
            tone[i] = tone[i] + (VEL_TONE * v * (1.0 - float(eff[i]))
                                 * float(self._rx_mod[3]))

        return tone, rest

    def _ax_ctx(self):
        """The whole geometric side of the axis drives, once per substep ---
        directions aimed, references transported, spans computed.  What stays
        in the pass loop is only the corrections against live positions: the
        scalar solver ran all of this 8x per substep and cost 651 ms a tick.
        """
        J3, pos, J = self._J3, self.pos, JIDX
        act, eff = self._ax_act, self._ax_eff
        tone, rest = self._reflexes(pos, eff)
        w = eff / (eff + tone)
        target = act * w + rest * (1.0 - w)
        gain = AXIS_GAIN * (eff + tone * (1.0 - eff))
        deg = self._ax_lo + self._ax_span * target

        frames = (self._frames.pelvis(pos), self._frames.chest(pos))
        up = np.stack([frames[i][0] for i in self._dir_frame])
        across = np.stack([frames[i][1] for i in self._dir_frame])
        fwd = np.stack([frames[i][2] for i in self._dir_frame])

        a_deg = np.radians(deg[self._dir_iA])[:, None]
        b_deg = np.radians(deg[self._dir_iB])[:, None]
        lat = across * self._dir_lat
        aim_ball = (-up) * np.cos(a_deg) + (
            lat * np.cos(b_deg) + fwd * np.sin(b_deg)) * np.sin(a_deg)
        aim_fold = _spin_rows(_spin_rows(up, -across, a_deg), fwd, b_deg)
        aim = np.where(self._dir_ball[:, None], aim_ball, aim_fold)
        aim /= np.maximum(_lens(aim), 1e-9)
        k_dir = ((gain[self._dir_iA] + gain[self._dir_iB]) * 0.5)[:, None]

        axis, ref, ok = self._az_world(pos, frames)
        ang = np.radians(deg[self._az_i] + self._ax_zero[self._az_i])[:, None]
        tdir = _spin_rows(ref, axis, ang)
        k_az = gain[self._az_i][:, None] * ok[:, None]

        inside = np.radians(180.0 - deg[self._h3_i])
        # law of cosines; the two bone terms are birth constants (`_h3_uu`,
        # `_h3_ul`), not something to recompute ninety times a tick
        want = np.sqrt(np.maximum(
            self._h3_uu - self._h3_ul * np.cos(inside), 1e-8))
        k_h = (0.5 * gain[self._h3_i])

        iT = self._ax_spine[2]
        spine = None
        if gain[iT] > 0.0:
            axis1, ref1 = self._sp_ref(pos, frames[0])
            if ref1 is not None:
                spine = (axis1,
                         J3._spin(ref1, axis1,
                                  np.radians(float(deg[iT]
                                                   + self._ax_zero[iT]))),
                         float(gain[iT]))
        return (aim, k_dir, axis, tdir, k_az, want, k_h, spine)

    def _solve_axes(self, ctx) -> None:
        """Every joint drawn toward what she ordered, inside the same passes
        as her bones, pairwise --- what a limb gains its trunk absorbs.

        Effort is the stiffness: a drive with no effort corrects almost
        nothing, TONE draws every axis gently toward her birth pose (and
        LIGAMENT holds the twist and roll axes, which have no bony stop), so
        limp is curled rather than splayed and an order takes over exactly as
        fast as she insists on it.  Strength caps EFFORT upstream and never
        the order --- all of an axis's range is hers to aim at.
        """
        if ctx is None:
            ctx = self._ax_ctx()
        aim, k_dir, axis, tdir, k_az, want, k_h, spine = ctx
        pos = self.pos
        # WHO IS FREE TO MOVE is a fact about the hand on her, not about the
        # pass: it changes at most once a tick and was being re-gathered eight
        # times per substep.  `_hold` rebuilds these the moment `pinned` moves.
        f = self._free

        delta = (pos[self._dir_origin] + aim * self._dir_len
                 - pos[self._dir_tip]) * k_dir
        half = delta * 0.5
        pos[self._dir_tip] += half * f[0]
        pos[self._dir_react] -= half * f[1]

        v = pos[self._az_tip] - pos[self._az_b]
        vpar = axis * (v * axis).sum(axis=1, keepdims=True)
        n = _lens(v - vpar)
        delta = (pos[self._az_b] + vpar + tdir * n - pos[self._az_tip])             * (k_az * (n > 1e-6))
        half, quarter = delta * 0.5, delta * 0.25
        pos[self._az_tip] += half * f[2]
        pos[self._az_backa] -= quarter * f[3]
        pos[self._az_backb] -= quarter * f[4]

        dvec = pos[self._h3_b] - pos[self._h3_a]
        cur = np.maximum(_lens(dvec, keepdims=False), 1e-6)
        corr = dvec * ((cur - want) / cur * k_h)[:, None]
        pos[self._h3_a] += corr * f[5]
        pos[self._h3_b] -= corr * f[6]

        if spine is not None:
            # HER SPINE'S TWIST, and it is the most expensive line in this
            # method --- 31.8 us a pass, 22.9 ms a tick, entirely on 3-element
            # vectors.  Nothing here is vectorisable; what it had was six
            # re-fetches of two shoulders and a half-vector built twice.
            axis1, tdir1, ktw = spine
            pR, pL = pos[self._i_shR], pos[self._i_shL]
            v1 = pR - pL
            vpar1 = axis1 * (v1 @ axis1)
            n2 = _len(v1 - vpar1)
            if n2 > 1e-6:
                mid = (pR + pL) * 0.5
                hv = (tdir1 * n2 + vpar1) * 0.5
                if f[7]:
                    pR += (mid + hv - pR) * ktw
                if f[8]:
                    pL += (mid - hv - pL) * ktw

    def _solve_hinges(self, fwd: np.ndarray) -> None:
        # The stops act AFTER the muscles have pulled --- a muscle that would
        # fold a knee backwards gets overruled by the knee, exactly as it is
        # overruled in a real leg by two bones meeting.
        # in C when knit.dll has it (`hinges`, the same arithmetic in the same
        # order, proved bit-identical over 600 ticks); the numpy below stays
        # the definition and the fallback
        if (_HINGES_C is not None and self.pos.dtype == np.float32
                and self.pos.flags.c_contiguous):
            fwd32 = np.ascontiguousarray(fwd, np.float32)       # lives through the call
            a32, b32, c32, s32, nh = self._c_hinges
            f9, f10, f11 = self._c_masks[1]
            _HINGES_C(self.pos.ctypes.data, a32, b32, c32, s32, nh, fwd32.ctypes.data,
                      float(HINGE_SLACK), float(HINGE_K), f9, f10, f11)
            return
        a, b, c, s = self._hinge_a, self._hinge_b, self._hinge_c, self._hinge_s
        off = np.dot(self.pos[b] - (self.pos[a] + self.pos[c]) * 0.5, fwd) * s
        bad = off < -HINGE_SLACK
        if not bad.any():
            return
        # `np.outer` is `x[:, None] * y`, and each of the three masks below was
        # being built TWICE per line, six times a pass, 5,760 times a tick.
        push = ((-HINGE_SLACK - off) * s * HINGE_K)[:, None] * fwd
        half = push * -0.5
        f = self._free
        mb, ma, mc = bad & f[9], bad & f[10], bad & f[11]
        self.pos[b[mb]] += push[mb]
        self.pos[a[ma]] += half[ma]
        self.pos[c[mc]] += half[mc]

    def _solve_flesh(self) -> None:
        """No two of her in one place --- sphere against sphere, soft as
        flesh (half a correction a pass, like a hinge), the pinned end of
        any pair standing its ground."""
        # in C when knit.dll has it (`flesh`, the same arithmetic in the same
        # order, proved bit-identical over 600 ticks); the numpy below stays
        # the definition and the fallback
        if (_FLESH_C is not None and self.pos.dtype == np.float32
                and self.pos.flags.c_contiguous):
            a32, b32, mn, nf = self._c_flesh
            _FLESH_C(self.pos.ctypes.data, a32, b32, mn, nf, self._c_masks[2])
            return
        a, b = self._flesh_a, self._flesh_b
        d = self.pos[a] - self.pos[b]
        n = np.sqrt((d * d).sum(axis=1))
        short = self._flesh_min - n
        hit = (short > 0.0) & (n > 1e-9)
        if not np.any(hit):
            return
        ha, hb = a[hit], b[hit]
        push = d[hit] * (short[hit] / n[hit] * 0.5)[:, None]
        freea = ~self.pinned[ha]
        freeb = ~self.pinned[hb]
        wa = np.where(freea & freeb, 0.5, np.where(freea, 1.0, 0.0))
        wb = np.where(freea & freeb, 0.5, np.where(freeb, 1.0, 0.0))
        np.add.at(self.pos, ha, push * wa[:, None])
        np.add.at(self.pos, hb, -push * wb[:, None])

    def _solve_spine(self) -> None:
        """Her trunk runs out of travel, the same way her neck does.

        A real spine stops.  Hers did not: measured 2026-09-02 over 900 ticks,
        `spine.bend` read **-169.8 to 171.9 degrees** --- nearly a full circle
        --- against the -20..60 its own range gives it, so **94% of her ticks
        were outside it**.  `_ax_read` divides by that range, `reading` clips
        to 0..1, and the result was a spindle **pinned at 0.000 for 749 live
        ticks while every other axis of her trunk and neck moved**.  She had
        no signal at all from the one joint that sits her up, which is exactly
        what her scale reports as "trunk never upright unaided".

        HIS CHOICE, 2026-09-02, over widening the range: constrain the joint.
        A girl whose spine bends 170 degrees is not the fix; her median is
        23.3 degrees and inside the range already --- it is the excursions
        that are not anatomy.

        ONE-SIDED, LIKE THE NECK: her muscles own the whole range, and this
        only catches the point where a real spine would already have stopped.
        Her sideways lean is untouched --- only the bend is limited, because
        only the bend is what ran away.
        """
        a, b = JIDX["pelvis"], JIDX["chest"]
        up, across, fwd = self._frames.pelvis(self.pos)
        d = self.pos[b] - self.pos[a]
        reach = _len(d)
        if reach < 1e-6:
            return
        u, f, side = float(d @ up), float(d @ fwd), float(d @ across)
        ang = np.degrees(np.arctan2(f, u))
        # her own range arrays, so there is one answer to "how far can it bend"
        iB = self._ax_spine[0]
        lo = float(self._ax_lo[iB])
        hi = lo + float(self._ax_span[iB])
        if lo <= ang <= hi:
            return
        # ...bent back to the nearer limit, keeping its length and its lean
        want = np.radians(lo if abs(ang - lo) < abs(ang - hi) else hi)
        flat = float(np.sqrt(max(u * u + f * f, 1e-12)))
        fixed = (up * (flat * float(np.cos(want)))
                 + fwd * (flat * float(np.sin(want))) + across * side)
        push = (self.pos[a] + fixed - self.pos[b]) * SPINE_K
        if not self.pinned[b]:
            self.pos[b] += push
        if not self.pinned[a]:
            self.pos[a] -= push * 0.5

    def _solve_neck(self, forward: np.ndarray) -> None:
        """Her neck runs out of travel.

        A one-sided limit, like a knee: her gaze may be anywhere within a cone
        around her trunk's own forward, and nowhere behind it.  Her neck muscles
        move her freely inside that cone --- this only catches the point where a
        real neck would already have stopped.
        """
        head, face = self._i_head, self._i_face
        aim = self.pos[face] - self.pos[head]
        reach = _len(aim)
        if reach < 1e-6:
            return
        short = NECK_CONE * reach - float(np.dot(aim, forward))
        if short <= 0.0:
            return
        push = forward * (short * NECK_K)
        f = self._free
        if f[12]:
            self.pos[face] += push
        if f[13]:
            self.pos[head] -= push * 0.5
        # ...AND HER HEAD CANNOT HANG BELOW HER CHEST.  The cone limits her
        # GAZE; nothing limited her head's PLACE, so hanging in the walk
        # helper her head swung a full neck-length under the chest --- 6 cm
        # below her own pelvis: "her head somtime aner her ass" (his,
        # 2026-08-27, watching).  A real cervical spine stops at
        # chin-on-chest, the head still above the chest along the trunk.
        # One-sided like everything here: her muscles own the whole range
        # above the stop.
        up = self.pos[JIDX["chest"]] - self.pos[JIDX["pelvis"]]
        n2 = _len(up)
        if n2 > 1e-6:
            up = up / n2
            neck = self.pos[head] - self.pos[JIDX["chest"]]
            low = HEAD_MIN * _rest_len("chest", "head") - float(np.dot(neck, up))
            if low > 0.0:
                # a skeleton stop pushes like bone, not like muscle ---
                # at NECK_K the weight of the hanging head argued it back
                # to -0.078 below the chest every pass
                lift = up * low
                if f[13]:
                    self.pos[head] += lift
                self.pos[JIDX["chest"]] -= lift * 0.5

    # --- keeping her ------------------------------------------------------------

    def save(self) -> dict:
        """Her body as it stands.  Everything a resume needs and nothing it
        does not --- her strength and fatigue belong to `muscles.py`."""
        return {"pos": self.pos.tolist(), "prev": self.prev.tolist(),
                "act": self._ax_act.tolist(), "eff": self._ax_eff.tolist()}

    def load(self, state: dict) -> None:
        self.pos = np.array(state["pos"], np.float32)
        self.prev = np.array(state["prev"], np.float32)
        act = np.array(state["act"], np.float32)
        eff = np.array(state.get("eff", state["act"]), np.float32)
        # A SAVE FROM THE SPAN BODY cannot mean anything on axes --- different
        # count, different meaning --- so an old save restores her POSE and
        # lets the orders start quiet, which is what waking up limp is.
        self._ax_act = (act if act.size == len(AXES)
                        else np.zeros(len(AXES), np.float32))
        self._ax_eff = (eff if eff.size == len(AXES)
                        else np.zeros(len(AXES), np.float32))
        self._ax_start = self._ax_read()
        self._prev_frame = self.frame()

    def __repr__(self) -> str:  # pragma: no cover - display only
        _, _, fwd = self.frame()
        return (f"<Ragdoll head={self.pos[JIDX['head']].round(3).tolist()} "
                f"facing={fwd.round(2).tolist()}>")
