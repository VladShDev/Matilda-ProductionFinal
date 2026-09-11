"""Her eye: what her room looks like from where her face actually is.

One ray per retinal cell, cast from her face into the room, against a handful of
analytic surfaces.  Not a renderer --- about three thousand dot products, no GPU,
no driver, and **deterministic**, so a test reproduces exactly and a moment that
went wrong can be replayed.

**SHE DOES NOT RECEIVE A VIDEO FEED.**  She sees HER ROOM, and the window is a
bright rectangle moving in it with whatever you are showing on its face.  Fly
closer and she sees better, because the rectangle covers more of her retina and
more cells fall on it.  **DETAIL IS A CONSEQUENCE OF GEOMETRY.  There is no
resolution setting anywhere and there must never be one.**

**THE PICTURE, NOT THE PICTURE'S EDGES** (2026-08-14, the owner's: *"we have to
see everything that she sees in that moment --- the room, every object there"*).
A cell reports what is in front of it.  It used to report how much it differed
from its own neighbourhood and then how much THAT differed from what it had been
saying lately --- two derivatives stacked on each other --- and between them a
solid thing sitting still in front of her was in her picture nowhere at all: a
flat-painted sphere became a ring of fixed thickness with a dead middle, so the
fraction of it she could see was ~8/radius and FELL as it came closer, reaching
zero by the time it filled her field.  Measured, an object crossing 0.387 m ->
0.135 m lit 40% -> 30% -> 12% -> **0%** of the cells it covered.

**AND WHAT SHE HANDS OVER IS A VIEW, NOT A COUNT OF ANYTHING.**  A tick of
seeing is one picture and a list of what was in it (`brain/parts.py`, which has
the whole argument).  Every decision this file ever made to keep a row count
down --- edges instead of a picture, change instead of what is there, four
photographs instead of one --- cost her eyesight and saved nothing that mattered:
a view is 703 bytes, measured on a real life.  Do not reason about her sight in
lines.

**ONE PHOTOGRAPH A TICK.**  It was four, and the owner cut it: motion belongs
between ticks, where she has a tape to compare them on.
"""

from __future__ import annotations

import numpy as np

try:                                     # THE KERNEL (locked decision 10):
    import cupy as _gpu                  # same math, same code, run on the
    _gpu.zeros(1)                        # GPU when one answers.  `_mod(a)`
except Exception:                        # picks numpy or cupy from the array
    _gpu = None                          # itself, so there is exactly ONE
                                         # implementation of her light.


def _mod(a):
    return _gpu.get_array_module(a) if _gpu is not None else np


def _lift(a):
    """To the device, if there is one.  **The dtype survives.**

    It read `np.asarray(a, np.float32)` on the no-GPU branch, so a bool mask
    became floats and `inked & _lift(_WEARS)[win]` raised --- her eye could not
    run at all without a device, which is what every probe of it died on.  The
    GPU branch has always preserved dtype; this is the same promise kept on both
    sides.
    """
    if _gpu is not None:
        return _gpu.asarray(a)
    a = np.asarray(a)
    return a if a.dtype == bool or a.dtype.kind in "iu" else np.asarray(a, np.float32)


def _home(a):
    """Back to the host, always float32, wherever it lived."""
    return _gpu.asnumpy(a) if _gpu is not None and _mod(a) is _gpu         else np.asarray(a, np.float32)

#: HER RETINA.  Settled at 512 --- measured 2026-08-25 on her own 4060 at
#: **11.87 ms a frame**, which fits inside a single 33 ms tick.  See
#: `body/EYE.md`.  Not a contract any more: it is a fact about her eye and it
#: lives with her eye.
RETINA_H = 512
RETINA_W = 512
#: red, green, blue --- three kinds of cone
CONES = 3
#: how many looks make one tick's picture
EYE_SLIDES = 1

from . import wordmark
from .room import BED, COT, MIRROR, PEN, ROOM, SIDES

#: which of `_SIDES` wear the mark, in that tuple's own order
_WEARS = np.array(wordmark.WEARS, bool)



#: How wide she sees.  Generous, because a newborn's field is wide and her
#: acuity is poor --- the opposite way round from a camera.
FIELD = np.radians(100.0)

#: Below how much contrast a cell says nothing at all.
#:
#: **HER EAR HAS ALWAYS HAD ONE OF THESE AND HER EYE HAD NONE.**  `sound.GATE`
#: drops a slide quieter than a number, so saying nothing gives her real silence;
#: her retina reported every cell it had, however faintly, and "faintly" is what
#: a room made of one soft colour is almost entirely made of.
#:
#: **AND IT IS NOT HERE TO KEEP A ROW COUNT DOWN.**  It was, and the whole
#: paragraph that used to sit here counted how many of her lines fired and what
#: share of her tape they were.  That question is retired (see the top of
#: `brain/parts.py`): a tick of seeing is a picture, her eye is not rows, and
#: every decision this project made to protect a row count cost her eyesight.
#:
#: What a threshold is for is that a real ganglion cell has one --- below some
#: level a cell says nothing, and hers should too.  `sound.FAINTEST` is 0.02 for
#: the same reason in the same words.  `test_a_gate_does_not_cost_her_an_edge`
#: flies the window at her and checks the gate did not eat what arrived.
GATE = 0.002

#: How far a cell is compared against to find its own local contrast.  Small
#: enough that a real edge stands out sharply, large enough that a single noisy
#: cell does not.
#:
#: **IT IS AN ANGLE, NOT A COUNT OF CELLS**, and writing it as a count is what
#: broke her sight when her retina doubled (2026-08-14).  At 32x32 a radius of
#: 2 compared a cell against 5/32 of her field; at 64x64 the same 2 compares it
#: against half that angle, so her contrast operator silently became twice as
#: sharp and anything broad --- a surface, a face close up --- read as flat.
#: Measured on the gate that catches exactly this
#: (`test_flying_closer_puts_more_of_you_on_her_retina_and_in_more_detail`):
#: green at 32x32, red at 64x64 at EVERY fovea setting including none, because
#: the fovea was never what did it.
BLUR = max(2, round(RETINA_W * 2 / 32))

#: What her room is made of.  Its surfaces are neutral on purpose --- see the
#: note inside --- and NOT to keep anything sparse, which is what this line used
#: to say.
COLOURS = {
    # THE ROOM IS NEUTRAL AND THE THINGS IN IT ARE NOT.  Her four surfaces used
    # to carry a faint warm or cool tint each, which put the BED 0.091 from
    # SKIN in colour while the room's own surfaces sat 0.013-0.081 from one
    # another --- so an arm against a cot was no more different than a cot is
    # from a ceiling.  They keep their brightnesses, which is what gives the
    # room its corners and gives the checker its optic flow; what they give up
    # is hue, which now means exactly one thing: something is there.  A nursery
    # is painted in neutrals and the things in it are the colourful part, which
    # is the same arrangement.
    "floor": (0.28, 0.28, 0.28),
    "ceiling": (0.79, 0.79, 0.79),
    "wall": (0.60, 0.60, 0.60),
    # FOUR WALLS, FOUR HUES --- his order, 2026-09-09: *"she has to see more
    # colourful life"*.  They were one grey, so turning her head changed the
    # geometry in her eye and nothing else; now WHICH WALL she is facing is
    # itself a thing she can read, which is the same want as his fence: a
    # bearing.  Every one keeps the old 0.60 brightness, so the floor 0.28 and
    # ceiling 0.79 still bracket them exactly as measured, and every one stays
    # LOW IN SATURATION on purpose --- his 2026-08-14 rule stands, a real
    # object is not the colour of the paint behind it, so her things
    # (red-orange), her toy (blue), her bottle (amber) and her fences (green)
    # remain the only saturated things in her world and `parts.find` still
    # picks them out of the wall.  Green is not among them: green is the edge
    # of where she may go.
    "wall_west": (0.66, 0.56, 0.56),      # rose
    "wall_east": (0.54, 0.59, 0.67),      # slate blue
    "wall_south": (0.67, 0.63, 0.52),     # sand
    "wall_north": (0.61, 0.56, 0.66),     # lilac
    "bed": (0.46, 0.46, 0.46),
    # HER FENCE IS GREEN, AND MARKED --- his order, 2026-09-09: *"her fence
    # around her playing ground is invisible for her, so make it visible in
    # green colour for her for contrast, and draw some lines on it so she is
    # able orienting more fast"*.
    #
    # It was grey (0.44) on my rule that a colour means SOMETHING IS THERE and
    # furniture is not a something.  His answer is that the edge of the world
    # she can move in is not furniture: it is the one fixed thing she can take
    # a bearing from, and she could not see it.  A HUE, not a brightness ---
    # his 2026-08-14 rule, because `parts.find` groups her field by colour and
    # a fence at the wall's brightness is a fence she cannot pick out.  Green
    # is the one channel nothing else in her room uses: her things are
    # red-orange, her toy blue, her bottle amber, everything else neutral.
    # Its brightness still sits between the floor 0.28 and the wall 0.60, so
    # it keeps the edge against both that the grey had.
    "cot": (0.24, 0.60, 0.30),
    # HIS TRAINING PEN'S FENCES, the same green --- his order, 2026-09-09,
    # after his screenshot: the cot was not the fence he meant.  `room.PEN`
    # has stopped her skin at two planes since 2026-08-30 ("THE PEN'S TWO
    # FENCES") and the word `pen` did not appear in this file at all --- a
    # wall she could walk into and never see, under a comment of ours that
    # called it *"real to her body, real to her eye"*.  The same green as her
    # cot on purpose: both are the edge of where she may go, and one meaning
    # is one colour.
    "fence": (0.24, 0.60, 0.30),
    "thing": (0.80, 0.34, 0.22),
    # 2026-08-10, the owner's question that decided it: "how can she
    # recognise ME if she doesn't recognise different objects?"  Until today
    # every object in her eye was the SAME red-orange --- her mother, the
    # bottle and the toy were told apart by nothing but size.  Real objects
    # have real colours; now a thing's kind is visible the way it is in any
    # nursery.  The toy is the only blue in her whole room on purpose: one
    # channel that means exactly one thing.
    # ...AND A COLOUR HAS TO BE A COLOUR, NOT A BRIGHTNESS (2026-08-14, the
    # owner's: *"your testing room doesn't contain objects, so solve it"*).  The
    # bottle was (0.94, 0.93, 0.88) and the wall is (0.62, 0.60, 0.58): the same
    # grey at two brightnesses, 0.011 apart once light is divided out.  Her
    # brain groups her field by colour (`parts.find`), because grouping it by
    # brightness picked the WALL on every tick of a real life --- so a bottle
    # painted in wall-grey was a bottle she could not find, and the bottle is
    # the one object in her world she has to learn to reach for.  A real one is
    # not the colour of the paint behind it.  Amber, and a window the blue of
    # daylight rather than another off-white rectangle.
    "bottle": (0.96, 0.84, 0.46),
    "toy": (0.20, 0.45, 0.75),
    "mother": (0.85, 0.55, 0.62),
    "window": (0.72, 0.84, 1.00),
    # SHE CAN SEE HERSELF (owner's order, 2026-08-11: "she must see her
    # body").  Until today her retina drew the room, the bed, every thing
    # and the window --- and not one atom of her: a hand crossing her own
    # face was invisible.  Seeing your own limb answer your own order is
    # the visual half of the loop her voice already closes (her sound
    # reaches her own ears); an infant stares at her own hands for weeks
    # because that contingency is the first thing sight can prove.  One
    # skin colour, hers alone in the room, the way the toy owns blue.
    "skin": (0.92, 0.74, 0.62),
    "nothing": (0.0, 0.0, 0.0),
}

_FAR = 1e9


#: HOW MUCH FINER THE MIDDLE OF HER EYE IS THAN THE EDGE.  Cell spacing at the
#: centre as a fraction of evenly-spaced: 0.35 means the middle resolves 2.9x
#: finer and the far edge 2.3x coarser, over the SAME field of view, with the
#: SAME cells and the SAME number of rays.  Nothing costs anything here.
#:
#: **AN EVENLY-SPACED RETINA IS THE PART THAT IS NOT HUMAN.**  Every eye in
#: nature is dense in the middle and coarse at the edge --- that is what a fovea
#: IS, and it is why you look AT things.  Hers sampled her whole field evenly
#: until today, so a thing held up in front of her face got exactly as many
#: cells as the wall behind her ear, and 32x32 spread over a whole field is too
#: coarse to make out anything held in a hand.
#:
#: **AND IT IS A 16x SAVING, NOT A COST** --- the owner asked what an evenly
#: spaced retina was actually buying, and the answer is nothing.  Measured on a
#: small toy at 35 cm (`measure.acuity`): 32x32 evenly spaced puts 2x2 cells on
#: it; 32x32 at 0.20 puts 8x8 on it for the same rays; matching that evenly
#: needs 128x128, which is SIXTEEN times the rays and lines.  Her field is 100
#: degrees wide and a held object covers about a tenth of it, so even spacing
#: spends ~89% of her cells on wall.
#:
#: What it really costs is the edge --- ~2.7x coarser than even spacing at 0.20
#: --- so she localises motion at the rim of her view less precisely.  That is
#: what a neck and a saccade are for, and she has both.
#:
#: 0.20 is still mild; a real fovea is nearer a HUNDRED times the periphery.
#: **1.0 --- NO LENS AT ALL** (2026-08-15, the owner's call, made with the
#: numbers below in front of him: *"remove this shit and retrieve video from
#: cam in good quality and find me on the pic finally"*).
#:
#: What it cost, measured with `py -m measure.acuity` on both settings, cells
#: landing on a thing held in front of her:
#:
#:                      FOVEA 0.20      FOVEA 1.0
#:     a fingertip     2392 (48x48)     112 (10x10)
#:     a thumb         5012 (70x70)     232 (15x15)
#:     a small toy    21224 (145x145)  1436 (37x37)
#:     his face       69196 (263x263) 10928 (104x104)
#:
#: What it bought, and why he is right for the world she is actually in: the
#: lens magnified her MIDDLE five-fold and crushed her rim, and she cannot
#: point her eye (`ragdoll._glimpse` returns the head frame; the pan/tilt is
#: in git awaiting that day).  A fovea you cannot aim is a magnifier bolted to
#: the middle of your head.  Worse, everything downstream had to undo it: her
#: sheet was warped here and un-warped again for the display, so what she
#: gained in the middle was thrown away before anyone saw it, and boxes found
#: on a camera frame did not lie where the warp had put things.  Straight
#: through, one geometry, no undoing.
#:
#: The day she can aim her eye, this is the number that makes the fovea earn
#: its keep, and the table above is what to expect back.
FOVEA = 1.0


def _spread(n: int, reach: float) -> np.ndarray:
    """Cell offsets across her field, packed toward the middle.

    `u * (FOVEA + (1 - FOVEA) * u^2)`: odd, so it stays symmetric; 1 at the
    edge, so the field of view is untouched; slope `FOVEA` at the centre, which
    IS the density gain.
    """
    u = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    return (reach * u * (FOVEA + (1.0 - FOVEA) * u * u)).astype(np.float32)


def _rays(up, right, fwd) -> np.ndarray:
    """Where each retinal cell is looking, in her own frame.  `(cells, 3)`."""
    reach = np.tan(FIELD * 0.5)
    x = _lift(_spread(RETINA_W, reach))
    y = _lift(-_spread(RETINA_H, reach))
    xp = _mod(x)
    sx, sy = xp.meshgrid(x, y)
    fwd, right, up = _lift(fwd), _lift(right), _lift(up)
    aim = (fwd[None, None, :] + right[None, None, :] * sx[:, :, None]
           + up[None, None, :] * sy[:, :, None])
    aim = aim.reshape(-1, 3)
    # explicit, not `linalg.norm`: the same three squares in the same order
    # on either chip, so CPU and GPU pictures can be compared byte for byte
    size = xp.sqrt(aim[:, 0] * aim[:, 0] + aim[:, 1] * aim[:, 1]
                   + aim[:, 2] * aim[:, 2])
    return aim / size[:, None]


def _closer(best_t, best_c, t, colour, hit, near=None):
    """Keep whichever surface a ray met first --- but no nearer than `near`.

    `near` is per-ray and exists for one reason: A MIRROR IS MOUNTED ON
    SOMETHING, and that something is behind the glass.  `_reflect` traces the
    reflection from the eye mirrored through the plane, so everything between
    that virtual eye and the glass --- the wall the mirror hangs on, the floor
    under a downward ray --- lies in front of it and would be struck first,
    which is the mirror showing you the back of its own fixing.  Measured with
    no clip at all: 682 cells of glass and ZERO of her in it.

    It is a floor on `t` rather than a list of things to skip, because the list
    is not knowable --- it depends on where she is standing and which way each
    ray tilts.  `None` costs nothing and is what every direct look passes.
    """
    xp = _mod(best_t)
    nearer = hit & (t < best_t)
    if near is not None:
        nearer = nearer & (t > near)
    best_t = xp.where(nearer, t, best_t)
    best_c = xp.where(nearer[:, None], colour, best_c)
    return best_t, best_c


#: The wall/floor pattern: two tones in bands this wide, differing this much.
#: The owner's order ("add walls --- she has to know why she stops") named a
#: real blindness: a flat-coloured surface filling her view is zero contrast,
#: so a wall at her nose was invisible and she stopped against nothing she
#: could see.  A pattern gives a wall edges at ANY distance, and gives the
#: floor OPTIC FLOW when she travels --- the signal locomotion is learned on.
#: Static bands adapt into her retinal baseline like all standing edges;
#: what stays loud is pattern that MOVES past her, which is exactly travel.
#:
#: **0.7 -> 1.4, the owner's word, 2026-08-17**
#: (`measured/checker_and_the_named_boxes.md`).  At 0.7 her whole resting view
#: WAS this pattern and nothing else: lying on her back her eye is 0.42 m up,
#: the ceiling 2.08 m above that, and a 100 deg field spans
#: `2 * 2.08 * tan(50) = 4.96 m` of it --- 7.1 squares across.  Predicted from
#: that geometry alone: one square is 0.1412 of her view wide.  Measured,
#: `parts.find` on that render: **64 things, 8 distinct x centres by 8
#: distinct y, medW 0.1406**, their centres exactly one square apart.  Every
#: thing she had was one square of her own wallpaper.
#:
#: `things = ceil(span/STRIPE)^2`, exact at every step, so the period is the
#: only lever here and it is a clean one.  CONTRAST IS NOT: 0.085 down to
#: 0.002 is a **42x** reduction for byte-identical output, because `find`'s
#: rule is comparative and a step of any size against a flat interior is an
#: edge.  Below that it falls off a cliff to one piece.  Nobody should reach
#: for `TONE` to quieten her room; it does not have that shape.
#:
#: WHAT THIS DOES NOT COST.  With the television in her view her grouping
#: covers it **95.8% with 0.5% spill** at 0.7, and **95.8% / 0.5% at 1.4** ---
#: unchanged --- while the fragments fall 194 -> 81.  Turning the pattern OFF
#: was the tempting move and it is the harmful one: at `TONE 0` the room is
#: one piece, the screen merges into the wall behind it, and 96.9% of the lump
#: around it is not the screen.  That is the exact failure `measure/binding.py`
#: exists to prevent.
#:
#: WHAT IT DOES COST, and it is the honest half.  A surface shows her no edge
#: at all once it is nearer than `STRIPE / (2 * tan(50))`, because she is
#: inside a single square: **0.294 m at 0.7, 0.587 m at 1.4**.  That radius
#: doubles.  It is affordable only because of what her eye actually renders
#: --- floor, ceiling, four walls at 10 m, the bed slab, and the window ---
#: and `one_look` does not read `Room.surfaces()`, so her cot's 38 uprights
#: were never in her picture to lose.
STRIPE = 1.4
#: HER FENCE'S OWN BANDING --- his order of 2026-09-09.  A quarter of the rail's
#: height gives four bands up a side she can count, and the same width along it;
#: the tone is his "for contrast", against the walls' deliberately quiet 0.085.
COT_BAND = COT["rail"] * 0.25
COT_TONE = 0.25
#: ...and his pen's fences the same way, at THEIR height: four bands up a
#: fence 0.40 high, the same width along it, so the mark moves both when she
#: turns her head and when she travels the fence.
FENCE_BAND = PEN["h"] * 0.25
FENCE_TONE = COT_TONE
TONE = 0.085


#: The six sides, and the paint on each --- one table, so the search below can
#: name a face by number and leave the painting until it knows which one won.
_SIDES = ((1, 0.0, "floor"), (1, ROOM["y"], "ceiling"),
          (0, -ROOM["x"], "wall_west"), (0, ROOM["x"], "wall_east"),
          (2, -ROOM["z"], "wall_south"), (2, ROOM["z"], "wall_north"))
_PAINT = np.array([COLOURS[w] for _, _, w in _SIDES], np.float32)


def _planes(o, d, best_t, best_c, near=None):
    """The six sides of the room.  A ray leaving the room meets exactly one.

    ...and that sentence is the whole shape of this function, which it did not
    used to have.  It painted ALL SIX and let `_closer` throw five away: the
    checker --- two divisions, two floors, an add and a modulo over every ray,
    then a colour built from it --- ran SIX TIMES a look and five of the
    answers were never seen by anything.  Now the face is found first and
    painted once.

    BYTE-IDENTICAL, and it has to be: `_closer` keeps the FIRST strictly
    nearest, so picking that same winner among the six and comparing it once
    against what came in is the same face, the same `t`, and the same float.
    Checked on her real 262,144 rays --- from her own glimpse and from three
    other places in the room, and with something already 2.0 m away so the
    incoming `best_t` is not `_FAR`: 0 differing cells of 786,432, max |diff|
    0.0, her whole render identical, and `parts.find` giving the same 23 things
    with identical rows.
    """
    xp = _mod(d)
    win_t = xp.full(len(d), _FAR, xp.float32)
    win = xp.zeros(len(d), xp.int64)             # which of the six
    win_ax = xp.zeros(len(d), xp.int64)          # and along which axis it lies
    won = xp.zeros(len(d), bool)
    for i, (axis, at, _what) in enumerate(_SIDES):
        along = d[:, axis]
        going = xp.abs(along) > 1e-9
        t = xp.where(going, (at - float(o[axis])) / xp.where(going, along, 1.0),
                     _FAR)
        take = going & (t > 1e-4) & (t < win_t)
        win_t = xp.where(take, t, win_t)
        win = xp.where(take, i, win)
        win_ax = xp.where(take, axis, win_ax)
        won = won | take
    # every face carries the pattern, the CEILING INCLUDED: a lying baby
    # faces up, and in the 16 x 12 m room a flat ceiling left her whole
    # resting view blank --- the still-room audit settled 3 places of 48.
    hit = _lift(o)[None, :] + d * win_t[:, None]
    # the two axes the winning face lies in --- `(1,2)` for an x face, `(0,2)`
    # for a y face, `(0,1)` for a z face.  That is
    # `a1, a2 = (i for i in range(3) if i != axis)` written for an axis that is
    # now per ray instead of per loop.
    p1 = xp.where(win_ax == 0, hit[:, 1], hit[:, 0])
    p2 = xp.where(win_ax == 2, hit[:, 1], hit[:, 2])
    checker = (xp.floor(p1 / STRIPE) + xp.floor(p2 / STRIPE)) % 2.0
    colour = _lift(_PAINT)[win] * (1.0 + TONE * (checker - 0.5) * 2.0)[:, None]

    # ...AND THE MARK ON THEM.  `body/wordmark.py` has been in the tree, whole
    # and measured, IMPORTED BY NOTHING; this is the line it was written for.
    #
    # WHY A ROOM NEEDS WRITING ON IT.  The checker gives her edges and it gives
    # her optic flow, and it is IDENTICAL EVERYWHERE: every square is the same
    # square, so it can tell her that something is there and never WHICH part of
    # a wall she is looking at or how far she turned.  A letterform can.
    # Measured on her own eye, a 20 degree turn: **0.0331 -> 0.1586 per cell,
    # 4.8x**.  The owner's sentence for what it is for: *"the room for her is
    # nothing, she doesn't see it at all, but she tries."*
    #
    # ONE TILE PER WALL, and it repeats across the ceiling rather than being
    # stretched to fit it --- *"without geometry / words broken"* is his line
    # and a squashed letter would break it.  The tile is the room's own width
    # by its own height, which is the 8:1 `wordmark._tile` was built for.
    # The floor does not wear it: she lies on the floor.
    u = xp.where(win_ax == 0, hit[:, 2], hit[:, 0])
    v = xp.where(win_ax == 1, hit[:, 2], hit[:, 1])
    sheet = _lift(wordmark.SHEET)
    h, w = wordmark.SHEET.shape
    ix = (((u / (ROOM["x"] * 2.0) + 0.5) % 1.0) * w).astype(xp.int64)
    iy = ((1.0 - (v / ROOM["y"]) % 1.0) * h).astype(xp.int64)
    inked = (sheet[xp.clip(iy, 0, h - 1), xp.clip(ix, 0, w - 1)] > 0.5)
    inked = inked & _lift(_WEARS)[win]
    colour = xp.where(inked[:, None], xp.float32(wordmark.INK), colour)
    return _closer(best_t, best_c, win_t, colour, won, near)


def _box(o, d, at, half, colour, best_t, best_c, near=None):
    """Her bed --- the slab method, which is six planes that agree with each
    other about where their edges are."""
    xp = _mod(d)
    at, half, colour = _lift(at), _lift(half), _lift(colour)
    here = _lift(o)
    lo = (at - half - here) / xp.where(xp.abs(d) > 1e-9, d, 1e-9)
    hi = (at + half - here) / xp.where(xp.abs(d) > 1e-9, d, 1e-9)
    # `hit_t` and not `near`: the slab's own entry distance, which used to be
    # called `near` and now would shadow the clip that shares the name.
    hit_t = xp.max(xp.minimum(lo, hi), axis=1)
    far = xp.min(xp.maximum(lo, hi), axis=1)
    return _closer(best_t, best_c, hit_t, colour, (far >= xp.maximum(hit_t, 0.0))
                   & (far > 1e-4) & (hit_t > 1e-4), near)


def _sphere(o, d, at, size, colour, best_t, best_c, near=None):
    xp = _mod(d)
    away = o - np.asarray(at, np.float32)            # tiny, host arithmetic
    # the dot spelled out, same order both chips (see `_rays`)
    b = (d[:, 0] * float(away[0]) + d[:, 1] * float(away[1])
         + d[:, 2] * float(away[2]))
    c = float(away @ away) - size * size
    under = b * b - c
    real = under > 0.0
    t = xp.where(real, -b - xp.sqrt(xp.maximum(under, 0.0)), _FAR)
    return _closer(best_t, best_c, t, colour, real & (t > 1e-4), near)


def _quad(o, d, window, best_t, best_c, near=None):
    """You: a flat rectangle with your camera on the face turned toward her.

    A ray that lands on it reads the pixel it landed on, so coming closer really
    does show her more of what you are holding up --- the same picture spread
    over more of her cells.
    """
    xp = _mod(d)
    normal = np.asarray(window.look, np.float32)
    # the dots spelled out, same order both chips (see `_rays`)
    facing = (d[:, 0] * float(normal[0]) + d[:, 1] * float(normal[1])
              + d[:, 2] * float(normal[2]))
    going = xp.abs(facing) > 1e-9
    gap = np.asarray(window.at, np.float32) - o
    t = xp.where(going, float(gap @ normal)
                 / xp.where(going, facing, 1.0), _FAR)

    on = _lift(o)[None, :] + d * t[:, None] \
        - _lift(np.asarray(window.at, np.float32))
    # **THE SCREEN'S OWN RIGHT, WHICH IS ALSO THE VIEWER'S** (2026-08-15).  This
    # was `cross(normal, up)`, and the window's normal points OUT of the screen
    # toward whoever is watching --- so it came out as the exact opposite of her
    # `right` (`ragdoll.frame` builds hers as `cross(fwd, up)` and her `fwd` is
    # `-normal`), and **everything ever shown to her arrived left-right
    # mirrored.**  Found by putting a face on the screen and seeing it come back
    # on the wrong side.  `down` is rebuilt from the new `across` rather than
    # from the normal, or fixing the mirror would flip her vertically instead.
    # **AND THE SCREEN'S OWN UP** (2026-08-17).  This read `cross(worldup,
    # normal)`, which rebuilt the roll out of the only thing left when nobody
    # carried it --- and the rebuild is amplified by `1 / |worldup x normal|`,
    # so a screen lying face-down has no roll to rebuild from.  `window.up` is
    # already square with `look` (`window.upright`), and with the world's up in
    # it this line IS the old line.
    across = np.cross(np.asarray(window.up, np.float32), normal)
    if np.linalg.norm(across) < 1e-6:
        across = np.array([1.0, 0.0, 0.0], np.float32)
    across /= np.linalg.norm(across)
    down = np.cross(across, normal)
    u = (on[:, 0] * float(across[0]) + on[:, 1] * float(across[1])
         + on[:, 2] * float(across[2]))
    v = (on[:, 0] * float(down[0]) + on[:, 1] * float(down[1])
         + on[:, 2] * float(down[2]))
    half_w, half_h = window.size[0] * 0.5, window.size[1] * 0.5
    inside = going & (t > 1e-4) & (xp.abs(u) <= half_w) & (xp.abs(v) <= half_h)

    colour = xp.tile(_lift(np.array(COLOURS["window"], np.float32)), (len(d), 1))
    shows = window.shows
    if shows is not None and shows.size:
        pix = np.asarray(shows, np.float32)
        if pix.ndim == 2:
            pix = np.repeat(pix[:, :, None], CONES, axis=2)
        if float(pix.max(initial=0.0)) > 1.0:
            pix = pix / 255.0
        pix = _lift(pix)
        h, w = pix.shape[0], pix.shape[1]
        px = xp.clip(((u / half_w + 1.0) * 0.5 * (w - 1)).astype(xp.int64), 0, w - 1)
        py = xp.clip(((v / half_h + 1.0) * 0.5 * (h - 1)).astype(xp.int64), 0, h - 1)
        colour = xp.where(inside[:, None], pix[py, px, :CONES], colour)
    return _closer(best_t, best_c, t, colour, inside, near)


def _bars(room, o, d, best_t, best_c, near=None):
    """Her cot: four sides, each ONE plane test and a modulo.

    **SHE COULD NOT SEE THE ONE THING THAT STOPS HER.**  This function did not
    exist.  `one_look` carried its own copy of the floor, the walls and the bed
    and never read `Room.surfaces()` --- which builds all 42 of her cot's boxes,
    and which had ZERO CALLERS.  `room.py` says it in its own words beside them:
    *"she cannot presently see what now stops her."*

    MEASURED, on her real eye in her cot, before this: **38.1% of her field was
    pointed at a bar**, and she read that part of the world as empty space
    **3.64 m** away when it was **0.10 m** away.  The nearest thing she could
    see was 2.02 m; the nearest thing that exists was 0.03 m.  She was not
    walking into a wall she could see --- the walls she can see are 10 m off ---
    she was pushing forward because forward looked empty.

    WHY IT IS FOUR PLANES AND NOT FORTY-TWO BOXES.  Her bars are regular, so a
    side is a plane and a modulo, which is exactly the trick `_planes` already
    uses for its checker.  The alternative was measured and is unaffordable
    (CPU, her 512 x 512 eye, one look):

        as she was                                 5.6 ms
        42 boxes, one `_box` call each           126.3 ms
        all 42 batched into one slab pass        614.7 ms   (33 M floats)
        four periodic planes                      43.2 ms

    A bar is where the hit lands near an upright's centre and below the band; a
    rail is the bar lying across their tops.  Both come off `room`'s own arrays,
    so the side a person has DROPPED is drawn dropped --- `_axle` and `_band`
    move when `rail` does, and this reads them every look.
    """
    xp = _mod(d)
    sl = COT["slat"]
    band, axle = room._band.reshape(-1), room._axle.reshape(-1)
    paint = _lift(np.array(COLOURS["cot"], np.float32))
    for i, side in enumerate(SIDES):
        a = side["axis"]
        b = 2 - a
        dv = d[:, a]
        t = (float(side["at"]) - o[a]) / xp.where(xp.abs(dv) > 1e-9, dv, 1e-9)
        py = o[1] + d[:, 1] * t
        pb = o[b] + d[:, b] * t
        on_side = ((t > 0.0) & (xp.abs(pb) <= side["along"] + sl)
                   & (py >= 0.0) & (py <= float(axle[i]) + sl))
        # how far this hit is from the nearest upright's centre
        off = xp.abs((pb - float(side["slats"][0])) % float(side["step"]))
        off = xp.minimum(off, float(side["step"]) - off)
        hit = on_side & (((off <= sl) & (py <= float(band[i])))
                         | (xp.abs(py - float(axle[i])) <= sl))
        # ...AND MARKED, SO SHE CAN TAKE A BEARING (his, 2026-09-09).  The same
        # two-tone banding her walls already carry, at her fence's own scale
        # and at his contrast: bands a quarter of her rail's height, crossed
        # along the side as well as up it, so the mark changes both when she
        # turns her head and when she moves along the fence.  A plain colour
        # gives her ONE line to read (is it there); a banded one gives her a
        # ruler, which is what orienting needs.  No number here is invented:
        # the band is `COT["rail"] / 4` and the tone is the one he asked for
        # --- three times her walls' 0.085, because a wall is meant to be
        # quiet and this is meant to be found.
        marks = (xp.floor(py / COT_BAND) + xp.floor(pb / COT_BAND)) % 2.0
        shade = paint * (1.0 + COT_TONE * (marks - 0.5) * 2.0)[:, None]
        best_t, best_c = _closer(best_t, best_c, t, shade, hit, near)
    return best_t, best_c


def _fences(o, d, best_t, best_c, near=None):
    """HIS PEN'S TWO FENCES, IN HER EYE --- his order, 2026-09-09.

    Two axis-aligned planes, bounded exactly the way her BODY is bounded by
    them in `room.py`: the west fence at `x0` across `z0..z1`, the south fence
    at `z0` across `x0..x1`, both ending at `h`, the height above which a
    carried body passes.  Every number is read off `PEN`, so a fence her skin
    has is a fence her eye has, and neither can drift from the other.
    """
    xp = _mod(d)
    paint = _lift(np.array(COLOURS["fence"], np.float32))
    for a9, at9, lo9, hi9 in ((0, PEN["x0"], PEN["z0"], PEN["z1"]),
                              (2, PEN["z0"], PEN["x0"], PEN["x1"])):
        b9 = 2 - a9
        dv = d[:, a9]
        t = (float(at9) - o[a9]) / xp.where(xp.abs(dv) > 1e-9, dv, 1e-9)
        py = o[1] + d[:, 1] * t
        pb = o[b9] + d[:, b9] * t
        hit = ((t > 0.0) & (pb >= float(lo9)) & (pb <= float(hi9))
               & (py >= 0.0) & (py <= float(PEN["h"])))
        marks = (xp.floor(py / FENCE_BAND) + xp.floor(pb / FENCE_BAND)) % 2.0
        shade = paint * (1.0 + FENCE_TONE * (marks - 0.5) * 2.0)[:, None]
        best_t, best_c = _closer(best_t, best_c, t, shade, hit, near)
    return best_t, best_c


def _scene(room, window, o, d, body, best_t=None, best_c=None, near=None):
    """Everything in her room, once, from one point.

    Pulled out of `one_look` so it can be traced TWICE --- see `_reflect`.  It
    is the same code in the same order and nothing about a single look changed.
    """
    xp = _mod(d)
    if best_t is None:
        best_t = xp.full(len(d), _FAR, xp.float32)
        best_c = xp.tile(_lift(np.array(COLOURS["nothing"], np.float32)),
                         (len(d), 1))
    best_t, best_c = _planes(o, d, best_t, best_c, near)
    best_t, best_c = _bars(room, o, d, best_t, best_c, near)
    best_t, best_c = _fences(o, d, best_t, best_c, near)
    best_t, best_c = _box(o, d, np.array([0.0, BED["top"] * 0.5, 0.0], np.float32),
                          np.array([BED["x"], BED["top"] * 0.5, BED["z"]], np.float32),
                          np.array(COLOURS["bed"], np.float32), best_t, best_c,
                          near)
    for thing in room.things.values():
        best_t, best_c = _sphere(o, d, thing.at, thing.size,
                                 _lift(np.array(COLOURS.get(thing.what,
                                                            COLOURS["thing"]),
                                                np.float32)),
                                 best_t, best_c, near)
    if body is not None:
        parts, sizes = body
        skin = _lift(np.array(COLOURS["skin"], np.float32))
        for at, size in zip(parts, sizes):
            best_t, best_c = _sphere(o, d, np.asarray(at, np.float32),
                                     float(size), skin, best_t, best_c, near)
    # HER ROOM MAY HAVE MORE THAN ONE SCREEN.  His, 2026-09-01: *"make one
    # more TV ... on your training spot wall and just rotate some images there
    # and tell names when she look there because I can't sit in front of my
    # laptop every time when she need to see me."*  His own television stays
    # his; the board is a second flat thing with its own picture and its own
    # name.  One screen or a list of them --- every caller that passes one is
    # untouched.
    for one in (() if window is None else
                (window if isinstance(window, (list, tuple)) else (window,))):
        if one is not None:
            best_t, best_c = _quad(o, d, one, best_t, best_c, near)
    return best_t, best_c


def _reflect(room, window, o, d, body, best_t, best_c):
    """Her mirror, and it is a mirror --- it shows her the room, and her.

    **A MIRROR THAT DOES NOT REFLECT IS A GREY RECTANGLE.**  This one was two
    square metres of flat blue drawn only in the spectator's page, on a wall ten
    metres off, in no `.py` file at all: she could not see it, it reflected
    nothing, and her body did not know it was there.

    ONE BOUNCE, AND IT COSTS ONE MORE TRACE.  The trick that makes it cheap is
    that a FLAT mirror needs no per-ray origin: what you see in it is exactly
    the room as seen from the eye REFLECTED THROUGH ITS PLANE.  So the second
    trace has a single origin like the first --- which matters, because every
    surface in this file takes `o` as one point and reads `float(o[axis])`, and
    a true secondary bounce would have needed all of them rewritten to carry an
    origin per ray.

    Her own body is in that second trace, which is the entire point: **this is
    the first surface in her world that can show her her own face.**  She has
    been able to see her hands since 2026-08-11 and never her head, because
    nobody sees their own head.

    It does not darken what it reflects.  A real mirror does, a little, and a
    tint here would be a constant chosen in this file to no measured purpose ---
    what it is FOR is that the thing in it moves when she moves.
    """
    m = MIRROR
    xp = _mod(d)
    ax, other = int(m["axis"]), 2 - int(m["axis"])
    along = d[:, ax]
    going = xp.abs(along) > 1e-9
    t = xp.where(going, (float(m["at"]) - float(o[ax]))
                 / xp.where(going, along, 1.0), _FAR)
    py = float(o[1]) + d[:, 1] * t
    pz = float(o[other]) + d[:, other] * t
    on = (going & (t > 1e-4) & (t < best_t)
          & (xp.abs(py - float(m["y"])) <= float(m["half_h"]))
          & (xp.abs(pz - float(m["z"])) <= float(m["half_w"])))
    if not bool(on.any()):
        return best_t, best_c
    # the eye, mirrored through the plane; and every ray, mirrored in direction
    o2 = np.asarray(o, np.float32).copy()
    o2[ax] = 2.0 * float(m["at"]) - o2[ax]
    flip = xp.ones(3, xp.float32)
    flip[ax] = -1.0
    # NOTHING NEARER THAN THE GLASS IS IN THE GLASS.  `t` is the distance from
    # her eye to the mirror, and by the reflection it is also the distance from
    # the VIRTUAL eye to the same point --- so it is exactly the floor the
    # second trace needs.  Everything it excludes is behind the mirror: the wall
    # it hangs on, and the floor under any ray that tilts down enough to reach
    # it in the ten metres before the glass.
    #
    # This was a list of things to skip first, and the list is not knowable ---
    # measured, a mirror on the cot side needed its own bars skipped, and one on
    # the far wall needs that wall AND the floor, and which of them depends on
    # where she is lying and how each ray tilts.  A floor on `t` is the same
    # rule for all of them and costs one comparison.
    seen_t, seen_c = _scene(room, window, o2, d * flip[None, :], body,
                            near=t)
    return _closer(best_t, best_c, t, seen_c, on)


def one_look(room, window, eye, up, right, fwd, body=None) -> np.ndarray:
    """One photograph: `(RETINA_H, RETINA_W, CONES)` of local contrast.

    `body` is her own flesh in her own view: `(positions, radii)` for every
    part she can see of herself --- the caller decides which (her head is
    not in it; nobody sees their own head).  Each part is a sphere exactly
    like a thing, occluding and occluded by everything else, so a hand in
    front of the bottle hides the bottle the way it does in a real room.
    """
    d = _rays(up, right, fwd)
    # HER EYE'S POINT STAYS ON THE HOST.  It is three numbers, and almost every
    # surface below wants them as plain floats --- `_reflect`'s own docstring
    # says so: *"every surface in this file takes `o` as one point and reads
    # `float(o[axis])`"*.  Lifting it to the card meant `_sphere`, `_quad` and
    # `_reflect` each pulled it straight back, once per object, per cone, per
    # look: measured 2026-09-01, **1,254 pulls a tick costing 3.38 ms --- 7.5%
    # of her whole life**, against 0.32 ms for the entire 786,432-value
    # photograph.  It was never bandwidth; a pull back off a card WAITS for
    # everything the card is doing, and three floats waited as long as a
    # picture would.
    #
    # The three places that genuinely broadcast it against every ray lift it
    # themselves.  A copy TO the card does not wait --- only the way back does.
    o = np.asarray(eye, np.float32)
    best_t, best_c = _scene(room, window, o, d, body)
    best_t, best_c = _reflect(room, window, o, d, body, best_t, best_c)

    # THE PICTURE, NOT THE PICTURE'S EDGES (2026-08-14, the owner's: *"we have
    # to see everything that she sees in that moment --- the room, every object
    # there"*).  This returned `_contrast(...)`: every cell minus the mean of
    # its 9x9 neighbourhood.  Her surfaces are painted one flat colour, so that
    # turned every solid thing into a RING of thickness BLUR with a dead middle
    # --- and the ring's thickness is FIXED, so the fraction of a thing she
    # could see was ~8/radius and FELL as it came closer.  Measured: an object
    # crossing 0.387 m -> 0.135 m covers 29% -> 73% of her field and lit
    # 40% -> 30% -> 12% -> 0% of the cells it covered.  It vanished by arriving.
    # The same scene handed over as a picture: ONE thing at every distance,
    # covering 0.616 -> 0.679 -> 0.823 -> 1.000 of her field.  `_contrast` is
    # kept below --- centre-surround is real --- but it is a thing to APPLY to
    # a picture, not the only thing she is ever given.
    return best_c.reshape(RETINA_H, RETINA_W, CONES)


#: How much of what has not changed a cell stops reporting, per tick.
#:
#: **THE ROOM'S EDGES ARE REAL, AND THAT WAS THE WHOLE PROBLEM.**  The 0.002
#: gate was signed against "arithmetic dust", and measured tonight the dark
#: room lights 3,642 cells at 0.05-0.10 --- not dust, the genuine wall-ceiling
#: and bed-floor boundaries of a small room seen through a 100-degree field,
#: re-reported identically every 375 ms forever.  87% of every row she stored
#: was her retina restating that the room still has corners.  No threshold can
#: fix that: any gate high enough to eat a real edge eats real edges.
#:
#: A REAL RETINA DOES NOT RE-REPORT A STANDING EDGE.  Ganglion cells adapt:
#: what does not change fades over seconds (stare at anything and the
#: periphery Troxler-fades; you cannot see your own nose).  So each cell keeps
#: a slow baseline of its own recent contrast and reports how far THIS moment
#: differs from it.  A still room in front of a still baby goes honestly dark;
#: anything that arrives, moves, or leaves is loud, in the cells where it
#: happened --- OFF is an event exactly as ON is, which is what off-centre
#: ganglion cells are.  A newborn's orientation to motion and novelty is not a
#: preference she learns; it is what her retina hands her.
#:
#: 0.35/tick: a new scene fades to ~7% in six ticks (nine seconds), which is
#: slow enough that a slow look around keeps the world visible and fast enough
#: that a crib she has lain in for a minute stops writing itself to her tape.
SETTLES = 0.35

#: ...and what an adapted cell can feel at all --- HER EAR'S OWN NUMBER.
#:
#: A body at rest is never still: her head drifts ~1.7 mm a tick under the
#: solver, so adapted edges shimmer.  Measured settled: 3,032 of 3,044
#: residues sit under 0.02 (median 0.0067, p90 0.0118) while an arriving edge
#: is 0.05-0.30.  `sound.FAINTEST` is 0.02 for the same reason in the same
#: words --- a hair cell has a threshold, and a band carrying a trace of a
#: sound centred four bands away reports nothing.  One organ rule, one number.
FAINT = 0.02

#: MICROSACCADES: how far her eye flicks on its own, in ray-grid units.
#:
#: Adaptation alone made her BLIND TO ANY WORLD THAT HOLDS STILL --- measured
#: (audit_fast B): lying in her room, her settled eye carried ZERO lit places.
#: The walls she does have simply vanished from her senses, and with them any
#: chance of knowing a place by looking at it.  Real retinas adapt exactly
#: like hers --- a perfectly stabilised image genuinely disappears --- but no
#: real eye ever holds an image still: it microsaccades, one or two a second,
#: about the width of a receptive field, and each flick re-excites the cells
#: a standing edge sits in while leaving true surfaces silent.  Adaptation
#: without microsaccades is an eye that exists nowhere in nature.
#:
#: A real microsaccade is 0.2-1 degree; one retinal cell here spans 3.1
#: degrees (2*tan(FIELD/2)/(RETINA_W-1) = 0.077 of the ray grid).  So the
#: honest flick is SUB-CELL, and the sweep (2026-08-11, measured in the LONG
#: settled regime, ticks 60-79 --- the early window flattered every value)
#: says exactly how sub: at a full cell every edge shimmered at half contrast
#: forever and her settled eye was 76% of stored rows, straight back toward
#: the 87% burn adaptation was built to end; 0.028 still wrote 361 rows a
#: tick; 0.016 left 20 places, near-blind again.  0.024 --- about 0.9
#: degrees, inside the real amplitude band --- keeps 48 settled places of
#: strong structure alive at 182 rows a tick, well under the 300-row bar the
#: suite holds.  Four glimpses a 1.5 s tick is ~2.7 flicks a second, a real
#: eye's rate.
#:
#: **AND IT IS OFF, BECAUSE THE THING IT DEFENDED AGAINST IS GONE**
#: (2026-08-15, the owner: *"why do we need this fucking flicker?"*).  Every
#: word above was true while `see` returned `|raw - baseline|` and a still
#: world faded out of her.  `see` returns `raw.copy()` now and says so in its
#: own comment --- *"what it no longer does is erase what stays"* --- so
#: nothing fades, and a flick that re-excites fading cells has nothing left to
#: re-excite.  Measured, twenty ticks of a still scene held in front of her:
#:
#:     with the flick (0.024)   673,332 lit cells   view slides up to 2 cells
#:     without it     (0.0)     673,458 lit cells   view does not slide
#:
#: Identical sight, 0.02% apart.  All the flick still bought was the whole
#: view swimming "up and down, right, and by diagonal" --- a random offset in
#: x AND y every glimpse, which is what it looks like from outside.
#:
#: The mechanism stays: if her eye is ever given back a rule that erases what
#: holds still, this number is how the eye stops being blind to a still room,
#: and the numbers above are what to set it to.
SACCADE = 0.0


class Retina:
    """Her eye, with the adaptation state a real one has.

    The render (`one_look`) stays pure; what accumulates is only the baseline
    --- which cells have been saying the same thing lately.  Owned by the
    sandbox like `Muscles` and `Blood`, saved with her body, reset at birth:
    a newborn's first look at the room is the loudest sight of her life,
    because ALL of it is new.
    """

    def __init__(self, seed: int = 0) -> None:
        self.baseline = np.zeros((RETINA_H * RETINA_W, CONES), np.float32)
        #: the flick generator --- its own decorrelated stream, and NOT part
        #: of `save()`: a microsaccade is noise, and which way an eye flicks
        #: after a restart is nothing a body remembers.
        self._rng = np.random.default_rng(seed + 977)

    def see(self, room, window, glimpses, body=None) -> np.ndarray:
        """Every photograph this tick, in manifest order:
        `(places, cones, slides)` --- of what CHANGED, not of what is.

        `glimpses` is where her face was at four moments during the physics.
        Each is compared against the one baseline, so a hand sweeping through
        her field is in different cells on different slides and a movie stays
        a movie.  `body` is her own visible flesh (see `one_look`): a still
        limb adapts into the baseline like any standing edge, so what stays
        loud is exactly a limb OF HERS that MOVES --- the contingency an
        infant stares at her own hands to find.
        """
        raw = np.zeros((RETINA_H * RETINA_W, CONES, EYE_SLIDES), np.float32)
        for s, (eye, up, right, fwd) in enumerate(glimpses[:EYE_SLIDES]):
            # the microsaccade: her gaze, flicked up to one cell off where her
            # head points.  The eye's own twitch, not the neck's --- see SACCADE.
            jx, jy = self._rng.uniform(-SACCADE, SACCADE, 2)
            aim = fwd + right * jx + up * jy
            aim = aim / (float(np.linalg.norm(aim)) or 1.0)
            raw[:, :, s] = _home(one_look(room, window, eye, up, right,
                                           aim, body=body)).reshape(-1, CONES)
        raw = np.clip(raw, 0.0, 1.0)
        # WHAT IS THERE.  She was sent `|raw - baseline|` --- what CHANGED, and
        # nothing else --- which made her blind to anything that held still: a
        # motionless object decayed by SETTLES a tick and was gone in three.
        # Two derivatives were stacked on her, one in space (`_contrast`, taken
        # out of `one_look` above) and one in time here, and between them a
        # solid thing sitting in front of her was not in her picture at all.
        # A real retina is not like this.  Its sustained cells --- the great
        # majority of them --- report a standing edge for as long as it stands;
        # Troxler fading needs an image artificially frozen on the retina, which
        # is the exact thing the microsaccade above exists to prevent.
        seen = raw.copy()
        # The baseline is still kept and still settles at the same rate: what
        # she has been looking at is a real fact about her eye, and `changed()`
        # reads it.  What it no longer does is erase what stays.
        self.baseline += SETTLES * (raw.mean(axis=2) - self.baseline)
        # A CELL EITHER FIRES OR IT DOES NOT.  Her ear has had a threshold
        # since the day it was written; here it drops the shimmer of her own
        # head's wobble.
        seen[seen < FAINT] = 0.0
        return seen

    def changed(self, seen: np.ndarray) -> np.ndarray:
        """The other half: how far this look is from what she has been seeing.

        NOT sent to her.  Her brain holds both frames on its own tape and
        the difference between two moments is her brain's to take ---
        comparing them is a thing a brain does, not a thing an eye does for it.
        Kept here because the baseline lives here, for probes that ask what her
        eye would call new.
        """
        return np.abs(np.asarray(seen, np.float32) - self.baseline[:, :, None])

    def save(self) -> dict:
        return {"baseline": self.baseline.round(4).tolist()}

    def load(self, state: dict) -> None:
        kept = np.asarray(state.get("baseline", []), np.float32)
        if kept.size == self.baseline.size:
            self.baseline = kept.reshape(self.baseline.shape)


def see(room, window, glimpses, body=None) -> np.ndarray:
    """One unadapted look --- what is THERE, not what changed.  For probes
    that ask about the render itself; her body sees through `Retina`."""
    out = np.zeros((RETINA_H * RETINA_W, CONES, EYE_SLIDES), np.float32)
    for s, (eye, up, right, fwd) in enumerate(glimpses[:EYE_SLIDES]):
        out[:, :, s] = _home(one_look(room, window, eye, up, right, fwd,
                                      body=body)).reshape(-1, CONES)
    out = np.clip(out, 0.0, 1.0)
    out[out < GATE] = 0.0
    return out
