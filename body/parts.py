"""Her eye is A PICTURE.  Not lines, not rows, not a count of anything.

================================================================================
**READ THIS FIRST, AND STOP MEASURING HER SIGHT IN LINES.**  The owner, 2026-08-
15: *"why do we still speak about pixels and huge row quantities in the db --- for
us it is an image that comes from her eyes, already ready for the brain."*  And:
*"remove this wrong meaning about pixels and lines, to avoid repeating the same
mistake every time."*

A TICK OF SEEING IS ONE VIEW.  It has exactly two parts:

    the file            one compressed picture, the shape a camera makes
    the representations one record per thing in it: where, how big, what colour

That is the whole model.  Anything phrased as "how many of her lines fired",
"how sparse is her retina", "how many rows a tick does her eye cost" is asking
about a shape she does not have, and every one of those questions has cost this
project a day.  The list, so it is not repeated:

  - Her eye was made to report EDGES so the row count would stay small.  It
    made her blind: a solid thing came out as a ring of fixed thickness with a
    dead middle, and the fraction of it she could see FELL as it came closer,
    reaching zero as it filled her field.
  - Her retina was made to report only CHANGE, for the same reason.  Anything
    that held still was gone in three ticks.
  - A tick was made FOUR photographs, quadrupling the lines, and nothing ever
    consumed the four.
  - The same lines were then written to `lived` a second time --- 100 million
    rows for a 2,000-tick life --- beside the picture that already held them.

Every one of those was a decision made to protect a row count.  The row count
was never the thing.  **The picture is small: 703 bytes a view, measured on a
real life. Storage is not the constraint and never was.**
================================================================================

A frame is `uint8` height x width x 3.  So is a frame off a webcam.  Storing
hers in that exact shape means a real camera feed is not converted into anything
to reach her --- it IS the picture, and everything downstream of here already
works on it.

**AND DELIBERATELY IN BAD QUALITY.**  float32 -> uint8 is 4x off before any
compression and is all the fidelity a 32x32 retina carries.  zlib on top, not
JPEG: at this size JPEG's fixed header is about a fifth of the whole raw image,
so it would cost a dependency to save almost nothing.  Swapping it in later is
one function if her retina ever gets big enough for it to pay.

**THE GRAINS ARE CUT ON READ, NOT ON WRITE.**  Halving a stored frame is a
numpy reshape and a mean --- microseconds --- so keeping every grain on the tape
would be storing what is free to recompute.  One frame in, any grain out.

================================================================================
**WHY THE PICTURE IS KEPT AT ALL --- READ THIS BEFORE CHANGING ANY OF IT.**
The owner, 2026-08-15, because it keeps being forgotten:

    *"I just don't want to forget what we need to store a screenshot of her view
    for --- compressed, and then be able to RESTORE it to investigate it again,
    in case we already have some similarity pattern to focus on again."*

The object list (`find`, the `things` table) is an INDEX.  The picture is the
EVIDENCE.  They are not two copies of one thing and the picture is not a
convenience.

Here is the sequence the whole store exists to serve.  Something happens now.
She looks back and finds an earlier moment that resembles it --- and the match is
made on the summary, because a summary is what an index can be searched by.  But
a summary is only ever as good as the question that was being asked when it was
written, and the question she has NOW was not being asked then.  So having found
the moment, she must be able to **open the actual picture again and look harder
at it**: at the part of the field the match pointed to, for the detail no object
row ever carried.

A stored summary cannot answer a question that had not been thought of when it
was stored.  A stored picture can.  That is the entire argument, and it is why
`forget` keeps six hours of real frames and not a longer index --- for six hours
a moment can still be RE-EXAMINED, and after that only its structure survives.

Anything here that makes the picture cheaper but unopenable has destroyed the
reason it is stored.
================================================================================
"""

from __future__ import annotations

import os

import zlib

import numpy as np

#: ONE ANSWER TO "HOW ALIKE", and it is the same one her hearing uses.  Her
#: sound index quantises centre, spread and peak at `ALIKE_STEPS` each; her
#: sight quantises colour direction and proportion the same way.  It left the
#: contract because it is a fact about how finely she tells things apart, not a
#: fingerprint anyone signs --- and it lives beside `alike_of` so there is
#: exactly one of it.
from .alike import ALIKE_STEPS as _ALIKE_STEPS

#: The grains a reader can ask for, coarsest last.  Not a claim about her ---
#: every halving there is between her retina and a single number.  Picking ONE
#: block size would be a claim about how big a thing is, before she has met any
#: things.
LEVELS: tuple[int, ...] = (16, 8, 4, 2, 1)


def shot(eye: np.ndarray) -> tuple[np.ndarray, float]:
    """One tick of her eye as a picture: (H, W, bands) uint8, and its gain.

    The four slides are averaged first --- they are four photos of one tick, and
    a thing in the world is not a thing in one exposure.

    **EXPOSED PER FRAME, BECAUSE HER EYE IS DARK.**  Her retina reports CHANGE
    against an adapting baseline (`light.Retina.see`), not brightness, and
    measured on a real life its firing cells sit at median 0.029 and max 0.15.
    Scaling 0..1 onto 0..255 therefore used **35 of 256 levels** and threw away
    3% of her median signal for nothing.  Each frame is scaled by its own
    brightest cell instead, and the gain is stored beside it, so the picture
    keeps its full range and `unpack` returns exactly what she saw.  This is
    what a camera's auto-exposure is, and for the same reason.
    """
    frame = np.asarray(eye, np.float32)
    if frame.ndim == 4:
        frame = frame.mean(axis=3)
    gain = float(np.max(frame)) if frame.size else 0.0
    if gain <= 0.0:
        return np.zeros(frame.shape, np.uint8), 0.0
    return np.clip(frame / gain * 255.0 + 0.5, 0, 255).astype(np.uint8), gain


def pack(frame: np.ndarray) -> bytes:
    """The screenshot, small.  Plain bytes --- no header, no format, because the
    shape is a column on the same row.

    **LEVEL 6, AND NOW IT IS MEASURED** (2026-08-15).  It had been 6 since the
    day this was written, chosen by nobody, and the owner asked for the best
    FAST way.  Measured on 40 real frames of her real life, 12,288 bytes raw:

        zlib 0     12,299 b   100.1%   0.007 ms pack   0.001 ms open
        zlib 1      1,191 b     9.7%   0.013           0.007
        zlib 3        648 b     5.3%   0.032           0.008
        zlib 6        501 b     4.1%   0.063           0.005   <-- this
        zlib 9        480 b     3.9%   0.256           0.005
        lzma p1       602 b     4.9%   0.604           0.048
        bz2 1         434 b     3.5%   0.222           0.045

    Level 9 buys 4% of the size for 4x the time. bz2 is the smallest and is 9x
    slower to OPEN, which is the number that actually matters here: the whole
    reason a picture is kept is that she must be able to re-open it later and
    look harder at it, so opening has to be cheap and packing only has to be
    affordable.  At level 6 a re-opened moment costs **5 microseconds** and a day
    of her life is 16 MB.
    """
    return zlib.compress(np.ascontiguousarray(frame, np.uint8).tobytes(), 6)


def unpack(blob, h: int, w: int, bands: int, gain: float = 1.0) -> np.ndarray:
    """The picture back, in the units she saw it in."""
    raw = np.frombuffer(zlib.decompress(bytes(blob)), np.uint8).reshape(h, w, bands)
    return raw.astype(np.float32) * (float(gain) / 255.0)


def blocks(frame: np.ndarray, side: int) -> np.ndarray:
    """The same picture at a coarser grain: (side, side, bands).

    Vectorised, so asking for a grain costs nothing worth storing to avoid.
    """
    f = np.asarray(frame, np.float32)
    h, w = f.shape[0], f.shape[1]
    if side > h or side > w or h % side or w % side:
        raise ValueError(f"{h}x{w} does not divide into {side}x{side}")
    return f.reshape(side, h // side, side, w // side, f.shape[2]).mean(axis=(1, 3))






#: HOW BIG A PIECE GETS BEFORE IT STARTS REFUSING TO GROW, per cell of the
#: frame.  Not a colour bar --- see `find` for why there cannot be one.  It is
#: per-cell because the rule below weighs `scale / size` with `size` counted in
#: CELLS, so a frame with four times the cells needs four times the number to
#: be the same rule; leaving it fixed is a units mistake that reads exactly
#: like a bug (measured: 22 object-sized pieces became 7, and one swallowed 72%
#: of her view).
#:
#: **FOUR TIMES WHAT IT WAS, BECAUSE THE CAMERA GOT SHARP** (2026-08-15).  It
#: was calibrated against a 320-wide camera packed to 256 texels and stretched
#: over her sheet --- a soft picture, where a person was already nearly one
#: colour.  Asked for 1280x720 and packed at 512, his skin arrives with its own
#: texture, shading and hair in it, and the rule split him into a dozen 1-2%
#: fragments: 393 pieces, the biggest 4.87% of her view, and not one of them
#: him.  A sharper picture has bigger real differences in it, so the size a
#: piece must reach before it starts refusing has to grow with it.  Swept on
#: the frame he was standing in:
#:
#:     x1    282 pieces   biggest 4.80%   biggest in his half 22x41% of view
#:     x4    126 pieces   biggest 17.6%   29x85%   <- him, one piece
#:     x16    53 pieces   biggest 21.5%   29x85%
#:     x64    26 pieces   biggest 41.6%   63x100%  <- he takes the room with him
#:
#: **TWENTY-EIGHT TIMES AGAIN, BECAUSE THE SCREEN CAME TO HER FACE**
#: (2026-08-17, the owner: *"why 254 if we have better resolution?"*).  The 4x
#: above was calibrated when his screen was a small rectangle across her room.
#: It now flies 0.30 m off her face and `covers` HALF HER RETINA, so the same
#: sharpness arrives at four times the size and the rule shattered it again ---
#: 254 pieces on the panel with him on the screen against 23 on a plain
#: ceiling.  **His face made her vision worse, not better.**
#:
#: Swept on HER OWN LIVE RETINA (30 frames pulled off a running body's `/eye`
#: and put back in her levels with `parts.unpack`; `find` on the
#: reconstruction returns her body's own `found` to within 2%), and on a
#: `Sandbox` with a real 1280x720 photograph of a person on the screen, her
#: verified in her cot.  Median over frames, "him" from `measure.ruler`:
#:
#:     x     SCALE      live    rest    him: one piece   its spill   screen spill
#:     x1    9.2e-5      156      21      0 of 17 looks       ---          10.9%
#:     x4    3.7e-4       79      21      0 of 17            ---            4.7%
#:     x16   1.5e-3       46      21      2 of 17           12.5%           2.8%
#:     x24   2.2e-3       34      21      8 of 17           14.7%           2.9%
#:     x28   2.6e-3       31      20      9 of 17           14.7%           3.7%  <- here
#:     x32   2.9e-3       28      20     14 of 17           22.5%          20.9%
#:     x64   5.9e-3       20      19     17 of 17           32.9%          30.3%  <- room
#:
#: **AT REST NOTHING MOVES AT ALL**: her biggest piece is 18.57% of her view at
#: every value from x1 to x64, because her room's own floor/wall/ceiling joins
#: are edges no size preference reaches.  The old "he takes the room with him"
#: failure is now a SPILL, not a count: past x28 the lump around the screen
#: starts taking the wall behind it (3.7% -> 20.9% of the lump lying off the
#: screen), and that is the ceiling on this constant, not her room falling in.
#:
#: What it bought, on her live view with his camera on the screen: 156 -> 31
#: pieces, the biggest 11.3% -> 33.7%, and `measure.screen`'s boundary score
#: went the right way --- the top box lay on the screen 33% of its area before
#: and **2% after**, landing on HER 36% -> 85%.  `find` also got cheaper,
#: because the shape loop is per piece.
SCALE = 2.6e-3
#: Dust: a piece smaller than this share of the frame is not a thing, and joins
#: whichever neighbour it most resembles.
#:
#: NOT MOVED WITH `SCALE`, and measured rather than assumed (2026-08-17): a
#: bigger size preference removes the dust by itself.  On one of her real
#: frames the pieces under 0.5% of her view went 43 to 6 and the share of her
#: view they held went 5.6% to 0.9%.  There was nothing left for `LEAST` to do.
LEAST = 4.6e-4
#: HOW MUCH BRIGHTNESS COUNTS, and why she has it back.
#:
#: This file used to divide brightness out entirely and wrote down the price:
#: *"a thing that differs from what is behind it ONLY in brightness is not
#: separated --- her bottle, at (0.94, 0.93, 0.88) against a wall at (0.62,
#: 0.60, 0.58), is 0.011 away in colour and would merge with the wall.  That is
#: a genuine hole, not a solved problem."*  It was divided out because adding a
#: relative brightness term put a checkerboard's 8.5% step into the same
#: population as an object's edge and the count went 5 things a tick to 768.
#:
#: **That was the fixed bar failing, not brightness.**  A face against a wall
#: differs mostly in brightness --- which is why, measured 2026-08-15 with the
#: owner on her screen, she could not pick him out of the wall behind him at
#: all.  Under the comparative rule the term is safe: a shading gradient earns
#: tolerance from its own varied interior, a sharp step at the edge of a face
#: does not.  At 0.0 his face and the wall are one piece; at 3.0 his face, his
#: torso, the doorway, the couch, the cushion and the picture on the wall are
#: each their own thing.
SHADE = 3.0
#: Work at about this many cells a side.  Her retina is foveal --- the rim is
#: crushed and reading it back upsamples it --- so beyond this much of a sheet
#: is interpolation rather than anything she resolved.  Measured on her real
#: view, with the walk compiled: 128 costs 63 ms against 256's 181 and finds
#: MORE of an object's size (49 against 45), because the rim it drops was
#: interpolation that only ever fragmented.  Below this it starts losing them
#: again (90 -> 45).
GRAIN = 128


try:                                    # same chip rule as body/light.py:
    import cupy as _gpu                 # one implementation, and the arrays
    _gpu.zeros(1)                       # say where it runs
except Exception:                       # noqa: BLE001 - any missing CUDA
    _gpu = None

#: THE WALK, COMPILED --- and deliberately on ONE thread.
#:
#: Joining pieces cheapest-pair-first is sequential and cannot honestly be made
#: otherwise: what a pair is measured against is `scale / size`, and size is
#: whatever the earlier pairs already built.  Letting a round of pairs merge at
#: once was tried and measured --- a chain joins three pieces on a test that
#: weighed two, and since a lone cell's threshold is `scale` itself, ROUND ONE
#: swallows the frame: 15 pieces, one covering 49% of her view, agreeing with
#: the true answer on 23% of cells.  Restricted to pieces that chose each other
#: it stops over-merging and under-merges instead.  There is no parallel form
#: of this rule that is this rule.
#:
#: So the kernel does not parallelise it --- it COMPILES it.  105,000 pointer
#: chases are perhaps 20 real instructions each; in the Python interpreter that
#: is a quarter of a second, and it is interpreter overhead, not work.  One GPU
#: thread is slower than one CPU core and still turns that into milliseconds,
#: and the answer is the SAME answer, cell for cell, because the arithmetic is
#: the same arithmetic: the threshold is compared in double exactly as Python
#: promotes it, and `inside` is stored back as float32 exactly as the array is.
_WALK = _gpu.RawKernel(r"""
extern "C" __global__ void walk(
        const int* u, const int* v, const float* gap, const int m,
        int* parent, int* size, float* inside,
        const double scale, const int least) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < m; ++i) {
        int a = u[i];
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        int b = v[i];
        while (parent[b] != b) { parent[b] = parent[parent[b]]; b = parent[b]; }
        if (a == b) continue;
        double d = (double)gap[i];
        double ta = (double)inside[a] + scale / (double)size[a];
        double tb = (double)inside[b] + scale / (double)size[b];
        if (d <= (ta < tb ? ta : tb)) {
            if (size[a] < size[b]) { int t = a; a = b; b = t; }
            parent[b] = a; size[a] += size[b]; inside[a] = (float)d;
        }
    }
    for (int i = 0; i < m; ++i) {          // dust is not a thing
        int a = u[i];
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        int b = v[i];
        while (parent[b] != b) { parent[b] = parent[parent[b]]; b = parent[b]; }
        if (a != b && (size[a] < least || size[b] < least)) {
            if (size[a] < size[b]) { int t = a; a = b; b = t; }
            parent[b] = a; size[a] += size[b];
        }
    }
}
""", "walk") if _gpu is not None else None


def _settle(parent: np.ndarray) -> np.ndarray:
    """Every cell's final piece, by jumping pointers until nothing moves.

    The same answer as asking `root` for each cell in turn, which was 65,536
    Python calls for one frame.
    """
    while True:
        nxt = parent[parent]
        if bool((nxt == parent).all()):
            return parent
        parent = nxt


#: the largest label there is, so a neighbour that is NOT joined can lose every
#: minimum without a mask of its own
_NEVER = np.iinfo(np.int32).max

#: HER KNITTING IN C, IF IT IS BUILT --- `body/native/knit.c`, which says why.
#: A sweep in numpy is O(passes) whole-array passes; a union-find in C is one
#: pass over the edges.  Loaded with `ctypes`, so no Python header and no ABI
#: to break, and **if it is missing or it fails, the numpy below runs exactly
#: as it always has**.  It is not a second implementation of her grouping: it
#: is the same partition, and `measure/knit.py` holds the two against each
#: other on 28 masks and on frames from her own life.
#:
#:     cd body/native && cl /O2 /LD knit.c /Fe:knit.dll
_INC = None
try:                                                        # pragma: no cover
    import ctypes as _ct

    _lib = _ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "native", "knit.dll"))
    _lib.knit.restype = _ct.c_int
    _lib.knit.argtypes = [
        _ct.c_int, _ct.c_int,
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")]
    _INC = _lib.knit
    _lib.walk.restype = None
    _lib.walk.argtypes = [
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        _ct.c_int,
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        _ct.c_double, _ct.c_int]
    _INC_WALK = _lib.walk
except Exception:                                           # noqa: BLE001
    _INC = None
    _INC_WALK = None


def _knit(h: int, w: int, down: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Join the neighbours that need no test, with no Python loop and no
    scatter.  `down` and `right` say which vertical and horizontal pairs are
    joined; the answer is every cell carrying the LOWEST index it can reach.

    Every cell takes the lowest label it can reach, repeatedly, until nothing
    changes.  Used for the neighbours that are EXACTLY equal --- 87% of a real
    frame, which the rule in `find` would join unconditionally anyway --- so
    the one-pair-at-a-time walk only ever sees pairs where the question is
    real.  Measured: 1,676 ms to 787 ms, with the grouping unchanged.

    **AND THEN IT SPENT HALF OF HER LOOKING IN `np.minimum.at`.**  Measured
    2026-09-01 on a frame out of her real life: `_knit` **29.2 ms, 51% of the
    whole of `find`** --- almost all of it in numpy's scatter, which walks an
    index list one element at a time and is the slowest primitive it has.

    It took a list of pairs, and the pairs were never a list: **they are a
    GRID**.  Every cell's neighbours are the cells beside it, so the whole
    scatter is four whole-array minimums against a shifted copy, masked where
    the pair is joined.  **29.2 ms to 15.6 ms, labels identical cell for
    cell**, and the pieces `find` returns identical value for value on both a
    reference room and a live frame.

    The card was tried first and lost twice: the exact algorithm on the GPU is
    **0.6x** (its convergence questions stall it every pass), and a fixed pass
    count that avoids the stalls got **1,042 of 16,384 cells wrong** on a real
    frame because it had not converged.  Her grouping is not a place to trade
    correctness for a millisecond.
    """
    if _INC is not None and h > 0 and w > 0:
        got = np.empty(h * w, np.int32)
        if _INC(h, w,
                np.ascontiguousarray(down, np.uint8),
                np.ascontiguousarray(right, np.uint8), got) == 0:
            return got

    lab = np.arange(h * w, dtype=np.int32).reshape(h, w)
    for _ in range(64):
        was = lab
        low = lab.copy()
        np.minimum(low[:-1], np.where(down, lab[1:], _NEVER), out=low[:-1])
        np.minimum(low[1:], np.where(down, lab[:-1], _NEVER), out=low[1:])
        np.minimum(low[:, :-1], np.where(right, lab[:, 1:], _NEVER),
                   out=low[:, :-1])
        np.minimum(low[:, 1:], np.where(right, lab[:, :-1], _NEVER),
                   out=low[:, 1:])
        line = low.reshape(-1)
        lab = line[line]
        while True:
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt
        lab = lab.reshape(h, w)
        if np.array_equal(lab, was):
            break
    return lab.reshape(-1)


#: How coarse the shape kept for a thing is, per side.  4x4 = 16 numbers.
#:
#: It is the thing's OWN mask, cropped to its own box and squashed to this ---
#: so it says what shape it is and not where it was or how big, which is the
#: only way "the same thing again, nearer" can match "the same thing, further".
SHAPE_SIDE = 4

#: How many buckets each part of the similarity index gets.  Colour is a
#: direction (two numbers once brightness is out), and proportion is one.
ALIKE_STEPS = _ALIKE_STEPS       # one copy, and it lives in the manifest
                                 # now that it is a territory's shape

#: A representation is these, in this order.  `parts.WHERE`..`parts.ALIKE` name
#: the columns so nothing downstream counts on its fingers.
WHERE, SIZE, COLOUR = slice(0, 2), slice(2, 5), slice(5, 8)
SHAPE = slice(8, 8 + SHAPE_SIDE * SHAPE_SIDE)
ALIKE = 8 + SHAPE_SIDE * SHAPE_SIDE
COLUMNS = ALIKE + 1


def find(frame: np.ndarray) -> np.ndarray:
    """The things in one frame --- one ROW EACH, `(n, 25)`:

        0..1    x, y        where it is in her field, -1..1
        2       area        how much of the view it covers
        3..4    w, h        its dimensions
        5..7    r, g, b     its colours
        8..23   shape       its own outline, 4x4, cropped and scaled
        24      alike       the SIMILARITY INDEX --- things that look alike
                            share it, so "have I seen something like this?"
                            is one indexed lookup and not a scan

    The owner's shape, asked for repeatedly and written down here so it stops
    being re-derived: *"object view: {id, file, representations{objId, colors,
    dimensions, similarity index, and the rest we need and are able to retrieve
    from pic}}"*.  `id` and `file` are the view (`tape.views`); this is the
    representations.

    Connected cells that stand out together, which is the only sense in which
    she has "an object": something that hangs together in her field.  `x` and
    `y` are where it sits in her viewport, -1..1 --- HER coordinates, not the
    room's, because where a thing is in the room is exactly what her brain is
    not allowed to be told.  `area` is how much of her field it covers, so the
    SAME thing coming closer grows, and that is distance without a sensor for
    distance.

    **JOINED BY COMPARISON, NOT BY A BAR** (2026-08-15, the owner's: *"it don't
    recognize nothing... when I hold beer can in front of camera it didn't
    border it at all"*).  This used to ask one question of every neighbouring
    pair --- are you closer in colour than `tol`, where `tol` was Otsu's split
    of the frame's own neighbour distances.  Measured on her real view with his
    face and his room on her screen, no `tol` exists that works:

        tol      pieces   biggest piece   pieces of an object's size (1%..40%)
        0.005    13156        40.9%                    2
        0.010     5515        74.8%                    2
        0.020     1106        94.7%                    1     <- what Otsu chose
        0.050       23       100.0%                    0
        0.120        1       100.0%                    0

    At her own setting ONE piece covered 79.2% of her view with a box spanning
    the whole frame, the median piece was 23 millionths of her view, and TWO
    pieces were an object's size: his head, the doorway and the couch were all
    inside the blob, and her marks were a shoulder edge and a cushion.  Tighten
    the bar and a shaded wall shatters into thirteen thousand specks while a
    41% blob survives anyway; loosen it and everything is one thing.  A fixed
    bar cannot know that a slow shading gradient across a wall is ONE surface
    while a smaller step at the edge of a can is TWO things.

    So the question is comparative (Felzenszwalb & Huttenlocher 2004 --- hand
    built, no training, no net, and its own theorem is that the result is
    provably neither too fine nor too coarse):

        join two pieces only if what separates them is small compared to the
        variation ALREADY INSIDE each of them.

    A gradient tolerates its own next step because its inside is already that
    varied.  A flat can refuses the wall behind it because its inside is flat,
    so any real edge is large by comparison.  `SCALE` is a preference for how
    big a piece should get before it starts refusing --- a size, not a colour,
    which is why one value spans scenes no single threshold could.  Same frame,
    same instant: 200 pieces, biggest 5.7%, **45 of an object's size**, at 380
    ms against the bar's 330 --- his face, his torso, the doorway, the couch,
    the cushion and the picture on the wall, each its own thing.

    Nothing is discarded as background --- the wall is a thing in her view too,
    and a big dull one is exactly what a wall should look like to her.

    Four-connected.  Cheapest pair first, so the walk is one sort and a
    union-find --- which is the shape `find`'s kernel is built from.
    """
    f = np.asarray(frame, np.float32)
    if f.shape[0] < 2 or f.shape[1] < 2:
        return np.zeros((0, COLUMNS), np.float32)
    # AT HER GRAIN, NOT HER SHEET'S.  See `GRAIN`; the size terms come with it
    # or the rule silently changes.
    step = max(1, min(f.shape[0], f.shape[1]) // GRAIN)
    if step > 1:
        f = f[::step, ::step]
    h, w = f.shape[0], f.shape[1]
    bands = f.shape[2]
    n = h * w
    scale = SCALE * n
    least = max(1, int(LEAST * n))

    # HOW DIFFERENT TWO NEIGHBOURS ARE, IN THE TWO WAYS THAT MATTER.
    #
    # In COLOUR: each cell as a DIRECTION, with how brightly it happens to be
    # lit divided out.  Her walls and floor carry a checkerboard
    # (`light.STRIPE`/`TONE`) which is the same paint at two brightnesses, and
    # her room is shaded besides; comparing raw colour cut every surface up
    # along its own light and shade and returned 716 "things" in one tick, not
    # one of them an object.  A shadow across a wall does not make two walls,
    # and that is not a convenience --- it is what colour constancy is for.
    #
    # AND IN BRIGHTNESS, weighted by `SHADE`, which is where the hole in this
    # file used to be --- see `SHADE` for what it cost her and why the
    # comparative rule can carry it when the bar could not.
    unit = f / np.maximum(np.sqrt((f ** 2).sum(axis=2, keepdims=True)), 1e-6)
    down = np.sqrt(((unit[1:] - unit[:-1]) ** 2).sum(axis=2))
    right = np.sqrt(((unit[:, 1:] - unit[:, :-1]) ** 2).sum(axis=2))
    if SHADE > 0.0:
        lum = f.mean(axis=2)
        down = down + SHADE * np.abs(lum[1:] - lum[:-1])
        right = right + SHADE * np.abs(lum[:, 1:] - lum[:, :-1])

    seq = np.arange(n, dtype=np.int32).reshape(h, w)
    u = np.concatenate([seq[:-1].ravel(), seq[:, :-1].ravel()])
    v = np.concatenate([seq[1:].ravel(), seq[:, 1:].ravel()])
    gap = np.concatenate([down.ravel(), right.ravel()])

    # THE PAIRS THAT NEED NO TEST, FIRST AND WITHOUT A LOOP.  A flat frame
    # falls out here as one piece, which is what it is.
    flat = gap <= 0.0
    # ...and the joined pairs go in as the grid they are, not as a list of
    # indices --- `down` and `right` ARE `flat`, before it was flattened out
    parent = _knit(h, w, down <= 0.0, right <= 0.0)
    size = np.bincount(parent, minlength=n).astype(np.int32)
    inside = np.zeros(n, np.float32)          # the roughest step within a piece

    u, v, gap = u[~flat], v[~flat], gap[~flat]
    order = np.argsort(gap, kind="stable")
    u, v, gap = u[order], v[order], gap[order]

    if _INC_WALK is not None:
        # FABLE'S WALK, ON THE HOST.  The same loop his kernel runs, in the
        # same order and the same types --- see `body/native/knit.c`.  He put
        # it on ONE GPU thread because CUDA compiles at runtime and there was
        # no compiler; his own note says a CPU core is faster, and this drops
        # the round trip off the card as well.  `_WALK` below still runs when
        # the library is missing, and the Python after that when neither is.
        u = np.ascontiguousarray(u, np.int32)
        v = np.ascontiguousarray(v, np.int32)
        gap = np.ascontiguousarray(gap, np.float32)
        parent = np.ascontiguousarray(parent, np.int32)
        size = np.ascontiguousarray(size, np.int32)
        inside = np.ascontiguousarray(inside, np.float32)
        _INC_WALK(u, v, gap, int(u.size), parent, size, inside,
                  float(scale), int(least))
    elif _WALK is not None:
        d_u, d_v = _gpu.asarray(u), _gpu.asarray(v)
        d_gap = _gpu.asarray(gap)
        d_par, d_size = _gpu.asarray(parent), _gpu.asarray(size)
        d_in = _gpu.asarray(inside)
        _WALK((1,), (1,), (d_u, d_v, d_gap, np.int32(u.size),
                           d_par, d_size, d_in,
                           np.float64(scale), np.int32(least)))
        parent = _gpu.asnumpy(d_par)
    else:
        def root(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        walk = order.size and range(u.size)
        for i in walk or ():
            a, b = root(int(u[i])), root(int(v[i]))
            if a == b:
                continue
            d = float(gap[i])
            if d <= min(inside[a] + scale / size[a],
                        inside[b] + scale / size[b]):
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                inside[a] = d
        # DUST IS NOT A THING.
        for i in walk or ():
            a, b = root(int(u[i])), root(int(v[i]))
            if a != b and (size[a] < least or size[b] < least):
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]

    tags, inv = np.unique(_settle(parent), return_inverse=True)
    m = int(tags.size)
    area = np.bincount(inv, minlength=m).astype(np.float64)
    ys, xs = np.divmod(np.arange(n), w)
    ys, xs = ys.astype(np.float64), xs.astype(np.float64)
    x0, x1 = np.full(m, np.inf), np.full(m, -np.inf)
    y0, y1 = np.full(m, np.inf), np.full(m, -np.inf)
    np.minimum.at(x0, inv, xs); np.maximum.at(x1, inv, xs)
    np.minimum.at(y0, inv, ys); np.maximum.at(y1, inv, ys)
    colour = [np.bincount(inv, weights=f[:, :, min(c, bands - 1)].ravel()
                          .astype(np.float64), minlength=m) / area
              for c in range(3)]
    got = np.stack([
        np.bincount(inv, weights=xs, minlength=m) / area / (w - 1) * 2.0 - 1.0,
        np.bincount(inv, weights=ys, minlength=m) / area / (h - 1) * 2.0 - 1.0,
        area / float(n),                             # how much of her field
        (x1 - x0 + 1.0) / w,
        (y1 - y0 + 1.0) / h,
        colour[0], colour[1], colour[2]], axis=1).astype(np.float32)

    # ITS OWN SHAPE, cropped to its own box and squashed to SHAPE_SIDE --- so
    # the same thing nearer has the same shape as the same thing further, which
    # is the only thing that makes them matchable at all.
    label = inv.reshape(h, w)
    shape = np.zeros((m, SHAPE_SIDE * SHAPE_SIDE), np.float32)
    for k in range(m):
        a, b = int(y0[k]), int(y1[k]) + 1
        c, d = int(x0[k]), int(x1[k]) + 1
        box = (label[a:b, c:d] == k).astype(np.float32)
        ry = np.linspace(0, box.shape[0], SHAPE_SIDE + 1).astype(int)
        rx = np.linspace(0, box.shape[1], SHAPE_SIDE + 1).astype(int)
        for i in range(SHAPE_SIDE):
            for j in range(SHAPE_SIDE):
                cell = box[ry[i]:max(ry[i] + 1, ry[i + 1]),
                           rx[j]:max(rx[j] + 1, rx[j + 1])]
                shape[k, i * SHAPE_SIDE + j] = float(cell.mean()) if cell.size else 0.0

    got = np.concatenate([got, shape,
                          _alike(got).reshape(-1, 1).astype(np.float32)], axis=1)
    return got[np.argsort(got[:, 2])[::-1]]          # biggest first


def _alike(rows: np.ndarray) -> np.ndarray:
    """THE SIMILARITY INDEX --- one integer that things which look alike share.

    Built out of what does NOT change when a thing moves or comes closer: the
    direction of its colour (brightness divided out, the same quantity `find`
    groups by) and its proportion.  Where it is and how big it looks are
    deliberately not in it --- those are exactly what differ between two sights
    of one thing.

    It is a BUCKET, not a distance.  "Have I seen something like this?" becomes
    `WHERE alike = ?`, which an index answers, and `like()` then ranks what
    comes back by their full representations.  A scan over every thing she has
    ever seen is the thing this exists to avoid.
    """
    if not len(rows):
        return np.zeros(0, np.int64)
    col = rows[:, COLOUR].astype(np.float64)
    unit = col / np.maximum(np.linalg.norm(col, axis=1, keepdims=True), 1e-6)
    tall = rows[:, 4] / np.maximum(rows[:, 3], 1e-6)
    part = np.clip((unit[:, :2] * ALIKE_STEPS).astype(np.int64), 0, ALIKE_STEPS - 1)
    thin = np.clip((np.arctan(tall) / (np.pi * 0.5) * ALIKE_STEPS).astype(np.int64),
                   0, ALIKE_STEPS - 1)
    return (part[:, 0] * ALIKE_STEPS + part[:, 1]) * ALIKE_STEPS + thin


def like(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    """How unlike `one` representation each of `many` is --- smaller is closer.

    Colour and shape, both of which survive a thing moving; not position and
    not size, which do not.  For ranking what an `alike` lookup handed back.
    """
    if not len(many):
        return np.zeros(0, np.float32)
    a, b = np.atleast_2d(one), np.atleast_2d(many)
    col = np.linalg.norm(b[:, COLOUR] - a[:, COLOUR], axis=1)
    shape = np.abs(b[:, SHAPE] - a[:, SHAPE]).mean(axis=1)
    return (col + shape).astype(np.float32)


