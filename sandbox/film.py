"""The buffer: computed frames on their way to a browser, and the bytes they
travel as.

ONE FRAME IS 118 BYTES.  Seventeen joints and the flying window, three
coordinates each, one signed 16-bit integer per axis in tenths of a millimetre.
At 30 frames a second that is 3.5 KB/s — less than one photograph a minute —
which is why this works over a phone's connection.

A frame is A SUBSTEP THAT WAS COMPUTED.  Nothing here resamples, interpolates or
invents one; the only thing the codec does is round to 0.1 mm, which is a
hundred times finer than a screen can show.

    header  20 bytes   magic | first | count | latest | wide | scale
    then    count x wide   int16, little-endian

`wide` is how many numbers a frame carries, so a reader never has to assume the
body it is looking at.  The same header is a recording's header, so a streamed
chunk and a clip on disk are one format read two ways.
"""

from __future__ import annotations

import struct
import threading

MAGIC = b"NIF2"
HEAD = struct.Struct("<4sIIIHH")
#: 0.5 mm per unit.  int16 tops out at 32767 units, and the room's widest
#: axis is what has to fit under that, not the 2.5 m ceiling this used to be
#: sized against: half-extent 10.0 m needs 20000 units at this scale, a
#: comfortable margin below the ceiling.  At the old SCALE (10000, sized for
#: a room whose floor was still ~3.2 m across) that same 10.0 m needed
#: 100000 units --- four times int16's range --- so every coordinate past
#: 3.2767 m silently clamped to the encoder's max and stuck there, identical
#: for every joint that crossed it. That is not a wall: it is every reading
#: past this line in this file agreeing on the same wrong number. Found by
#: grepping the codebase for the literal repeating value (3.2767...) a
#: "collapsed" and "wall-stopped" baby kept reading at.
SCALE = 2000

#: What rides along after the joints: the window's position, where it is aimed,
#: and how big it is.  IT IS COMPUTED HERE AND MIRRORED THERE — the page draws
#: where you are; it does not work out where you are.
WINDOW_VALUES = 8


def encode(values) -> bytes:
    """One computed pose -> bytes."""
    return struct.pack(f"<{len(values)}h", *[
        max(-32768, min(32767, int(round(v * SCALE)))) for v in values])


def decode(raw: bytes) -> list[float]:
    """Bytes -> one pose, in metres."""
    return [v / SCALE for v in struct.unpack(f"<{len(raw) // 2}h", raw)]


def pack(first: int, latest: int, frames: list[bytes], wide: int) -> bytes:
    """A chunk of the buffer, ready to send or to save."""
    # `latest` IS -1 UNTIL THE FIRST FRAME EXISTS (`Film.latest` is
    # `_next - 1`) AND THE HEADER FIELD IS UNSIGNED, so every request that
    # arrived in the gap between the server binding its port and physics
    # producing frame 0 died in `struct.pack` --- one crashed thread per
    # poll, and a viewer left open across a restart polls hard.  0 is safe
    # to send in its place because the FRAME COUNT is what the viewer
    # believes: it gets 0 frames, plays nothing, and asks again from 0,
    # which is where the first frame will be.
    return (HEAD.pack(MAGIC, first, len(frames), max(0, latest), wide, SCALE)
            + b"".join(frames))


def unpack(blob: bytes) -> tuple[int, int, int, list[bytes]]:
    """(first, latest, wide, frames) — for `play` mode and the checks."""
    magic, first, count, latest, wide, scale = HEAD.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a {MAGIC.decode()} film: {magic!r}")
    if scale != SCALE:
        raise ValueError(f"that film is scaled {scale}, this build reads {SCALE}")
    step = wide * 2
    body = memoryview(blob)[HEAD.size:]
    if len(body) < count * step:
        raise ValueError(f"film says {count} frames and carries {len(body) // step}")
    return first, latest, wide, [bytes(body[i * step:(i + 1) * step])
                                 for i in range(count)]


class Film:
    """A ring of the most recent frames, indexed by an absolute frame number
    that never resets.

    ABSOLUTE INDICES ARE THE POINT.  A client asks for "everything after 1841"
    and either gets it or is told it has fallen too far behind — it can never be
    handed different frames believing they are the ones it asked for, which is
    what an offset into a wrapping buffer eventually does.
    """

    def __init__(self, wide: int, keep: int = 900) -> None:
        self.wide = wide
        self.keep = keep
        self._ring: list[bytes | None] = [None] * keep
        self._next = 0                     # index the next appended frame gets
        self._lock = threading.Lock()

    def append(self, frame: bytes) -> int:
        with self._lock:
            at = self._next
            self._ring[at % self.keep] = frame
            self._next = at + 1
            return at

    @property
    def latest(self) -> int:
        """The newest frame index that exists.  -1 before the first one."""
        return self._next - 1

    @property
    def oldest(self) -> int:
        return max(0, self._next - self.keep)

    def since(self, first: int, most: int = 120) -> tuple[int, list[bytes]]:
        """Frames from `first` on, up to `most` of them.

        Returns where the answer actually starts — later than asked when the
        caller has fallen out of the ring.  Silently starting somewhere else
        would make a viewer play a jump as though it were motion.
        """
        with self._lock:
            start = max(first, self.oldest)
            end = min(self._next, start + max(0, most))
            out = [self._ring[i % self.keep] for i in range(start, end)]
            return start, [f for f in out if f is not None]

    def chunk(self, first: int, most: int = 120) -> bytes:
        start, frames = self.since(first, most)
        return pack(start, self.latest, frames, self.wide)
