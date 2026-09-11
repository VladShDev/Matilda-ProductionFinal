"""WHAT HE SAID TO HER, KEPT AS A FILE HE CAN PLAY.

    lives/<her tape>_you.wav

**IT IS THE WATCHER'S, NOT HERS.**  `brain/` and `body/` do not import it and
never will: she does not keep a recording of anybody, and the whole reason her
ears reduce a sound to one integer is that she must not.  This is a tap on the
wire between his microphone and her room, put there so HE can hear what actually
arrived --- the same job the old tree's `what_she_heard_of_you.wav` did.

WHERE IT TAPS, AND WHY THERE.  His microphone reaches her through
`POST /window -> flying() -> says.pcm -> window.say`, which is the last point at
which the samples still exist: one tick later her ears have turned them into a
band split and an id, and the waveform is gone for good.  `POST /say` is tapped
too, because `measure` and a second page use it.

THE HEADER IS REWRITTEN ON EVERY WRITE.  A wav whose sizes are only fixed at
close is a wav that is corrupt whenever the thing writing it was killed --- and
a live body is killed rather than closed almost every time.  Two seeks and eight
bytes is nothing against losing the recording.
"""

from __future__ import annotations

import io
import os
import struct
import threading

import numpy as np


class Heard:
    """His voice, appended as it arrives.  Float32 in, 16-bit PCM on disk."""

    def __init__(self, where: str, rate: int = 16000) -> None:
        self.where = where
        self.rate = int(rate)
        self.frames = 0
        self.seconds = 0.0
        #: HIS OWN LOCK.  Her body's lock is for her body --- taking it to write
        #: a file would put a disk behind her tick, which is the fault that cost
        #: 26 seconds of waiting in 35 seconds of wall time when `/frames` held
        #: it.
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(where)) or ".", exist_ok=True)
        self.fh = io.open(where, "wb+")
        self.fh.write(self._header(0))
        self.fh.flush()

    def _header(self, n: int) -> bytes:
        """A 44-byte canonical wav header for `n` frames of one 16-bit channel."""
        data = n * 2
        return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, self.rate, self.rate * 2, 2, 16)
                + b"data" + struct.pack("<I", data))

    def add(self, pcm, rate: float | None = None) -> int:
        """One arrival.  Returns how many frames are on the file now.

        A rate that is not hers is RESAMPLED rather than written wrong: a wav
        that plays at the wrong speed is a recording that lies about his voice,
        and the page's rate is whatever his hardware gave it.
        """
        got = np.asarray(pcm, np.float32).ravel()
        if got.size == 0:
            return self.frames
        if rate and abs(float(rate) - self.rate) > 1.0:
            # nearest sample --- this is a record for his ear, not a measurement
            at = np.linspace(0.0, got.size - 1.0,
                             max(1, int(round(got.size * self.rate / float(rate)))))
            got = got[np.rint(at).astype(np.int64).clip(0, got.size - 1)]
        # ...AND IT IS CLIPPED, NOT WRAPPED.  int16 overflow turns a loud word
        # into a burst of noise that sounds exactly like a broken microphone.
        block = np.clip(got, -1.0, 1.0)
        block = (block * 32767.0).astype("<i2")
        with self.lock:
            self.fh.seek(0, os.SEEK_END)
            self.fh.write(block.tobytes())
            self.frames += int(block.size)
            self.seconds = self.frames / float(self.rate)
            here = self.fh.tell()
            self.fh.seek(0)
            self.fh.write(self._header(self.frames))
            self.fh.seek(here)
            self.fh.flush()
        return self.frames

    def close(self) -> None:
        with self.lock:
            try:
                self.fh.close()
            except Exception:                                   # noqa: BLE001
                pass

    def __repr__(self) -> str:                 # pragma: no cover - display only
        return f"<Heard {self.seconds:.1f}s at {self.rate} Hz -> {self.where}>"
