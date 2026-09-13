"""HER MEMORY, SAID BACK BY HER OWN MOUTH --- the voicing test.

    python -m measure.saidback                     # her newest life's record
    python -m measure.saidback NAME.duckdb         # ...or one by name
    python -m measure.saidback --from HIS.wav      # his recording, through her gate

TESTING ONLY.  His word, 2026-09-12: *"this separate script has to use her
memory ... prelearning her experience, from which she will build her bigger
experiences to reproduce the longest steps, full words."*  Nothing in `body/` or
`mind/` imports this, and nothing ever may.

WHAT IT DOES, AND ALL OF IT IS HERS:
  * The levels come out of HER RECORD --- `sound.similarity` and `sound.lvl` a
    tick, her two sound lines, exactly as her ear stored them.  With `--from`
    they come out of his recording through the same one gate instead (`door`,
    `align`, `SOUND_HOPS`, at her rate), which is what her ear would have stored.
  * The table is her alphabet (`measure.alphabet`): rows she lived in her own
    sweep, each with what her ear said of it.  It stands in for the experiences
    she has not lived yet.
  * For each tick, the row her ear called this level is looked up --- nearest
    by her two sound lines, nothing else --- and said through ONE mouth that
    carries from tick to tick and is never reset.  No search, no trying, no
    climbing, no breath helper: a gap is the floor, the floor is a level, and
    the row that answers it is the row she lived when she was not breathing.

WHAT COMES OUT: `measured/saidback_her.wav` (and `measured/saidback_you.wav`
with `--from`), and praat's word on it --- her larynx, her voice, her rhythm,
and, beside his, how the two move together.
"""
from __future__ import annotations

import glob
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body import speech                                            # noqa: E402
from body.hearing import (LISTENS, SOUND_BANDS, SOUND_HOPS,        # noqa: E402
                          SOUND_SLIDES, TICK_SECONDS, door, grabFor, toHer)
from body.muscles import Voice                                     # noqa: E402
from measure.alphabet import PARTS, heard, load                    # noqa: E402

HER = os.path.join("measured", "saidback_her.wav")
YOU = os.path.join("measured", "saidback_you.wav")


# ------------------------------------------------------------------ the levels
def fromRecord(tape: str) -> tuple:
    """Her two sound lines, every tick of her record."""
    import dataclasses
    from measure.record import frames                              # noqa: E402
    # A RECORD WRITTEN WHILE A FIELD EXISTED THAT NO LONGER DOES IS STILL HERS
    # TO READ.  The store builds each sound as `Echo(**row)`; fields the struct
    # has since lost (`spread`, `peak`, 2026-09-12) are dropped here, in the
    # instrument, so her memory of him from that morning can be replayed.
    for name, mod in list(sys.modules.items()):
        if name.split(".")[-1] == "store" and hasattr(mod, "_life"):
            for kind in ("Echo", "Heard"):
                cls = getattr(mod, kind, None)
                if cls is None or not dataclasses.is_dataclass(cls):
                    continue
                keep = {f.name for f in dataclasses.fields(cls)}
                setattr(mod, kind, (lambda c, k: (lambda **kw: c(**{a: b for a, b in kw.items() if a in k})))(cls, keep))
    level, loud = [], []
    for t in frames(tape):
        s = t.life.input.sound
        level.append(float(getattr(s, "similarity", 0.0) or 0.0))
        loud.append(float(s.lvl))
    return np.asarray(level, np.float64), np.asarray(loud, np.float64), None


def fromHis(path: str, seconds: float = 8.0) -> tuple:
    """His recording through her gate: the two lines her ear would have stored,
    one tick at a time, and the piece of his air they came from."""
    with wave.open(path, "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        r0 = float(w.getframerate())
    rate = float(speech.RATE)
    n = int(round(len(y) * rate / r0))
    y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    # INTO HER REGISTER FIRST, ONCE --- as her body does where his air arrives
    # (`window.say`); only then the 11 ms cut.  His order, 2026-09-12.
    y = toHer(y)
    # the loudest stretch of him, so the test is on words and not on room
    half = int(rate * 0.5)
    power = np.array([float(np.abs(y[k:k + half]).mean()) for k in range(0, max(1, len(y) - half), half)])
    if power.size:
        at = int(np.argmax(np.convolve(power, np.ones(int(seconds * 2)) / (seconds * 2), "same")))
        y = y[max(0, at * half - int(rate * seconds / 2)):][:int(rate * seconds)]
    # each 11 ms piece named from the last LISTENS of air, as her ear does
    grab, hop = grabFor(rate, LISTENS), int(round(rate * TICK_SECONDS))
    y = np.concatenate([np.zeros(grab - hop, np.float32), y])
    level, loud = [], []
    for e in range(grab, len(y) + 1, hop):
        lv, ld = heard(door(y[e - grab:e], rate, LISTENS, SOUND_HOPS))
        level.append(lv); loud.append(ld)
    return np.asarray(level, np.float64), np.asarray(loud, np.float64), y


# ------------------------------------------------------------------ her mouth
def sayBack(level, loud, table) -> tuple:
    """`(pcm, made)` --- her air, one carried mouth, and the level her ear gave
    each tick of it."""
    rows, tlevel, tloud = table
    v = Voice(SOUND_BANDS, SOUND_SLIDES)
    out, made = [], []
    for lv, ld in zip(level, loud):
        j = int(np.argmin(np.abs(tlevel - lv) + np.abs(tloud - ld)))   # the row her ear called this
        got = v.say(rows[j].reshape(PARTS, 1))
        made.append(heard(got)[0])
        out.append(np.asarray(v.pcm, np.float32).ravel().copy())
    if not out:
        return np.zeros(0, np.float32), np.zeros(0)
    w = max(len(x) for x in out)
    pcm = np.concatenate([np.pad(x, (0, w - len(x))) if len(x) < w else x[:w] for x in out])
    return pcm, np.asarray(made, np.float64)


def write(path: str, pcm, rate: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    y = pcm / peak * 0.85 if peak > 0 else pcm
    with wave.open(path, "wb") as g:
        g.setnchannels(1); g.setsampwidth(2); g.setframerate(int(rate))
        g.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


# ------------------------------------------------------------------ praat
def praat(her: str, his: str | None) -> None:
    """The outside program's word --- never ours."""
    try:
        import parselmouth as pm
        from parselmouth.praat import call
    except ImportError:
        print("  (praat-parselmouth is not installed; no outside check)")
        return
    B = pm.Sound(her)
    f = B.to_pitch(pitch_floor=200, pitch_ceiling=700).selected_array["frequency"]
    vv = f[f > 0]
    pt = call(B, "To PointProcess (periodic, cc)", 200, 700)
    print("  her voice (praat):  F0 %.0f Hz   voiced %.0f%%   jitter %.1f%%   HNR %.1f dB" % (
        float(np.median(vv)) if vv.size else 0.0, 100.0 * vv.size / max(1, f.size),
        100.0 * call(pt, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3),
        call(B.to_harmonicity(), "Get mean", 0, 0)))

    def envelope(s):
        i = s.to_intensity(time_step=0.01)
        return np.nan_to_num(np.array([i.get_value(t) for t in np.arange(0.02, s.duration - 0.02, 0.01)]), nan=0.0)

    def rhythm(e):
        e = e - e.mean(); fr = np.fft.rfftfreq(len(e), 0.01); p = np.abs(np.fft.rfft(e)) ** 2
        return 100.0 * p[(fr >= 3) & (fr <= 8)].sum() / max(p[fr > 0.5].sum(), 1e-9)

    eb = envelope(B)
    line = "  her rhythm: %.0f%% of her loudness in the syllable band (3-8 Hz)" % rhythm(eb)
    if his and os.path.exists(his):
        A = pm.Sound(his)
        ea = envelope(A)
        n = min(len(ea), len(eb))
        qa, qb = ea < (ea.max() - 25), eb < (eb.max() - 25)
        line += "   yours %.0f%%" % rhythm(ea)
        print(line)
        print("  together: loudness %+.3f   quiet at the same moment %.0f%%" % (
            float(np.corrcoef(ea[:n], eb[:n])[0, 1]), 100.0 * np.mean(qa[:n] == qb[:n])))
    else:
        print(line)


# ------------------------------------------------------------------ main
def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    args = sys.argv[1:]
    his = None
    if "--from" in args:
        his = args[args.index("--from") + 1]
        args = [a for a in args if a != "--from" and a != his]
    table = load()
    if his:
        level, loud, air = fromHis(his)
        source = his
    else:
        tapes = [a for a in args if a.endswith(".duckdb")]
        tape = tapes[0] if tapes else (sorted(glob.glob(os.path.join("mind", "lives", "*_life.duckdb")),
                                              key=os.path.getmtime) or [None])[-1]
        if not tape:
            print("no life to read"); return 1
        level, loud, air = fromRecord(tape)
        source = tape
    live = level > 0.0
    print("SAID BACK --- from %s" % source)
    print("  ticks %d, sounding %d;  her table: %d rows, %d distinct levels" % (
        len(level), int(live.sum()), len(table[0]), len(np.unique(np.round(table[1] * 512)))))
    if live.any():
        lo, hi = float(table[1][table[1] > 0].min()), float(table[1].max())
        print("  levels asked %.4f..%.4f;  her mouth reaches %.4f..%.4f;  %.0f%% of the asked ticks inside" % (
            level[live].min(), level[live].max(), lo, hi,
            100.0 * np.mean((level[live] >= lo) & (level[live] <= hi))))
    pcm, made = sayBack(level, loud, table)
    write(HER, pcm, speech.RATE)
    if air is not None:
        write(YOU, air, speech.RATE)
    if live.any():
        miss = np.abs(made[live] - level[live])
        print("  her ear on what came back: within her grain (0.01) %.0f%%, median miss %.4f, follows the asked level %+.3f" % (
            100.0 * np.mean(miss <= 0.01), float(np.median(miss)),
            float(np.corrcoef(level[live], made[live])[0, 1]) if live.sum() > 2 else 0.0))
    print("  written %s%s" % (HER, ("  and " + YOU) if air is not None else ""))
    praat(HER, YOU if air is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
