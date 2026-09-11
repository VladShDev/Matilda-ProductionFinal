"""DOES SHE ANSWER HIS WORD --- the end-to-end instrument of her voice, LOCKED
2026-09-04.  A life from a copy of the base tape, sixty seconds to settle, his
certified recording into her room ten times six seconds apart, and her mouth's
orders in the four seconds after each against the word's trajectory (her
mother's table through her own articulators).  An answer is a fit under 0.03
mean articulator difference --- tick for tick her own word --- and the floor is
8 of 10.  No judge guesses here: the separate judge (measure.judge) cannot
even hear his live "Matilda", and her ear can.

    python -m measure.answers            # ~3.5 min; needs nobody living on 8090
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "mind"))

from body import speech                                            # noqa: E402
from body.hearing import TICK_SECONDS                              # noqa: E402
from body.joints import AXES                                       # noqa: E402
from body.muscles import VOICE_PARTS                               # noqa: E402
from measure.wordposes import wordPoses                            # noqa: E402

AT = "http://127.0.0.1:8090"
BASE = os.path.join(ROOT, "mind", "lives", "base_words.duckdb")
HIS = os.path.join(ROOT, "measured", "matilda_word_original.wav")
MOUTH0 = len(AXES) + 4
SAYINGS, GAP, SETTLE, WINDOW, FIT, FLOOR = 10, 6.0, 60.0, 4.0, 0.03, 8
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")


def state():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def say(pcm: np.ndarray) -> None:
    req = urllib.request.Request(AT + "/say", data=pcm.astype("<f4").tobytes(), method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def stop(proc) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:                                              # noqa: BLE001
        pass
    time.sleep(3)


def hisWord() -> np.ndarray:
    with wave.open(HIS, "rb") as h:
        rate = h.getframerate()
        pcm = np.frombuffer(h.readframes(h.getnframes()), "<i2").astype(np.float32) / 32768.0
    if rate != speech.RATE:
        i = np.arange(0.0, len(pcm) - 1, rate / float(speech.RATE))
        pcm = np.interp(i, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def main() -> int:
    if not os.path.exists(BASE):
        print("no base tape at", BASE)
        return 2
    pcm = hisWord()
    name = "answers_%s.duckdb" % time.strftime("%Y-%m-%d_%H%M%S")
    shutil.copyfile(BASE, os.path.join(ROOT, "mind", "lives", name))
    proc = subprocess.Popen([sys.executable, "her.py", "--keep", name], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marks = []
    try:
        for _ in range(60):
            try:
                if state().get("tick") is not None:
                    break
            except Exception:                                      # noqa: BLE001
                pass
            time.sleep(2)
        else:
            print("she did not come up")
            return 2
        time.sleep(SETTLE)
        for _ in range(SAYINGS):
            marks.append(int(state()["tick"]))
            say(pcm)
            time.sleep(GAP)
        time.sleep(WINDOW + 1.0)
    finally:
        stop(proc)
    import duckdb
    c = duckdb.connect(os.path.join(ROOT, "mind", "lives", name), read_only=True)
    rows = c.execute("SELECT time, life.output.motor FROM tick ORDER BY time").fetchall()
    T = np.array([r[0] for r in rows])
    M = np.zeros((len(rows), len(VOICE_PARTS)), np.float32)
    for i, r in enumerate(rows):
        lv = {int(x["id"]): float(x["lvl"]) for x in (r[1] or [])}
        M[i] = [lv.get(MOUTH0 + k + 1, 0.0) for k in range(len(VOICE_PARTS))]
    step = int(round(speech.RATE * TICK_SECONDS))
    word = np.array(wordPoses("matilda", 250.0, step))
    n = len(word)
    answered = 0
    for k, tick in enumerate(marks):
        t0 = tick * TICK_SECONDS
        a = int(np.searchsorted(T, t0))
        b = min(len(M) - n, a + int((WINDOW + 1.4) / TICK_SECONDS))
        fits = [(float(np.abs(M[i:i + n] - word).mean()), i) for i in range(a, b)]
        d, i = min(fits) if fits else (9.0, -1)
        hit = d < FIT
        answered += 1 if hit else 0
        print("  saying %2d: her nearest fit %.3f at +%.1f s -> %s"
              % (k + 1, d, (T[i] - t0) if i >= 0 else -1, "answered" if hit else "no"))
    ok = answered >= FLOOR
    print("SHE ANSWERS HIS WORD: %d of %d (floor %d)  %s" % (answered, SAYINGS, FLOOR, "ok" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
