"""A SEPARATE JUDGE FOR HER VOICE --- vosk, an offline speech recogniser that
knows nothing of her, her ear or her bank.

His word, 2026-09-04, after the twentieth fix of her voice by her own
yardsticks: *"use some separate app to analyse your results and notify me
when I'll be able to recognise at least something in her word producing"*.
Her ear's names are three coarse digits of a spectrum and many sounds share
a name, so "her ear names it as his word" was never the same as a word.

What the judge established that night, in order:
  * his recording -> "matilda" 1.00; Windows' own voice -> 1.00 (the control)
  * her renders -> nothing, in every joining, at her newborn register
  * her MOTHER's words through the same synthesiser -> mommy 1.00, milk 1.00
  * the same synthesiser with the mother's "matilda" -> 1.00 at an adult
    tract, 1.00 at 1.3 x adult with the larynx at 250 Hz, "banana" at 1.4,
    nothing at the half-adult tract she had --- no recogniser hears a word
    from a newborn's tract, hers or a real one's; he chose the toddler's
  * her own articulators at that tract -> nothing, then "mommy", then
    "matilda" 0.86 as three seams of her mouth were closed: a mouth that
    restarted its glottis and every resonator ninety times a second, lines
    that stepped 11 ms at a time instead of sliding, and an F3 law 800 Hz
    too high at front vowels.

Usage:
    python -m measure.judge FILE.wav [FILE2.wav ...]
        free transcription and the word-list verdict ("matilda" against
        distractors and [unk]), with confidences.

The model: lives/vosk-model-small-en-us-0.15 (68 MB, not in git; from
https://alphacephei.com/vosk/models).  A wav is 16 kHz mono 16-bit, as her
mouth makes them; the judge pads 0.7 s of silence around a clip and levels
it to 0.7, which is what it needs to segment a word.
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "lives", "vosk-model-small-en-us-0.15")
WORDS = ["matilda", "tilda", "mommy", "mama", "dada", "milk", "ball", "rattle", "bear", "hello",
         "banana", "water", "mother", "baby", "tea", "your", "[unk]"]
RATE = 16000


def _model():
    import vosk
    vosk.SetLogLevel(-1)
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError("the judge's model is missing: " + MODEL_DIR)
    return vosk.Model(MODEL_DIR)


def hear(pcm, rate: int = RATE, words=WORDS, model=None):
    """`(free_text, [(word, conf)...], listed_text, [(word, conf)...])` for
    float pcm in -1..1: what the judge hears unconstrained, and against the
    word list.  `conf` of "matilda" in the list verdict is the number."""
    import vosk
    model = model or _model()
    y = np.asarray(pcm, np.float32)
    y = y / max(float(np.abs(y).max()), 1e-9) * 0.7
    pad = np.zeros(int(0.7 * rate), np.float32)
    y = np.concatenate([pad, y, pad])
    data = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    out = []
    for grammar in (None, json.dumps(list(words))):
        rec = (vosk.KaldiRecognizer(model, rate, grammar) if grammar
               else vosk.KaldiRecognizer(model, rate))
        rec.SetWords(True)
        rec.AcceptWaveform(data)
        got = json.loads(rec.FinalResult())
        out.append(got.get("text", ""))
        out.append([(w["word"], round(float(w["conf"]), 2))
                    for w in (got.get("result") or [])])
    return tuple(out)


def matilda(pcm, rate: int = RATE, model=None) -> float:
    """The judge's confidence that this is his word, 0..1."""
    _, _, _, listed = hear(pcm, rate, model=model)
    return max([c for w, c in listed if w == "matilda"] + [0.0])


def main() -> int:
    model = _model()
    for path in sys.argv[1:]:
        with wave.open(path) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767
        free, fw, listed, lw = hear(pcm, rate, model=model)
        print("%-44s free: %-24s %s" % (os.path.basename(path), '"%s"' % free, fw))
        print("%-44s list: %-24s %s" % ("", '"%s"' % listed, lw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
