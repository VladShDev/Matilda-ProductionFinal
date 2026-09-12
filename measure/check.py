"""EVERY CHECK SHE HAS, IN ONE COMMAND --- and it writes down what it found.

    python -m measure.check          run them all, record the answer
    python -m measure.check --fast   skip the slow ones (no word test)

**THIS EXISTS BECAUSE HE ASKED THE RIGHT QUESTION:** *"what if next time ill
forget to push you to run it"*.  A check that has to be remembered is a check
that will be skipped on exactly the night it mattered --- and this project's
whole failure mode is a green suite over a girl who does nothing.

So it does two things nobody has to remember:

  1. runs every instrument in one command, and
  2. **writes the answers to `measured/state.json` with the commit they were
     true at** --- so a claim can be checked against the code it was made about.

`.githooks/pre-commit` then refuses a commit that touches `brain/` or `body/`
when the recorded answer is older than the change.  Nobody has to remember; the
commit will not go through.

Nothing here touches her.  Every instrument it runs is a read or a fresh life.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WROTE = os.path.join(HERE, "measured", "state.json")

#: WHAT TO RUN, WHAT IT MUST SAY, AND WHETHER IT IS SLOW.
#:
#: Each is `(name, module args, the pattern that means it PASSED, slow)`.  The
#: pattern is matched against everything it printed, so an instrument that dies
#: silently fails here rather than being read as a pass.
CHECKS = [
    # `wiring` always returns 0 --- it LISTS, it does not judge --- so its
    # pattern is the whole of its check: it must have printed a census.
    ("wiring", ["measure.wiring"], r"never called,", False),
    # HER VOICE IS A MONOLITH --- his word, 2026-08-31: *"you should mark
    # everywhere that our voice is like monolith"*, after it had been broken
    # and rebuilt four times in one night.  It needs no body and no GPU: it
    # cuts her mouth from his certified recording and asks whether she can
    # still say his word back.  Fast, so it runs even with --fast.
    ("voice", ["measure.voice"], r"THE VOICE HOLDS", False),
]

#: RETIRED 2026-08-30, on his word ("Retire the five, commit") --- each of
#: these measured an ORGAN HE REPLACED, so none can ever pass again, and a
#: red that can never green is not a gate, it is a bypass being trained.
#: Cancelled with the reason written down, the project's own rule:
#:
#:   "nothing lost" / "sight reaches her" / "a thing's id is an appearance"
#:       all three import `reasonOf` from `brain.brain.lookup` --- the
#:       IN-PROCESS brain, retired when her mind moved OUTSIDE (matilda4
#:       over HTTP, his 2026-08-26 architecture).  The organ they probed
#:       runs in no life anymore.  The living successors read the outside
#:       mind: the Matilda Scale (measure.scale) and the guardian.
#:   "she names her own sounds" / "she says his word"
#:       both crash on her mouth's shape --- they assume the retired
#:       synthesized voice, replaced by the clip player of his recorded
#:       pieces (his decision, certified by his ear).  The scale's voice
#:       items (repertoire, imitation, answers, twice-law) replace them.
#:
#: Outside-brain equivalents of "nothing lost" and "sight reaches her"
#: are on the gap list; until then the scale is the yardstick.


#: INSTRUMENTS THAT CANNOT RUN ARE RED, WITH THEIR REASON WRITTEN DOWN.
#:
#: His, 2026-09-01, after a night of finding dead measurements: *"we still has
#: a lot of things that do not works at all"* --- and the machine that kept
#: telling him she was ready was THIS FILE, going green while four instruments
#: could not execute a line.
#:
#: A dead instrument does not report red.  It reports NOTHING, and nothing
#: looks exactly like fine.  That is the whole loop: build, measure with a
#: broken meter, hear "ready", find her slow or mute, start again.
#:
#: So every module under `measure/` is IMPORTED here.  An import error is a
#: red mark with the module's name, unless it is listed below with the reason
#: it is dead --- and a listed one that starts importing again is ALSO red,
#: because a retirement that quietly came back is a lie in the other
#: direction.  Nothing is skipped silently, ever.
#: DEAD ON IMPORT --- these must FAIL to import.  One that starts importing
#: again is red: a retirement that quietly came back is a lie the other way.
RETIRED_IMPORT = {}

#: DEAD WHEN DRIVEN --- they import cleanly and die on their first tick,
#: because they build a STANDALONE body and a standalone body runs the
#: retired in-process brain.  Proving it costs a body and minutes, so the gate
#: does not run them; it PRINTS them, every single check, so the number is
#: never out of his sight and can only be made smaller.
RETIRED_RUN = {
}
RETIRED_INSTRUMENTS = dict(RETIRED_IMPORT, **RETIRED_RUN)


#: WHAT MUST NOT ONLY IMPORT, BUT RUN.
#:
#: His, 2026-09-02, and he was right to be angry: *"we already spoke that you
#: has to write by code, not by documents ... did you run it or you just trace
#: it and decided, oh, that function, I don't give a fuck, and goes further."*
#:
#: `instruments()` below walks `measure/` and IMPORTS each module.  Both halves
#: of that leaked:
#:
#:   * it never looked at `body/` at all, and
#:   * an import is not a call.  A method that does not exist is looked up when
#:     it is USED, so a module can import green for ever and die the moment
#:     anything asks it to work.
#:
#: MEASURED 2026-09-02, which is why this exists: `body/sounds.py` --- the
#: table of everything her mouth can make --- called `voice.say()` in three
#: places, and `Voice.say` had been deleted on 2026-08-26.  `import
#: body.sounds` succeeded every single day; calling it raised AttributeError.
#: Her alphabet could not be regenerated for a week and the suite was green.
#:
#: Each line here is a snippet that must run without raising.  They are small
#: on purpose --- this is "does the wiring carry current", not a test of what
#: she can do.  Anything expensive belongs in its own instrument.
RUNS = (
    ("her mouth can SAY",
     "import numpy as np;"
     "from body.muscles import Voice;"
     "from body.hearing import SOUND_BANDS, SOUND_SLIDES;"
     "v=Voice(SOUND_BANDS, SOUND_SLIDES);"
     "v.say(np.full((v.parts, 9), 0.5, np.float32));"
     "assert len(v.pcm) > 0"),
    ("her alphabet can be swept",
     "import numpy as np;"
     "from body import sounds;"
     "from body.muscles import Voice;"
     "from body.hearing import SOUND_SLIDES;"
     "sounds.held(Voice(24, SOUND_SLIDES), np.full(7, 0.5, np.float32))"),
    ("her ear can name a sound",
     "import numpy as np;"
     "from body.hearing import bands_from_pcm;"
     "from body.alike import alike_of;"
     "import numpy as np;"
     "b=bands_from_pcm(np.sin(np.arange(16000)*0.1).astype(np.float32));"
     "alike_of(b)"),
    ("her balance answers",
     "from body.ragdoll import Ragdoll;"
     "from body.room import Room;"
     "from body import balance;"
     "her=Ragdoll(Room());"
     "ear=balance.Balance();"
     "ear.read(her); pulled, turning = ear.read(her);"
     "assert len(pulled) == 6 and len(turning) == 6"),
    ("her eye can look and her pieces be found",
     "import numpy as np;"
     "from body import light, parts;"
     "from body.room import Room;"
     "from body.window import Window;"
     "up=np.array([0,1,0],np.float32);rt=np.array([1,0,0],np.float32);"
     "fw=np.array([0,0,-1],np.float32);"
     "pic=np.asarray(light._home(light.see(Room(), [Window()],"
     " [(np.array([0.2,0.42,-0.34],np.float32), up, rt, fw)])))"
     "[:, :, 0].reshape(light.RETINA_H, light.RETINA_W, light.CONES);"
     "assert len(parts.find(pic)) >= 0"),
)


def bodyRuns() -> list:
    """Every snippet in `RUNS`, actually executed.  Returns the broken ones."""
    import subprocess
    bad = []
    for what, code in RUNS:
        got = subprocess.run([sys.executable, "-c", code], cwd=HERE,
                             capture_output=True, text=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if got.returncode != 0:
            why = got.stderr.strip().splitlines()[-1][:70] if got.stderr.strip() \
                else "did not run"
            bad.append((what, why))
    return bad


def bodyImports() -> list:
    """...and every module of her BODY imports.  `instruments()` only ever
    looked at `measure/`, so a break in `body/` was invisible to it."""
    import subprocess
    here = os.path.join(HERE, "body")
    bad = []
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py") or name.startswith("__"):
            continue
        got = subprocess.run(
            [sys.executable, "-c", "import body.%s" % name[:-3]], cwd=HERE,
            capture_output=True, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if got.returncode != 0:
            bad.append((name[:-3], got.stderr.strip().splitlines()[-1][:70]
                        if got.stderr.strip() else "did not import"))
    return bad


def instruments() -> tuple:
    """Import every instrument.  `(broken, resurrected)` --- both are red."""
    import importlib
    here = os.path.join(HERE, "measure")
    broken, back = [], []
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py") or name.startswith("__"):
            continue
        mod = name[:-3]
        if mod == "check":
            continue
        got = subprocess.run(
            [sys.executable, "-c", "import measure.%s" % mod], cwd=HERE,
            capture_output=True, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        ok = got.returncode == 0
        if not ok and mod not in RETIRED_INSTRUMENTS:
            broken.append((mod, got.stderr.strip().splitlines()[-1][:70]
                           if got.stderr.strip() else "did not import"))
        if ok and mod in RETIRED_IMPORT:
            back.append(mod)
    return broken, back


def run(args, patience=1200):
    began = time.perf_counter()
    got = subprocess.run([sys.executable, "-m"] + args, cwd=HERE,
                         capture_output=True, text=True, timeout=patience,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return got.stdout + got.stderr, (time.perf_counter() - began), got.returncode


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the slow ones")
    got = ap.parse_args()
    # THE BARE INTERPRETER IS REFUSED.  Only one python on this machine has
    # duckdb, parselmouth and cupy (her python, requirements.txt); the
    # one on PATH is a bare 3.14 with numpy.  Under it four instruments were
    # unrunnable for a week behind a green suite, because an import error in
    # a child looks like any other failure.  His rule (step 8, 2026-09-03).
    try:
        import duckdb  # noqa: F401
        import parselmouth  # noqa: F401
    except ImportError as why:
        print("THIS IS NOT HER INTERPRETER:", sys.executable)
        print("  ", why)
        print("   run me with her python: the one that has requirements.txt installed")
        return 2
    print("interpreter:", sys.executable)

    print(f"{'check':<32}{'':>6}{'seconds':>9}")
    found, bad = {}, 0
    for name, args, wants, slow in CHECKS:
        if slow and got.fast:
            print(f"  {name:<30}{'skip':>6}")
            continue
        try:
            # **THE INSTRUMENT'S OWN VERDICT, NOT A REGEX.**  This matched a
            # pattern in what was printed --- and for the word that pattern was
            # a COUNT, so it reported `ok` on she-said-it-once-by-accident,
            # p = 0.130.  The shuffle inside `measure.words` exists precisely to
            # stop a count being read as success, and this check walked around
            # it.  Every instrument here returns 0 only when its own answer is
            # yes, so that is what is asked; the pattern stays as a second
            # guard, because a process that dies before printing anything
            # returns nonzero AND matches nothing.
            out, took, code = run(args)
            ok = code == 0 and re.search(wants, out) is not None
        except Exception as why:                            # noqa: BLE001
            out, took, ok = f"{type(why).__name__}: {why}", 0.0, False
        found[name] = bool(ok)
        bad += 0 if ok else 1
        print(f"  {name:<30}{'ok' if ok else 'NO':>6}{took:>9.0f}")
        if not ok:
            for line in out.strip().splitlines()[-3:]:
                print(f"      {line[:76]}")

    # WHAT HAS EVER HELD, so a REGRESSION can be told from a goal.
    #
    # "she says his word" is red and will be red until she says it --- that is
    # the thing she is for, not a fault in what was built.  A gate that refuses
    # every commit until she talks is a gate that gets bypassed on the first
    # night, and then it is not a gate.  So what is remembered is the BEST each
    # check has ever managed, and only a check that once held and now does not
    # is a regression.
    ever = {}
    if os.path.exists(WROTE):
        try:
            with open(WROTE, encoding="utf-8") as f:
                ever = json.load(f).get("ever", {})
        except Exception:                                   # noqa: BLE001
            ever = {}
    broke = sorted(k for k, ok in found.items() if ever.get(k) and not ok)
    for k, ok in found.items():
        ever[k] = bool(ever.get(k)) or bool(ok)

    os.makedirs(os.path.dirname(WROTE), exist_ok=True)
    with open(WROTE, "w", encoding="utf-8") as f:
        json.dump({"at": head(), "when": time.strftime("%Y-%m-%d %H:%M"),
                   "fast": bool(got.fast), "found": found, "ever": ever,
                   "broke": broke}, f, indent=2)
    if broke:
        print("")
        print("  REGRESSED --- these held before and do not now: "
              + ", ".join(broke))
    # ...AND WHETHER THE INSTRUMENTS THEMSELVES CAN RUN AT ALL
    broken, back = instruments()
    print(f"\n{'instruments':<32}{'':>6}{'':>9}")
    print("  %-30s%6s" % ("every measure/* imports",
                          "ok" if not broken and not back else "NO"))
    for mod, why in broken:
        bad += 1
        print("      %s CANNOT RUN --- %s" % (mod, why))
    for mod in back:
        bad += 1
        print("      %s is listed as retired but imports again --- un-retire "
              "it or fix the list" % mod)
    print("  %-30s%6s   %s"
          % ("instruments that cannot run", len(RETIRED_INSTRUMENTS),
             ", ".join(sorted(RETIRED_INSTRUMENTS))))
    print("      they are not red --- a red that can never green is a bypass "
          "being trained --- but the number is here every run and may only "
          "get smaller.")
    found["instruments"] = not (broken or back)

    # ...AND HER BODY, IMPORTED AND ACTUALLY RUN.  See `RUNS` for why both:
    # `body/sounds.py` called a method that had been deleted, imported green
    # every day for a week, and died the moment anything asked it to work.
    hurt = bodyImports()
    print("  %-30s%6s" % ("every body/* imports", "ok" if not hurt else "NO"))
    for mod, why in hurt:
        bad += 1
        print("      body.%s CANNOT IMPORT --- %s" % (mod, why))
    found["her body imports"] = not hurt
    dead = bodyRuns()
    print("  %-30s%6s   %d wires" % ("...and her body RUNS",
                                     "ok" if not dead else "NO", len(RUNS)))
    for what, why in dead:
        bad += 1
        print("      %s --- %s" % (what, why))
    found["her body runs"] = not dead

    print(f"\n  written to measured/state.json at {head()[:8] or '(no git)'}")
    print("  " + ("everything she is checked for holds."
                  if not bad else f"{bad} of {len(found)} say NO."))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
