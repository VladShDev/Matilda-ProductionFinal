"""HIS CHECK ON ME --- one command, row by row, and he runs it, not I.

His words, 2026-09-11: *"how I can check you? ... what option I have to check
finally and confirm that the script works fine without some, you know, you're
quickly finding of some mistake."*

So this is not a report I write.  It is a machine that runs her, and every row
is a thing the machine did, with the number it got.  A row that says HOLDS was
executed.  A row that says BROKEN prints the exact line to show him.

WHY IT EXISTS.  `measure.check` never imports her mind, so a mind that could
not even be loaded went green three times.  And nothing ever woke a life from
its own record, so a line that crashes every `--keep` sat in her for as long as
it took him to try one.  Both are rows here.

    py.cmd -m measure.verify            every row (about two minutes; it stops any live body)
    py.cmd -m measure.verify --still    only the rows that do not need to run her
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HER = os.path.join(ROOT, "mind", "lives", "verify.duckdb")
AT = "http://127.0.0.1:8090"
STOP = (r"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        r"'sandbox[\\/]app\.py|life\.py lives|her\.py' } | ForEach-Object { "
        r"Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")

rows: list = []


def row(name, ok, said) -> bool:
    rows.append((name, bool(ok), str(said)))
    print("  %-32s %-7s %s" % (name, "HOLDS" if ok else "BROKEN", said), flush=True)
    return bool(ok)


def state(timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(AT + "/state", timeout=timeout) as r:
        return json.loads(r.read())


def hers() -> list:
    """Every file of hers, so no row can quietly skip one."""
    out = []
    for sub in ("body", "mind", "measure", "sandbox", "tests"):
        for base, _, files in os.walk(os.path.join(ROOT, sub)):
            if "__pycache__" in base:
                continue
            out += [os.path.join(base, f) for f in files if f.endswith(".py")]
    return out + [os.path.join(ROOT, "her.py")]


# ---------------------------------------------------------------- the rows
def compiles() -> bool:
    bad = []
    for f in hers():
        try:
            ast.parse(open(f, encoding="utf-8", errors="replace").read(), f)
        except SyntaxError as e:
            bad.append("%s:%s %s" % (os.path.relpath(f, ROOT), e.lineno, e.msg))
    return row("every file of hers parses", not bad,
               "%d files" % len(hers()) if not bad else bad[0])


def loads() -> bool:
    """HER MIND IMPORTS.  `measure.check` does not do this and never did."""
    got = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'mind'); "
         "import life, hormones, discover, store, structure, sandbox"],
        cwd=ROOT, capture_output=True, text=True)
    last = (got.stderr or "").strip().splitlines()
    return row("her mind imports", got.returncode == 0,
               "six files" if got.returncode == 0 else (last[-1] if last else "?"))


def wired() -> bool:
    got = subprocess.run([sys.executable, "-m", "measure.check"],
                         cwd=ROOT, capture_output=True, text=True)
    said = (got.stdout or "") + (got.stderr or "")
    ok = "everything she is checked for holds" in said
    tail = [l for l in said.strip().splitlines() if l.strip()]
    return row("her body's wires", ok,
               "every wire" if ok else (tail[-1] if tail else "?"))


def voice() -> bool:
    got = subprocess.run([sys.executable, "-m", "measure.voice"],
                         cwd=ROOT, capture_output=True, text=True)
    said = (got.stdout or "") + (got.stderr or "")
    ok = "THE VOICE HOLDS" in said
    tail = [l for l in said.strip().splitlines() if l.strip()]
    return row("her mouth, sample for sample", ok,
               "his certified word" if ok else (tail[-1] if tail else "?"))


def unreached() -> bool:
    """WHAT NOTHING CALLS.  Not a pass or a fail --- a list, so he can hold me
    to leaving nothing of an older her in the folder.  A name defined in her
    body, her mind or her room and never written anywhere else in the whole
    folder is dead by definition."""
    # ...EXCEPT WHAT PYTHON ITSELF CALLS.  A handler's `do_GET` is never
    # written anywhere in her folder and runs on every request: `http.server`
    # reaches for it by name.  Naming those here is the whole exception, and it
    # is short on purpose --- anything else on this list is really dead.
    theirs = {"do_GET", "do_POST", "do_HEAD", "log_message", "log_error",
              "handle_one_request", "setup", "finish"}
    made, said = {}, set(theirs)
    owns = tuple(os.path.join(ROOT, s) for s in ("body", "mind", "sandbox"))
    for f in hers():
        try:
            tree = ast.parse(open(f, encoding="utf-8", errors="replace").read(), f)
        except SyntaxError:
            continue
        own = f.startswith(owns)
        for n in ast.walk(tree):
            if own and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not n.name.startswith("__"):
                    made.setdefault(n.name, "%s:%d" % (os.path.relpath(f, ROOT), n.lineno))
            elif isinstance(n, ast.Attribute):
                said.add(n.attr)
            elif isinstance(n, ast.Name):
                said.add(n.id)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                said.update(n.value.replace("(", " ").replace(".", " ").split())
    dead = sorted(k for k in made if k not in said)
    print("  %-32s %-7s %d of %d defined" % ("nothing of an older her", "LIST",
                                             len(dead), len(made)), flush=True)
    for k in dead:
        print("        nothing calls %-24s %s" % (k, made[k]), flush=True)
    rows.append(("nothing of an older her", True, "%d never called" % len(dead)))
    return True


# --------------------------------------------------- the rows that run her
def quiet() -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    for _ in range(40):
        try:
            state(2.0)
            time.sleep(0.5)
        except Exception:
            return


def start():
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    return subprocess.Popen([sys.executable, "her.py", "--keep", "verify.duckdb"],
                            cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def lives(proc, seconds: float) -> dict:
    """Wait for her, then watch her.  Her mind is a child of her body: if it
    dies her body still answers /state, so the proof her mind is alive is that
    HER OWN numbers are there and move --- `closes`, `feelMs`."""
    was = None
    for _ in range(90):
        if proc.poll() is not None:
            return {"died": (proc.stderr.read() or b"").decode("utf8", "replace")}
        try:
            was = state()
            if was.get("tick") is not None:
                break
        except Exception:
            pass
        time.sleep(2)
    if was is None or was.get("tick") is None:
        return {"died": "she never answered"}
    began, feels, behinds = time.time(), [], []
    while time.time() - began < seconds:
        time.sleep(2)
        try:
            now = state()
        except Exception:
            continue
        if now.get("feelMs") is not None:
            feels.append(float(now["feelMs"]))
        if now.get("behind") is not None:
            behinds.append(float(now["behind"]))
        was = now
    return {"state": was, "feels": feels, "behinds": behinds}


def born() -> dict:
    quiet()
    for old in (HER, HER + ".wal"):
        if os.path.exists(old):
            os.remove(old)
    proc = start()
    try:
        got = lives(proc, 45.0)
        if "died" in got:
            row("she is born", False,
                (got["died"].strip().splitlines() or ["?"])[-1])
            return got
        st, feels = got["state"], got["feels"]
        row("she is born", bool(feels) and st.get("closes", 0) > 0,
            "%d cuts, %d trials, %d replays"
            % (st.get("closes", 0), st.get("laddered", 0), st.get("replays", 0)))
        return got
    finally:
        proc.kill()
        quiet()


def pace(got: dict) -> bool:
    """HER CLOCK.  A tick is 1/90 s = 11.1 ms.  If feeling one costs her more
    than that she is living slower than her own life, and every number after it
    is a number from a different baby.  `behind` is her body's own word for it."""
    feels = got.get("feels") or []
    behinds = got.get("behinds") or []
    if not feels:
        return row("she keeps her own clock", False, "she never felt a tick")
    worst, slip = max(feels), (max(behinds) if behinds else -1.0)
    return row("she keeps her own clock", worst < 11.1 and slip <= 0.0,
               "%.1f ms a tick (her clock is 11.1), behind %.3f" % (worst, slip))


def wakes(got: dict) -> bool:
    """THE WAKE.  The life just born, opened again.  This is the row that was
    never here: every experience she holds is read back through `Store.hands()`
    in `Life._remember`, and nothing ran that line until he tried --keep."""
    if not os.path.exists(HER):
        return row("she wakes with her memory", False, "no life to wake")
    before = (got.get("state") or {}).get("closes", 0)
    proc = start()
    try:
        now = lives(proc, 25.0)
        if "died" in now:
            return row("she wakes with her memory", False,
                       (now["died"].strip().splitlines() or ["?"])[-1])
        st = now["state"]
        kept = st.get("closes", 0)
        return row("she wakes with her memory",
                   bool(now["feels"]) and kept >= before,
                   "%d cuts carried of %d" % (kept, before) if now["feels"]
                   else "her body answers, her mind is dead")
    finally:
        proc.kill()
        quiet()


def main() -> int:
    still = "--still" in sys.argv[1:]
    print("\nHIS CHECK --- every row below is a thing that ran.\n")
    compiles()
    loads()
    wired()
    voice()
    unreached()
    if not still:
        print()
        got = born()
        pace(got)
        wakes(got)
    broke = [n for n, ok, _ in rows if not ok]
    print()
    if broke:
        print("SHE DOES NOT HOLD: " + ", ".join(broke))
        return 1
    print("SHE HOLDS --- every row above ran and came back right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
