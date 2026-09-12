"""HER, IN ONE COMMAND.

    python her.py                 she lives --- body, mind, and her room
    python her.py --keep NAME     ...continuing a life instead of a new one
    python her.py --port 8091     ...somewhere else

His, 2026-09-01, and it is the whole reason this file exists: *"i will rest
only when we build her proper and i can run her from her own independent
rep/folder just by one command run and she work as needed proper way withou
dancig with drums around her day by day."*

WHAT THE DRUMS WERE.  Starting her meant remembering: which flags (`--brain
outside --mind ../matilda4`), a FRESH tape name (an old one parks her forever
on "waiting for the body to be restored"), that her mouth must exist before
she can speak, and then four `POST /world` calls to put her room right --- her
mother on, no helper, her bottle armed.  Forget one and she lives a wrong life
that looks exactly like a right one.  Every one of those is now here.

WHAT IT REFUSES TO DO.  If her voice is broken it does not start her.
`measure.voice` is her monolith's wall (his: *"our voice is like monolith"*)
and it takes under a second; a girl whose mouth has regressed should not spend
an hour babbling before anybody notices.

HER ROOM, AS HE SETTLED IT:

    her mother   ON.  She is his button, and he has pressed it.
    helper       NONE.  His, 2026-09-01: *"it is not halps her at all even
                 worse it training her just make lags atraight and relax and
                 everithing will move around and you can be layzy"*.  The
                 WALKER (`{"helper": "walk"}`) is different and earns its
                 place --- squat, rails and a drifting floor made 137 steps in
                 four minutes --- but it is a thing done TO her for a stretch,
                 not how she lives.
    bottle       NONE.  His, 2026-09-02, on the audit: *"autofeed off"* ---
                 the armed bottle fed her at hunger 0.35 from her own posted
                 number, before her mother's earned milk (a word-shape, then
                 the feed) was ever needed, so calling for food could never be
                 the thing that brought it.  Starving still pins her state at
                 -1.000, and a pinned state is a run that never closes; that
                 is reported, not hidden.

It leaves the terminal attached: Ctrl-C stops her body and her mind together.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
#: HER MIND LIVES INSIDE HER NOW.  One folder, one command --- his ask,
#: 2026-09-02: *"she has to be monolith ... able to run by one comment"*.
#: It was `../matilda4`, a sibling folder, which meant she could only be
#: started from a tree that had one beside it.
MIND = os.path.abspath(os.path.join(HERE, "mind"))
#: NO BASE TAPE, NO GUIDE (his, 2026-09-06, on the bootstrap of 2026-09-04: a
#: helper, "a wrong brick at the base"): every life starts from nothing but
#: her body and her mother, or continues one with --keep.

#: her room, once, so nobody has to remember it
ROOM = ({"teacher": True}, {"helper": None}, {"autofeed": None})


def say(line: str) -> None:
    print(line, flush=True)


def post(at: str, path: str, body: dict) -> None:
    req = urllib.request.Request(at + path, data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=15) as h:
        h.read()


def get(at: str, path: str):
    with urllib.request.urlopen(at + path, timeout=15) as h:
        return json.loads(h.read())


def check() -> bool:
    """Her voice, before she is born.  Under a second, and it is the one
    thing that has been broken and rebuilt four times."""
    got = subprocess.run([sys.executable, "-m", "measure.voice"], cwd=HERE,
                         capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    ok = got.returncode == 0 and "THE VOICE HOLDS" in got.stdout
    for line in got.stdout.strip().splitlines()[-7:]:
        say("   " + line)
    return ok


def performanceMask() -> int:
    """THE PERFORMANCE CORES of a hybrid CPU (Windows), as an affinity mask;
    0 when the machine is not hybrid or cannot say.  Found 2026-09-04: with
    Task Manager, Docker and the overlays open, Windows scheduled her body
    onto the efficiency cores and her tick went from 11 ms to 20 (physics 17
    ms against 7, her look 300 ms against 130) --- half her clock, on which
    every constant of hers is measured.  Pinned to the performance cores she
    was back at 10.7 ms within seconds.  Nothing of hers changes here: this
    is where the machine puts her."""
    if sys.platform != "win32":
        return 0
    import ctypes
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    n = wintypes.DWORD(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))      # RelationProcessorCore
    buf = ctypes.create_string_buffer(int(n.value))
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
        return 0
    cores, off = [], 0
    while off + 8 <= n.value:
        rel, size = struct.unpack_from("II", buf, off)
        if rel == 0 and size >= 48:
            eff = struct.unpack_from("B", buf, off + 9)[0]
            mask = struct.unpack_from("Q", buf, off + 32)[0]
            cores.append((eff, mask))
        off += max(size, 8)
    if not cores:
        return 0
    top = max(e for e, _ in cores)
    if all(e == top for e, _ in cores):
        return 0
    mask = 0
    for e, m in cores:
        if e == top:
            mask |= m
    return mask


def childrenOf(pid: int) -> list:
    """The processes a process started (Windows).  The venv's python.exe is a
    LAUNCHER: it starts the real interpreter as its child, so a pin on the
    launcher alone lands on 4 MB and leaves her body where it was."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    out = []
    if snap in (0, -1):
        return out
    e = Entry(); e.dwSize = ctypes.sizeof(Entry)
    if k.Process32First(snap, ctypes.byref(e)):
        while True:
            if int(e.th32ParentProcessID) == int(pid):
                out.append(int(e.th32ProcessID))
            if not k.Process32Next(snap, ctypes.byref(e)):
                break
    k.CloseHandle(snap)
    return out


def pinToPerformance(pids, mask: int) -> int:
    """Her processes onto the performance cores, her body at high priority ---
    each pid and the processes it started.  Returns how many were pinned."""
    if not mask or sys.platform != "win32":
        return 0
    import ctypes
    k = ctypes.windll.kernel32
    done = 0
    whole = []
    for i, pid in enumerate(pids):
        if pid:
            whole.append((i, int(pid)))
            whole += [(i, c) for c in childrenOf(int(pid))]
    for i, pid in whole:
        h = k.OpenProcess(0x0200 | 0x0400, False, int(pid))
        if not h:
            continue
        ok = k.SetProcessAffinityMask(h, ctypes.c_size_t(mask))
        if i == 0:
            k.SetPriorityClass(h, 0x80)                                # HIGH_PRIORITY_CLASS, her body
        k.CloseHandle(h)
        done += 1 if ok else 0
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--keep", default=None,
                    help="continue this life (a .duckdb in mind/lives)")
    ap.add_argument("--fast", action="store_true",
                    help="her body does not wait for the clock: she lives as fast as it "
                         "computes (his fast-living harness, sprint); every law in her seconds")
    ap.add_argument("--no-check", action="store_true",
                    help="start her even if her voice is red")
    got = ap.parse_args()
    os.chdir(HERE)
    at = "http://127.0.0.1:%d" % got.port

    say("checking her voice")
    if not check() and not got.no_check:
        say("\nHER VOICE IS BROKEN AND SHE IS NOT STARTED.  Restore it:")
        say("  git checkout her-voice-monolith -- body/muscles.py "
            "body/alive.py measure/mouth.py")
        say("  python -m measure.mouth")
        say("(--no-check starts her anyway.)")
        return 1

    # NO TAPE.  Her body kept a pickle of every tick beside her mind's record
    # and nothing ever read it (699 MB on 2026-09-02 alone); her record is her
    # mind's DuckDB in `mind/lives/`, and her body keeps only the window
    # `/ticks` serves.  One life, one record --- his rule 4.
    args = [sys.executable, "-u", "sandbox/app.py", "--port", str(got.port),
            "--mind", MIND]
    if got.keep:
        args += ["--mind-db", got.keep]
    # ...else SHE IS BORN FROM NOTHING --- no tape, no guide, no map.  His,
    # 2026-09-06: *"it shouldn't be any tape, any experience she learned"*;
    # her mouth she finds by her own trials, her echo naming each sound.
    if got.fast:
        args += ["--sprint"]
        say("her body will not wait for the clock: she lives as fast as it computes")
    say("\nstarting her body and her mind")
    her = subprocess.Popen(args, cwd=HERE)

    for _ in range(60):
        try:
            if get(at, "/state").get("tick") is not None:
                mask = performanceMask()
                if mask:
                    got2 = pinToPerformance([her.pid, get(at, "/state").get("mindPid")], mask)
                    say("her body and her mind on the performance cores (mask %#x): %d pinned" % (mask, got2))
                break
        except Exception:                                   # noqa: BLE001
            time.sleep(1)
    else:
        her.terminate()
        say("she did not come up --- see the output above")
        return 1

    for one in ROOM:
        try:
            post(at, "/world", one)
        except Exception as why:                            # noqa: BLE001
            say("could not set her room (%s)" % why)

    a = get(at, "/state")
    time.sleep(10)
    b = get(at, "/state")
    tch = (b.get("teacher") or {})
    say("")
    say("SHE IS LIVING --- watch her at %s" % at)
    say("   %.1f ticks a second (her clock is 90)"
        % ((b["tick"] - a["tick"]) / 10.0))
    try:
        eye = str(get(at, "/meta").get("eye", "?"))
    except Exception:                                       # noqa: BLE001
        eye = "?"
    say("   her eye is on the %s%s" % (eye.upper(), "" if eye == "gpu" else
        "   <-- NOT the GPU: she will fall behind the clock; the GPU driver or CUDA_VISIBLE_DEVICES is wrong"))
    say("   her mother is %s, no helper, no bottle --- milk is earned"
        % ("here" if tch.get("on") else "NOT here"))
    say("   state %+.3f   hunger %.2f   her mouth is her own, %d muscles in all"
        % (b.get("state", 0.0), b.get("hunger", 0.0),
           int(get(at, "/able").get("motors", 0))))
    try:
        now = [one["name"] for one in get(at, "/lives").get("lives", ())
               if one.get("living")]
        say("   her life is mind/lives/%s" % (now[0] if now else "..."))
    except Exception:                                       # noqa: BLE001
        pass
    say("")
    say("Ctrl-C stops her.")
    try:
        her.wait()
    except KeyboardInterrupt:
        say("\nstopping her")
        try:
            her.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt"
                            else signal.SIGTERM)
        except Exception:                                   # noqa: BLE001
            her.terminate()
        try:
            her.wait(timeout=20)
        except Exception:                                   # noqa: BLE001
            her.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
