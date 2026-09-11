"""A BIRTH FROM NOTHING, WATCHED --- no tape, no guide, no map: her body, her
mother, and the clock not waiting (his fast-living harness, `sprint`).  His,
2026-09-06: *"let's do this honest ... run this process ... be precise as
possible."*  What is counted, hour by hour of HER time: her own sounds
(distinct echo ids), her mother's answers and milks and namings, her
experiences closed, her replays and trials, her hunger and mood, her mind's
tick cost and how far her body fell behind.  Read-only on the record.

    python tests/birthwatch.py <name> [wall minutes=20]
"""
import json, os, subprocess, sys, time, urllib.request
import duckdb
ROOT = os.getcwd()
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "birth"
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(AT + path, data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


tape = "mind/lives/%s.duckdb" % name
for old in (tape, tape + ".wal"):
    if os.path.exists(old):
        os.remove(old)
subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
# ...AND WAIT FOR THE OLD BODY TO BE GONE.  A body from a run before may still
# answer /state for a second after it is told to stop; a test that took that
# answer for the new body then met "connection refused" while the new one was
# still checking her voice (2026-09-06, test 1's first rerun).
for _ in range(40):
    try:
        st(); time.sleep(0.5)
    except Exception:
        break
# HER BODY GETS THE GPU (his rule: run her only on GPU).  An instrument hides the
# GPU from itself (CUDA_VISIBLE_DEVICES=""), and her.py must not inherit that:
# on 2026-09-06 every life started by a runner ran her eye on the CPU.
ENV = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
proc = subprocess.Popen([sys.executable, "her.py", "--keep", name + ".duckdb", "--fast"], cwd=ROOT, env=ENV,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rows = []
try:
    for _ in range(90):
        try:
            if st().get("tick") is not None:
                break
        except Exception:
            pass
        time.sleep(2)
    for _ in range(60):
        s = st()
        if (s.get("teacher") or {}).get("on"):
            break
        time.sleep(0.5)
    t0 = time.time()
    print("%7s %7s %7s %6s %6s %6s %6s %5s %5s %5s %6s %5s" % (
        "wall s", "her s", "closes", "replay", "trials", "hunger", "mood", "answ", "milks", "named", "feelMs", "behind"), flush=True)
    nxt = 0.0
    while time.time() - t0 < MINUTES * 60.0:
        try:
            s = st()
        except Exception:
            time.sleep(1.0)
            continue
        if time.time() - t0 >= nxt:
            m = ((s.get("teacher") or {}).get("meter") or {})
            print("%7.0f %7.0f %7s %6s %6s %6.3f %6.3f %5s %5s %5s %6s %5.1f" % (
                time.time() - t0, s.get("seconds", 0), s.get("closes"), s.get("guided"), s.get("laddered"),
                s.get("hunger", 0), s.get("state", 0), m.get("answers"), m.get("milks"), m.get("named"),
                s.get("feelMs"), float(s.get("behind", 0))), flush=True)
            rows.append(s)
            nxt += 60.0
        time.sleep(2.0)
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

from measure.record import frames
n = 0; t0r = None; firsts = None; ids = set(); sounding = 0; heard = 0
for t in frames(tape):
    if t0r is None: t0r = t.time
    n += 1
    if t.life.input.echo.lvl > 0:
        ids.add(int(t.life.input.echo.id)); sounding += 1
        if firsts is None: firsts = t.time - t0r
    if t.life.input.sound.lvl > 0: heard += 1
hours = [(0, len(ids), sounding, heard, n)]
allsounds = len(ids)
c = duckdb.connect(tape, read_only=True); exps = c.execute("select count(*) from exp").fetchone()[0]; c.close()
print("\nborn from nothing: %d ticks (%.1f min of her time) in %.0f wall minutes; %d experiences closed"
      % (n, n / 90.0 / 60.0, MINUTES, exps))
print("her first sound at +%s s of her time; %d distinct sounds of her own in all"
      % ("%.1f" % firsts if firsts is not None else "never", allsounds))
print("her hour  distinct sounds  ticks sounding  ticks hearing her mother  ticks")
for h, snd, sng, hrd, tk in hours:
    print("   %2d        %5d           %6d            %6d             %6d" % (h, snd or 0, sng or 0, hrd or 0, tk))
print("final: hunger %.3f  mood %.3f  closes %s  replays %s  trials %s  mother answers %s milks %s named %s  feelMs %s  behind %s"
      % (fin.get("hunger", 0), fin.get("state", 0), fin.get("closes"), fin.get("guided"), fin.get("laddered"),
         ((fin.get("teacher") or {}).get("meter") or {}).get("answers"), ((fin.get("teacher") or {}).get("meter") or {}).get("milks"),
         ((fin.get("teacher") or {}).get("meter") or {}).get("named"), fin.get("feelMs"), fin.get("behind")))
