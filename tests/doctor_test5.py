"""THE DOCTOR'S TEST 11 --- MOVES BOTH SIDES ALIKE, SMOOTHLY (HINE: movement
quantity and quality, the most predictive items at every age of the first
year; CDC 2 months: "moves both arms and both legs").

A doctor lays the baby down unwrapped and watches her move for a while.
Born from nothing, UNSWADDLED, her mother on, the room's hum first, then a
quiet ten minutes.  Read from her record, from outside: her spindles (where
each joint axis is, 0..1), tick by tick.

  QUANTITY   how much she moves: mean |change of a spindle| per tick, per
             minute, for her left side, her right side and her midline
             (spine, neck).  Every minute she moves at all on both sides is a
             moving minute.  PASS if at least 8 of the 10 minutes are moving
             minutes on both sides.
  SYMMETRY   over the visit, the side that moves less moves at least half as
             much as the side that moves more.  PASS if so.
  SMOOTHNESS the share of her moving ticks where the change reverses
             direction from the tick before (jerk), per side; reported, no bar
             (a bar needs his word: what is jerky for her).

Control: her own minutes against each other; a wrapped minute (the last
minute, swaddled) as the floor of "not moving".

    python tests/doctor_test5.py <name> [quiet seconds]   (repo root, her python)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from body.hearing import TICK_SECONDS
from body.joints import AXES
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test5"
QUIET = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
FPS = 1.0 / TICK_SECONDS
tape = "mind/lives/%s.duckdb" % name


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body or {}).encode()
    req = urllib.request.Request(AT + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


for old in (tape, tape + ".wal"):
    if os.path.exists(old):
        os.remove(old)
subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
for _ in range(40):
    try:
        st(); time.sleep(0.5)
    except Exception:
        break
ENV = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
proc = subprocess.Popen([sys.executable, "her.py", "--keep", name + ".duckdb"], cwd=ROOT, env=ENV,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(90):
        try:
            if st().get("tick") is not None:
                break
        except Exception:
            pass
        time.sleep(2)
    for _ in range(60):
        try:
            if (st().get("teacher") or {}).get("on"):
                break
        except Exception:
            pass
        time.sleep(0.5)
    # UNWRAPPED: no helper at all, read back
    for _ in range(40):
        try:
            post("/world", {"helper": None})
            time.sleep(0.5)
            if st().get("helper") in (None, "", "None"):
                break
        except Exception:
            time.sleep(0.5)
    assert st().get("helper") in (None, "", "None"), "she is still held: %s" % st().get("helper")
    t0 = time.time()

    def until(sec):
        while time.time() - t0 < sec:
            time.sleep(0.2)
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    until(30 + QUIET)
    # the floor of "not moving": one wrapped minute at the end
    post("/world", {"helper": "cradle"})
    until(30 + QUIET + 60)
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

# ------------------------------------------------------------- her record
from measure.record import frames
rows = []
for t in frames(tape):
    sp = sorted(((int(s.id), float(s.lvl)) for s in t.life.input.spindle), key=lambda x: x[0])
    rows.append([lvl for i, lvl in sp if i <= len(AXES)])        # spindle ids 1..26 are her body's axes; her eyes and mouth follow
S = np.array(rows, float)                                      # ticks x 26, spindle id k+1 = AXES[k]
N = len(S)
names = list(AXES)
left = np.array(["_left" in n for n in names]); right = np.array(["_right" in n for n in names])
mid = ~(left | right)
d = np.abs(np.diff(S, axis=0)); d = np.r_[np.zeros((1, S.shape[1])), d]
sgn = np.sign(np.diff(S, axis=0)); rev = np.r_[np.zeros((1, S.shape[1])), np.r_[np.zeros((1, S.shape[1])), (sgn[1:] * sgn[:-1] < 0).astype(float)]]
start = int(round(50 * FPS)); quietEnd = int(round((30 + QUIET) * FPS))
sec = lambda k: int(round(k * FPS))
print("\n=== TEST 11: MOVES BOTH SIDES ALIKE, SMOOTHLY (%d s of her unwrapped, her mother on; then one wrapped minute) ===" % int(QUIET))
print("minute   left     right    midline   jerk L  jerk R   (mean |spindle change| per tick x1000; jerk = share of moving ticks that reverse)")
mins = []
for m in range(int(QUIET // 60)):
    a, b = start + sec(60 * m), start + sec(60 * (m + 1))
    if b > quietEnd:
        break
    L = d[a:b][:, left].mean() * 1000; R = d[a:b][:, right].mean() * 1000; M = d[a:b][:, mid].mean() * 1000
    mvL = d[a:b][:, left] > 0; mvR = d[a:b][:, right] > 0
    jL = rev[a:b][:, left][mvL].mean() if mvL.any() else 0.0; jR = rev[a:b][:, right][mvR].mean() if mvR.any() else 0.0
    mins.append((L, R, M, jL, jR))
    print("  %2d    %6.2f   %6.2f   %6.2f    %.2f    %.2f" % (m + 1, L, R, M, jL, jR))
wrapped = d[quietEnd + sec(5):quietEnd + sec(60)]
wL = wrapped[:, left].mean() * 1000; wR = wrapped[:, right].mean() * 1000
print("  wrapped %6.2f   %6.2f   (the floor: swaddled)" % (wL, wR))
mv = [(L > wL and R > wR) for L, R, M, jL, jR in mins]
totL = sum(L for L, R, M, jL, jR in mins); totR = sum(R for L, R, M, jL, jR in mins)
ratio = min(totL, totR) / max(totL, totR) if max(totL, totR) > 0 else 0.0
print("QUANTITY   moving minutes (both sides above the wrapped floor): %d of %d   %s" % (sum(mv), len(mv), "PASS" if len(mv) >= 5 and sum(mv) >= 0.8 * len(mv) else "NOT YET"))
print("SYMMETRY   the lesser side moves %.0f%% of the greater (left %.2f, right %.2f)   %s" % (100 * ratio, totL, totR, "PASS" if ratio >= 0.5 else "NOT YET"))
print("SMOOTHNESS jerk: left %.2f, right %.2f (reported, no bar)" % (np.mean([jL for L, R, M, jL, jR in mins]), np.mean([jR for L, R, M, jL, jR in mins])))
print("the visit: closes %s replays %s trials %s hunger %.3f; ticks %d" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0), N))
