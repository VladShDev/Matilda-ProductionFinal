"""THE DOCTOR'S TEST 12 --- HOLDS HER HEAD UP WHEN HELD (CDC 4 months: "holds
head steady without support when you are holding him"; HINE: ventral
suspension --- held horizontally under the trunk, the head's position noted).

The doctor lifts the baby with one hand under her chest and one under her
hips and watches the head.  Her room has exactly that hand: the "crawl"
helper (one hand under her chest, one under her hips, the head hers).  Born
from nothing, unwrapped, her mother on, the room's hum first; then three holds
of ten seconds with twenty seconds free between them.

Read from outside, 5 times a second, from her frames: her head's height above
her chest.  Control: the first second of each hold (she is lifted limp).
Reading: the last five seconds of the hold.  PASS (provisional bar, his word
to set) if in two of the three holds her head sits at least 2 cm higher above
her chest than in the first second.

    python tests/doctor_test6.py <name>          (repo root, her python)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind"), os.path.join(ROOT, "sandbox")]
from body import speech
from body.ragdoll import JIDX
import film
JOINTS = list(JIDX)
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test6"
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
tape = "mind/lives/%s.duckdb" % name


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body or {}).encode()
    req = urllib.request.Request(AT + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


class Pose:
    """Her joints' positions from her frames, the latest row only."""
    def __init__(self):
        self.k = 0; self.last = None

    def __call__(self):
        with urllib.request.urlopen(AT + "/frames?from=%d" % self.k, timeout=5) as r:
            blob = r.read()
        first, latest, wide, rows = film.unpack(blob)
        if rows:
            v = film.decode(bytes(rows[-1]))
            self.last = {n: np.array(v[3 * i:3 * i + 3], float) for i, n in enumerate(JOINTS)}
            self.k = latest
        return self.last


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
holds = []
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
    for _ in range(40):
        try:
            post("/world", {"helper": None}); time.sleep(0.5)
            if st().get("helper") in (None, "", "None"):
                break
        except Exception:
            time.sleep(0.5)
    t0 = time.time()

    def until(sec):
        while time.time() - t0 < sec:
            time.sleep(0.2)
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    pose = Pose()
    for h in range(3):
        until(40 + 30 * h)
        post("/world", {"helper": "crawl"})
        rows = []
        t1 = time.time()
        while time.time() - t1 < 10.0:
            p = pose()
            if p is not None:
                rows.append((time.time() - t1, p["head"][1] - p["chest"][1], p["chest"][1] - p["pelvis"][1], p["head"][1]))
            time.sleep(0.2)
        post("/world", {"helper": None})
        holds.append(rows)
    until(40 + 30 * 3)
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

print("\n=== TEST 12: HOLDS HER HEAD UP WHEN HELD (three ten-second holds under chest and hips, unwrapped, her mother on) ===")
passes = 0
for h, rows in enumerate(holds):
    if not rows:
        print("hold %d: no frames read" % (h + 1)); continue
    a = np.array(rows)
    first = a[a[:, 0] < 1.0]; last = a[a[:, 0] >= 5.0]
    up0 = first[:, 1].mean() if len(first) else float("nan"); up1 = last[:, 1].mean() if len(last) else float("nan")
    trunk = last[:, 2].mean() if len(last) else float("nan")
    ok = (up1 - up0) >= 0.02
    passes += int(ok)
    print("hold %d: head above chest %.3f m in the first second -> %.3f m in the last five (%+.3f); chest above pelvis %.3f m (0 = trunk level); head height %.2f m   %s"
          % (h + 1, up0, up1, up1 - up0, trunk, last[:, 3].mean() if len(last) else float("nan"), "up" if ok else "not up"))
print("HOLDS HER HEAD UP WHEN HELD   %d of 3 holds   %s   (provisional bar: +2 cm over the first second, his word to set)"
      % (passes, "PASS" if passes >= 2 else "NOT YET"))
print("the visit: closes %s replays %s trials %s hunger %.3f" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0)))
