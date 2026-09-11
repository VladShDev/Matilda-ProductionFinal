"""THE DOCTOR'S ITEM 6 --- CALMS WHEN HELD (CDC 2 months: "calms down when
spoken to or picked up").

She has no cry (his word: babbling is her cry), so the doctor's flow is: a
free baby, then picked up, and what is seen from outside before and after.
Picked up = carried under her chest and hips (the crawl hand; the first run
used the wrap, which pins the very joints it read).  Her limbs hang free, so
what is read is her limbs' moving (arms and legs, 20 spindles) and her
sounding.

Born from nothing, unwrapped, her mother on, the room's hum first.  Minutes 1-3
free; picked up at minute 3, held for a minute.  Read: her limbs' moving and
her sounding in the 60 s before and the 60 s after.  Control: the same
before/after across minute 2 -> minute 3, free.  PASS (provisional bar, his
word to set) if her limbs' moving falls more across the pick-up than across
the control boundary; her sounding is reported beside it (a quiet baby has
nothing to fall).

    python tests/doctor_test8.py <name>          (repo root, her python)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from body.hearing import TICK_SECONDS
from body.joints import AXES
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test8"
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
    until(30 + 180)
    wrapTick = st().get("tick")
    post("/world", {"helper": "crawl"})
    until(30 + 240)
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

from measure.record import frames
limb = np.array([any(k in n for k in ("shoulder", "elbow", "hip", "knee", "ankle")) for n in AXES])
S, E, T = [], [], []
for t in frames(tape):
    sp = sorted(((int(s.id), float(s.lvl)) for s in t.life.input.spindle), key=lambda x: x[0])
    S.append([lvl for i, lvl in sp if i <= len(AXES)]); E.append(float(t.life.input.echo.lvl) > 0)   # ids 1..26 are her body's axes
S = np.array(S, float); E = np.array(E); N = len(S)
d = np.r_[np.zeros((1, S.shape[1])), np.abs(np.diff(S, axis=0))]
neckMove = d[:, limb].mean(axis=1)
sec = lambda k: int(round(k * FPS))
w = N - sec(60)                                   # the wrap fell 60 s before the end
def block(a, b):
    return neckMove[a:b].mean() * 1000, 100 * E[a:b].mean()
bN, bS = block(w - sec(60), w); aN, aS = block(w + sec(5), w + sec(60))
cN0, cS0 = block(w - sec(120), w - sec(60)); cN1, cS1 = block(w - sec(60), w)
print("\n=== ITEM 6: CALMS WHEN HELD (free three minutes, then carried under chest and hips for one; her limbs and her voice) ===")
print("across the pick-up:  her limbs' moving %.2f -> %.2f (x1000/tick); her sounding %.1f%% -> %.1f%% of ticks" % (bN, aN, bS, aS))
print("control, free:       minute 2 -> minute 3: limbs %.2f -> %.2f; sounding %.1f%% -> %.1f%%" % (cN0, cN1, cS0, cS1))
fallN = (bN - aN) - (cN0 - cN1); fallS = (bS - aS) - (cS0 - cS1)
print("CALMS WHEN HELD   limbs fall %+.2f more than the control, sounding %+.1f points more   %s   (provisional bar: the limbs fall more than the control, his word)"
      % (fallN, fallS, "PASS" if fallN > 0 else "NOT YET"))
print("the visit: closes %s replays %s trials %s hunger %.3f; ticks %d" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0), N))
