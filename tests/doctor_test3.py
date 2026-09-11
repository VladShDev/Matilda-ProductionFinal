"""THE DOCTOR'S TEST 3 --- FOLLOWING.  A doctor holds a bright thing in front
of the baby's face and moves it slowly to one side and back, and watches
whether the eyes go with it.

Born from nothing, swaddled, her mother on, the room's hum first (as tests 1
and 2).  The ball is drawn on the flying screen 0.4 m along her eyes' axis
(read from her state), held still for ten seconds, then swept slowly: to 30
degrees on her right over ten seconds, back to the middle, to 30 degrees on
her left, back --- twice.  Her eyes' axis is read five times a second from
her state (her body's own geometry, the one answer to where she looks) and
set against where the card is at that moment.

  FIXATES THE MOVING CARD   PASS if her eyes' axis is within 10 degrees of the
                            card more of the sweeps than of the still minute
                            before the card came.
  FOLLOWS                   PASS if, over the sweeps, her gaze's sideways angle
                            goes with the card's (correlation above zero) and
                            her gaze moved at all (its range above her eyes'
                            resolution).

Nothing of her chemistry is read; nothing in her is told.
    python tests/doctor_test3.py <name>          (repo root, her venv)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from measure import cards
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test3"
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


def showOnScreen(name):
    img = cards.draw(name) if name != "blank" else np.full((cards.SIZE, cards.SIZE, 3), 235, np.uint8)
    req = urllib.request.Request(AT + "/show", data=np.ascontiguousarray(img).tobytes(), method="POST",
                                 headers={"X-Width": str(img.shape[1]), "X-Height": str(img.shape[0])})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def turned(face, deg):
    """Her face's line turned `deg` toward HER RIGHT (up x face), in degrees."""
    right = np.cross(np.array([0.0, 1.0, 0.0]), face); right /= max(1e-6, np.linalg.norm(right))
    th = np.radians(deg)
    v = np.cos(th) * np.asarray(face, float) + np.sin(th) * right
    return v / max(1e-6, np.linalg.norm(v))


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
herlog = open(tape + ".her.log", "w", encoding="utf-8")
proc = subprocess.Popen([sys.executable, "-u", "her.py", "--keep", name + ".duckdb"], cwd=ROOT, env=ENV,
                        stdout=herlog, stderr=subprocess.STDOUT)
samples = []          # (seconds, block, card angle in degrees, her gaze dir)
block = ["start"]; cardDeg = [0.0]
HEAD = None; FACE = None
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
            post("/world", {"helper": "cradle"})
            time.sleep(0.5)
            if st().get("helper") == "cradle":
                break
        except Exception:
            time.sleep(0.5)
    assert st().get("helper") == "cradle", "the wrap did not take"
    g = st().get("gaze") or {}
    HEAD = np.array(g.get("head") or [0.20, 0.42, -0.34], float)
    FACE = np.array(g.get("dir") or [-0.75, -0.56, 0.35], float); FACE /= max(1e-6, np.linalg.norm(FACE))
    t0 = time.time()
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()

    def place(deg):
        d = turned(FACE, deg)
        post("/window", {"at": (HEAD + 0.4 * d).tolist(), "look": (-d).tolist(), "size": [0.30, 0.22]})
        cardDeg[0] = deg

    def until(sec):
        while time.time() - t0 < sec:
            try:
                s = st()
                samples.append((round(time.time() - t0, 1), block[0], cardDeg[0], (s.get("gaze") or {}).get("dir")))
            except Exception:
                pass
            time.sleep(0.2)

    def sweep(a, b, seconds, start):
        steps = int(seconds / 0.5)
        for k in range(steps + 1):
            place(a + (b - a) * k / steps)
            until(start + seconds * k / steps)
    place(0.0); showOnScreen("blank")
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    block[0] = "still-blank"; until(90)
    block[0] = "still-ball"; showOnScreen("ball"); print("ball up, still, at %.0f s" % (time.time() - t0), flush=True)
    until(100)
    t = 100.0
    for rnd in (1, 2):
        for a, b in ((0, 30), (30, 0), (0, -30), (-30, 0)):
            block[0] = "sweep%d %+d>%+d" % (rnd, a, b); sweep(a, b, 10.0, t); t += 10.0
    print("sweeps done at %.0f s" % (time.time() - t0), flush=True)
    block[0] = "end"; place(0.0); until(t + 5)
    try:
        showOnScreen("blank"); fin = st()
    except Exception as e:                                          # noqa: BLE001
        print("at the end her body did not answer: %s" % e, flush=True)
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)
json.dump(samples, open(tape + ".follow.json", "w", encoding="utf-8"))

# --------------------------------------------------------------- the sheet
good = [s for s in samples if s[3]]
deg = np.array([s[2] for s in good]); blk = np.array([s[1] for s in good])
dirs = np.array([s[3] for s in good], float)


def angleTo(cardDegs):
    out = []
    for d, c in zip(dirs, cardDegs):
        cd = turned(FACE, c); out.append(np.degrees(np.arccos(np.clip(float(d @ cd), -1, 1))))
    return np.array(out)


ang = angleTo(deg)
# her gaze's own sideways angle, in the same frame as the card's: signed, right positive
side = np.array([turned(FACE, 90.0) @ d for d in dirs]); gazeDeg = np.degrees(np.arcsin(np.clip(side, -1, 1)))
still = blk == "still-blank"; sweeps = np.array([b.startswith("sweep") for b in blk])
print("\n=== TEST 3: FOLLOWING (what a doctor sees: her eyes against the moving card) ===")
print("her head at %s, her eyes' axis at rest %s; samples %d" % (np.round(HEAD, 2).tolist(), np.round(FACE, 2).tolist(), len(good)))
print("her eyes within 10 degrees of the card: still blank minute %.0f%%, still ball %.0f%%, during the sweeps %.0f%%"
      % (100 * (ang[still] < 10).mean() if still.any() else 0, 100 * (ang[blk == "still-ball"] < 10).mean() if (blk == "still-ball").any() else 0,
         100 * (ang[sweeps] < 10).mean() if sweeps.any() else 0))
r = float(np.corrcoef(deg[sweeps], gazeDeg[sweeps])[0, 1]) if sweeps.sum() > 2 and gazeDeg[sweeps].std() > 0 else 0.0
rng = float(gazeDeg[sweeps].max() - gazeDeg[sweeps].min()) if sweeps.any() else 0.0
for b in sorted(set(blk[sweeps]), key=lambda x: [s[1] for s in good].index(x)):
    m = blk == b
    print("  %-16s card %+3.0f..%+3.0f deg: her gaze %+5.1f..%+5.1f deg, median angle to the card %4.1f" % (b, deg[m][0], deg[m][-1], gazeDeg[m].min(), gazeDeg[m].max(), np.median(ang[m])))
print("her eyes within 10 deg of the moving card %.0f%% of the sweeps vs %.0f%% of the still minute   (reported, no bar: fixation dropped on his word 2026-09-07)" % (100 * (ang[sweeps] < 10).mean() if sweeps.any() else 0.0, 100 * (ang[still] < 10).mean() if still.any() else 0.0))
print("FOLLOWS                   %s   (her gaze with the card: correlation %+.2f; her gaze's range over the sweeps %.1f degrees)" % ("PASS" if r > 0 and rng > 2.0 else "NOT YET", r, rng))
