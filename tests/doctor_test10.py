"""THE DOCTOR'S ITEM 9 --- REACHES FOR A THING (CDC 4 months: "uses his arm to
swing at toys"; 6 months: "reaches to grab a toy she wants").

The doctor holds a toy near the baby's hand, in her view, and watches the
hand.  Here the toy is the ball on the flying screen, flown to 12 cm above
her right hand and a little toward her face.  Born from nothing, unwrapped,
her mother on, the room's hum first.  Two free minutes (the control: her
right hand's distance to the place the ball will be); then the ball there
for three minutes.  Read from her frames, 5 times a second: her right hand's
distance to the ball, and touches (within 5 cm).  PASS (provisional bar, his
word to set) if her hand is on average nearer the ball while it is there
than it was to that place before, by more than 3 cm, or touches it.

    python tests/doctor_test10.py <name>          (repo root, her python)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind"), os.path.join(ROOT, "sandbox")]
from body import speech
from body.ragdoll import JIDX
from measure import cards
import film
JOINTS = list(JIDX)
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test10"
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


class Pose:
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
before, during = [], []
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
    pose = Pose(); until(35)
    p = pose()
    g = st().get("gaze") or {}
    HEAD = np.array(g.get("head"), float)
    toFace = HEAD - p["haR"]; toFace[1] = 0.0; toFace /= max(1e-6, np.linalg.norm(toFace))
    BALL = p["haR"] + np.array([0.0, 0.12, 0.0]) + 0.06 * toFace
    LOOK = HEAD - BALL; LOOK /= max(1e-6, np.linalg.norm(LOOK))
    # the control: two free minutes, her hand against the place the ball will be
    while time.time() - t0 < 35 + 120:
        q = pose()
        if q is not None:
            before.append(float(np.linalg.norm(q["haR"] - BALL)))
        time.sleep(0.2)
    post("/window", {"at": BALL.tolist(), "look": LOOK.tolist(), "size": [0.12, 0.12]})
    showOnScreen("ball")
    while time.time() - t0 < 35 + 120 + 180:
        q = pose()
        if q is not None:
            during.append(float(np.linalg.norm(q["haR"] - BALL)))
        time.sleep(0.2)
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

b, d = np.array(before), np.array(during)
print("\n=== ITEM 9: REACHES FOR A THING (the ball 12 cm above her right hand for three minutes; two free minutes before as the control) ===")
if len(b) and len(d):
    print("her right hand to the ball's place: before %.3f m (nearest %.3f), with the ball %.3f m (nearest %.3f); touches (<5 cm): %d of %d samples"
          % (b.mean(), b.min(), d.mean(), d.min(), int((d < 0.05).sum()), len(d)))
    ok = (b.mean() - d.mean() > 0.03) or (d < 0.05).any()
    print("REACHES FOR A THING   %s   (provisional bar: 3 cm nearer on average, or a touch; his word)" % ("PASS" if ok else "NOT YET"))
else:
    print("NOT TESTED: no frames read")
print("the visit: closes %s replays %s trials %s hunger %.3f" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0)))
