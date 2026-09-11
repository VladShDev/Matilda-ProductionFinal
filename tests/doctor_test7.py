"""THE DOCTOR'S TESTS 13, 14, 15 --- ON HER FRONT, SITTING, TIPPED: the three
hands (CDC 2-6 months: holds head up on tummy, rolls; 6-9 months: sits with
support, sits alone; HINE: forward parachute / arm protection).

The doctor's hands are the page's own hand on her joints (`/world {"hold":
{joint: [x,y,z]}}`, a hold releases every other pin; `{"let_go": true}` is
every hand off).  Nothing in her body is added.  Born from nothing,
unwrapped, her mother on, the room's hum first.  Then, in order:

  13  ON HER FRONT   she is turned over (her shoulders and hips swapped left
                     for right through the hand, four seconds) and let go for
                     a minute.  Read: is she on her front (her trunk's front
                     points down), her head's height above her chest (head up),
                     and whether she rolls back (the front turns up again).
  14  SITTING        her pelvis is held where it is and her chest lifted above
                     it by her spine's length, four seconds.  Then WITH SUPPORT:
                     only her pelvis held, ten seconds.  Then ALONE: every hand
                     off, ten seconds.  Read: the share of the time her chest
                     stays above her pelvis by 70% of her spine's length.
  15  TIPPED         sat again, then her chest carried forward 12 cm in half a
                     second and let go.  Read: her hands' travel forward and
                     down in the next second, against the half second before.

Bars are provisional and his to set: 13 head up 5 s of 60; 14 with support
5 s of 10, alone 3 s of 10; 15 hands forward 3 cm more than before.
Expected today: NOT YET on all three; nothing has paid for posture yet.  What
matters is the number, and its slope across lives.

    python tests/doctor_test7.py <name>          (repo root, her python)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind"), os.path.join(ROOT, "sandbox")]
from body import speech
from body.ragdoll import JIDX
import film
JOINTS = list(JIDX)
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test7"
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


def front(p):
    """Her trunk's front, a unit vector: across her shoulders crossed with up her spine."""
    f = np.cross(p["shR"] - p["shL"], p["chest"] - p["pelvis"])
    n = np.linalg.norm(f)
    return f / n if n > 1e-9 else np.zeros(3)


def hold(d, seconds, rate=10.0):
    """A hand on the named joints, posted `rate` times a second for `seconds` (a post moves a joint at most 15 cm)."""
    t1 = time.time()
    while time.time() - t1 < seconds:
        post("/world", {"hold": {k: [float(x) for x in v] for k, v in d.items()}})
        time.sleep(1.0 / rate)


def watch(pose, seconds, fn):
    rows = []; t1 = time.time()
    while time.time() - t1 < seconds:
        p = pose()
        if p is not None:
            rows.append((time.time() - t1,) + tuple(fn(p)))
        time.sleep(0.2)
    return np.array(rows) if rows else np.zeros((0, 4))


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
out = {}
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
    until(35)
    p0 = pose()
    out["birth_front_y"] = float(front(p0)[1])
    # THE SIGN IS CALIBRATED AT BIRTH (first run, 2026-09-07: it read "down" on a baby lying on her
    # back): she is born on her back, so "on her front" is the front vector turned the other way.
    SIGN = -1.0 if out["birth_front_y"] < 0 else 1.0
    spine = float(np.linalg.norm(p0["chest"] - p0["pelvis"]))
    # 13. ON HER FRONT
    hold({"shR": p0["shL"], "shL": p0["shR"], "hiR": p0["hiL"], "hiL": p0["hiR"]}, 4.0)
    post("/world", {"let_go": True})
    out["prone"] = watch(pose, 60.0, lambda p: (SIGN * front(p)[1], p["head"][1] - p["chest"][1], p["head"][1]))
    # 14. SITTING
    p1 = pose()
    seat = p1["pelvis"].copy(); up = seat + np.array([0.0, spine, 0.0])
    hold({"pelvis": seat, "chest": up}, 4.0)
    out["sit_supported"] = watch(pose, 0.0, lambda p: (0, 0, 0))
    t1 = time.time(); rows = []
    while time.time() - t1 < 10.0:                                   # only her pelvis held
        post("/world", {"hold": {"pelvis": [float(x) for x in seat]}})
        p = pose()
        if p is not None:
            rows.append((time.time() - t1, (p["chest"][1] - p["pelvis"][1]) / spine, p["head"][1], 0))
        time.sleep(0.2)
    out["sit_supported"] = np.array(rows)
    post("/world", {"let_go": True})
    out["sit_alone"] = watch(pose, 10.0, lambda p: ((p["chest"][1] - p["pelvis"][1]) / spine, p["head"][1], 0))
    # 15. TIPPED
    p2 = pose()
    seat = p2["pelvis"].copy(); up = seat + np.array([0.0, spine, 0.0])
    hold({"pelvis": seat, "chest": up}, 4.0)
    f = SIGN * front(pose()); f[1] = 0.0                        # her true front, the sign from birth
    f = f / (np.linalg.norm(f) or 1.0)
    before = watch(pose, 0.5, lambda p: (float(((p["haR"] + p["haL"]) * 0.5 - (p["shR"] + p["shL"]) * 0.5) @ f),
                                         float(((p["haR"] + p["haL"]) * 0.5 - (p["shR"] + p["shL"]) * 0.5)[1]), 0))
    t1 = time.time()
    while time.time() - t1 < 0.5:
        s = (time.time() - t1) / 0.5
        post("/world", {"hold": {"pelvis": [float(x) for x in seat], "chest": [float(x) for x in (up + f * 0.12 * s)]}})
        time.sleep(0.1)
    post("/world", {"let_go": True})
    after = watch(pose, 1.0, lambda p: (float(((p["haR"] + p["haL"]) * 0.5 - (p["shR"] + p["shL"]) * 0.5) @ f),
                                        float(((p["haR"] + p["haL"]) * 0.5 - (p["shR"] + p["shL"]) * 0.5)[1]), 0))
    out["tip_before"], out["tip_after"] = before, after
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

print("\n=== TESTS 13, 14, 15: ON HER FRONT, SITTING, TIPPED (the page's hand; unwrapped, her mother on) ===")
print("at birth she lies on her back (front.y %+.2f, the sign calibrated there)" % out["birth_front_y"])
pr = out["prone"]
if len(pr):
    onFront = pr[:, 1] < 0
    print("   her front over the minute: min %+.2f, max %+.2f (negative = facing down)" % (pr[:, 1].min(), pr[:, 1].max()))
    # the control: where her head rests in the first three seconds after the hand lets go (her head's
    # radius is larger than her chest's, so at rest it already sits above it); "up" is 2 cm above THAT
    settle = pr[pr[:, 0] < 3.0]; rest = settle[:, 2].mean() if len(settle) else 0.0
    print("   her head above her chest at rest (first 3 s) %.3f m; over the minute mean %.3f, max %.3f" % (rest, pr[:, 2].mean(), pr[:, 2].max()))
    headUp = (pr[:, 2] > rest + 0.02) & onFront & (pr[:, 0] >= 3.0)
    rolled = bool(onFront[: max(1, len(pr) // 6)].mean() > 0.5 and (~onFront[len(pr) // 2:]).mean() > 0.5)
    print("13 ON HER FRONT   on her front %.0f%% of the minute; head up (2 cm above its rest) %.1f s of 57; rolled back %s   %s"
          % (100 * onFront.mean(), headUp.sum() * 0.2, "yes" if rolled else "no",
             "PASS" if headUp.sum() * 0.2 >= 5.0 else ("NOT YET" if onFront.mean() > 0.5 else "NOT TESTED: the hand did not turn her")))
ss, sa = out["sit_supported"], out["sit_alone"]
if len(ss) and len(sa):
    upS = (ss[:, 1] > 0.7).sum() * 0.2; upA = (sa[:, 1] > 0.7).sum() * 0.2
    print("14 SITTING        with support (pelvis held): chest up %.1f s of 10 (upright share %.2f -> %.2f); alone: %.1f s of 10 (%.2f -> %.2f)   support %s, alone %s"
          % (upS, ss[0, 1], ss[-1, 1], upA, sa[0, 1], sa[-1, 1], "PASS" if upS >= 5 else "NOT YET", "PASS" if upA >= 3 else "NOT YET"))
tb, ta = out.get("tip_before"), out.get("tip_after")
if tb is not None and len(tb) and len(ta):
    fwd = ta[:, 1].max() - tb[:, 1].mean(); down = tb[:, 2].mean() - ta[:, 2].min()
    print("15 TIPPED         her hands went forward %+.3f m and down %+.3f m past where they were   %s" % (fwd, down, "arms out (PASS)" if fwd >= 0.03 else "NOT YET"))
print("the visit: closes %s replays %s trials %s hunger %.3f" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0)))
