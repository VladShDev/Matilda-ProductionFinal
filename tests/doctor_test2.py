"""THE DOCTOR'S TEST 2 --- LOOKING: fixation, habituation, and the new.
The oldest infant test there is (looking time).  A doctor holds a card in
front of the baby and watches her eyes: does she look at it, does she look
less when it comes back and back, does she look again when a new card comes.

Born from nothing, swaddled, her mother on, ten seconds of the room's own hum
first (as test 1).  Then the ball on the flying screen held 0.4 m along her face's line for 30 s,
the board blank for 15 s, four times over; then the bear for 30 s.  What is
read: what her gaze holds, her body's own geometry (`/state` "looking", the
one answer to where she is looking), sampled five times a second from
outside, the way a doctor reads a baby's eyes.  Nothing of her chemistry is
read, nothing in her is told.

  FIXATES          PASS if she looks at the first ball more of its 30 s than
                   at the blank board in the 15 s before it.
  HABITUATES       PASS if she looks at the fourth ball less than at the first.
  NOTICES THE NEW  PASS if she looks at the bear more than at the fourth ball.

    python tests/doctor_test2.py <name>          (repo root, her venv)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from measure import cards
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test2"
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
tape = "mind/lives/%s.duckdb" % name


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def showOnScreen(name):
    """The doctor's card on the flying screen (no name: her mother does not
    name it; this test needs no word), or a blank pale screen."""
    img = cards.draw(name) if name != "blank" else np.full((cards.SIZE, cards.SIZE, 3), 235, np.uint8)
    req = urllib.request.Request(AT + "/show", data=np.ascontiguousarray(img).tobytes(), method="POST",
                                 headers={"X-Width": str(img.shape[1]), "X-Height": str(img.shape[0])})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


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
herlog = open(tape + ".her.log", "w", encoding="utf-8")     # her body's own words, kept: a crash must be readable
proc = subprocess.Popen([sys.executable, "-u", "her.py", "--keep", name + ".duckdb"], cwd=ROOT, env=ENV,
                        stdout=herlog, stderr=subprocess.STDOUT)
looks = []            # (seconds, what her gaze held, the block's name)
block = ["start"]
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
    # THE CARD WHERE SHE LOOKS.  The first run of this test (2026-09-06) put the
    # card on the board 34 degrees above her skull's line; her eyes stayed
    # within a few degrees of straight ahead the whole visit and her gaze
    # never held it --- 0% on every showing.  A doctor holds the card in the
    # baby's line of sight.  So: the flying screen is flown to 0.4 m along her
    # face's line (her head and face in the cot as measured in body/alive.py),
    # facing her, and the card is drawn on it.  Her gaze on it reads "window".
    # ...HER LINE OF SIGHT AS IT IS NOW, from her state: her head and her eyes'
    # axis (the second run of this test used a constant for her face and her
    # gaze held nothing again: 0%).  The screen goes 0.4 m along it, facing her.
    g = st().get("gaze") or {}
    HEAD = np.array(g.get("head") or [0.20, 0.42, -0.34], float)
    FACE = np.array(g.get("dir") or [-0.75, -0.56, 0.35], float); FACE /= max(1e-6, np.linalg.norm(FACE))
    # ...AND A LITTLE TO ONE SIDE.  Held straight on her line of sight, her
    # resting eyes were "on it" 90% of the time whatever it showed, blank
    # included (the third run): the sheet was reading where her eyes rest,
    # not where they go.  A doctor holds the card off to the side, inside
    # the baby's reach, and watches the eyes move to it.  So: the card 25
    # degrees to her right of her eyes' axis (her eyes reach 35), 0.4 m.
    RIGHT = np.cross(np.array([0.0, 1.0, 0.0]), FACE); RIGHT /= max(1e-6, np.linalg.norm(RIGHT))   # her right: up x face
    th = np.radians(25.0)
    SIDE = np.cos(th) * FACE + np.sin(th) * RIGHT
    AT_CARD = (HEAD + 0.4 * SIDE).tolist(); LOOK_BACK = (-SIDE).tolist()
    post("/window", {"at": AT_CARD, "look": LOOK_BACK, "size": [0.30, 0.22]})
    time.sleep(1.0)
    w = (st().get("window") or {})
    print("her head at %s, her eyes' axis %s; the card flown to %s (the screen reports %s)"
          % (np.round(HEAD, 2).tolist(), np.round(FACE, 2).tolist(), np.round(AT_CARD, 2).tolist(), [round(v, 2) for v in (w.get("at") or [])]), flush=True)
    t0 = time.time()
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()

    def until(sec):
        while time.time() - t0 < sec:
            try:
                s = st()
                looks.append((round(time.time() - t0, 1), s.get("looking"), block[0], (s.get("gaze") or {}).get("dir")))
            except Exception:
                pass
            time.sleep(0.2)
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    block[0] = "blank0"; until(60)
    for k in range(1, 5):
        block[0] = "ball%d" % k; showOnScreen("ball"); print("ball %d up at %.0f s" % (k, time.time() - t0), flush=True)
        until(60 + (k - 1) * 45 + 30)
        block[0] = "blank%d" % k; showOnScreen("blank")
        until(60 + k * 45)
    block[0] = "bear"; showOnScreen("bear"); print("bear up at %.0f s" % (time.time() - t0), flush=True)
    until(60 + 4 * 45 + 30)
    # the visit is over: whatever her body does now, the sheet is written
    try:
        showOnScreen("blank")
        fin = st()
    except Exception as e:                                       # noqa: BLE001
        print("at the end her body did not answer: %s" % e, flush=True)
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)
json.dump(looks, open(tape + ".looks.json", "w", encoding="utf-8"))


def share(blockName, card):
    got = [l for l in looks if l[2] == blockName]
    return (sum(1 for l in got if l[1] == card) / len(got)) if got else 0.0, len(got)


print("\n=== TEST 2: LOOKING (what a doctor sees: where her eyes are, five times a second) ===")
s0, n0 = share("blank0", "window")
rows = []
for k in range(1, 5):
    sh, n = share("ball%d" % k, "window"); rows.append(sh)
    print("ball %d up 30 s: her gaze on it %3.0f%% of the time (%d looks)" % (k, 100 * sh, n))
sb, nb = share("bear", "window")
held = {}
for l in looks:
    held[l[1]] = held.get(l[1], 0) + 1
print("bear up 30 s:   her gaze on it %3.0f%% of the time (%d looks);  blank board before the first ball: on the board %3.0f%%"
      % (100 * sb, nb, 100 * share("blank0", "None")[0]))
print("what her gaze held over the whole visit: %s" % ", ".join("%s %d" % (k, v) for k, v in sorted(held.items(), key=lambda kv: -kv[1])[:6]))
dirs = np.array([l[3] for l in looks if len(l) > 3 and l[3]], float)
if len(dirs):
    card = np.array(AT_CARD) - HEAD; card /= np.linalg.norm(card)
    ang = np.degrees(np.arccos(np.clip(dirs @ card, -1, 1)))
    print("her eyes' axis against the card's direction over the visit: median %.0f degrees, within 25 degrees %.0f%% of samples (the naming cone)" % (np.median(ang), 100 * (ang < 25).mean()))
print("her eyes on the first ball %.0f%% of its 30 s vs the blank's %.0f%%   (reported, no bar: fixation dropped on his word 2026-09-07 --- no fovea, one look is the memory)" % (100 * rows[0], 100 * s0))
print("HABITUATES       %s   (ball 1 %.0f%% -> ball 4 %.0f%%)" % ("PASS" if rows[3] < rows[0] else "NOT YET", 100 * rows[0], 100 * rows[3]))
print("NOTICES THE NEW  %s   (ball 4 %.0f%% -> bear %.0f%%)" % ("PASS" if sb > rows[3] else "NOT YET", 100 * rows[3], 100 * sb))
