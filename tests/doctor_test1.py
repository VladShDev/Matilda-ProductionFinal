"""THE DOCTOR'S TEST 1 --- TURNING TOWARD A SOUND, AND HABITUATION.
His, 2026-09-06: a doctor watches; she does not draw blood.  So this reads
only what a doctor sees from outside: her eyes and her neck, which her record
holds as her own motor lines, tick by tick.  Nothing of her chemistry is read.

Born from nothing, swaddled, her mother on.  Ten seconds of the room's own low
hum at 20 s (her ear takes its usual from it), then, after two minutes, a chirp
she has never heard, three times, fifteen seconds apart.  The side it comes from is
what her own two ears report at the chirp (the same difference her startle
turns on).  For each chirp: how far her gaze and her neck moved in the two
seconds after it, against the two seconds before it, and whether toward the
sound's side.

  TURNS TOWARD A SOUND   PASS if, at the first chirp, her eyes or her neck
                         move within 2 s more than in the 2 s before, and
                         toward the side the sound came from.
  HABITUATION            PASS if the turn at the third chirp is smaller than
                         at the first.

    python tests/doctor_test1.py <name>          (repo root, her venv)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np, duckdb
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from body.hearing import TICK_SECONDS
from body.joints import AXES
from body import orient
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test1"
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
FPS = 1.0 / TICK_SECONDS
RES = 0.05                                  # her body's resolution: a move she can make
EYE_L, EYE_R = len(AXES) + 1, len(AXES) + 2   # her eyes' horizontal pair, as motor ids
NECK = next((k + 1 for k, a in enumerate(AXES) if a.startswith("neck") and "side" in a), None) \
    or next((k + 1 for k, a in enumerate(AXES) if a.startswith("neck")), None)
tape = "mind/lives/%s.duckdb" % name


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body or {}).encode()
    req = urllib.request.Request(AT + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


def chirp() -> bytes:
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    return (0.3 * np.sin(2 * np.pi * (700.0 * tt + 500.0 * tt * tt)) * np.hanning(len(tt))).astype("<f4").tobytes()


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
ENV = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
proc = subprocess.Popen([sys.executable, "her.py", "--keep", name + ".duckdb"], cwd=ROOT, env=ENV,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
marks = []
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
    t0 = time.time()
    NEW = chirp()

    gazes = []      # (seconds, her gaze's direction, head + eyes, from her state)

    def until(sec):
        while time.time() - t0 < sec:
            try:
                s9 = st(); gazes.append((time.time() - t0, (s9.get("gaze") or {}).get("dir")))
            except Exception:
                pass
            time.sleep(0.2)
    # THE RATTLE IS HELD TO ONE SIDE.  The chirp comes through the flying screen,
    # and the screen stays where the last test left it (lives/window): after
    # test 3 it sat dead ahead of her, the chirp reached both ears alike, the
    # sheet read "neither" and the turn had no side (2026-09-06).  A doctor
    # holds the rattle 30 degrees to the baby's right, half a metre off.
    FACE = None
    g = st().get("gaze") or {}
    HEAD = np.array(g.get("head") or [0.20, 0.42, -0.34], float)
    FACE = np.array(g.get("dir") or [-0.75, -0.56, 0.35], float); FACE /= max(1e-6, np.linalg.norm(FACE))
    RIGHT = np.cross(np.array([0.0, 1.0, 0.0]), FACE); RIGHT /= max(1e-6, np.linalg.norm(RIGHT))   # her right: up x face
    th = np.radians(30.0)
    SIDE = np.cos(th) * FACE + np.sin(th) * RIGHT
    post("/window", {"at": (HEAD + 0.5 * SIDE).tolist(), "look": (-SIDE).tolist(), "size": [0.3, 0.22]})
    # THE ROOM'S OWN SOUND FIRST.  Her ear compares a sound with what it is
    # used to, and that usual is built only from sounds she has heard.  Her
    # mother speaks two percent of the time, so two quiet minutes gave her
    # ear under three seconds of sound and it was still waiting at every
    # chirp (2026-09-06).  His word: do not touch her, give the test a room.
    # So: ten seconds of a low hum, quieter than the chirp, the way a real
    # room hums around a newborn; her ear takes its usual from it itself.
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    chirpAt = []
    for k in (1, 2, 3):
        until(120 + 15 * (k - 1)); chirpAt.append(time.time() - t0)       # two minutes with her mother first: her ear's usual needs 6.7 s of heard sound (STARTLE_WARM)
        s = st(); marks.append((int(s["tick"]), int(s.get("startles", 0)), k))
        post("/say", raw=NEW)
        print("chirp %d at her tick %d (startles so far %s)" % (k, marks[-1][0], marks[-1][1]), flush=True)
    until(180)
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
rows = [([{"id": s.id, "lvl": s.lvl} for s in t.life.input.spindle], [{"id": s.id, "lvl": s.lvl} for s in t.life.input.sensor]) for t in frames(tape)]
N = len(rows)
T_END = int(fin["tick"])


def row(tick):
    return max(0, min(N - 1, N - 1 - (T_END - int(tick))))


# WHAT HER EYES AND NECK DID, not what she ordered: her spindles.  A reflex
# writes the executed order and arrives on her spindles like any pull, while
# her posted order (`output.motor`) is her mind's alone --- the first run of
# this test read the order and saw no reflex at all (2026-09-06).
gaze = np.zeros(N); neck = np.zeros(N); earL = np.zeros(N); earR = np.zeros(N)
for i, (spindle, sensor) in enumerate(rows):
    lv = {int(m["id"]): float(m["lvl"]) for m in (spindle or [])}
    gaze[i] = lv.get(EYE_R, 0.0) - lv.get(EYE_L, 0.0)          # + aims at her right (orient.py)
    neck[i] = lv.get(NECK, 0.5) if NECK else 0.5
    sv = {int(x["id"]): float(x["lvl"]) for x in (sensor or [])}
    earL[i], earR[i] = sv.get(50, 0.0), sv.get(51, 0.0)


def sec(k):
    return int(round(k * FPS))


print("\n=== TEST 1: TURNING TOWARD A SOUND (what a doctor sees: her eyes and her neck) ===")
print("her eyes: motors %d and %d; her neck: motor %s (%s); side by her ears (lines 50, 51); a move is %.2f, her resolution"
      % (EYE_L, EYE_R, NECK, AXES[NECK - 1] if NECK else "none", RES))
turn = []
for tick, startles, k in marks:
    i = row(tick)
    a0, a1 = max(0, i - sec(2)), i
    b0, b1 = i, min(N, i + sec(2))
    heard = slice(i, min(N, i + sec(1)))
    side = float(np.sum(earR[heard] - earL[heard]))            # + the right ear heard more
    sideName = "right" if side > 0 else ("left" if side < 0 else "neither")
    gBefore = float(np.abs(gaze[a0:a1] - gaze[a0]).max()) if a1 > a0 else 0.0
    gAfter = float(np.abs(gaze[b0:b1] - gaze[b0]).max())
    gDir = float(gaze[b0:b1][np.argmax(np.abs(gaze[b0:b1] - gaze[b0]))] - gaze[b0])
    nBefore = float(np.abs(neck[a0:a1] - neck[a0]).max()) if a1 > a0 else 0.0
    nAfter = float(np.abs(neck[b0:b1] - neck[b0]).max())
    nDir = float(neck[b0:b1][np.argmax(np.abs(neck[b0:b1] - neck[b0]))] - neck[b0])
    # toward: her neck's own sign for "toward the louder ear" is orient.NECK_SIGN
    # (the startle turns to 0.5 + 0.5 * NECK_SIGN * side); her gaze's pan is
    # INVERTED against it --- a positive pan aims at her right (orient.py's
    # own measurement), and a positive side is her right ear hearing more
    towardN = (nDir * orient.NECK_SIGN * side) > 0
    towardG = (gDir * side) > 0
    # HER EYES ARE NEVER STILL --- swaddled, they swing their whole range every
    # look (her own orders, and the eye's lean to what stands out), so the
    # size of a move says nothing.  What a doctor judges on a restless baby is
    # the side: does her gaze, on average, shift toward the sound in the two
    # seconds after it, against the two seconds before.
    sgn = 1.0 if side > 0 else (-1.0 if side < 0 else 0.0)
    shift = float((gaze[b0:b1].mean() - gaze[a0:a1].mean()) * sgn) if a1 > a0 else 0.0
    onSide = (float((gaze[a0:a1] * sgn > 0).mean()) if a1 > a0 else 0.0, float((gaze[b0:b1] * sgn > 0).mean()))
    moved = max(0.0, shift)          # a shift away from the sound is no turn; habituation compares turns toward it
    turn.append((moved, gAfter, nAfter, gBefore, nBefore, towardG, towardN, sideName))
    print("chirp %d from her %s: eyes moved %.3f in the 2 s after (%.3f in the 2 s before) %s; neck moved %.3f after (%.3f before) %s"
          % (k, sideName, gAfter, gBefore, "TOWARD it" if towardG and gAfter > RES else "", nAfter, nBefore,
             "TOWARD it" if towardN and nAfter > RES else ""))
# HER GAZE, HEAD AND EYES TOGETHER --- what a doctor sees.  Her eye spindles are
# her eyes against her head; once her head helps her eyes (2026-09-06), a turn
# of the gaze can leave the eyes centred.  So the turn is read from her gaze's
# direction, sampled five times a second from her state: its angle toward the
# rattle's side in the 2 s after each chirp against the 2 s before.
gz = [(t, np.array(d, float)) for t, d in gazes if d]
if gz and FACE is not None:
    right = np.cross(np.array([0.0, 1.0, 0.0]), FACE); right /= max(1e-6, np.linalg.norm(right))   # her right: up x face
    turn = []
    for k, tc in enumerate(chirpAt):
        before = [np.degrees(np.arcsin(np.clip(d @ right, -1, 1))) for t, d in gz if tc - 2 <= t < tc]
        after = [np.degrees(np.arcsin(np.clip(d @ right, -1, 1))) for t, d in gz if tc <= t < tc + 2]
        shift = (np.mean(after) - np.mean(before)) if before and after else 0.0
        moved = max(0.0, shift)
        turn.append((moved, 0, 0, 0, 0, shift > 0, False, "right"))
        print("chirp %d: her gaze (head and eyes) %+.1f deg to her right in the 2 s before, %+.1f in the 2 s after -> shift toward the rattle %+.1f deg"
              % (k + 1, np.mean(before) if before else 0, np.mean(after) if after else 0, shift))
    RES_DEG = 2.0    # two degrees: below what a doctor can see
    m1 = turn[0][0]
    print("TURNS TOWARD A SOUND   %s" % ("PASS" if m1 > RES_DEG else "NOT YET"))
    print("HABITUATION            %s   (the turn: 1st %.1f, 2nd %.1f, 3rd %.1f deg)" % ("PASS" if (m1 > RES_DEG and turn[2][0] < m1) else "NOT YET", m1, turn[1][0], turn[2][0]))
    print("startles her body counted at the three chirps: %s;  the whole life: %d ticks (%.0f s), swaddled, born from nothing"
          % ([s for _, s, _ in marks] + [fin.get("startles")], N, N * TICK_SECONDS))
    sys.exit(0)
m1, g1, n1, gb1, nb1, tg1, tn1, side1 = turn[0]
turned = ((g1 > RES and g1 > gb1 and tg1) or (n1 > RES and n1 > nb1 and tn1))
print("TURNS TOWARD A SOUND   %s" % ("PASS" if turned else "NOT YET"))
habit = turned and turn[2][0] < m1
print("HABITUATION            %s   (the turn: 1st %.3f, 2nd %.3f, 3rd %.3f)" % ("PASS" if habit else "NOT YET", m1, turn[1][0], turn[2][0]))
print("startles her body counted at the three chirps: %s;  the whole life: %d ticks (%.0f s), swaddled, born from nothing"
      % ([s for _, s, _ in marks] + [fin.get("startles")], N, N * TICK_SECONDS))
