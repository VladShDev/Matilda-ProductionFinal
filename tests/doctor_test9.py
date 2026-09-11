"""THE DOCTOR'S ITEM 7 --- PREFERS HER MOTHER'S VOICE (CDC 6 months: "knows
familiar people"; 4 months: "turns head towards the sound of your voice").

The doctor plays the mother's voice and a stranger's from the same side and
watches which one the baby turns to.  Her mother's voice is a recording of
her mother saying "mama" (`measured/mama_mother.wav`), which the baby hears
about thirty times, straight ahead, over six free minutes first (the first run
took her mother's answers from `/mom` and found nothing: the answers are not
kept as sound).  The stranger is the same sound with its pitch lowered a
fifth and its length kept, from the same place at the same loudness.

Born from nothing, unwrapped, her mother on, the room's hum first; six free
minutes in which the recording is played to her; then, from 30 degrees to her right
at 0.5 m, alternately her mother's sound and the stranger's, three of each,
20 s apart.  Read (as test 1): her gaze's angle to the source in the 2 s
after each sound against the 2 s before; the turn toward.  PASS (provisional
bar, his word to set) if her mean turn to her mother's sound is larger than
to the stranger's.

    python tests/doctor_test9.py <name>          (repo root, her python)
"""
import base64, json, os, subprocess, sys, time, urllib.request, zlib
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test9"
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
tape = "mind/lives/%s.duckdb" % name


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(AT + path, timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body or {}).encode()
    req = urllib.request.Request(AT + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


def momPcm(after):
    got = get("/mom?after=%d" % after)
    z = got.get("z") or ""
    pcm = np.frombuffer(zlib.decompress(base64.b64decode(z)), "<i2").astype(np.float32) / 32767.0 if z else np.zeros(0, np.float32)
    return int(got.get("tick", after)), pcm


def angleTo(g, src):
    head = np.array(g.get("head"), float); d = np.array(g.get("dir"), float); d /= max(1e-6, np.linalg.norm(d))
    v = src - head; v /= max(1e-6, np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(d @ v, -1, 1))))


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
turns = {"mother": [], "stranger": []}; sample = None
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
    # six free minutes: her mother's recorded "mama" about thirty times, straight ahead
    import wave
    wv = wave.open("measured/mama_mother.wav"); rec = np.frombuffer(wv.readframes(wv.getnframes()), "<i2").astype(np.float32) / 32767.0; wv.close()
    sample = rec
    g0 = st().get("gaze") or {}
    H0 = np.array(g0.get("head"), float); F0 = np.array(g0.get("dir"), float); F0 /= max(1e-6, np.linalg.norm(F0))
    post("/window", {"at": (H0 + 0.4 * F0).tolist(), "look": (-F0).tolist(), "size": [0.30, 0.22]})
    rng = np.random.default_rng(7); nextAt = 35.0; heard = 0
    while time.time() - t0 < 30 + 360:
        time.sleep(0.5)
        if time.time() - t0 >= nextAt:
            post("/say", raw=(rec / (float(np.abs(rec).max()) or 1.0) * 0.3).astype(np.float32).tobytes()); heard += 1
            nextAt = time.time() - t0 + float(rng.uniform(8.0, 14.0))
    print("she heard her mother's \"mama\" %d times in six minutes" % heard, flush=True)
    if sample is None or len(sample) < int(0.05 * speech.RATE):
        print("no recording")
    else:
        sample = sample[: int(1.5 * speech.RATE)]
        peak = float(np.abs(sample).max()) or 1.0
        mother = (sample / peak * 0.3).astype(np.float32)
        # the stranger: pitch a fifth lower, the same length and loudness
        n = len(mother); xs = np.linspace(0, n - 1, int(n * 1.5))
        low = np.interp(xs, np.arange(n), mother)[:n].astype(np.float32)
        stranger = (low / (float(np.abs(low).max()) or 1.0) * 0.3).astype(np.float32)
        g = st().get("gaze") or {}
        HEAD = np.array(g.get("head"), float); FACE = np.array(g.get("dir"), float); FACE /= max(1e-6, np.linalg.norm(FACE))
        RIGHT = np.cross(np.array([0.0, 1.0, 0.0]), FACE); RIGHT /= max(1e-6, np.linalg.norm(RIGHT))
        th = np.radians(30.0); SIDE = np.cos(th) * FACE + np.sin(th) * RIGHT
        SRC = HEAD + 0.5 * SIDE
        post("/window", {"at": SRC.tolist(), "look": (-SIDE).tolist(), "size": [0.30, 0.22]})
        order = ["mother", "stranger", "stranger", "mother", "mother", "stranger"]
        t1 = time.time()
        for i, who in enumerate(order):
            while time.time() - t1 < 20 * i + 5:
                time.sleep(0.1)
            before = []
            tb = time.time()
            while time.time() - tb < 2.0:
                before.append(angleTo(st().get("gaze") or {}, SRC)); time.sleep(0.2)
            post("/say", raw=(mother if who == "mother" else stranger).tobytes())
            afterA = []
            ta = time.time()
            while time.time() - ta < 2.0:
                afterA.append(angleTo(st().get("gaze") or {}, SRC)); time.sleep(0.2)
            turns[who].append(float(np.mean(before) - np.mean(afterA)))
    fin = st()
finally:
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(4)

print("\n=== ITEM 7: PREFERS HER MOTHER'S VOICE (her mother's mama, heard thirty times first, and a stranger's, 30 degrees to her right) ===")
if sample is None:
    print("NOT TESTED: no recording")
else:
    print("her mother's sound: %.2f s; the turns toward it (deg, + = toward): %s" % (len(sample) / speech.RATE, [round(v, 1) for v in turns["mother"]]))
    print("the stranger's: %s" % [round(v, 1) for v in turns["stranger"]])
    m, s = np.mean(turns["mother"]), np.mean(turns["stranger"])
    print("PREFERS HER MOTHER'S VOICE   mean turn to her mother %+.1f deg, to the stranger %+.1f   %s   (provisional bar: mother > stranger, his word)"
          % (m, s, "PASS" if m > s else "NOT YET"))
print("the visit: closes %s replays %s trials %s hunger %.3f" % (fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0)))
