"""THE NIGHT, HIS WAY (2026-09-07, 01:40): THE BOTTLE HUNT.

His words: no automatic feeding; "put bottle in her room ... on open space and
her also there. Each time when she reach that bottle she has to get some milk,
but count how many exactly ... something around ten percent at the beginning
... that place where you put bottle every time has to be different ... on
twenty or thirty times make this bottle a little bit higher, to she would have
reason to try to get up a bit ... till she will get up fully on her legs."
And: no helpers (no hands) through the night; answer her sounds with his
words --- her name, mama, ball, milk, rattle, bear (his recordings; papa, up,
down were never recorded and are not said).

What this does, all through her room's doors, nothing in her touched:
  - a fresh life; her mother on; no hand; she is carried once from her cot to
    the open floor and let go there (the page's own hand), then never held;
  - a bottle (a real thing of her room, `{"put": ...}`, her eye sees it and it
    yields to her body) at a new place each time: 0.30-0.45 m from her chest,
    toward where her face points, on the floor at first; after GRABS_PER_LEVEL
    grabs the bottle stands STEP higher, and again after each level;
  - a grab = her hand or her face within REACH of the bottle: she gets MILK
    (a share of a mouthful, `{"give": {"milk": MILK}}`; her mother says the
    milk's name herself at every give), the bottle goes and a new one comes;
  - his six words are said to her once each at the start, then a word answers
    a sound of hers at most once every WORD_REST seconds.  (The first night,
    2026-09-07 02:30: a word after EVERY sound, 424 posts in 34 minutes --- her
    mother keeps every word said into her room with no cap and rewrites her
    whole memory to disk in her tick at each one: 3.2 MB, her mother's stage
    0.03 -> 10 ms, her body at 53 ticks a second.  The cap is her mother's to
    get, on his word; this is the caregiver keeping quiet.)
  - one JSON row a minute to the log, and a row per grab and per level.

    python tests/night.py [life name] [log path] [words dir]      (repo root, her python)
    defaults: night, measured/<name>.log, measured/words
Stop: a file named STOP beside the log (measured/STOP), or the PowerShell line in CLAUDE.md.
Continue the same life another night: the same life name (her.py --keep is what it runs).
"""
import base64, glob, json, os, subprocess, sys, time, urllib.request, wave, zlib
import numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "sandbox")]
from body import speech
from body.ragdoll import JIDX
from body.room import BED
import film
JOINTS = list(JIDX)
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "night"
logPath = sys.argv[2] if len(sys.argv) > 2 else os.path.join("measured", name + ".log")
wordsDir = sys.argv[3] if len(sys.argv) > 3 else os.path.join("measured", "words")     # his six recorded words
MILK = 0.3                 # a share of a mouthful per grab (his "around ten percent" of a feed, to be lowered when she hunts)
REACH = 0.10               # m: hand or face this close to the bottle's centre is a grab
GRABS_PER_LEVEL = 25       # his "twenty or thirty"
STEP = 0.10                # m: how much higher the bottle stands at each level
WORD_REST = 600.0          # s between words said to her after the first six (see the docstring)
FLOOR_SPOT = np.array([1.6, 0.0, 0.0])      # open floor, 1.6 m from her cot (the cot is at the room's centre)
STOP = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
        "'sandbox[\\\\/]app\\.py|life\\.py lives' } | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")


def st():
    with urllib.request.urlopen(AT + "/state", timeout=5) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(AT + path, timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None, raw=None, headers=None):
    data = raw if raw is not None else json.dumps(body or {}).encode()
    req = urllib.request.Request(AT + path, data=data, method="POST", headers=headers or {})
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.read()


def loadWord(path):
    w = wave.open(path); rate = w.getframerate(); n = w.getnframes(); ch = w.getnchannels()
    pcm = np.frombuffer(w.readframes(n), "<i2").astype(np.float32) / 32767.0; w.close()
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if rate != speech.RATE:
        xs = np.linspace(0, len(pcm) - 1, int(len(pcm) * speech.RATE / rate)); pcm = np.interp(xs, np.arange(len(pcm)), pcm)
    return (pcm / (float(np.abs(pcm).max()) or 1.0) * 0.5).astype(np.float32)


WORDS = {os.path.basename(f).replace("_mother", "").replace(".wav", ""): loadWord(f) for f in sorted(glob.glob(os.path.join(wordsDir, "*.wav")))}
ORDER = [w for w in ("matilda", "mama", "milk", "ball", "rattle", "bear") if w in WORDS]


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


def herSaid(after):
    got = get("/said?after=%d" % after)
    z = got.get("z") or ""
    pcm = np.frombuffer(zlib.decompress(base64.b64decode(z)), "<i2").astype(np.float32) / 32767.0 if z else np.zeros(0, np.float32)
    return int(got.get("tick", after)), pcm


def bottleAt(s):
    for t in (s.get("room") or {}).get("things") or []:
        if t.get("what") == "bottle":
            return np.array(t.get("at"), float)
    return None


def log(row):
    with open(logPath, "a", encoding="utf-8") as h:
        h.write(json.dumps(row) + "\n")


def hold(d, seconds, rate=10.0):
    t1 = time.time()
    while time.time() - t1 < seconds:
        post("/world", {"hold": {k: [float(x) for x in v] for k, v in d.items()}}); time.sleep(1.0 / rate)


rng = np.random.default_rng(int(time.time()) % 100000)


def newPlace(p, level, last):
    """A new place for the bottle: 0.30-0.45 m from her chest, toward where her face points (within 70 deg), at the level's height; not within 0.25 m of the last."""
    g = st().get("gaze") or {}
    face = np.array(g.get("dir") or [1.0, 0.0, 0.0], float); face[1] = 0.0
    if np.linalg.norm(face) < 1e-6:
        face = np.array([1.0, 0.0, 0.0])
    face /= np.linalg.norm(face)
    for _ in range(30):
        # 20-70 degrees to EITHER SIDE of where her face points, never on her axis (his, 02:50: the
        # bottle sat at the centre of her view, where the first night put it, and looked like a fault)
        th = float(rng.uniform(20, 70)) * (1.0 if rng.random() < 0.5 else -1.0); r = float(rng.uniform(0.30, 0.45))
        c, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))
        d = np.array([c * face[0] - s_ * face[2], 0.0, s_ * face[0] + c * face[2]])
        at = p["chest"] + d * r; at[1] = 0.06 + STEP * level
        # NEVER INSIDE HER BED (his, 2026-09-08: "bottle stuck under bed so it
        # hasn't to appear there").  Her cot is a solid box at the room's
        # centre; a bottle put inside it is one she can see and never reach.
        if (abs(at[0]) < BED["x"] + 0.06 and abs(at[2]) < BED["z"] + 0.06
                and at[1] < BED["top"]):
            continue
        if last is None or np.linalg.norm(at - last) >= 0.25:
            return at
    return at


subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
for _ in range(40):
    try:
        st(); time.sleep(0.5)
    except Exception:
        break
ENV = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
proc = subprocess.Popen([sys.executable, "her.py", "--keep", name + ".duckdb"], cwd=ROOT, env=ENV,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
said = {w: 0 for w in WORDS}; answered = 0; grabs = 0; level = 0; grabsAtLevel = 0; t0 = time.time(); lastAt = None
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
    post("/world", {"autofeed": None})
    pose = Pose(); time.sleep(3.0)
    p = pose()
    # ONCE: carried from her cot to the open floor, and let go there
    off = FLOOR_SPOT - p["pelvis"]; off[1] = 0.0
    hold({"pelvis": p["pelvis"] + off + [0, 0.08, 0], "chest": p["chest"] + off + [0, 0.08, 0]}, 4.0)
    post("/world", {"let_go": True}); time.sleep(3.0)
    p = pose()
    at = newPlace(p, level, None); lastAt = at
    post("/world", {"put": {"what": "bottle", "at": at.tolist(), "size": 0.06}})
    s = st()
    log({"t": 0, "event": "start", "life": name, "helper": s.get("helper"), "autofeed": s.get("autofeed"),
         "her chest": [round(v, 2) for v in p["chest"]], "bottle": [round(v, 2) for v in at], "milk per grab": MILK,
         "reach": REACH, "grabs per level": GRABS_PER_LEVEL, "step": STEP, "words": ORDER,
         "things": [t.get("what") for t in (s.get("room") or {}).get("things") or []]})
    # his six words, once each, three seconds apart: they enter her mother's memory once
    for w in ORDER:
        post("/say", raw=WORDS[w].tobytes()); said[w] += 1; time.sleep(3.0)
    after = s.get("tick") or 0; lastWord = time.time(); k = 0; lastMinute = 0; lastGrab = 0.0
    while not os.path.exists(os.path.join(os.path.dirname(logPath), "STOP")):
        time.sleep(0.4)                     # lighter on her body: the probe showed her tick behind the clock under 0.2 s polling
        try:
            # her sounds, answered with his words
            after, pcm = herSaid(after)
            if pcm.size and float(np.abs(pcm).max()) > 0.01 and time.time() - lastWord > WORD_REST:
                w = ORDER[k % len(ORDER)]
                time.sleep(0.3); post("/say", raw=WORDS[w].tobytes())
                said[w] += 1; answered += 1; lastWord = time.time(); k += 1
            # the bottle: reached?
            p = pose(); s = st(); b = bottleAt(s)
            if p is not None and b is not None and time.time() - lastGrab > 3.0:
                near = min(float(np.linalg.norm(p[j] - b)) for j in ("haR", "haL", "face"))
                if near <= REACH + 0.06:
                    hungerBefore = s.get("hunger")
                    post("/world", {"give": {"milk": MILK}})
                    grabs += 1; grabsAtLevel += 1; lastGrab = time.time()
                    post("/world", {"take": "bottle"})
                    log({"t": round(time.time() - t0), "event": "grab", "n": grabs, "level": level, "by": min((j for j in ("haR", "haL", "face")), key=lambda j: float(np.linalg.norm(p[j] - b))),
                         "hunger before": hungerBefore, "bottle": [round(v, 2) for v in b]})
                    if grabsAtLevel >= GRABS_PER_LEVEL:
                        level += 1; grabsAtLevel = 0
                        log({"t": round(time.time() - t0), "event": "level", "level": level, "height": round(0.06 + STEP * level, 2)})
                    time.sleep(1.0)
                    p = pose() or p
                    at = newPlace(p, level, lastAt); lastAt = at
                    post("/world", {"put": {"what": "bottle", "at": at.tolist(), "size": 0.06}})
            m = int((time.time() - t0) // 60)
            if m > lastMinute:
                lastMinute = m
                s = st(); tb = s.get("teacher") or {}; p = pose(); b = bottleAt(s)
                log({"t": m * 60, "tick": s.get("tick"), "hunger": s.get("hunger"), "closes": s.get("closes"), "replays": s.get("guided"),
                     "trials": s.get("laddered"), "behind": s.get("behind"), "helper": s.get("helper"), "grabs": grabs, "level": level,
                     "answered": answered, "said": dict(said),
                     "mother": dict(tb.get("meter") or {}, named=tb.get("named"), namedWhat=tb.get("namedWhat")),
                     "her chest": [round(v, 2) for v in p["chest"]] if p else None, "her head y": round(float(p["head"][1]), 2) if p else None,
                     "bottle": [round(v, 2) for v in b] if b is not None else None,
                     "hand to bottle": round(min(float(np.linalg.norm(p[j] - b)) for j in ("haR", "haL", "face")), 3) if (p and b is not None) else None,
                     "startles": s.get("startles"), "looking": s.get("looking")})
        except Exception as e:                                          # noqa: BLE001
            log({"t": round(time.time() - t0), "event": "error", "what": str(e)[:160]})
            time.sleep(2)
            if proc.poll() is not None:
                log({"t": round(time.time() - t0), "event": "her body is gone; the night ends"}); break
finally:
    log({"t": round(time.time() - t0), "event": "stop", "grabs": grabs, "level": level, "answered": answered, "said": said})
    subprocess.run(["powershell", "-NoProfile", "-Command", STOP], capture_output=True)
    try:
        proc.kill()
    except Exception:
        pass
