"""THE DOCTOR'S TESTS 4, 5 AND 8 --- HER SOUNDS, in one visit.
A doctor sits with the baby and her mother for a few minutes and listens.

Born from nothing, swaddled, her mother on, the room's hum first.  Then four
quiet minutes: her mother speaks when she speaks (she names what the baby
looks at, and answers her), the doctor does nothing.  Read from her record:

  4  MAKES SOUNDS OF HER OWN   distinct sounds her own mouth made (her echo
                               line's ids).  PASS if at least one; the count
                               is the number.
  5  QUIETS TO HER MOTHER      while her mother's voice sounds, her moving
                               (how much her spindles change a tick) and her
                               sounding are lower than in the 2 s before each
                               of her mother's utterances.  PASS if moving falls.
  8  ANSWERS A SOUND           after her mother's utterance begins, a sound of
                               hers within 3 s, more often than in the same
                               3 s windows taken where her mother was silent.
                               PASS if more often, on at least two utterances.
                               (Any sound of hers counts --- never his word;
                               that was the old lock's question.)

    python tests/doctor_test4.py <name> [quiet seconds]   (repo root, her python)
    the quiet minutes default to 240 s; the sheet's quarter hour is 900 (his word, 2026-09-06 night)
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np, duckdb
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, "mind")]
from body import speech
from body.hearing import TICK_SECONDS
AT = "http://127.0.0.1:8090"
name = sys.argv[1] if len(sys.argv) > 1 else "test4"
QUIET = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0     # the quiet minutes with her mother
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
            post("/world", {"helper": "cradle"})
            time.sleep(0.5)
            if st().get("helper") == "cradle":
                break
        except Exception:
            time.sleep(0.5)
    assert st().get("helper") == "cradle", "the wrap did not take"
    t0 = time.time()

    def until(sec):
        while time.time() - t0 < sec:
            time.sleep(0.2)
    tt = np.arange(int(speech.RATE * 1.0)) / speech.RATE
    HUM = (0.15 * np.sin(2 * np.pi * 400.0 * tt)).astype("<f4").tobytes()
    for k in range(10):
        until(20 + k); post("/say", raw=HUM)
    until(30 + QUIET)
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
rows = [(t.life.input.echo.id, t.life.input.echo.lvl, t.life.input.sound.lvl, [{"id": s.id, "lvl": s.lvl} for s in t.life.input.spindle]) for t in frames(tape)]
N = len(rows)
echoId = np.array([int(r[0]) for r in rows]); echo = np.array([float(r[1]) > 0 for r in rows])
heard = np.array([float(r[2]) > 0 for r in rows])
sp = np.array([[float(s["lvl"]) for s in (r[3] or [])] for r in rows], float) if rows and rows[0][3] else np.zeros((N, 1))
moving = np.r_[0.0, np.abs(np.diff(sp, axis=0)).mean(axis=1)]            # how much her spindles changed this tick
start = int(round(50 * FPS))                                             # after the hum: the quiet minutes
sec = lambda k: int(round(k * FPS))

print("\n=== TESTS 4, 5, 8: HER SOUNDS (what a doctor hears and sees, %d s of her, her mother on) ===" % int(N * TICK_SECONDS))
# 4. makes sounds of her own
ids = sorted(set(echoId[start:][echo[start:]].tolist()))
print("4  MAKES SOUNDS OF HER OWN   %d distinct sounds, on %.1f%% of her ticks   %s"
      % (len(ids), 100 * echo[start:].mean(), "PASS" if len(ids) >= 1 else "NOT YET"))
# her mother's utterances: the sound line rising from silence
on = np.where(heard[1:] & ~heard[:-1])[0] + 1
on = on[on > start + sec(2)]
# 5. quiets to her mother's voice
mvIn, mvBefore, sdIn, sdBefore = [], [], [], []
for i in on:
    j = i
    while j < N and heard[j]:
        j += 1
    if j - i < sec(0.2):
        continue
    mvIn.append(moving[i:j].mean()); mvBefore.append(moving[i - sec(2):i].mean())
    sdIn.append(echo[i:j].mean()); sdBefore.append(echo[i - sec(2):i].mean())
if mvIn:
    print("5  QUIETS TO HER MOTHER      her moving while her mother speaks %.4f vs the 2 s before %.4f; her sounding %.1f%% vs %.1f%% (%d utterances)   %s"
          % (np.mean(mvIn), np.mean(mvBefore), 100 * np.mean(sdIn), 100 * np.mean(sdBefore), len(mvIn),
             "PASS" if np.mean(mvIn) < np.mean(mvBefore) else "NOT YET"))
else:
    print("5  QUIETS TO HER MOTHER      NOT TESTED: her mother did not speak")
# 8. answers a sound with a sound
ans, nullHits, nullN = 0, 0, 0
for i in on:
    if echo[i:i + sec(3)].any():
        ans += 1
quiet = np.ones(N, bool)
for i in on:
    quiet[max(0, i - sec(3)):i + sec(3)] = False
for i in range(start, N - sec(3), sec(3)):
    if quiet[i:i + sec(3)].all():
        nullN += 1; nullHits += int(echo[i:i + sec(3)].any())
pa = ans / max(1, len(on)); pn = nullHits / max(1, nullN)
print("8a from her mother's ONSET   a sound of hers within 3 s of her mother's onset: %d of %d (%.0f%%); in 3 s windows of silence: %.0f%% of %d   (confounded: her mother answers her 0.3 s after HER sound, so this window holds her own sound)"
      % (ans, len(on), 100 * pa, 100 * pn, nullN))
# 8. FROM HER MOTHER'S END (2026-09-07, after the quarter-hour visit showed the onset window
# counting her mother's answers to her): her sound 0.3-3 s after the utterance ENDS, against
# the same windows placed 10 s later where her mother is silent.
ends = []
for i in on:
    j = i
    while j < N and heard[j]:
        j += 1
    ends.append(j)
ansE = sum(int(echo[j + sec(0.3):j + sec(3)].any()) for j in ends)
nullE = nullEN = 0
for j in ends:
    k = j + sec(10)
    if k + sec(3) < N and quiet[k:k + sec(3)].all():
        nullEN += 1; nullE += int(echo[k + sec(0.3):k + sec(3)].any())
pe = ansE / max(1, len(ends)); pne = nullE / max(1, nullEN)
print("8  ANSWERS A SOUND           a sound of hers 0.3-3 s after her mother's utterance ENDS: %d of %d (%.0f%%); the same windows 10 s later in silence: %.0f%% of %d   %s"
      % (ansE, len(ends), 100 * pe, 100 * pne, nullEN,
         "PASS" if (ansE >= 2 and nullEN >= 2 and pe > pne) else ("NOT TESTED" if len(ends) == 0 else "NOT YET")))
print("her mother spoke on %.1f%% of her ticks; %d utterances; the visit: closes %s replays %s trials %s hunger %.3f"
      % (100 * heard[start:].mean(), len(on), fin.get("closes"), fin.get("guided"), fin.get("laddered"), fin.get("hunger", 0)))
