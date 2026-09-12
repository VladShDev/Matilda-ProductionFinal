# MATILDA — this folder, and nothing outside it

Copyright (c) 2026 Vladyslav Shovhenia. All rights reserved. No license is
granted: see `LICENSE` and `COPYRIGHT.md`.

She is a baby: a body with real physics and real perceptual limits (`body/`),
and a mind that learns only from what that body reports (`mind/`). Nothing in
her is ever told anything about the world. She is born from nothing — no tape,
no guide, no map. What she says and does she finds by her own trials.

---

# THE FIRST RULE, AND IT OVERRIDES EVERY OTHER INSTINCT YOU HAVE

**This folder is the whole world. Nothing outside it exists.**

Do not read, open, copy from, search, or reason from:

- `..\Matilda-Production090626`, `..\matilda`, `..\matilda_clean`,
  `..\practice`, `..\core`, or any other folder on this machine
- any git history, any older repository, any earlier commit
- any memory, any summary, any note from a previous session

Every one of those is an **older version of her with a different
architecture**. Reading one is how a session starts guessing, and guessing is
what has broken her before. If a fact is not in this folder, **it is not a
fact**. Say you do not know and ask him.

The only thing outside this folder is the interpreter that runs her, below.

**And her mind is finished.** `mind/` is the centre of the eye. His word,
2026-09-11: *"we don't need to change her mind from that point."* Changes from
here are her room, her things, what she can see — `sandbox/`, `body/room.py`,
the page. **Never `mind/` without his explicit word in the session**, and your
default answer to your own idea about her mind is no.

---

# HOW TO RUN HER

**The one interpreter.** `python` on PATH is bare and has none of her packages.
Every command below uses this one, and `run.cmd` / `py.cmd` set it for you:

```
"C:\Users\tscen\Documents\antropic\practice\project Matilda\.venv\Scripts\python.exe"
```

*(It lives under an old folder. It is an interpreter, not a link — nothing of
hers is read from there.)*

**Turn her on** — one command, from this folder. It sets the interpreter and
the GPU for her eye:

```
run.cmd                        (or: run.cmd --keep NAME.duckdb to continue a life)
```

**Every test or probe** runs on her interpreter with the GPU hidden:

```
py.cmd -m measure.check        every wire of her body; must print "everything she is checked for holds"
py.cmd -m measure.voice        her mouth still says his certified word (six floors)
py.cmd tests\doctor_test1.py NAME
```

`CUDA_VISIBLE_DEVICES=""` for every probe you start; never for her body.
**Never GPU work beside a live body.**

**Watch her:** `http://127.0.0.1:8090` — her view and her ear bars, nothing
else on the page by his word. Her state: `curl -s http://127.0.0.1:8090/state`.
Read `behind` (must stay 0), `closes`, `replays`, `laddered`, `hunger`,
`blood`, `floor`, `lift`.

**If port 8090 is busy or a body was left running:**

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'sandbox[\\/]app\.py|life\.py lives|her\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

**Normal, not a bug:** `UserWarning` from numpy/cupy/duckdb at start; a
`ConnectionAbortedError: [WinError 10053]` traceback from `socketserver` (a
client closed its connection); `lives/window` showing as untracked; `warning:
LF will be replaced by CRLF` at a commit.

---

# HOW SHE WORKS

His architecture, confirmed 2026-09-11. This is the specification; her code is
what must match it, and it does.

### The row

1. **One row of levels every tick.** What arrives and what she sends, side by
   side. Nothing is named to her — the fixed slots are the app's bookkeeping so
   her body fills the same place each tick; she only ever sees a number in a
   position. **Her mind never branches on which channel a line came from.**
2. **A thing seen is what it is and where** — similarity, x, y (one picture, one
   level, three lines; never per-object). **A sound is four** — how alike
   (`heard,0`), how loud (`heard,1`), and her two ears (sensors 50, 51), which
   carry which-side and by-how-much between them. (Balance-as-a-sign was dropped
   2026-09-12: two ear levels already hold it.) Never one line per id.
3. **Her hormones are lines like any other. Her own orders are lines like any
   other.**

### A line stays where she put it

**NOTHING MOVES A LINE BUT HER.** Her output carries forward whole, and no part
of her body puts it back toward rest. That is what makes the chain possible:
the next experience opens where the last one closed, so she sets one line, then
another, and both are still there --- a whole pose built out of one-line trials.

Her motor lines used to drop toward rest every tick, so a line was gone in
0.22 s. Nothing she did survived to her next experience, her trials come seconds
apart, and her mouth was therefore always one shape: measured 2026-09-12, of the
seven articulators she used 55 different values of `loud` and exactly one of
everything else. That is "oh, oh, oh" and it is the only sound one shape can
make. **Output(0) cannot become Output(+1) if her body moves the lines between
them.**

### One step, and it is 11 ms

**ELEVEN MILLISECONDS IS THE STEP FOR EVERYTHING, NAMING INCLUDED.** One piece
of sound is one tick of real time; one level, one row in her memory, per piece.
The STEP from one piece to the next is her tick and nothing else.

**Her ear LISTENS further back than it steps** (`hearing.LISTENS`, four ticks).
A frequency cannot be told in less time than about one of its cycles: in 11 ms
of air nothing closer than ~90 Hz is separable, and fifteen of her 24 bands are
narrower than that. Measured 2026-09-12 with her own judge (vosk): named from
11 ms of air, 6 of his 19 words survived her ear; named from the last 44 ms,
12. The piece is 11 ms, the step is 11 ms; what her ear listens to for each
piece is the last four --- his air and her own alike. A cochlea does the same.

**The door only cuts.** His order, 2026-09-12: *"first converted to her
register, then cut to eleven millisecond pieces --- just once."* The outside is
brought into her register where it arrives (`window.say`, by `toHer`:
frequencies by REGISTER, time untouched) and `door` after it only cuts. The
measured register for the mouth she has is **1.0** --- every step above it
made her worse on every count --- so today `toHer` is the identity and the
structure stands ready for any register he names.

### One gate

**EVERY SOUND IS NAMED BY THE SAME EAR, HER OWN INCLUDED.** Her ears are the
spindle for her mouth, and she has one line for what a sound is. His air and
hers both become her bands by `bands_from_pcm`, over the same `LISTENS` of air,
and both are named by the same `align`. His enters through `door` (the door
only cuts; the register, if any, was applied once at `window.say`); her own air
does not pass the door --- it is already hers --- and is mixed into her ear
bands with everyone else's (`alive.py`, *"one input, and everything is in
it"*). Two voices at once are one level, because that is what the room sounded
like. There is no echo channel and no ruler.

### An experience

4. **An experience is two states**: its start whole, and its end kept as only
   the lines that differ. Nothing between them is stored — a tick where zero
   became zero carries no information.
5. **The next begins where the last ended**, so it holds the last inside
   itself. The nesting is not a structure that gets built; it falls out of
   starting where you stopped.
6. An experience holds the one it began from (`Exp.exp`, `parts`), down to the
   smallest, which holds pieces of input and output as levels.
7. **Novelty makes it one**: a value a line has never shown bursts by the size
   of the change, and from the next try it is usual. That is her habituation
   and nothing else is needed for it.
8. **A step that brings nothing makes no experience** — she keeps discovering
   until something changes.
9. **Nothing is thrown away and nothing is forgotten.** What is old is merely
   many turns away. No cap, no drop; her whole memory is written to the record
   and read back at a wake.

### What she can tell apart

**HER GRAIN IS HER MIND'S ONE ANSWER TO "DID ANYTHING CHANGE?"** --- and his
word for it, 2026-09-12, is **0.01** (`self.resolution` in `body/alive.py`,
served to her mind). Her mind uses it on every line for four things: a value a
line has never shown (novelty), a change worth keeping in an experience, the
rise that closes one, and the step her record is written at (`store._q`).

It is **not** the step her muscles move by. A trial moves by her deficit,
`max(resolution, (1 - state) / 2)` in `discover.py`, normally 0.05 to 0.5 --- so
her smoothness is her body's, and this number almost never touches it.

Measured on her own state, tick to tick, 2026-09-12:

- at **1/512** (0.002 --- the count of the *old* ear names, which no longer
  exist; her similarity is a continuous level now) half of all her ticks read
  as a change, an experience closed every ~100 ms, and her mind felt only
  68-76 of her 90 ticks a second: her state's own wobble was being read as
  news.
- at **0.05**, her muscle's step, one tick in 140 read as a change, and his
  speech (levels 0.12-0.87) held only ~15 distinguishable sounds.
- at **0.01** (his word), one tick in 22, ~75 distinguishable sounds, and her
  mind felt 85-86 of 90. Two sounds closer than 0.01 in level are one sound to
  her.

**A number that serves her mind's sense of change is hers to set by how her
state actually behaves, not by how finely her ear or her muscles move.**

### What moves her

10. **ONE VALUE: her state above the floor she has learned she can feel.** That
    one value is what cuts an experience — a rise she can feel, then a fall she
    can feel. **Value is not computed and not stored**: her hormone lines are
    already in the row, so what an experience did to her is already in the
    record. There is no rating, no weight, no worth.
11. **The floor only rises** (`Hormones.floor()` and dopamine's own set-point,
    `_usualEase`; his word 2026-09-12: *"once she felt better, her state is
    never on the floor again --- as much profit before, as much she wants"*).
    The same comfort stops paying and she must reach further. A rise still
    counts at the worst of her life because the cut measures the rise over the
    run's OWN lowest, not over the floor. Where the law has hands is
    dopamine's set-point: sitting below it drags her state, and a lower state
    makes her trials bigger --- the craving.
12. **Boredom works like hunger**: nothing changing → her state falls → her
    deficit rises → she works harder.
13. **Effort is how far a level moves and how often.** Her hormones set her
    speed and her effort, **never her decisions**. Her deficit is
    `(1 - state) / 2` — a half of her at neutral, all of her at her worst, and
    never zero.

### Recreating one

14. **She picks the experience with the best profit** --- the state she felt
    the moment it switched. No closeness, no song rule (his word, 2026-09-12).
    An experience's end is stored as the CHANGE on each line that moved (0.1 ->
    0.5 is kept as 0.4). Its profit falls to what she actually felt when
    replaying it brought less --- including when there was nothing of hers in
    it to replay --- so a wrong experience sinks and the next gets its turn.
    No short and long memory, no 26 and 26: the tree keeps her from the wrong
    turns by itself.
15. **Retrying it is ONE STEP**: she posts the stored change, from where she is
    now, and her flesh walks the way. Her muscles cap force, not destination.
    Retrying pays nothing (novelty), so the run stays open and she goes on
    discovering --- the same line one step further, or the next --- until
    something new closes a bigger experience that holds the retry inside it.
16. **If a change came from outside and she has never made that output, she
    does not have that piece.** It is a level she has seen and cannot produce,
    until discovery gives her the output that makes it.

### Her voice

17. **Her ears are the spindle for her mouth.** There is no echo channel — her
    own air arrives at her ears with everyone else's, and what makes it hers is
    her own `loud` muscle sitting in the same row.
18. **Her voice is muscles.** She practises it exactly like her body: one
    muscle to its end, then the next — and **the shape is a chain, not a held
    pose**. Repeating one alone brings nothing, so she chains them.
19. `loud` (motor 31) is her breath. Her folds are a **glottal flow pulse**
    (not a sawtooth; `speech._glottis`, 2026-09-12), which her lips
    differentiate — that is what gives her a voice's spectral slope instead of
    the harsh buzz. Noise lives at her folds as well as at a pinch, so she can
    breathe and make unvoiced sounds.
20. Her tract reaches the **whole adult vowel table × 1.3** (`muscles.py`:
    F1 350–1000, F2 1100–3000, F3 riding with `front`) — a child's range,
    including the back vowels. Her larynx is `PITCH_HZ` 250–600, a newborn's.
21. **No prepared table, no naming a sound, no aiming, and no ruler.** A sound
    is named by the shape of the piece itself.

### And what is deliberately absent

**No alarm. No `NEW` line. No prediction. No judge. No helper. No salience. No
credit, no averaging, no `worth`. No trajectory replay.** Every one of these was
removed on his word. If you find yourself wanting one back, you have
misunderstood something — ask him.

---

# RULES THAT DECIDE EVERY CHANGE

1. **No designed switches in her mind; no helper; nothing named to her.**
2. **Never hand her a conclusion.** A line stores what her body carries.
3. **No hard-coded numbers in her.** A threshold is her body's resolution, a
   rate is a share, a time is his seconds through `TICK_SECONDS`. A number
   somebody picked is a defect even if it works.
4. **Trace the code that exists** before stating how she works. Run before
   quoting a number. Never report dead code as if it runs.
5. **Measure before and after.** A change without a number on both sides is an
   opinion.
6. **Her laws move only on his explicit word**, in the session. `mind/` and
   `body/` are his.
7. **Commits need his confirm**, with `measure.check` green at HEAD first.
8. **A zero must prove it had data. A constant signal is no signal.** One life
   is not a bar — two births read differently.

---

# THE GATES

Before you say anything is finished, and before any commit he confirms:

```
py.cmd -m measure.check    ->  "everything she is checked for holds"
py.cmd -m measure.voice    ->  "THE VOICE HOLDS"
```

**`measure.check` does not import `mind/`.** It covers her body and the
instruments. A broken mind passes it. So also run:

```
py.cmd -c "import sys; sys.path.insert(0,'mind'); import life, hormones, discover, store, structure, sandbox"
```

Her voice is locked sample-for-sample against
`measured/matilda_through_her_2026-09-12.wav` (the flow folds and the open
tract, on his word 2026-09-12); the 09-04 and 09-11 renderings are kept beside
it. **Any change to her mouth fails that lock by design** — it needs his ear
and his word; only then is the new rendering made the lock, and the other two
floors (rattle, length) must still hold.

---

# THE SHAPE

```
her.py                    one command
  └─ sandbox/app.py       HER BODY + the HTTP server (8090): /state /ticks /act /felt /world /life /log
       ├─ body/alive.py   her body assembled; oneTick()
       ├─ body/teacher.py her mother (off unless he turns her on)
       └─ mind/life.py    HER MIND, a child process: reads /ticks, feels every tick, posts to /act
```

`mind/` is six files: `life`, `hormones`, `discover`, `store`, `structure`,
`sandbox`. `mind/life.py` holds the architecture above (HOW SHE WORKS) and
nothing else.

**Instruments, testing only** (`measure/`; she never imports them):
`alphabet` --- her possibilities, swept by her own law with one carried mouth;
`saidback` --- her memory said back through her own mouth, judged by praat;
`hervoice` --- HIS WORDS IN HER VOICE, the path he chose 2026-09-12 (his air ->
her ear's 24 levels a piece -> her own mouth with three lines following his
voicing and pitch -> his envelope on a small child's tract, `--tract 1.5`).
Every knob is at the top of that file with what it does; change it there and
nothing else moves. It is the measure of what her memory would give if a sound
reached it as her ear's levels instead of one number.

The record is DuckDB in `mind/lives/`: every tick of her life, written as a
start point and then changes. Her page writes its own exceptions and its heap
into her log through `/log` — `grep "^PAGE" <her log>` is the autopsy when the
browser dies.

---

# WHAT IS STILL OPEN

Honest, so nobody rediscovers it as a surprise:

- **The bottle.** `sandbox/app.py` places it from a gaze vector with its
  vertical component discarded, and the placement direction is fixed across all
  30 retries — so on many placements it spawns inside the mattress at 6 cm and
  she can neither see nor reach it. She starves on long runs. **This is the one
  thing standing between her and a real test.** His word needed; it is her
  world, not her mind.
- **Her sound in memory is one level per piece — the one open voice decision.**
  Measured 2026-09-12 with her own judge (vosk): her ear keeps 12 of his 19
  words, her memory keeps 1. So her own mouth gives back his rhythm, loudness
  and pauses in her voice, and none of his words. Driving her own folds and
  tract from her ear's 24 levels a piece instead of one gave 20 of 41 words
  back in a child's voice (`measure.hervoice`). Whether a sound reaches her
  memory as several levels rather than one is his to decide; it is the only
  thing between her and copying his words.
- **Her look render.** She now attempts a look every tick (`LOOKS_PER_SECOND =
  1/TICK_SECONDS`); the eye runs on its own thread and `_eyeBusy` holds it off
  until the last look is done, so it never blocks her clock. But the full-room
  render (her body, two windows, every thing, the reflection) is ~12–65 ms of
  unfused GPU passes, so under load `behind` bounces ~0.4–2.0 s and recovers.
  Two ways to hold `behind` at 0: fuse the render into one kernel
  (`body/light.py`), or set `LOOKS_PER_SECOND` to ~30. His word.
- **The bottle.** `sandbox/app.py` places it from a gaze vector with its
  vertical component discarded, direction fixed across all 30 retries — so it
  often spawns inside the mattress at 6 cm, unseeable and unreachable, and she
  starves on long runs. Her world, not her mind; his word.
- Numbers still his to set: `RESTS`' 1.5, `PRESS`'s build/drain balance,
  `FLOOR_HALF_LIFE`, and whether asphyxia should outrank hunger.

Fixed 2026-09-12, no longer open: the `--keep` wake crash (`store.hands()`
tuple); balance dropped so sound-in is four lines; her folds and tract; her
grain 0.01; the register applied once at the door; the voice re-lock.
