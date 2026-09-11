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
2. **A thing seen is what it is and where** — similarity, x, y. **A sound is
   the same three** — how alike, how loud, which side. Never one line per id.
3. **Her hormones are lines like any other. Her own orders are lines like any
   other.**

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

### What moves her

10. **ONE VALUE: her state above the floor she has learned she can feel.** That
    one value is what cuts an experience — a rise she can feel, then a fall she
    can feel. **Value is not computed and not stored**: her hormone lines are
    already in the row, so what an experience did to her is already in the
    record. There is no rating, no weight, no worth.
11. **The floor follows her state** (`Hormones.floor()`), so the same comfort
    stops paying and she must reach further. His addiction law. It is also why
    the measure is floor-relative: the floor comes down with her, so a rise is
    a rise wherever she is.
12. **Boredom works like hunger**: nothing changing → her state falls → her
    deficit rises → she works harder.
13. **Effort is how far a level moves and how often.** Her hormones set her
    speed and her effort, **never her decisions**. Her deficit is
    `(1 - state) / 2` — a half of her at neutral, all of her at her worst, and
    never zero.

### Recreating one

14. She looks for the experience whose **start is nearest her now**.
15. **Retrying it is ONE STEP**: she posts the stored difference and her flesh
    walks the way. Her muscles cap force, not destination.
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
19. `loud` (motor 31) is her breath. Noise lives at her folds as well as at a
    pinch, so she can breathe and make unvoiced sounds.
20. **No prepared table, no naming a sound, no aiming, and no ruler.** A sound
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
`measured/matilda_through_her_2026-09-11.wav`, re-locked on his word that day
after he heard it against his 2026-09-04 rendering, which is kept beside it.
**Any change to her mouth fails that lock by design** — it needs his ear and
his word, not a re-lock.

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
`sandbox`. `mind/life.py` is 492 lines and holds the twenty points above and
nothing else.

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
- **Her view is still many blobs per tick, not one picture.** His design is one
  picture, one similarity, x and y. `body/parts.py` and `body/bind.py` do
  segmentation his architecture does not need. Discussed 2026-09-11, not
  decided.
- Numbers still his to set: `RESTS`' 1.5, `PRESS`'s build/drain balance,
  `FLOOR_HALF_LIFE`, and whether asphyxia should outrank hunger.
