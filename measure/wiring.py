"""WHAT IS BUILT AND JOINED TO NOTHING --- read off the code, not off memory.

    python -m measure.wiring

**THIS PROJECT'S STANDING FAILURE IS NOT BUGS.  IT IS CORRECT MECHANISMS THAT
NOBODY CALLS**, or that are called and whose answer nobody reads.  They never
raise, the suite stays green, and the thing sits there for weeks looking done.
Counted in one night, 2026-08-25/26, in a tree eight files long:

    Her.mind          existed since her body was assembled, NOTHING SERVED IT
    Watched.oneTick   renamed out from under its override --- no frames recorded
    Her.snap/pose     raise NameError; nothing calls them, so nothing said
    enterExp          re-entered a run and made 2 experiences out of 952 changes
    align's offset    computed every tick and dropped on the floor
    KEEPS/FORGETS     per-LOOK constants left over from the 1.5 s clock

So this asks three questions of the code itself:

    1. WHAT IS NEVER CALLED     a def in `body/` or `mind/` that nothing names
    2. WHAT IS THROWN AWAY      a returned value assigned to `_`
    3. WHAT IS WRITTEN, NEVER READ    a `self.x = ...` nothing ever reads back

It is deliberately dumb.  A name matched anywhere counts as a caller, so it
UNDER-reports --- everything it does list is worth looking at, and a clean run
does not mean everything is joined.  `measure/works.py` and `measure/words.py`
answer the other half: whether the joined-up thing does anything.
"""
from __future__ import annotations

import ast
import collections
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: HER, and only her.  `measure/` is instruments and `sandbox/` is the watcher;
#: neither is allowed to be the reason something of hers looks alive.
MINE = ("body", "mind")        # brain/ was deleted 2026-09-02; mind/ scanned since 2026-09-03
#: ...but a caller anywhere in the tree counts, including the watcher.
ALL = ("body", "mind", "sandbox", "measure")

SKIP = {"main", "__init__", "__enter__", "__exit__", "__repr__", "__post_init__"}


def files(where):
    for root, _, names in os.walk(os.path.join(HERE, where)):
        if os.path.basename(root) == "tests":       # pytest names those
            continue
        if "__pycache__" in root:
            continue
        for n in sorted(names):
            if n.endswith(".py"):
                yield os.path.join(root, n)


def main() -> int:
    defs: dict = {}
    used: collections.Counter = collections.Counter()
    dropped: list = []
    written: dict = {}
    read: collections.Counter = collections.Counter()

    for where in ALL:
        for path in files(where):
            rel = os.path.relpath(path, HERE).replace("\\", "/")
            with open(path, encoding="utf-8") as f:
                src = f.read()
            try:
                tree = ast.parse(src)
            except SyntaxError as why:
                print(f"  {rel} will not parse: {why}")
                continue
            mine = where in MINE
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if mine and node.name not in SKIP and not node.name.startswith("__"):
                        defs.setdefault(node.name, f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Name):
                    used[node.id] += 1
                elif isinstance(node, ast.Attribute):
                    used[node.attr] += 1
                    # A READ IS A READ WHEREVER IT IS.  This counted only loads
                    # through `self.`, and reported `window.up` as written and
                    # never read --- while HER EYE reads it at `light.py:482` to
                    # rebuild the screen's roll, which is a measured fix.  An
                    # attribute is not private to the class that writes it.
                    if isinstance(node.ctx, ast.Load):
                        read[node.attr] += 1
                    elif mine and isinstance(node.value, ast.Name)                             and node.value.id == "self":
                        written.setdefault(node.attr, f"{rel}:{node.lineno}")
                # a returned value assigned to `_`
                if isinstance(node, ast.Assign) and mine:
                    for t in node.targets:
                        names = (t.elts if isinstance(t, (ast.Tuple, ast.List))
                                 else [t])
                        for k, one in enumerate(names):
                            if isinstance(one, ast.Name) and one.id == "_":
                                call = node.value
                                what = ""
                                if isinstance(call, ast.Call):
                                    f = call.func
                                    what = (f.attr if isinstance(f, ast.Attribute)
                                            else getattr(f, "id", ""))
                                dropped.append((f"{rel}:{node.lineno}", what, k))

    print("1. NEVER CALLED --- a def in body/ or mind/ that nothing names\n")
    # ZERO, NOT ONE.  This read `<= 1` and listed 85 things including `align`,
    # `hear`, `openExp` and `settle` --- all of which are called exactly once, so
    # a threshold of one flags every mechanism with a single caller, which in a
    # tree with one implementation of everything is most of them.  An instrument
    # that cries at everything is an instrument nobody reads.
    lonely = sorted((n, w) for n, w in defs.items() if used[n] == 0)
    for name, where in lonely:
        print(f"   {where:<34} {name}")
    if not lonely:
        print("   nothing")

    print(f"\n2. THROWN AWAY --- a returned value assigned to `_`\n")
    for where, what, k in dropped:
        print(f"   {where:<34} value {k} of {what or 'a call'}()")
    if not dropped:
        print("   nothing")

    print(f"\n3. WRITTEN AND NEVER READ BACK --- `self.x = ...` nothing loads\n")
    quiet = sorted((n, w) for n, w in written.items() if not read[n])
    for name, where in quiet:
        print(f"   {where:<34} self.{name}")
    if not quiet:
        print("   nothing")

    print(f"\n  {len(lonely)} never called, {len(dropped)} thrown away, "
          f"{len(quiet)} written and never read.")
    print("  It UNDER-reports on purpose: a name matched anywhere counts as a")
    print("  caller.  Everything listed is worth looking at; a clean run is not")
    print("  a promise that everything is joined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
