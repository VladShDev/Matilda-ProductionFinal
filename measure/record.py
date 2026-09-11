"""HER RECORD, READ WHOLE --- for the instruments.

The record keeps a start point per experience and only the changes after it
(`mind/store.py`, his word of 2026-09-06); an instrument that wants ticks
gets them whole through `frames`, one at a time, in time order.  Nothing here
touches her.

    from measure.record import frames
    for t in frames("mind/lives/NAME.duckdb"):
        t.time, t.exp.id, t.plan.id, t.life.input..., t.life.output...
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "mind")):
    if p not in sys.path:
        sys.path.insert(0, p)

from store import Store                                            # noqa: E402


def frames(path: str):
    """Every tick of a life, whole, oldest first, streamed."""
    s = Store(path, read=True)
    try:
        for t in s.walk():
            yield t
    finally:
        s.close()


def count(path: str) -> int:
    s = Store(path, read=True)
    try:
        return s.count()
    finally:
        s.close()
