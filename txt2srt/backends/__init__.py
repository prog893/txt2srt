"""Alignment backends.

A backend's whole job: given audio and a Doc, return a start and end time for
as many stream positions as it can. It never returns text.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..transcript import Doc

BACKENDS = ("whisper", "vad")


@dataclass
class Times:
    """Per-character times over Doc.stream. NaN means "this backend had nothing to say"."""
    start: np.ndarray            # (N,) float64
    end: np.ndarray              # (N,)
    prob: np.ndarray             # (N,) float32, backend-defined confidence
    backend: str
    model: str = ""

    @classmethod
    def empty(cls, n: int, backend: str, model: str = "") -> "Times":
        nan = np.full(n, np.nan)
        return cls(nan.copy(), nan.copy(), np.zeros(n, dtype=np.float32), backend, model)

    def set(self, i0: int, i1: int, start: float, end: float, prob: float = 0.0) -> None:
        self.start[i0:i1] = start
        self.end[i0:i1] = end
        self.prob[i0:i1] = prob

    @property
    def timed(self) -> int:
        return int((~np.isnan(self.start)).sum())

    def fill_gaps(self) -> None:
        """Give silent characters (punctuation, spaces, dropped glyphs) a position.

        Interpolated between the neighbours that do have times, so a cue that
        begins with an opening bracket still starts when its first spoken
        character does.
        """
        n = len(self.start)
        known = np.flatnonzero(~np.isnan(self.start))
        if known.size == 0:
            raise RuntimeError("no character received a timestamp")
        first, last = known[0], known[-1]
        self.start[:first] = self.start[first]
        self.end[:first] = self.start[first]
        self.start[last + 1:] = self.end[last]
        self.end[last + 1:] = self.end[last]
        for a, b in zip(known[:-1], known[1:]):
            if b == a + 1:
                continue
            lo, hi = self.end[a], self.start[b]
            span = max(hi - lo, 0.0)
            gap = b - a
            for k in range(1, gap):
                t = lo + span * (k - 0.5) / gap
                self.start[a + k] = t
                self.end[a + k] = t
        # Monotonicity: DTW-based backends can emit a word that starts before
        # its predecessor ended. Cues are built from min/max over a range, so a
        # single inversion would otherwise widen a cue across a neighbour.
        np.maximum.accumulate(self.start, out=self.start)
        np.maximum.accumulate(self.end, out=self.end)
        np.minimum(self.start, self.end, out=self.start)
        assert not np.isnan(self.start).any()


def align(backend: str, audio: np.ndarray, doc: Doc, **opts) -> Times:
    if backend == "whisper":
        from .whisper import align as fn
    elif backend == "vad":
        from .vad import align as fn
    else:
        raise SystemExit(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    times = fn(audio, doc, **opts)
    times.fill_gaps()
    return times
